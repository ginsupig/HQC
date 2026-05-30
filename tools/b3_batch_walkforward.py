"""
B3 — Batch walk-forward across B2 survivors at the deployed parameter set.

Roadmap reference: Workstream B, task B3. For every pair that B2 flagged
``screen_pass = 1``, runs the Kalman pairs walk-forward (the same harness
the deployed JPM/BAC pair was validated on) at a FIXED parameter set,
captures per-window OOS returns, and aggregates a pooled verdict.

What B3 does:
  1. Read the B2 output CSV. Optionally filter to ``screen_pass = 1``.
  2. For each surviving pair, locate the two leg CSVs in ``--data-dir``.
  3. Run walk-forward at the user-supplied (or default = deployed)
     parameter set: entry_z, exit_z, delta, ve, train/test days,
     slippage, borrow, max hold minutes.
  4. Per pair: bootstrap 95% CI on mean OOS per-window return; report
     Newey-West-corrected one-sample p(mean > 0).
  5. Pooled: take the per-pair means as observations; report bootstrap
     CI and Newey-West p of the pair-mean distribution. This is the
     family-corrected unit of analysis per the F1 §3.1 / roadmap rule
     ("decision is pooled across symbols at a fixed config").
  6. Write a CSV with one row per pair + a summary line; also write a
     per-pair returns JSON for downstream tools (B4 / A1).

What B3 does NOT do:
  - It does NOT sweep parameters. A2 already covers that on a single
    pair; B3's job is to test PARAMETER STABILITY across multiple pairs
    at the deployed config.
  - It does NOT make a deployment decision. A1's family-corrected bar
    (Bonferroni or BH-FDR at family-size = K survivors) is the gate.
    B4 pre-registers that bar BEFORE B3 results are seen.
  - It does NOT re-screen. B2 is the screen; B3 trusts B2's output.

Family-size accounting:
  K = number of pairs scored here. A1's corrected p-value bar uses K.
  If you want the deployed pair JPM/BAC to count toward A1's family
  alongside B3's new candidates, K must include it. The default behavior
  is to score every pair in the B2 CSV with ``screen_pass = 1`` and
  report K in the summary line so B4 can lock the corrected threshold.

Usage:
  python tools/b3_batch_walkforward.py \\
      --input b2_financials.csv \\
      --data-dir data/alpaca \\
      --workers 4 \\
      --output b3_financials.csv

  # Resume an interrupted run (same --output)
  python tools/b3_batch_walkforward.py \\
      --input b2_financials.csv --resume --output b3_financials.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the same backtest path A2 and the deployment use, so B3's per-pair
# scores are directly comparable to A1/A2/A3.
from analyze_walkforward import (  # noqa: E402
    _bootstrap_mean_ci,
    _one_sample_p_greater_than_zero,
)
from backtest_runner import BacktestConfig, run_backtest  # noqa: E402
from tools.b1_universe_clustering import _find_csv  # noqa: E402
from walkforward_pairs import _build_cfg, _date_col, _run_window_slice, _windows  # noqa: E402


# --------------------------------------------------------------------------- #
# Worker — one pair, full walk-forward
# --------------------------------------------------------------------------- #
def _worker_run_pair(task: dict) -> dict:
    """Run the full walk-forward for one (y, x) pair at fixed parameters.

    Module-level so multiprocessing.Pool can pickle it. Returns
    ``{y, x, n_windows, returns_pct, total_pnl, total_trades, error}``.
    """
    args = argparse.Namespace(**task["args"])
    y_csv = Path(task["csv_y"])
    x_csv = Path(task["csv_x"])
    out = {
        "y": args.symbol_y,
        "x": args.symbol_x,
        "n_windows": 0,
        "returns_pct": [],
        "total_pnl": 0.0,
        "total_trades": 0,
        "error": "",
    }
    try:
        df_y = pd.read_csv(y_csv)
        df_x = pd.read_csv(x_csv)
        days = sorted(
            set(_date_col(df_y).dropna().dt.normalize().tolist())
            & set(_date_col(df_x).dropna().dt.normalize().tolist())
        )
        windows = _windows(days, args.train_days, args.test_days)
        if not windows:
            out["error"] = "no_walkforward_windows"
            return out
        for (_train_start, _train_end, test_start, test_end) in windows:
            result = asyncio.run(
                _run_window_slice(df_y, df_x, args, test_start, test_end)
            )
            ret = result.get("total_return_pct")
            if isinstance(ret, (int, float)) and math.isfinite(ret):
                out["returns_pct"].append(float(ret))
            pnl = result.get("realized_pnl") or 0.0
            if isinstance(pnl, (int, float)):
                out["total_pnl"] += float(pnl)
            trades = result.get("trades") or 0
            if isinstance(trades, (int, float)):
                out["total_trades"] += int(trades)
        out["n_windows"] = len(windows)
    except Exception as e:  # noqa: BLE001 — surface any worker failure
        out["error"] = f"{type(e).__name__}:{e}"
    return out


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def _pair_stats(rets: List[float], alpha: float, n_boot: int) -> dict:
    if not rets:
        return {
            "mean_pct": "", "ci_lo": "", "ci_hi": "", "raw_p": "",
        }
    mean, ci_lo, ci_hi = _bootstrap_mean_ci(rets, n_boot=n_boot, alpha=alpha)
    p = _one_sample_p_greater_than_zero(rets)
    return {
        "mean_pct": f"{mean:.6f}",
        "ci_lo": f"{ci_lo:.6f}",
        "ci_hi": f"{ci_hi:.6f}",
        "raw_p": f"{p:.6f}",
    }


# --------------------------------------------------------------------------- #
# Resume helpers
# --------------------------------------------------------------------------- #
def _load_existing_outputs(output: Path) -> Dict[Tuple[str, str], dict]:
    if not output.exists():
        return {}
    out: Dict[Tuple[str, str], dict] = {}
    with open(output, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("y") or not row.get("x"):
                continue
            out[(row["y"], row["x"])] = row
    return out


def _append_row(output: Path, fieldnames: List[str], row: dict) -> None:
    write_header = not output.exists() or output.stat().st_size == 0
    with open(output, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _build_task_args(args: argparse.Namespace, y_symbol: str, x_symbol: str) -> dict:
    """Build the argparse.Namespace-shaped dict that walkforward_pairs'
    helpers expect (so _build_cfg + _run_window_slice work unchanged)."""
    return {
        "symbol_y": y_symbol,
        "symbol_x": x_symbol,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "initial_capital": args.initial_capital,
        "pair_entry_z": args.pair_entry_z,
        "pair_exit_z": args.pair_exit_z,
        "pair_delta": args.pair_delta,
        "pair_ve": args.pair_ve,
        "pair_max_leg_staleness_sec": args.pair_max_leg_staleness_sec,
        "pair_cooldown_seconds": args.pair_cooldown_seconds,
        "pair_nominal_stop_pct": args.pair_nominal_stop_pct,
        "pair_target_dollar_notional": args.pair_target_dollar_notional,
        "sim_max_hold_minutes": args.sim_max_hold_minutes,
        "sim_stop_buffer_ticks": args.sim_stop_buffer_ticks,
        "slippage_bps_per_side": args.slippage_bps_per_side,
        "commission_per_share": args.commission_per_share,
        "commission_min_per_trade": args.commission_min_per_trade,
        "sec_fee_rate": args.sec_fee_rate,
        "short_borrow_apr": args.short_borrow_apr,
        "skip_train": True,  # we don't use train results in pooled summary
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="B2 output CSV (b2_*.csv).")
    p.add_argument("--data-dir", type=Path,
                   default=_REPO_ROOT / "data" / "alpaca")
    p.add_argument("--all-rows", action="store_true",
                   help="Score every row in --input, not just screen_pass=1.")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--returns-json", type=Path, default=None,
                   help="If set, write per-pair per-window returns to this JSON. "
                        "Defaults to <output>.returns.json.")
    p.add_argument("--resume", action="store_true",
                   help="Skip pairs already present in --output.")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)

    # Pair-strategy parameter set (defaults match the deployed config).
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--pair-entry-z", type=float, default=1.5)
    p.add_argument("--pair-exit-z", type=float, default=0.4)
    p.add_argument("--pair-delta", type=float, default=1e-4)
    p.add_argument("--pair-ve", type=float, default=1e-3)
    p.add_argument("--pair-max-leg-staleness-sec", type=float, default=30.0)
    p.add_argument("--pair-cooldown-seconds", type=float, default=120.0)
    p.add_argument("--pair-nominal-stop-pct", type=float, default=0.02)
    p.add_argument("--pair-target-dollar-notional", type=float, default=10_000.0)
    p.add_argument("--sim-max-hold-minutes", type=int, default=240)
    p.add_argument("--sim-stop-buffer-ticks", type=float, default=0.0)
    p.add_argument("--slippage-bps-per-side", type=float, default=1.5)
    p.add_argument("--commission-per-share", type=float, default=0.0)
    p.add_argument("--commission-min-per-trade", type=float, default=0.0)
    p.add_argument("--sec-fee-rate", type=float, default=0.000008)
    p.add_argument("--short-borrow-apr", type=float, default=0.0025)

    args = p.parse_args()
    returns_json_path = args.returns_json or args.output.with_suffix(".returns.json")

    # Load B2 input
    with open(args.input, newline="", encoding="utf-8") as f:
        b2_rows = list(csv.DictReader(f))
    if not b2_rows:
        raise SystemExit(f"No rows in {args.input}")

    tasks: List[dict] = []
    skipped_filter = 0
    skipped_missing = 0
    for r in b2_rows:
        if not args.all_rows and r.get("screen_pass") != "1":
            skipped_filter += 1
            continue
        y, x = r["y"], r["x"]
        y_csv = _find_csv(args.data_dir, y)
        x_csv = _find_csv(args.data_dir, x)
        if y_csv is None or x_csv is None:
            print(f"  WARN: missing CSV for {y}/{x} — skipping", flush=True)
            skipped_missing += 1
            continue
        tasks.append({
            "args": _build_task_args(args, y, x),
            "csv_y": str(y_csv),
            "csv_x": str(x_csv),
        })

    print(
        f"B3: input={args.input}  tasks={len(tasks)}  "
        f"skipped_filter={skipped_filter}  missing_csv={skipped_missing}",
        flush=True,
    )

    existing = _load_existing_outputs(args.output) if args.resume else {}
    if existing:
        before = len(tasks)
        tasks = [t for t in tasks if (t["args"]["symbol_y"], t["args"]["symbol_x"]) not in existing]
        print(f"  resume: skipping {before - len(tasks)} already-scored pairs", flush=True)

    fieldnames = [
        "y", "x", "n_windows", "n_returns",
        "total_pnl", "total_trades",
        "mean_pct", "ci_lo", "ci_hi", "raw_p",
        "error",
    ]

    per_pair_returns: Dict[str, List[float]] = {}
    # Reload existing returns from json if resuming
    if existing and returns_json_path.exists():
        try:
            per_pair_returns = json.loads(returns_json_path.read_text(encoding="utf-8"))
        except Exception:
            per_pair_returns = {}

    pool_ctx = mp.get_context("spawn")
    t0 = time.time()
    completed = 0

    if tasks:
        with pool_ctx.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_worker_run_pair, tasks):
                completed += 1
                rets = result["returns_pct"]
                stats = _pair_stats(rets, args.alpha, args.bootstrap)
                row = {
                    "y": result["y"],
                    "x": result["x"],
                    "n_windows": result["n_windows"],
                    "n_returns": len(rets),
                    "total_pnl": f"{result['total_pnl']:.2f}",
                    "total_trades": result["total_trades"],
                    "error": result["error"],
                    **stats,
                }
                _append_row(args.output, fieldnames, row)
                per_pair_returns[f"{result['y']}|{result['x']}"] = rets
                returns_json_path.write_text(
                    json.dumps(per_pair_returns, default=str),
                    encoding="utf-8",
                )
                elapsed = time.time() - t0
                err_tag = f"  ERR={result['error']}" if result["error"] else ""
                print(
                    f"[B3] {completed}/{len(tasks)} {result['y']}/{result['x']}  "
                    f"n_windows={result['n_windows']}  mean={stats['mean_pct'] or 'n/a'}%  "
                    f"p={stats['raw_p'] or 'n/a'}  pnl=${result['total_pnl']:.0f}"
                    f"  t={elapsed:.0f}s{err_tag}",
                    flush=True,
                )

    _print_pooled_summary(args.output, args.alpha, args.bootstrap)


def _print_pooled_summary(output: Path, alpha: float, n_boot: int) -> None:
    if not output.exists():
        return
    with open(output, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("mean_pct")]
    if not rows:
        print("B3 summary: no usable rows in output.")
        return

    pair_means: List[float] = []
    for r in rows:
        try:
            pair_means.append(float(r["mean_pct"]))
        except (TypeError, ValueError):
            continue
    if not pair_means:
        print("B3 summary: no per-pair means could be parsed.")
        return

    pooled_mean, pooled_lo, pooled_hi = _bootstrap_mean_ci(
        pair_means, n_boot=n_boot, alpha=alpha
    )
    pooled_p = _one_sample_p_greater_than_zero(pair_means)

    print()
    print("=" * 70)
    print("B3 pooled verdict (family-size unit of analysis)")
    print("=" * 70)
    print(f"  K = {len(pair_means)} pairs scored")
    print(f"  pooled mean per-window return = {pooled_mean:+.4f}%")
    print(f"  bootstrap {int((1 - alpha) * 100)}% CI on pooled mean: "
          f"[{pooled_lo:+.4f}%, {pooled_hi:+.4f}%]")
    print(f"  Newey-West p(mean > 0) on pooled distribution = {pooled_p:.5f}")
    print()
    print("  Family-corrected thresholds (Bonferroni at family-size = K):")
    print(f"    alpha=0.05    -> per-pair threshold = {0.05 / max(1, len(pair_means)):.5f}")
    print(f"    alpha=0.01    -> per-pair threshold = {0.01 / max(1, len(pair_means)):.5f}")
    print()
    print("  B4 should freeze this K BEFORE inspecting any per-pair p-values.")


if __name__ == "__main__":
    main()
