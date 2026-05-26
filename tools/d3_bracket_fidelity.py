"""
D3 — Backtest vs live OTO-bracket fidelity audit.

The validated backtest uses backtest_runner._SimulatedExitEngine,
which closes positions on the FIRST of three triggers:
  (1) z-reversion / strategy-side close emission
  (2) price crosses stop_loss_price (mirrors Alpaca OTO bracket)
  (3) hard time stop after sim_max_hold_minutes (default 240 min)

Live trading via Alpaca uses native OTO bracket: only (1) and (2)
fire. There is NO time stop on a live bracket -- positions hold
until z-reversion or EOD liquidator (15:55 ET).

If a meaningful share of backtest trades close via (3), the live
realized PnL will differ from the validated backtest. D3 quantifies
that difference by re-running the walk-forward twice on the SAME
windows:

  with_ts    : --sim-max-hold-minutes 240    (validated config)
  without_ts : --sim-max-hold-minutes 999999 (effectively no time stop;
               only stop_loss + z-reversion + EOD fire)

Then diffs per-window total_return_pct. The expectation per the
README: "results would be ~50% optimistic WITHOUT sim-exit" -- but
that was about backtests with NO exits at all. With the bracket's
stop_loss still active, the live-vs-backtest gap should be much
smaller. D3 puts a number on it.

Usage:
  python tools/d3_bracket_fidelity.py \\
      --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \\
      --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \\
      --train-days 30 --test-days 10 \\
      --workers 4 \\
      --output d3_jpm_bac.csv

Each run = 2 walk-forwards (with_ts, without_ts) = 2 * 47 windows
per pair. ~30 min single-process, ~5-10 min with 4 workers.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest_runner import BacktestConfig, run_backtest  # noqa: E402
from walkforward_pairs import _date_col, _windows  # noqa: E402
from analyze_walkforward import _bootstrap_mean_ci, _one_sample_p_greater_than_zero  # noqa: E402


# Two configurations: validated backtest path vs live-bracket-semantics path.
# 999999 minutes (~1.9 years) is effectively no time stop within any
# walk-forward window. EOD liquidator will fire at 15:55 ET well before this.
SIM_MAX_HOLD_WITH_TS = 240
SIM_MAX_HOLD_WITHOUT_TS = 999999


@dataclass(frozen=True)
class FidelityCell:
    """One walk-forward configuration we want to run."""
    label: str        # "with_ts" or "without_ts"
    sim_max_hold_minutes: int

    def key(self) -> str:
        return self.label


CSV_COLUMNS = [
    "pair", "config", "window_idx",
    "test_start", "test_end",
    "trades", "total_pnl", "total_costs", "return_pct",
]


def _config_for_cell(
    cell: FidelityCell,
    symbol_y: str,
    symbol_x: str,
    args: argparse.Namespace,
) -> BacktestConfig:
    return BacktestConfig(
        symbol=symbol_y,
        benchmark_symbol=symbol_y,
        strategy="pairs",
        initial_capital=args.initial_capital,
        sim_max_hold_minutes=cell.sim_max_hold_minutes,
        slippage_bps_per_side=args.slippage_bps_per_side,
        commission_per_share=0.0,
        commission_min_per_trade=0.0,
        sec_fee_rate=args.sec_fee_rate,
        short_borrow_apr=args.short_borrow_apr,
        hedge_symbol=symbol_x,
        pair_entry_z=args.pair_entry_z,
        pair_exit_z=args.pair_exit_z,
        pair_delta=args.pair_delta,
        pair_ve=args.pair_ve,
        pair_max_leg_staleness_sec=args.pair_max_leg_staleness_sec,
        pair_cooldown_seconds=args.pair_cooldown_seconds,
        pair_nominal_stop_pct=args.pair_nominal_stop_pct,
        pair_target_dollar_notional=args.pair_target_dollar_notional,
    )


async def _walkforward_one_cell(
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    cell: FidelityCell,
    symbol_y: str,
    symbol_x: str,
    pair_label: str,
    args: argparse.Namespace,
) -> List[dict]:
    """Returns one row per OOS window with PnL + trade count + return."""
    sy = _date_col(df_y).dt.date
    sx = _date_col(df_x).dt.date
    cfg = _config_for_cell(cell, symbol_y, symbol_x, args)

    rows: List[dict] = []
    for idx, (_train_start, _train_end, test_start, test_end) in enumerate(windows):
        mask_y = (sy >= test_start.date()) & (sy <= test_end.date())
        mask_x = (sx >= test_start.date()) & (sx <= test_end.date())
        sliced = pd.concat([df_y[mask_y], df_x[mask_x]], ignore_index=True)
        result = await run_backtest(sliced, cfg)
        rows.append({
            "pair": pair_label,
            "config": cell.label,
            "window_idx": idx,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "trades": int(result.get("trades") or 0),
            "total_pnl": round(float(result.get("realized_pnl") or 0.0), 2),
            "total_costs": round(float(result.get("total_costs") or 0.0), 2),
            "return_pct": round(float(result.get("total_return_pct") or 0.0) * 100.0, 6),
        })
    return rows


# Multiprocessing worker state.
_W_DF_Y: Optional[pd.DataFrame] = None
_W_DF_X: Optional[pd.DataFrame] = None
_W_WINDOWS: Optional[List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]] = None
_W_ARGS: Optional[argparse.Namespace] = None


def _worker_init(csv_y: str, csv_x: str, train_days: int, test_days: int, args: argparse.Namespace) -> None:
    global _W_DF_Y, _W_DF_X, _W_WINDOWS, _W_ARGS
    _W_DF_Y = pd.read_csv(csv_y)
    _W_DF_X = pd.read_csv(csv_x)
    days_y = sorted(set(_date_col(_W_DF_Y).dropna().dt.normalize().tolist()))
    days_x = sorted(set(_date_col(_W_DF_X).dropna().dt.normalize().tolist()))
    days = sorted(set(days_y) & set(days_x))
    _W_WINDOWS = _windows(days, train_days, test_days)
    _W_ARGS = args


def _worker_run_cell(args_tuple: Tuple[FidelityCell, str, str, str]) -> List[dict]:
    cell, symbol_y, symbol_x, pair_label = args_tuple
    return asyncio.run(_walkforward_one_cell(
        _W_DF_Y, _W_DF_X, _W_WINDOWS,
        cell, symbol_y, symbol_x, pair_label, _W_ARGS,
    ))


def _append_rows(output_path: Path, rows: List[dict]) -> None:
    is_new = not output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _summarize(rows: List[dict], pair_label: str, alpha: float) -> None:
    """Diff per-window returns between with_ts and without_ts, report verdict."""
    by_config: Dict[str, Dict[int, dict]] = {"with_ts": {}, "without_ts": {}}
    for r in rows:
        if r["pair"] != pair_label:
            continue
        cfg = r["config"]
        if cfg not in by_config:
            continue
        by_config[cfg][r["window_idx"]] = r

    common_idx = sorted(set(by_config["with_ts"]) & set(by_config["without_ts"]))
    if not common_idx:
        print(f"\n[{pair_label}] no common windows between with_ts and without_ts; nothing to diff.")
        return

    rets_with = [by_config["with_ts"][i]["return_pct"] for i in common_idx]
    rets_without = [by_config["without_ts"][i]["return_pct"] for i in common_idx]
    deltas = [w - wo for w, wo in zip(rets_with, rets_without)]
    pnl_with = sum(by_config["with_ts"][i]["total_pnl"] for i in common_idx)
    pnl_without = sum(by_config["without_ts"][i]["total_pnl"] for i in common_idx)
    trades_with = sum(by_config["with_ts"][i]["trades"] for i in common_idx)
    trades_without = sum(by_config["without_ts"][i]["trades"] for i in common_idx)

    mean_with = statistics.mean(rets_with)
    mean_without = statistics.mean(rets_without)
    mean_delta = statistics.mean(deltas)
    delta_mean, delta_lo, delta_hi = _bootstrap_mean_ci(deltas, n_boot=10000, alpha=alpha)

    # Relative gap: how much of with_ts's mean return came from time-stop optimism?
    rel_gap_pct = (mean_delta / mean_with * 100.0) if abs(mean_with) > 1e-9 else float("inf")

    print()
    print("=" * 86)
    print(f"D3 bracket-fidelity audit  {pair_label}")
    print(f"  n_windows={len(common_idx)}  alpha={alpha}")
    print("=" * 86)
    print(f"  with_ts     (sim_max_hold={SIM_MAX_HOLD_WITH_TS}m, validated config):")
    print(f"    mean per-window return = {mean_with:+.4f}%   total PnL = ${pnl_with:+.2f}   trades = {trades_with}")
    print(f"  without_ts  (sim_max_hold={SIM_MAX_HOLD_WITHOUT_TS}m, live-bracket semantics):")
    print(f"    mean per-window return = {mean_without:+.4f}%   total PnL = ${pnl_without:+.2f}   trades = {trades_without}")
    print()
    print(f"  Per-window delta (with_ts - without_ts) = {mean_delta:+.4f}%")
    print(f"    bootstrap 95% CI on delta: [{delta_lo:+.4f}%, {delta_hi:+.4f}%]")
    if abs(mean_with) > 1e-9:
        print(f"    Relative gap: time-stop adds {rel_gap_pct:+.1f}% to the validated mean.")
    else:
        print(f"    Relative gap: n/a (with_ts mean ~ 0)")

    # Verdict thresholds:
    #   |rel_gap| < 10%   -> fidelity good
    #   10-30%            -> ATTENTION
    #   > 30%             -> FLAG (live PnL will diverge meaningfully)
    print()
    if mean_with <= 0:
        print(f"  VERDICT: pair has no positive backtest edge to audit; skip.")
    elif not math.isfinite(rel_gap_pct):
        print(f"  VERDICT: cannot compute relative gap.")
    elif abs(rel_gap_pct) < 10.0:
        print(f"  VERDICT: FIDELITY GOOD ({rel_gap_pct:+.1f}% gap).")
        print(f"           Live OTO-bracket behavior should approximately match the validated backtest.")
    elif abs(rel_gap_pct) < 30.0:
        print(f"  VERDICT: FIDELITY ATTENTION ({rel_gap_pct:+.1f}% gap).")
        print(f"           Modest divergence expected live; budget for ~{rel_gap_pct:+.0f}% PnL difference.")
    else:
        print(f"  VERDICT: FIDELITY FLAG ({rel_gap_pct:+.1f}% gap).")
        print(f"           Live realized PnL will materially diverge from the validated backtest.")
        if rel_gap_pct > 0:
            print(f"           Time-stop is concealing losses by closing trades early -- live edge will be SMALLER.")
        else:
            print(f"           Time-stop is closing winners early -- live edge will be LARGER.")
        print(f"           Either (a) re-validate with sim_max_hold disabled, or (b) implement a")
        print(f"           runtime time-stop in main_pairs.py to mirror the validated backtest.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-y", required=True)
    p.add_argument("--csv-x", required=True)
    p.add_argument("--symbol-y", required=True)
    p.add_argument("--symbol-x", required=True)
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--slippage-bps-per-side", type=float, default=1.5)
    p.add_argument("--sec-fee-rate", type=float, default=0.000008)
    p.add_argument("--short-borrow-apr", type=float, default=0.0025)
    p.add_argument("--pair-entry-z", type=float, default=1.5)
    p.add_argument("--pair-exit-z", type=float, default=0.4)
    p.add_argument("--pair-delta", type=float, default=1e-4)
    p.add_argument("--pair-ve", type=float, default=1e-3)
    p.add_argument("--pair-max-leg-staleness-sec", type=float, default=30.0)
    p.add_argument("--pair-cooldown-seconds", type=float, default=120.0)
    p.add_argument("--pair-nominal-stop-pct", type=float, default=0.02)
    p.add_argument("--pair-target-dollar-notional", type=float, default=10000.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    pair_label = f"{args.symbol_y.upper()}/{args.symbol_x.upper()}"
    cells = [
        FidelityCell("with_ts", SIM_MAX_HOLD_WITH_TS),
        FidelityCell("without_ts", SIM_MAX_HOLD_WITHOUT_TS),
    ]
    out = Path(args.output)
    done_keys: set[str] = set()
    if args.resume and out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("pair") or "").strip() == pair_label:
                    done_keys.add(str(row.get("config") or ""))
        cells = [c for c in cells if c.label not in done_keys]
    print(f"[D3] pair={pair_label} configs to run={[c.label for c in cells]}  workers={args.workers}", flush=True)
    if not cells:
        print("[D3] nothing to do.")
    else:
        started = time.time()
        work = [(c, args.symbol_y.upper(), args.symbol_x.upper(), pair_label) for c in cells]
        if args.workers == 1:
            _worker_init(args.csv_y, args.csv_x, args.train_days, args.test_days, args)
            for i, item in enumerate(work, 1):
                rows = _worker_run_cell(item)
                _append_rows(out, rows)
                print(
                    f"[D3] {i}/{len(work)} config={item[0].label} windows={len(rows)} "
                    f"mean_return={statistics.mean(r['return_pct'] for r in rows):+.4f}%",
                    flush=True,
                )
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=args.workers,
                initializer=_worker_init,
                initargs=(args.csv_y, args.csv_x, args.train_days, args.test_days, args),
            ) as pool:
                for i, rows in enumerate(pool.imap_unordered(_worker_run_cell, work, chunksize=1), 1):
                    _append_rows(out, rows)
                    print(
                        f"[D3] {i}/{len(work)} config={rows[0]['config']} windows={len(rows)} "
                        f"mean_return={statistics.mean(r['return_pct'] for r in rows):+.4f}%",
                        flush=True,
                    )
        print(f"[D3] runs done in {time.time() - started:.0f}s. Output: {out}")

    # Always re-summarize from the CSV (whether we just wrote it or it
    # was already complete and --resume short-circuited).
    all_rows: List[dict] = []
    with open(out, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["window_idx"] = int(row["window_idx"])
                row["trades"] = int(row["trades"])
                row["total_pnl"] = float(row["total_pnl"])
                row["total_costs"] = float(row["total_costs"])
                row["return_pct"] = float(row["return_pct"])
            except (TypeError, ValueError):
                continue
            all_rows.append(row)
    _summarize(all_rows, pair_label, args.alpha)


if __name__ == "__main__":
    main()
