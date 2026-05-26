"""
A3 — Cost-stress / breakeven walk-forward.

Re-runs the validated walk-forward harness for a pair across a grid
of transaction-cost assumptions. The deployed cost model assumes
1.5 bps/side slippage (=3 bps round-trip) and 0.25% APR short borrow,
both conservative for liquid large-cap names. A3 answers two
questions:

  1. Does the deployed cost level leave margin? If a 1 bp slippage
     increase or a 0.75% borrow increase wipes out the edge, the
     pair is cost-fragile and probably should not be sized up.
  2. What is the breakeven cost level? The grid is sparse, so we
     report the boundary cells where the verdict flips from EDGE+
     to fail.

Default grid (from the roadmap):
  slippage_bps_per_side ∈ {1.5, 3.0, 5.0}   (round-trip: 3, 6, 10 bps)
  short_borrow_apr      ∈ {0.0025, 0.01, 0.03}  (0.25%, 1%, 3% APR)
3 * 3 = 9 cells per pair. With 47 windows per cell, ~420 backtests --
~90 min single-process, ~15 min with 8 workers.

Usage:
  python tools/a3_cost_stress.py \\
      --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \\
      --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \\
      --train-days 30 --test-days 10 \\
      --workers 4 \\
      --output a3_jpm_bac.csv

  python tools/a3_analyze.py --input a3_jpm_bac.csv

The runner re-uses tools/a2_parameter_sensitivity.py's multiprocessing
infrastructure (per-worker CSV+window initialization, fsync-after-row
append, --resume). A3 holds the strategy parameters fixed at the
deployed values and varies cost only.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
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

from backtest_runner import BacktestConfig, run_backtest  # noqa: E402
from walkforward_pairs import _date_col, _windows  # noqa: E402
from analyze_walkforward import _bootstrap_mean_ci, _one_sample_p_greater_than_zero  # noqa: E402


@dataclass(frozen=True)
class CostCell:
    slippage_bps_per_side: float
    short_borrow_apr: float

    def key(self) -> str:
        return f"s{self.slippage_bps_per_side}_b{self.short_borrow_apr}"


# Roadmap grid.
DEFAULT_SLIPPAGE_BPS = (1.5, 3.0, 5.0)
DEFAULT_BORROW_APR = (0.0025, 0.01, 0.03)

CSV_COLUMNS = [
    "pair",
    "slippage_bps_per_side", "short_borrow_apr",
    "n_windows", "windows_positive", "windows_pf_gt_1", "avg_pf",
    "mean_pct", "ci_lo", "ci_hi", "raw_p",
    "total_pnl", "total_costs", "seconds",
]


def _build_grid(args: argparse.Namespace) -> List[CostCell]:
    slip = DEFAULT_SLIPPAGE_BPS
    borrow = DEFAULT_BORROW_APR
    if args.slippage_grid:
        slip = tuple(float(x) for x in args.slippage_grid.split(","))
    if args.borrow_grid:
        borrow = tuple(float(x) for x in args.borrow_grid.split(","))
    return [CostCell(slippage_bps_per_side=s, short_borrow_apr=b) for s, b in itertools.product(slip, borrow)]


def _load_done(output_path: Path, pair_label: str) -> Dict[str, dict]:
    if not output_path.exists():
        return {}
    done: Dict[str, dict] = {}
    try:
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("pair") or "").strip() != pair_label:
                    continue
                try:
                    cell = CostCell(
                        slippage_bps_per_side=float(row["slippage_bps_per_side"]),
                        short_borrow_apr=float(row["short_borrow_apr"]),
                    )
                except (KeyError, ValueError):
                    continue
                done[cell.key()] = row
    except Exception:
        return {}
    return done


def _config_for_cell(
    cell: CostCell,
    symbol_y: str,
    symbol_x: str,
    args: argparse.Namespace,
) -> BacktestConfig:
    return BacktestConfig(
        symbol=symbol_y,
        benchmark_symbol=symbol_y,
        strategy="pairs",
        initial_capital=args.initial_capital,
        sim_max_hold_minutes=args.sim_max_hold_minutes,
        slippage_bps_per_side=cell.slippage_bps_per_side,
        commission_per_share=0.0,
        commission_min_per_trade=0.0,
        sec_fee_rate=args.sec_fee_rate,
        short_borrow_apr=cell.short_borrow_apr,
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
    cell: CostCell,
    symbol_y: str,
    symbol_x: str,
    pair_label: str,
    args: argparse.Namespace,
) -> dict:
    test_returns: List[float] = []
    test_pnls: List[float] = []
    test_costs: List[float] = []
    test_pfs: List[float] = []
    windows_positive = 0

    sy = _date_col(df_y).dt.date
    sx = _date_col(df_x).dt.date
    cfg = _config_for_cell(cell, symbol_y, symbol_x, args)

    for _train_start, _train_end, test_start, test_end in windows:
        mask_y = (sy >= test_start.date()) & (sy <= test_end.date())
        mask_x = (sx >= test_start.date()) & (sx <= test_end.date())
        sliced = pd.concat([df_y[mask_y], df_x[mask_x]], ignore_index=True)
        result = await run_backtest(sliced, cfg)
        ret = float(result.get("total_return_pct") or 0.0)
        pnl = float(result.get("realized_pnl") or 0.0)
        cost = float(result.get("total_costs") or 0.0)
        pf = result.get("profit_factor")
        test_returns.append(ret)
        test_pnls.append(pnl)
        test_costs.append(cost)
        if pnl > 0:
            windows_positive += 1
        if isinstance(pf, (int, float)):
            test_pfs.append(float(pf))

    n = len(test_returns)
    mean, lo, hi = _bootstrap_mean_ci(test_returns, n_boot=args.bootstrap, alpha=args.alpha)
    p_pos = _one_sample_p_greater_than_zero(test_returns)
    avg_pf = sum(test_pfs) / len(test_pfs) if test_pfs else None
    return {
        "pair": pair_label,
        "slippage_bps_per_side": cell.slippage_bps_per_side,
        "short_borrow_apr": cell.short_borrow_apr,
        "n_windows": n,
        "windows_positive": windows_positive,
        "windows_pf_gt_1": sum(1 for x in test_pfs if x > 1.0),
        "avg_pf": round(avg_pf, 4) if avg_pf is not None else "",
        "mean_pct": round(mean * 100.0, 6),
        "ci_lo": round(lo * 100.0, 6),
        "ci_hi": round(hi * 100.0, 6),
        "raw_p": round(p_pos, 6) if math.isfinite(p_pos) else "",
        "total_pnl": round(sum(test_pnls), 2),
        "total_costs": round(sum(test_costs), 2),
    }


# Module-level worker state (multiprocessing pickling).
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


def _worker_run_cell(args_tuple: Tuple[CostCell, str, str, str]) -> dict:
    cell, symbol_y, symbol_x, pair_label = args_tuple
    t0 = time.time()
    row = asyncio.run(_walkforward_one_cell(
        _W_DF_Y, _W_DF_X, _W_WINDOWS,
        cell, symbol_y, symbol_x, pair_label, _W_ARGS,
    ))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def _append_row(output_path: Path, row: dict) -> None:
    is_new = not output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-y", required=True)
    p.add_argument("--csv-x", required=True)
    p.add_argument("--symbol-y", required=True)
    p.add_argument("--symbol-x", required=True)
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--sim-max-hold-minutes", type=int, default=240)
    p.add_argument("--sec-fee-rate", type=float, default=0.000008)
    # Strategy params held FIXED at the deployed values; A3 varies cost only.
    p.add_argument("--pair-entry-z", type=float, default=1.5)
    p.add_argument("--pair-exit-z", type=float, default=0.4)
    p.add_argument("--pair-delta", type=float, default=1e-4)
    p.add_argument("--pair-ve", type=float, default=1e-3)
    p.add_argument("--pair-max-leg-staleness-sec", type=float, default=30.0)
    p.add_argument("--pair-cooldown-seconds", type=float, default=120.0)
    p.add_argument("--pair-nominal-stop-pct", type=float, default=0.02)
    p.add_argument("--pair-target-dollar-notional", type=float, default=10000.0)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--slippage-grid", type=str, default="", help="CSV override e.g. 1.5,3,5,10")
    p.add_argument("--borrow-grid", type=str, default="", help="CSV override e.g. 0.0025,0.01,0.03")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    pair_label = f"{args.symbol_y.upper()}/{args.symbol_x.upper()}"
    grid = _build_grid(args)
    out = Path(args.output)
    done = _load_done(out, pair_label) if args.resume else {}
    to_run = [c for c in grid if c.key() not in done]
    print(
        f"[A3] pair={pair_label} grid={len(grid)} cells | done={len(done)} | "
        f"to run={len(to_run)} | workers={args.workers}",
        flush=True,
    )
    if not to_run:
        print("[A3] nothing to do.")
        return

    started = time.time()
    work_items: List[Tuple[CostCell, str, str, str]] = [
        (c, args.symbol_y.upper(), args.symbol_x.upper(), pair_label) for c in to_run
    ]

    if args.workers == 1:
        _worker_init(args.csv_y, args.csv_x, args.train_days, args.test_days, args)
        for i, item in enumerate(work_items, 1):
            row = _worker_run_cell(item)
            _append_row(out, row)
            print(
                f"[A3] {i}/{len(work_items)} cell=slip{row['slippage_bps_per_side']}bps/"
                f"borrow{row['short_borrow_apr']*100:.2f}%  "
                f"mean={row['mean_pct']:+.3f}%  p={row['raw_p']}  "
                f"pnl={row['total_pnl']:+.0f}  t={row['seconds']}s",
                flush=True,
            )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(args.csv_y, args.csv_x, args.train_days, args.test_days, args),
        ) as pool:
            for i, row in enumerate(
                pool.imap_unordered(_worker_run_cell, work_items, chunksize=1), start=1
            ):
                _append_row(out, row)
                print(
                    f"[A3] {i}/{len(work_items)} cell=slip{row['slippage_bps_per_side']}bps/"
                    f"borrow{row['short_borrow_apr']*100:.2f}%  "
                    f"mean={row['mean_pct']:+.3f}%  p={row['raw_p']}  "
                    f"pnl={row['total_pnl']:+.0f}  t={row['seconds']}s",
                    flush=True,
                )

    total = time.time() - started
    print(f"[A3] done in {total:.0f}s. Output: {out}")
    print(f"     Analyze with: python tools/a3_analyze.py --input {out}")


if __name__ == "__main__":
    main()
