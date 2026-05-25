"""
A2 — Parameter-sensitivity / overfitting-surface walk-forward.

Walks a 4-D grid of Kalman-pair parameters across the full validated
walk-forward harness, one cell at a time, and writes a CSV with the
OOS mean per-window return, bootstrap 95% CI, and Newey-West p-value
for each cell. Edge must be a contiguous PLATEAU around the deployed
parameters, not a knife-edge SPIKE — A1 left JPM/BAC as marginal
(passes strict Bonferroni by 0.0013, fails inclusive by 0.00075), and
the question A2 answers is whether the surviving edge is robust to
parameter perturbation or curve-fit to the single (entry_z=1.5,
exit_z=0.4, delta=1e-4, ve=1e-3) node.

Default grid from the roadmap:
  entry_z ∈ {1.25, 1.5, 1.75, 2.0}
  exit_z  ∈ {0.2, 0.4, 0.6}
  delta   ∈ {1e-5, 1e-4, 1e-3}
  ve      ∈ {1e-3, 1e-2}
4 * 3 * 3 * 2 = 72 cells. With 47 OOS windows per cell, ~3,400
walk-forward backtests total. Use --workers N for parallelism and
--smoke for a fast sanity-pass first.

Usage:
  # Smoke test (8 cells, ~10 min):
  python tools/a2_parameter_sensitivity.py \\
      --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \\
      --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \\
      --train-days 30 --test-days 10 \\
      --smoke --workers 4 \\
      --output a2_jpm_bac_smoke.csv

  # Full grid (overnight on 8 cores):
  python tools/a2_parameter_sensitivity.py \\
      --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \\
      --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \\
      --train-days 30 --test-days 10 \\
      --workers 8 \\
      --output a2_jpm_bac_full.csv

  # Resume an interrupted run (re-reads --output, skips done cells):
  ... --resume --output a2_jpm_bac_full.csv

Then analyze with tools/a2_analyze.py.
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

# Ensure the repo root is importable so this script works from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Inputs to the harness — reused as-is. A2 does not touch the validated
# code paths; it only iterates over parameter cells.
from backtest_runner import BacktestConfig, run_backtest  # noqa: E402
from walkforward_pairs import _date_col, _windows  # noqa: E402
from analyze_walkforward import _bootstrap_mean_ci, _one_sample_p_greater_than_zero  # noqa: E402


@dataclass(frozen=True)
class Cell:
    entry_z: float
    exit_z: float
    delta: float
    ve: float

    def key(self) -> str:
        return f"e{self.entry_z}_x{self.exit_z}_d{self.delta:.0e}_v{self.ve:.0e}"


# Roadmap full grid.
DEFAULT_ENTRY_Z = (1.25, 1.5, 1.75, 2.0)
DEFAULT_EXIT_Z = (0.2, 0.4, 0.6)
DEFAULT_DELTA = (1e-5, 1e-4, 1e-3)
DEFAULT_VE = (1e-3, 1e-2)

# Smoke subgrid: just enough to verify the runner end-to-end.
SMOKE_ENTRY_Z = (1.5, 2.0)
SMOKE_EXIT_Z = (0.2, 0.4)
SMOKE_DELTA = (1e-4,)
SMOKE_VE = (1e-3, 1e-2)


CSV_COLUMNS = [
    "entry_z", "exit_z", "delta", "ve",
    "n_windows", "windows_positive", "windows_pf_gt_1", "avg_pf",
    "mean_pct", "ci_lo", "ci_hi", "raw_p",
    "total_pnl", "total_costs", "seconds",
]


def _build_grid(args: argparse.Namespace) -> List[Cell]:
    entry_z = SMOKE_ENTRY_Z if args.smoke else DEFAULT_ENTRY_Z
    exit_z = SMOKE_EXIT_Z if args.smoke else DEFAULT_EXIT_Z
    delta = SMOKE_DELTA if args.smoke else DEFAULT_DELTA
    ve = SMOKE_VE if args.smoke else DEFAULT_VE
    if args.entry_z_grid:
        entry_z = tuple(float(x) for x in args.entry_z_grid.split(","))
    if args.exit_z_grid:
        exit_z = tuple(float(x) for x in args.exit_z_grid.split(","))
    if args.delta_grid:
        delta = tuple(float(x) for x in args.delta_grid.split(","))
    if args.ve_grid:
        ve = tuple(float(x) for x in args.ve_grid.split(","))
    cells: List[Cell] = []
    for ez, xz, d, v in itertools.product(entry_z, exit_z, delta, ve):
        # Skip degenerate cells where exit_z >= entry_z.
        if xz >= ez:
            continue
        cells.append(Cell(entry_z=ez, exit_z=xz, delta=d, ve=v))
    return cells


def _load_already_done(output_path: Path) -> Dict[str, dict]:
    if not output_path.exists():
        return {}
    done: Dict[str, dict] = {}
    try:
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    c = Cell(
                        entry_z=float(row["entry_z"]),
                        exit_z=float(row["exit_z"]),
                        delta=float(row["delta"]),
                        ve=float(row["ve"]),
                    )
                except (KeyError, ValueError):
                    continue
                done[c.key()] = row
    except Exception:
        return {}
    return done


def _config_for_cell(
    cell: Cell,
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
        slippage_bps_per_side=args.slippage_bps_per_side,
        commission_per_share=0.0,
        commission_min_per_trade=0.0,
        sec_fee_rate=args.sec_fee_rate,
        short_borrow_apr=args.short_borrow_apr,
        hedge_symbol=symbol_x,
        pair_entry_z=cell.entry_z,
        pair_exit_z=cell.exit_z,
        pair_delta=cell.delta,
        pair_ve=cell.ve,
        pair_max_leg_staleness_sec=args.pair_max_leg_staleness_sec,
        pair_cooldown_seconds=args.pair_cooldown_seconds,
        pair_nominal_stop_pct=args.pair_nominal_stop_pct,
        pair_target_dollar_notional=args.pair_target_dollar_notional,
    )


async def _walkforward_one_cell(
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    cell: Cell,
    symbol_y: str,
    symbol_x: str,
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
        "entry_z": cell.entry_z,
        "exit_z": cell.exit_z,
        "delta": cell.delta,
        "ve": cell.ve,
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


# Module-level worker so multiprocessing.Pool can pickle it.
_WORKER_DF_Y: Optional[pd.DataFrame] = None
_WORKER_DF_X: Optional[pd.DataFrame] = None
_WORKER_WINDOWS: Optional[List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]] = None
_WORKER_ARGS: Optional[argparse.Namespace] = None


def _worker_init(csv_y: str, csv_x: str, train_days: int, test_days: int, args: argparse.Namespace) -> None:
    global _WORKER_DF_Y, _WORKER_DF_X, _WORKER_WINDOWS, _WORKER_ARGS
    _WORKER_DF_Y = pd.read_csv(csv_y)
    _WORKER_DF_X = pd.read_csv(csv_x)
    days_y = sorted(set(_date_col(_WORKER_DF_Y).dropna().dt.normalize().tolist()))
    days_x = sorted(set(_date_col(_WORKER_DF_X).dropna().dt.normalize().tolist()))
    days = sorted(set(days_y) & set(days_x))
    _WORKER_WINDOWS = _windows(days, train_days, test_days)
    _WORKER_ARGS = args


def _worker_run_cell(args_tuple: Tuple[Cell, str, str]) -> dict:
    cell, symbol_y, symbol_x = args_tuple
    t0 = time.time()
    row = asyncio.run(_walkforward_one_cell(
        _WORKER_DF_Y, _WORKER_DF_X, _WORKER_WINDOWS,
        cell, symbol_y, symbol_x, _WORKER_ARGS,
    ))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def _append_row(output_path: Path, row: dict) -> None:
    """Append one cell's result. Creates file with header if missing.
    Flush + fsync after each row so an interrupted run is resumable."""
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
    p.add_argument("--csv-y", required=True, help="Primary leg OHLCV CSV (e.g. JPM).")
    p.add_argument("--csv-x", required=True, help="Hedge leg OHLCV CSV (e.g. BAC).")
    p.add_argument("--symbol-y", required=True)
    p.add_argument("--symbol-x", required=True)
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--sim-max-hold-minutes", type=int, default=240)
    p.add_argument("--slippage-bps-per-side", type=float, default=1.5)
    p.add_argument("--sec-fee-rate", type=float, default=0.000008)
    p.add_argument("--short-borrow-apr", type=float, default=0.0025)
    p.add_argument("--pair-max-leg-staleness-sec", type=float, default=30.0)
    p.add_argument("--pair-cooldown-seconds", type=float, default=120.0)
    p.add_argument("--pair-nominal-stop-pct", type=float, default=0.02)
    p.add_argument("--pair-target-dollar-notional", type=float, default=10000.0)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--smoke", action="store_true", help="Run a small sanity-check subgrid first.")
    p.add_argument("--entry-z-grid", type=str, default="", help="CSV override e.g. 1.25,1.5,1.75,2.0")
    p.add_argument("--exit-z-grid", type=str, default="")
    p.add_argument("--delta-grid", type=str, default="")
    p.add_argument("--ve-grid", type=str, default="")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--resume", action="store_true", help="Skip cells already present in --output.")
    p.add_argument("--output", required=True, help="CSV output path. Rows are appended; safe to resume.")
    args = p.parse_args()

    grid = _build_grid(args)
    out = Path(args.output)
    done = _load_already_done(out) if args.resume else {}
    to_run = [c for c in grid if c.key() not in done]
    print(
        f"[A2] grid={len(grid)} cells | already done={len(done)} | to run={len(to_run)} | "
        f"workers={args.workers}",
        flush=True,
    )
    if not to_run:
        print("[A2] nothing to do.")
        return

    started = time.time()
    work_items: List[Tuple[Cell, str, str]] = [(c, args.symbol_y.upper(), args.symbol_x.upper()) for c in to_run]

    if args.workers == 1:
        # Single-process path: useful for debugging the run, and avoids spawning
        # on platforms where mp.Pool has issues.
        _worker_init(args.csv_y, args.csv_x, args.train_days, args.test_days, args)
        for i, item in enumerate(work_items, 1):
            row = _worker_run_cell(item)
            _append_row(out, row)
            print(
                f"[A2] {i}/{len(work_items)} cell={row['entry_z']:.2f}/{row['exit_z']:.2f}/"
                f"{row['delta']:.0e}/{row['ve']:.0e}  "
                f"mean={row['mean_pct']:+.3f}%  p={row['raw_p']}  "
                f"pnl={row['total_pnl']:+.0f}  t={row['seconds']}s",
                flush=True,
            )
    else:
        # Multi-process path: each worker loads the CSVs + windows once
        # via _worker_init, then handles N cells.
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
                    f"[A2] {i}/{len(work_items)} cell={row['entry_z']:.2f}/{row['exit_z']:.2f}/"
                    f"{row['delta']:.0e}/{row['ve']:.0e}  "
                    f"mean={row['mean_pct']:+.3f}%  p={row['raw_p']}  "
                    f"pnl={row['total_pnl']:+.0f}  t={row['seconds']}s",
                    flush=True,
                )

    total = time.time() - started
    print(f"[A2] done in {total:.0f}s. Output: {out}")
    print(f"     Analyze with: python tools/a2_analyze.py --input {out}")


if __name__ == "__main__":
    main()
