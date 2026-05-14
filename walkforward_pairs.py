"""
Walk-forward validation for a Kalman pairs strategy.

Splits a primary + hedge CSV pair into rolling (train, test) windows
on calendar-date boundaries, runs the existing backtest_runner with
strategy=pairs on each test slice, and emits a result JSON in the same
shape as walkforward_basket so analyze_walkforward.py can consume it
unmodified.

The train window does not currently feed a parameter search — pairs
have so few user-tunable knobs (entry_z, exit_z, delta) that an in-
sample grid sweep on tiny windows is more noise than signal. The
train_result is run for symmetry with the single-symbol walk-forward
and is reported but not used for selection.

Usage:
    python walkforward_pairs.py \\
        --csv-y data/alpaca/jpm_183d_1m.csv --symbol-y JPM \\
        --csv-x data/alpaca/bac_183d_1m.csv --symbol-x BAC \\
        --train-days 30 --test-days 10 \\
        --pair-entry-z 1.5 --pair-exit-z 0.4 \\
        --sim-max-hold-minutes 240 \\
        --output result_wf_pair_jpm_bac.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from backtest_runner import BacktestConfig, run_backtest


def _date_col(df: pd.DataFrame) -> pd.Series:
    for c in ["timestamp", "datetime", "date", "time", "t"]:
        if c in df.columns:
            return pd.to_datetime(df[c], utc=True, errors="coerce")
    raise ValueError("CSV must contain a timestamp column.")


def _windows(days: List[pd.Timestamp], train_days: int, test_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    out: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    i = 0
    while i + train_days + test_days <= len(days):
        train_start = days[i]
        train_end = days[i + train_days - 1]
        test_start = days[i + train_days]
        test_end = days[i + train_days + test_days - 1]
        out.append((train_start, train_end, test_start, test_end))
        i += test_days
    return out


def _build_cfg(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        symbol=args.symbol_y.upper(),
        benchmark_symbol=args.symbol_y.upper(),
        strategy="pairs",
        initial_capital=args.initial_capital,
        sim_max_hold_minutes=args.sim_max_hold_minutes,
        sim_stop_buffer_ticks=args.sim_stop_buffer_ticks,
        slippage_bps_per_side=args.slippage_bps_per_side,
        commission_per_share=args.commission_per_share,
        commission_min_per_trade=args.commission_min_per_trade,
        sec_fee_rate=args.sec_fee_rate,
        hedge_symbol=args.symbol_x.upper(),
        pair_entry_z=args.pair_entry_z,
        pair_exit_z=args.pair_exit_z,
        pair_delta=args.pair_delta,
        pair_ve=args.pair_ve,
        pair_max_leg_staleness_sec=args.pair_max_leg_staleness_sec,
        pair_cooldown_seconds=args.pair_cooldown_seconds,
        pair_nominal_stop_pct=args.pair_nominal_stop_pct,
    )


async def _run_window_slice(
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, object]:
    """Slice both legs to [start, end] (inclusive on calendar date) and
    hand the concatenated frame to run_backtest with strategy=pairs."""
    sy = _date_col(df_y).dt.date
    sx = _date_col(df_x).dt.date
    mask_y = (sy >= start.date()) & (sy <= end.date())
    mask_x = (sx >= start.date()) & (sx <= end.date())
    sliced = pd.concat([df_y[mask_y], df_x[mask_x]], ignore_index=True)
    cfg = _build_cfg(args)
    return await run_backtest(sliced, cfg)


async def _main_async(args: argparse.Namespace) -> None:
    df_y = pd.read_csv(args.csv_y)
    df_x = pd.read_csv(args.csv_x)
    days_y = sorted(set(_date_col(df_y).dropna().dt.normalize().tolist()))
    days_x = sorted(set(_date_col(df_x).dropna().dt.normalize().tolist()))
    days = sorted(set(days_y) & set(days_x))  # only days both legs trade
    windows = _windows(days, args.train_days, args.test_days)
    if not windows:
        raise SystemExit("Not enough overlapping days to build walk-forward windows.")

    pair_label = f"{args.symbol_y.upper()}/{args.symbol_x.upper()}"
    print(f"[{pair_label}] {len(windows)} walk-forward windows", flush=True)

    window_results: List[Dict[str, object]] = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        if args.skip_train:
            train_result: Dict[str, object] = {"skipped": True}
        else:
            train_result = await _run_window_slice(df_y, df_x, args, train_start, train_end)
        test_result = await _run_window_slice(df_y, df_x, args, test_start, test_end)

        window_results.append(
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_result": train_result,
                "test_result": test_result,
            }
        )
        pnl = test_result.get("realized_pnl", 0.0)
        pf = test_result.get("profit_factor")
        n = test_result.get("trades", 0)
        print(
            f"[{pair_label}] window {i:>2}/{len(windows)} test {test_start.date()} -> "
            f"{test_end.date()}  trades={n}  pf={pf}  pnl={pnl:.2f}",
            flush=True,
        )

    # Mirror walkforward_basket's per-symbol summary so analyze_walkforward consumes it as-is.
    test_pnls = [float(w["test_result"].get("realized_pnl") or 0.0) for w in window_results]
    test_pfs = [
        float(w["test_result"].get("profit_factor"))
        for w in window_results
        if w["test_result"].get("profit_factor") is not None
    ]
    test_returns = [float(w["test_result"].get("total_return_pct") or 0.0) for w in window_results]

    sym_block = {
        "symbol": pair_label,
        "source_csv": f"{args.csv_y};{args.csv_x}",
        "windows": len(window_results),
        "total_test_pnl": round(sum(test_pnls), 2),
        "avg_test_pnl_per_window": round(sum(test_pnls) / max(1, len(test_pnls)), 2),
        "avg_test_profit_factor": round(sum(test_pfs) / len(test_pfs), 4) if test_pfs else None,
        "windows_with_pf_above_1": sum(1 for x in test_pfs if x and x > 1.0),
        "windows_with_positive_pnl": sum(1 for x in test_pnls if x > 0.0),
        "avg_test_return_pct": round(sum(test_returns) / max(1, len(test_returns)), 6),
        "results": window_results,
    }
    payload = {"basket": True, "results": [sym_block]}

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"wrote {out}")

    pf_disp = sym_block["avg_test_profit_factor"]
    print()
    print("=" * 60)
    print(f"{pair_label}  walk-forward summary")
    print("=" * 60)
    print(f"  windows                  : {sym_block['windows']}")
    print(f"  windows positive pnl     : {sym_block['windows_with_positive_pnl']}")
    print(f"  windows profit_factor>1  : {sym_block['windows_with_pf_above_1']}")
    print(f"  avg test profit_factor   : {pf_disp}")
    print(f"  avg test return per window: {sym_block['avg_test_return_pct']*100:.3f}%")
    print(f"  total test pnl           : ${sym_block['total_test_pnl']:.2f}")
    print("=" * 60)
    print("Run analyze_walkforward.py on the output for bootstrap CI + p-value.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-y", required=True, help="CSV for the primary leg (Y).")
    p.add_argument("--csv-x", required=True, help="CSV for the hedge leg (X).")
    p.add_argument("--symbol-y", required=True)
    p.add_argument("--symbol-x", required=True)
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--pair-entry-z", type=float, default=1.5)
    p.add_argument("--pair-exit-z", type=float, default=0.4)
    p.add_argument("--pair-delta", type=float, default=1e-4)
    p.add_argument("--pair-ve", type=float, default=1e-3)
    p.add_argument("--pair-max-leg-staleness-sec", type=float, default=30.0)
    p.add_argument("--pair-cooldown-seconds", type=float, default=120.0)
    p.add_argument("--pair-nominal-stop-pct", type=float, default=0.02)
    p.add_argument("--sim-max-hold-minutes", type=int, default=240)
    p.add_argument("--sim-stop-buffer-ticks", type=float, default=0.0)
    p.add_argument("--slippage-bps-per-side", type=float, default=1.5)
    p.add_argument("--commission-per-share", type=float, default=0.0)
    p.add_argument("--commission-min-per-trade", type=float, default=0.0)
    p.add_argument("--sec-fee-rate", type=float, default=0.000008)
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the in-sample train backtest per window (faster; we don't tune anyway).",
    )
    p.add_argument("--output", default="")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main_async(parse_args()))
