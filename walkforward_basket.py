"""
Run walk-forward validation across a basket of per-symbol CSV files.

For each CSV in --input, partition the bars into rolling (train, test)
windows, sweep the parameter grid on the in-sample period, pick the
configuration that maximizes the legacy objective, and re-run that
configuration on the next out-of-sample window. The point is to see
whether parameters that worked yesterday survive into tomorrow — if
profit factor stays < 1 in every OOS window, the strategy has no
robust edge regardless of how you tune it.

The default parameter grid is intentionally tiny (one or two values
per knob) so a 7-symbol run completes in minutes rather than hours.
Pass explicit ``--*-grid`` strings to widen.

Usage:
    python walkforward_basket.py --input data\\alpaca --strategy both \\
        --train-days 30 --test-days 10 --sim-max-hold-minutes 60 --summary-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from portfolio_batch_runner import _discover_csvs, _infer_symbol
from walkforward_runner import _date_col, _run_window, _window_ranges


async def _run_symbol(
    csv_path: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    symbol = _infer_symbol(csv_path, None)
    ts = _date_col(df)
    days = sorted(set(pd.Series(ts.dropna().dt.normalize()).tolist()))
    windows = _window_ranges(days, args.train_days, args.test_days)
    if not windows:
        return {
            "symbol": symbol,
            "source_csv": str(csv_path),
            "windows": 0,
            "skipped": "insufficient bars for any walk-forward window",
            "results": [],
        }

    # Build a per-symbol args namespace so _run_window's config builder
    # picks up the right benchmark / symbol.
    sym_args = argparse.Namespace(**vars(args))
    sym_args.symbol = symbol
    if not getattr(sym_args, "benchmark_symbol", "") or sym_args.benchmark_symbol == "":
        sym_args.benchmark_symbol = symbol

    window_results: List[Dict[str, object]] = []
    prior_selected: Dict[str, object] | None = None
    for train_start, train_end, test_start, test_end in windows:
        result = await _run_window(
            df=df,
            args=sym_args,
            symbol=symbol,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            prior_selected=prior_selected,
        )
        window_results.append(result)
        prior = result.get("selected_parameters")
        if isinstance(prior, dict):
            prior_selected = prior

    test_pnls = [float(r["test_result"].get("realized_pnl") or 0.0) for r in window_results]
    test_pfs = [
        float(r["test_result"].get("profit_factor"))
        for r in window_results
        if r["test_result"].get("profit_factor") is not None
    ]
    test_returns = [float(r["test_result"].get("total_return_pct") or 0.0) for r in window_results]

    return {
        "symbol": symbol,
        "source_csv": str(csv_path),
        "windows": len(window_results),
        "total_test_pnl": round(sum(test_pnls), 2),
        "avg_test_pnl_per_window": round(sum(test_pnls) / max(1, len(test_pnls)), 2),
        "avg_test_profit_factor": round(sum(test_pfs) / len(test_pfs), 4) if test_pfs else None,
        "windows_with_pf_above_1": sum(1 for x in test_pfs if x and x > 1.0),
        "windows_with_positive_pnl": sum(1 for x in test_pnls if x > 0.0),
        "avg_test_return_pct": round(sum(test_returns) / max(1, len(test_returns)), 6),
        "results": window_results,
    }


def _print_basket_summary(symbol_results: List[Dict[str, object]]) -> None:
    header = (
        f"{'symbol':<8} {'wins':>5} {'pos':>4} {'pfwins':>6} "
        f"{'avgPF':>7} {'avgRet%':>8} {'totalPnL':>11}"
    )
    print()
    print("=" * len(header))
    print("Walk-forward basket summary  (test-window aggregates per symbol)")
    print(
        "  wins=#windows; pos=#OOS windows with positive PnL; "
        "pfwins=#OOS windows with profit_factor>1"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    grand_total_pnl = 0.0
    grand_total_windows = 0
    grand_pf_above_1 = 0
    grand_positive = 0
    for r in symbol_results:
        sym = str(r.get("symbol") or "?")[:7]
        windows = int(r.get("windows", 0) or 0)
        if windows == 0:
            print(f"{sym:<8} {windows:>5} {'-':>4} {'-':>6} {'-':>7} {'-':>8} {'-':>11}  (skipped)")
            continue
        pos_wins = int(r.get("windows_with_positive_pnl") or 0)
        pf_wins = int(r.get("windows_with_pf_above_1") or 0)
        avg_pf = r.get("avg_test_profit_factor")
        avg_pf_str = f"{float(avg_pf):>7.2f}" if isinstance(avg_pf, (int, float)) else f"{'n/a':>7}"
        avg_ret = float(r.get("avg_test_return_pct") or 0.0) * 100.0
        total_pnl = float(r.get("total_test_pnl") or 0.0)
        grand_total_pnl += total_pnl
        grand_total_windows += windows
        grand_pf_above_1 += pf_wins
        grand_positive += pos_wins
        print(
            f"{sym:<8} {windows:>5} {pos_wins:>4} {pf_wins:>6} "
            f"{avg_pf_str} {avg_ret:>7.3f}% {total_pnl:>11.2f}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<8} {grand_total_windows:>5} {grand_positive:>4} {grand_pf_above_1:>6} "
        f"{'':>7} {'':>8} {grand_total_pnl:>11.2f}"
    )
    print("=" * len(header))


async def _main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    csv_files = _discover_csvs(input_path, args.glob)
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {input_path} with pattern {args.glob}")

    symbol_results: List[Dict[str, object]] = []
    for csv_path in csv_files:
        sym_result = await _run_symbol(csv_path, args)
        symbol_results.append(sym_result)
        # Stream a per-symbol line so the user sees progress on long runs.
        sym = sym_result.get("symbol")
        windows = sym_result.get("windows")
        avg_pf = sym_result.get("avg_test_profit_factor")
        total_pnl = sym_result.get("total_test_pnl")
        print(
            f"[{sym}] windows={windows} avg_pf={avg_pf} total_test_pnl={total_pnl}"
        )

    _print_basket_summary(symbol_results)

    if not args.summary_only:
        print(json.dumps({"basket": True, "results": symbol_results}, indent=2, default=str))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"basket": True, "results": symbol_results}, indent=2, default=str),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Directory of per-symbol CSV files (or a single CSV).")
    p.add_argument("--glob", default="*.csv", help="Glob for CSV discovery when input is a directory.")
    p.add_argument("--benchmark-symbol", default="", help="If set, used as benchmark for every symbol; otherwise each symbol benchmarks against itself.")
    p.add_argument("--strategy", choices=["orb", "vwap", "both"], default="both")
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=10)
    # Tiny defaults so a 7-symbol run completes in minutes; widen to actually
    # tune (each extra grid point multiplies wall-time per window).
    p.add_argument("--min-rank-score-grid", default="4.75")
    p.add_argument("--orb-range-minutes-grid", default="15")
    p.add_argument("--orb-breakout-buffer-grid", default="0.0005")
    p.add_argument("--orb-min-range-pct-grid", default="0.0025")
    p.add_argument("--vwap-tolerance-grid", default="0.002")
    p.add_argument("--vwap-momentum-grid", default="0.005")
    p.add_argument("--self-improve", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--adaptive-keep-per-param", type=int, default=2)
    p.add_argument(
        "--sim-max-hold-minutes",
        type=int,
        default=60,
        help="Backtest-only hard time-stop for open positions (minutes).",
    )
    p.add_argument(
        "--sim-stop-buffer-ticks",
        type=float,
        default=0.0,
        help="Backtest-only buffer added to stop levels before triggering (price units).",
    )
    p.add_argument("--slippage-bps-per-side", type=float, default=1.5)
    p.add_argument("--commission-per-share", type=float, default=0.0)
    p.add_argument("--commission-min-per-trade", type=float, default=0.0)
    p.add_argument("--sec-fee-rate", type=float, default=0.000008)
    p.add_argument("--output", default="")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the per-symbol summary table; skip the full JSON dump.",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main_async(parse_args()))
