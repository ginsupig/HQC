from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from backtest_runner import BacktestConfig, run_backtest


def _discover_csvs(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob(pattern))


def _infer_symbol(csv_path: Path, explicit_symbol: str | None = None) -> str:
    if explicit_symbol:
        return explicit_symbol.upper()
    stem = csv_path.stem.upper()
    for chunk in stem.replace("-", "_").split("_"):
        if chunk.isalpha() and 1 <= len(chunk) <= 6:
            return chunk
    return stem[:6]


async def _run_one(csv_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    symbol = _infer_symbol(csv_path, args.symbol)
    cfg = BacktestConfig(
        symbol=symbol,
        benchmark_symbol=args.benchmark_symbol or symbol,
        strategy=args.strategy,
        initial_capital=args.initial_capital,
        min_rank_score=args.min_rank_score,
        orb_range_minutes=args.orb_range_minutes,
        orb_max_trades=args.orb_max_trades,
        orb_cooldown_bars=args.orb_cooldown_bars,
        orb_min_range_pct=args.orb_min_range_pct,
        orb_breakout_buffer_pct=args.orb_breakout_buffer_pct,
        vwap_min_volume_shares=args.vwap_min_volume_shares,
        vwap_tolerance_pct=args.vwap_tolerance_pct,
        vwap_momentum_threshold_pct=args.vwap_momentum_threshold_pct,
        vwap_max_daily_trades=args.vwap_max_daily_trades,
        vwap_cooldown_bars=args.vwap_cooldown_bars,
        vwap_min_stop_pct=args.vwap_min_stop_pct,
        vwap_max_window_bars=args.vwap_max_window_bars,
        sim_max_hold_minutes=args.sim_max_hold_minutes,
        sim_stop_buffer_ticks=args.sim_stop_buffer_ticks,
        slippage_bps_per_side=args.slippage_bps_per_side,
        commission_per_share=args.commission_per_share,
        commission_min_per_trade=args.commission_min_per_trade,
        sec_fee_rate=args.sec_fee_rate,
    )
    result = await run_backtest(df, cfg)
    result["source_csv"] = str(csv_path)
    return result


async def _main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    csv_files = _discover_csvs(input_path, args.glob)
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {input_path} with pattern {args.glob}")

    symbol_results: List[Dict[str, object]] = []
    for csv_path in csv_files:
        symbol_results.append(await _run_one(csv_path, args))

    total_pnl = sum(float(r.get("realized_pnl", 0.0) or 0.0) for r in symbol_results)
    total_trades = sum(int(r.get("trades", 0) or 0) for r in symbol_results)
    total_wins = sum(int(r.get("wins", 0) or 0) for r in symbol_results)
    total_losses = sum(int(r.get("losses", 0) or 0) for r in symbol_results)

    payload = {
        "portfolio_batch": True,
        "symbols": len(symbol_results),
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "results": [
            {
                "symbol": r.get("symbol"),
                "test_result": r,
            }
            for r in symbol_results
        ],
    }

    _print_summary_table(symbol_results, total_pnl, total_trades, total_wins, total_losses)

    if not args.summary_only:
        print(json.dumps(payload, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary_table(
    results: List[Dict[str, object]],
    total_pnl: float,
    total_trades: int,
    total_wins: int,
    total_losses: int,
) -> None:
    """Compact per-symbol comparison so a multi-symbol run is readable at a glance."""
    header = (
        f"{'symbol':<8} {'trades':>6} {'wins':>5} {'losses':>6} {'win%':>6} "
        f"{'pf':>6} {'ret%':>7} {'maxDD%':>7} {'sharpe':>7} {'IR':>7} {'avgDur':>7}"
    )
    print()
    print("=" * len(header))
    print("Per-symbol summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        sym = str(r.get("symbol") or "?")[:7]
        trades = int(r.get("trades", 0) or 0)
        wins = int(r.get("wins", 0) or 0)
        losses = int(r.get("losses", 0) or 0)
        win_pct = (wins / trades * 100.0) if trades else 0.0
        pf = r.get("profit_factor")
        pf_str = f"{float(pf):>6.2f}" if isinstance(pf, (int, float)) else f"{'n/a':>6}"
        ret = float(r.get("total_return_pct") or 0.0) * 100.0
        dd = float(r.get("max_drawdown") or 0.0) * 100.0
        sharpe = r.get("annualized_sharpe")
        sharpe_str = f"{float(sharpe):>7.2f}" if isinstance(sharpe, (int, float)) else f"{'n/a':>7}"
        ir = r.get("information_ratio")
        ir_str = f"{float(ir):>7.2f}" if isinstance(ir, (int, float)) else f"{'n/a':>7}"
        dur = r.get("avg_duration_min")
        dur_str = f"{float(dur):>7.1f}" if isinstance(dur, (int, float)) else f"{'n/a':>7}"
        print(
            f"{sym:<8} {trades:>6} {wins:>5} {losses:>6} {win_pct:>5.1f}% "
            f"{pf_str} {ret:>6.2f}% {dd:>6.2f}% {sharpe_str} {ir_str} {dur_str}"
        )

    print("-" * len(header))
    win_pct_total = (total_wins / total_trades * 100.0) if total_trades else 0.0
    print(
        f"{'TOTAL':<8} {total_trades:>6} {total_wins:>5} {total_losses:>6} {win_pct_total:>5.1f}% "
        f"{'':>6} pnl={total_pnl:>10.2f}"
    )
    print("=" * len(header))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run batch portfolio backtests over a folder of per-symbol CSV files.")
    p.add_argument("--input", required=True, help="CSV file or directory of CSV files")
    p.add_argument("--glob", default="*.csv", help="Glob for CSV discovery when input is a directory")
    p.add_argument("--symbol", default="", help="Optional fixed symbol override for single-file mode")
    p.add_argument("--benchmark-symbol", default="", help="Benchmark symbol if present in each CSV")
    p.add_argument(
        "--strategy",
        choices=["orb", "vwap", "both", "gap_fade", "all"],
        default="both",
    )
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--min-rank-score", type=float, default=4.75)
    p.add_argument("--orb-range-minutes", type=int, default=15)
    p.add_argument("--orb-max-trades", type=int, default=2)
    p.add_argument("--orb-cooldown-bars", type=int, default=10)
    p.add_argument("--orb-min-range-pct", type=float, default=0.0025)
    p.add_argument("--orb-breakout-buffer-pct", type=float, default=0.0005)
    p.add_argument("--vwap-min-volume-shares", type=float, default=250000.0)
    p.add_argument("--vwap-tolerance-pct", type=float, default=0.002)
    p.add_argument("--vwap-momentum-threshold-pct", type=float, default=0.005)
    p.add_argument("--vwap-max-daily-trades", type=int, default=3)
    p.add_argument("--vwap-cooldown-bars", type=int, default=8)
    p.add_argument("--vwap-min-stop-pct", type=float, default=0.003)
    p.add_argument("--vwap-max-window-bars", type=int, default=8)
    p.add_argument(
        "--sim-max-hold-minutes",
        type=int,
        default=240,
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
        help="Print only the compact per-symbol comparison table and totals; "
             "skip the full JSON dump.",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main_async(parse_args()))
