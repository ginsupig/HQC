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

    print(json.dumps(payload, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run batch portfolio backtests over a folder of per-symbol CSV files.")
    p.add_argument("--input", required=True, help="CSV file or directory of CSV files")
    p.add_argument("--glob", default="*.csv", help="Glob for CSV discovery when input is a directory")
    p.add_argument("--symbol", default="", help="Optional fixed symbol override for single-file mode")
    p.add_argument("--benchmark-symbol", default="", help="Benchmark symbol if present in each CSV")
    p.add_argument("--strategy", choices=["orb", "vwap", "both"], default="both")
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
    p.add_argument("--output", default="")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main_async(parse_args()))
