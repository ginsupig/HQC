from __future__ import annotations

import argparse
import asyncio
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from backtest_runner import BacktestConfig, run_backtest


def _parse_float_grid(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_grid(raw: str) -> List[int]:
    return [int(float(x.strip())) for x in raw.split(",") if x.strip()]


def _objective(run: Dict[str, object]) -> float:
    pnl = float(run.get("realized_pnl", 0.0) or 0.0)
    win_rate = float(run.get("win_rate", 0.0) or 0.0)
    profit_factor = float(run.get("profit_factor", 0.0) or 0.0) if run.get("profit_factor") is not None else 0.0
    excess = float(run.get("excess_return_pct", 0.0) or 0.0)
    drawdown = abs(float(run.get("max_drawdown", 0.0) or 0.0))
    return pnl + (1000.0 * win_rate) + (500.0 * excess) + (50.0 * profit_factor) - (300.0 * drawdown)


def _narrow_float_grid(values: List[float], target: float, keep: int) -> List[float]:
    if not values:
        return values
    unique_vals = sorted(set(values))
    ranked = sorted(unique_vals, key=lambda x: abs(x - target))
    return sorted(ranked[: max(1, min(keep, len(ranked)))])


def _narrow_int_grid(values: List[int], target: int, keep: int) -> List[int]:
    if not values:
        return values
    unique_vals = sorted(set(values))
    ranked = sorted(unique_vals, key=lambda x: abs(x - target))
    return sorted(ranked[: max(1, min(keep, len(ranked)))])


def _candidate_configs(
    args: argparse.Namespace,
    symbol: str,
    prior_selected: Dict[str, object] | None = None,
) -> List[BacktestConfig]:
    rank_scores = _parse_float_grid(args.min_rank_score_grid)
    orb_range_minutes = _parse_int_grid(args.orb_range_minutes_grid)
    orb_buffers = _parse_float_grid(args.orb_breakout_buffer_grid)
    orb_min_ranges = _parse_float_grid(args.orb_min_range_pct_grid)
    vwap_tolerances = _parse_float_grid(args.vwap_tolerance_grid)
    vwap_momentums = _parse_float_grid(args.vwap_momentum_grid)

    if args.self_improve and prior_selected:
        keep = max(1, int(args.adaptive_keep_per_param))
        rank_scores = _narrow_float_grid(rank_scores, float(prior_selected.get("min_rank_score", rank_scores[0])), keep)
        orb_range_minutes = _narrow_int_grid(orb_range_minutes, int(prior_selected.get("orb_range_minutes", orb_range_minutes[0])), keep)
        orb_buffers = _narrow_float_grid(orb_buffers, float(prior_selected.get("orb_breakout_buffer_pct", orb_buffers[0])), keep)
        orb_min_ranges = _narrow_float_grid(orb_min_ranges, float(prior_selected.get("orb_min_range_pct", orb_min_ranges[0])), keep)
        vwap_tolerances = _narrow_float_grid(vwap_tolerances, float(prior_selected.get("vwap_tolerance_pct", vwap_tolerances[0])), keep)
        vwap_momentums = _narrow_float_grid(vwap_momentums, float(prior_selected.get("vwap_momentum_threshold_pct", vwap_momentums[0])), keep)

    configs: List[BacktestConfig] = []
    for min_rank_score, orb_range, orb_buffer, orb_min_range, vwap_tol, vwap_momentum in itertools.product(
        rank_scores,
        orb_range_minutes,
        orb_buffers,
        orb_min_ranges,
        vwap_tolerances,
        vwap_momentums,
    ):
        configs.append(
            BacktestConfig(
                symbol=symbol,
                benchmark_symbol=args.benchmark_symbol,
                strategy=args.strategy,
                initial_capital=args.initial_capital,
                min_rank_score=min_rank_score,
                orb_range_minutes=orb_range,
                orb_breakout_buffer_pct=orb_buffer,
                orb_min_range_pct=orb_min_range,
                vwap_tolerance_pct=vwap_tol,
                vwap_momentum_threshold_pct=vwap_momentum,
                sim_max_hold_minutes=args.sim_max_hold_minutes,
                sim_stop_buffer_ticks=args.sim_stop_buffer_ticks,
            )
        )
    return configs


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _date_col(df: pd.DataFrame) -> pd.Series:
    col = None
    for c in ["timestamp", "datetime", "date", "time", "t"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("CSV must include one of: timestamp, datetime, date, time, t")
    return pd.to_datetime(df[col], utc=True, errors="coerce")


def _window_ranges(days: List[pd.Timestamp], train_days: int, test_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    i = 0
    while i + train_days + test_days <= len(days):
        train_start = days[i]
        train_end = days[i + train_days - 1]
        test_start = days[i + train_days]
        test_end = days[i + train_days + test_days - 1]
        windows.append((train_start, train_end, test_start, test_end))
        i += test_days
    return windows


async def _run_window(
    df: pd.DataFrame,
    args: argparse.Namespace,
    symbol: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    prior_selected: Dict[str, object] | None = None,
) -> Dict[str, object]:
    ts = _date_col(df)
    work = df.copy()
    work["__ts__"] = ts

    train_df = work[(work["__ts__"].dt.date >= train_start.date()) & (work["__ts__"].dt.date <= train_end.date())].drop(columns=["__ts__"])
    test_df = work[(work["__ts__"].dt.date >= test_start.date()) & (work["__ts__"].dt.date <= test_end.date())].drop(columns=["__ts__"])

    candidate_configs = _candidate_configs(args, symbol, prior_selected=prior_selected)
    best_cfg = None
    best_score = float("-inf")
    best_train_result = None
    evaluations = 0

    for cfg in candidate_configs:
        train_result = await run_backtest(train_df, cfg)
        objective = _objective(train_result)
        evaluations += 1
        if objective > best_score:
            best_score = objective
            best_cfg = cfg
            best_train_result = train_result

    assert best_cfg is not None
    test_result = await run_backtest(test_df, best_cfg)

    return {
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "selected_parameters": best_train_result.get("parameters") if isinstance(best_train_result, dict) else None,
        "selection_objective": round(best_score, 4),
        "selection_profit_factor": best_train_result.get("profit_factor") if isinstance(best_train_result, dict) else None,
        "candidates_evaluated": evaluations,
        "train_result": best_train_result,
        "test_result": test_result,
    }


async def _main_async(args: argparse.Namespace) -> None:
    df = _load_csv(Path(args.csv))
    ts = _date_col(df)
    days = sorted(set(pd.Series(ts.dropna().dt.normalize()).tolist()))

    windows = _window_ranges(days, args.train_days, args.test_days)
    if not windows:
        raise RuntimeError("Not enough data to build walk-forward windows.")

    results: List[Dict[str, object]] = []
    prior_selected: Dict[str, object] | None = None
    for train_start, train_end, test_start, test_end in windows:
        result = await _run_window(
            df=df,
            args=args,
            symbol=args.symbol,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            prior_selected=prior_selected,
        )
        results.append(result)
        prior_selected = result.get("selected_parameters") if isinstance(result.get("selected_parameters"), dict) else prior_selected

    aggregate_test_pnl = sum(float(r["test_result"].get("realized_pnl", 0.0)) for r in results)
    aggregate_test_trades = sum(int(r["test_result"].get("trades", 0)) for r in results)
    aggregate_test_pf_values = [
        float(r["test_result"].get("profit_factor"))
        for r in results
        if r.get("test_result", {}).get("profit_factor") is not None
    ]
    total_candidates = sum(int(r.get("candidates_evaluated", 0) or 0) for r in results)
    aggregate = {
        "windows": len(results),
        "self_improve": bool(args.self_improve),
        "adaptive_keep_per_param": int(args.adaptive_keep_per_param),
        "total_candidates_evaluated": total_candidates,
        "total_test_pnl": round(aggregate_test_pnl, 2),
        "total_test_trades": aggregate_test_trades,
        "avg_test_profit_factor": round(sum(aggregate_test_pf_values) / len(aggregate_test_pf_values), 4)
        if aggregate_test_pf_values
        else None,
        "avg_test_pnl_per_window": round(aggregate_test_pnl / max(1, len(results)), 2),
        "results": results,
    }

    print(json.dumps(aggregate, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run walk-forward evaluation for HQC strategies.")
    p.add_argument("--csv", required=True, help="Path to OHLCV csv file.")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--benchmark-symbol", default="SPY")
    p.add_argument("--strategy", choices=["orb", "vwap", "both"], default="both")
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--train-days", type=int, default=20)
    p.add_argument("--test-days", type=int, default=5)
    p.add_argument("--min-rank-score-grid", default="3.5,4.25,4.75,5.25")
    p.add_argument("--orb-range-minutes-grid", default="10,15,20")
    p.add_argument("--orb-breakout-buffer-grid", default="0.0003,0.0005,0.0008")
    p.add_argument("--orb-min-range-pct-grid", default="0.002,0.0025,0.0035")
    p.add_argument("--vwap-tolerance-grid", default="0.0015,0.002,0.003")
    p.add_argument("--vwap-momentum-grid", default="0.003,0.005,0.007")
    p.add_argument("--self-improve", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adaptive-keep-per-param", type=int, default=2)
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
    p.add_argument("--output", default="")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main_async(parse_args()))
