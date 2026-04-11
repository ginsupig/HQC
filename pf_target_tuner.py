from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from backtest_runner import BacktestConfig, run_backtest
from performance_report import build_report
from walkforward_runner import _date_col, _load_csv, _window_ranges


LOGGER = logging.getLogger("PFTargetTuner")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for name in [
        "CandidateRanker",
        "EquityVWAPHunter",
        "EquityORB",
        "EventBus",
        "RiskPositionSizer",
        "AlpacaExecutionRouter",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


def _float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _int_list(raw: str) -> List[int]:
    return [int(float(x.strip())) for x in raw.split(",") if x.strip()]


def _ordered_unique(values: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _neighbor_values(values: Sequence[Any], target: Any, radius: int) -> List[Any]:
    ordered = list(values)
    if not ordered:
        return []
    try:
        idx = ordered.index(target)
    except ValueError:
        return ordered
    lo = max(0, idx - radius)
    hi = min(len(ordered), idx + radius + 1)
    return ordered[lo:hi]


def _build_spaces(args: argparse.Namespace) -> Dict[str, List[Any]]:
    return {
        "min_rank_score": _ordered_unique(_float_list(args.min_rank_scores)),
        "orb_range_minutes": _ordered_unique(_int_list(args.orb_range_minutes)),
        "orb_breakout_buffer_pct": _ordered_unique(_float_list(args.orb_breakout_buffers)),
        "orb_min_range_pct": _ordered_unique(_float_list(args.orb_min_range_pcts)),
        "vwap_min_volume_shares": _ordered_unique(_float_list(args.vwap_min_volumes)),
        "vwap_tolerance_pct": _ordered_unique(_float_list(args.vwap_tolerances)),
        "vwap_momentum_threshold_pct": _ordered_unique(_float_list(args.vwap_momentums)),
        "vwap_max_daily_trades": _ordered_unique(_int_list(args.vwap_max_daily_trades)),
        "vwap_cooldown_bars": _ordered_unique(_int_list(args.vwap_cooldowns)),
        "vwap_min_stop_pct": _ordered_unique(_float_list(args.vwap_min_stop_pcts)),
        "vwap_max_window_bars": _ordered_unique(_int_list(args.vwap_max_window_bars)),
    }


def _default_candidate(strategy: str) -> Dict[str, Any]:
    cfg = BacktestConfig(strategy=strategy)
    return {
        "strategy": strategy,
        "min_rank_score": cfg.min_rank_score,
        "orb_range_minutes": cfg.orb_range_minutes,
        "orb_breakout_buffer_pct": cfg.orb_breakout_buffer_pct,
        "orb_min_range_pct": cfg.orb_min_range_pct,
        "vwap_min_volume_shares": cfg.vwap_min_volume_shares,
        "vwap_tolerance_pct": cfg.vwap_tolerance_pct,
        "vwap_momentum_threshold_pct": cfg.vwap_momentum_threshold_pct,
        "vwap_max_daily_trades": cfg.vwap_max_daily_trades,
        "vwap_cooldown_bars": cfg.vwap_cooldown_bars,
        "vwap_min_stop_pct": cfg.vwap_min_stop_pct,
        "vwap_max_window_bars": cfg.vwap_max_window_bars,
    }


def _sample_candidate(
    strategy: str,
    spaces: Dict[str, List[Any]],
    rng: random.Random,
    seed_candidate: Dict[str, Any] | None,
    radius: int,
) -> Dict[str, Any]:
    defaults = _default_candidate(strategy)
    candidate = dict(defaults)
    for key, values in spaces.items():
        if key.startswith("orb_") and strategy == "vwap":
            candidate[key] = defaults[key]
            continue
        if key.startswith("vwap_") and strategy == "orb":
            candidate[key] = defaults[key]
            continue

        pool = list(values)
        if seed_candidate is not None and key in seed_candidate:
            narrowed = _neighbor_values(pool, seed_candidate[key], radius)
            if narrowed:
                pool = narrowed
        candidate[key] = rng.choice(pool)
    return candidate


def _candidate_key(candidate: Dict[str, Any]) -> tuple[Any, ...]:
    ordered_keys = [
        "strategy",
        "min_rank_score",
        "orb_range_minutes",
        "orb_breakout_buffer_pct",
        "orb_min_range_pct",
        "vwap_min_volume_shares",
        "vwap_tolerance_pct",
        "vwap_momentum_threshold_pct",
        "vwap_max_daily_trades",
        "vwap_cooldown_bars",
        "vwap_min_stop_pct",
        "vwap_max_window_bars",
    ]
    return tuple(candidate.get(key) for key in ordered_keys)


def _config_from_candidate(candidate: Dict[str, Any], args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        symbol=args.symbol,
        benchmark_symbol=args.benchmark_symbol,
        strategy=str(candidate["strategy"]),
        initial_capital=args.initial_capital,
        min_rank_score=float(candidate["min_rank_score"]),
        orb_range_minutes=int(candidate["orb_range_minutes"]),
        orb_min_range_pct=float(candidate["orb_min_range_pct"]),
        orb_breakout_buffer_pct=float(candidate["orb_breakout_buffer_pct"]),
        vwap_min_volume_shares=float(candidate["vwap_min_volume_shares"]),
        vwap_tolerance_pct=float(candidate["vwap_tolerance_pct"]),
        vwap_momentum_threshold_pct=float(candidate["vwap_momentum_threshold_pct"]),
        vwap_max_daily_trades=int(candidate["vwap_max_daily_trades"]),
        vwap_cooldown_bars=int(candidate["vwap_cooldown_bars"]),
        vwap_min_stop_pct=float(candidate["vwap_min_stop_pct"]),
        vwap_max_window_bars=int(candidate["vwap_max_window_bars"]),
    )


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = _date_col(df)
    work = df.copy()
    work["__ts__"] = ts
    return work[(work["__ts__"].dt.date >= start.date()) & (work["__ts__"].dt.date <= end.date())].drop(columns=["__ts__"])


async def _evaluate_candidate(
    df: pd.DataFrame,
    windows: List[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    candidate: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cfg = _config_from_candidate(candidate, args)
    results: List[Dict[str, Any]] = []
    for train_start, train_end, test_start, test_end in windows:
        train_df = _slice_window(df, train_start, train_end)
        test_df = _slice_window(df, test_start, test_end)
        train_result = await run_backtest(train_df, cfg)
        test_result = await run_backtest(test_df, cfg)
        results.append(
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "selected_parameters": test_result.get("parameters"),
                "selection_objective": None,
                "selection_profit_factor": train_result.get("profit_factor"),
                "candidates_evaluated": 1,
                "train_result": train_result,
                "test_result": test_result,
            }
        )

    payload: Dict[str, Any] = {
        "windows": len(results),
        "self_improve": False,
        "adaptive_keep_per_param": None,
        "total_candidates_evaluated": len(results),
        "candidate": candidate,
        "results": results,
    }
    report = build_report(payload, scope="test")
    summary = report["summary"] if isinstance(report.get("summary"), dict) else {}
    profit_factor = summary.get("profit_factor")
    if profit_factor is None:
        gross_profit = float(summary.get("gross_profit", 0.0) or 0.0)
        gross_loss = float(summary.get("gross_loss", 0.0) or 0.0)
        if gross_profit > 0 and gross_loss == 0:
            profit_factor = float("inf")

    return {
        "candidate": candidate,
        "payload": payload,
        "report": report,
        "profit_factor": profit_factor,
        "total_pnl": float(summary.get("total_pnl", 0.0) or 0.0),
        "total_trades": int(summary.get("total_trades", 0) or 0),
    }


def _score_evaluation(evaluation: Dict[str, Any]) -> float:
    profit_factor = evaluation.get("profit_factor")
    if profit_factor == float("inf"):
        pf_score = 1000.0
    else:
        pf_score = float(profit_factor or 0.0)
    total_pnl = float(evaluation.get("total_pnl", 0.0) or 0.0)
    total_trades = int(evaluation.get("total_trades", 0) or 0)
    return (pf_score * 1000.0) + total_pnl + (10.0 * total_trades)


async def _search(args: argparse.Namespace) -> Dict[str, Any]:
    df = _load_csv(Path(args.csv))
    ts = _date_col(df)
    days = sorted(set(pd.Series(ts.dropna().dt.normalize()).tolist()))
    windows = _window_ranges(days, args.train_days, args.test_days)
    if not windows:
        raise RuntimeError("Not enough data to build walk-forward windows for tuning.")

    spaces = _build_spaces(args)
    rng = random.Random(args.seed)
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    best: Dict[str, Any] | None = None
    history: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for strategy in strategies:
        strategy_best = best["candidate"] if best and best["candidate"].get("strategy") == strategy else None
        for iteration in range(args.iterations):
            radius = max(1, args.initial_radius - iteration)
            batch: List[Dict[str, Any]] = []
            while len(batch) < args.batch_size:
                seed_candidate = strategy_best if strategy_best is not None else (best["candidate"] if best is not None else None)
                candidate = _sample_candidate(strategy, spaces, rng, seed_candidate, radius)
                key = _candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                batch.append(candidate)

            for candidate in batch:
                evaluation = await _evaluate_candidate(df, windows, candidate, args)
                history.append(
                    {
                        "candidate": candidate,
                        "profit_factor": evaluation.get("profit_factor"),
                        "total_pnl": evaluation.get("total_pnl"),
                        "total_trades": evaluation.get("total_trades"),
                    }
                )
                if best is None or _score_evaluation(evaluation) > _score_evaluation(best):
                    best = evaluation
                    strategy_best = candidate
                    LOGGER.warning(
                        "New best strategy=%s pf=%s pnl=%.2f trades=%d",
                        candidate["strategy"],
                        "inf" if evaluation.get("profit_factor") == float("inf") else round(float(evaluation.get("profit_factor") or 0.0), 4),
                        float(evaluation.get("total_pnl", 0.0) or 0.0),
                        int(evaluation.get("total_trades", 0) or 0),
                    )

                profit_factor = evaluation.get("profit_factor")
                if profit_factor == float("inf") or (
                    profit_factor is not None
                    and float(profit_factor) >= float(args.target_pf)
                    and int(evaluation.get("total_trades", 0) or 0) >= int(args.min_trades)
                ):
                    return {
                        "target_pf": args.target_pf,
                        "achieved": True,
                        "best": evaluation,
                        "history": history,
                        "evaluations": len(history),
                    }

    return {
        "target_pf": args.target_pf,
        "achieved": False,
        "best": best,
        "history": history,
        "evaluations": len(history),
    }


def _serialize(data: Any) -> Any:
    if data == float("inf"):
        return "inf"
    if isinstance(data, dict):
        return {k: _serialize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize(v) for v in data]
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteratively search for walk-forward configurations that hit a target profit factor.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--train-days", type=int, default=4)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--target-pf", type=float, default=2.0)
    parser.add_argument("--min-trades", type=int, default=2)
    parser.add_argument("--strategies", default="vwap,both,orb")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--initial-radius", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rank-scores", default="0,1.5,3.0,4.75,6.0,7.0")
    parser.add_argument("--orb-range-minutes", default="5,10,15,20")
    parser.add_argument("--orb-breakout-buffers", default="0,0.0001,0.0003,0.0005,0.0008")
    parser.add_argument("--orb-min-range-pcts", default="0.0005,0.001,0.0015,0.002,0.0025")
    parser.add_argument("--vwap-min-volumes", default="5000,25000,50000,100000,250000")
    parser.add_argument("--vwap-tolerances", default="0.0005,0.0008,0.0012,0.0015,0.002,0.003")
    parser.add_argument("--vwap-momentums", default="0.0002,0.0005,0.0008,0.001,0.0015,0.002")
    parser.add_argument("--vwap-max-daily-trades", default="1,2,3")
    parser.add_argument("--vwap-cooldowns", default="2,4,8,12")
    parser.add_argument("--vwap-min-stop-pcts", default="0.0015,0.002,0.0025,0.003,0.004")
    parser.add_argument("--vwap-max-window-bars", default="3,5,8,12")
    parser.add_argument("--output", default="state/pf_target_tuner_result.json")
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    _configure_logging()
    result = await _search(args)
    serialized = _serialize(result)
    print(json.dumps(serialized, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    best = result.get("best")
    if isinstance(best, dict):
        payload_out = out.with_name(out.stem + "_walkforward.json")
        report_out = out.with_name(out.stem + "_report.json")
        payload_out.write_text(json.dumps(_serialize(best.get("payload")), indent=2), encoding="utf-8")
        report_out.write_text(json.dumps(_serialize(best.get("report")), indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(_main_async())
