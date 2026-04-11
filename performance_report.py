from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_walkforward(payload: Dict[str, object]) -> bool:
    return isinstance(payload.get("results"), list) and ("windows" in payload or payload.get("portfolio_batch") is True)


def _collect_runs(payload: Dict[str, object], scope: str) -> List[Tuple[str, Dict[str, object]]]:
    if not _is_walkforward(payload):
        return [("backtest", payload)]

    runs: List[Tuple[str, Dict[str, object]]] = []
    for idx, row in enumerate(payload.get("results", []), start=1):
        if scope in {"train", "both"} and isinstance(row.get("train_result"), dict):
            runs.append((f"train_{idx}", row["train_result"]))
        if scope in {"test", "both"} and isinstance(row.get("test_result"), dict):
            runs.append((f"test_{idx}", row["test_result"]))
    return runs


def _aggregate_strategy_breakdown(runs: List[Tuple[str, Dict[str, object]]]) -> Dict[str, Dict[str, object]]:
    aggregate: Dict[str, Dict[str, object]] = {}
    for _, run in runs:
        for strategy, row in (run.get("strategy_breakdown") or {}).items():
            bucket = aggregate.setdefault(
                strategy,
                {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, "exposure_minutes": 0.0},
            )
            bucket["trades"] += int(row.get("trades", 0) or 0)
            bucket["pnl"] += float(row.get("pnl", 0.0) or 0.0)
            bucket["wins"] += int(row.get("wins", 0) or 0)
            bucket["losses"] += int(row.get("losses", 0) or 0)
            bucket["exposure_minutes"] += float(row.get("exposure_minutes", 0.0) or 0.0)

    for strategy, row in aggregate.items():
        row["pnl"] = round(float(row["pnl"]), 2)
        row["win_rate"] = round((row["wins"] / row["trades"]) if row["trades"] else 0.0, 4)
        row["exposure_minutes"] = round(float(row["exposure_minutes"]), 2)
    return aggregate


def _aggregate_trades(runs: List[Tuple[str, Dict[str, object]]]) -> List[Dict[str, object]]:
    trades: List[Dict[str, object]] = []
    for run_name, run in runs:
        for trade in run.get("trade_log") or []:
            enriched = dict(trade)
            enriched["run"] = run_name
            trades.append(enriched)
    return trades


def build_report(payload: Dict[str, object], scope: str) -> Dict[str, object]:
    runs = _collect_runs(payload, scope=scope)
    if not runs:
        raise RuntimeError("No runs found for requested scope.")

    total_pnl = sum(float(run.get("realized_pnl", 0.0) or 0.0) for _, run in runs)
    total_trades = sum(int(run.get("trades", 0) or 0) for _, run in runs)
    total_wins = sum(int(run.get("wins", 0) or 0) for _, run in runs)
    total_losses = sum(int(run.get("losses", 0) or 0) for _, run in runs)
    total_gross_profit = sum(float(run.get("gross_profit", 0.0) or 0.0) for _, run in runs)
    total_gross_loss = sum(float(run.get("gross_loss", 0.0) or 0.0) for _, run in runs)
    total_exposure_minutes = sum(float(run.get("gross_exposure_minutes", 0.0) or 0.0) for _, run in runs)
    total_return_pct_values = [float(run.get("total_return_pct", 0.0) or 0.0) for _, run in runs if run.get("total_return_pct") is not None]
    benchmark_return_pct_values = [float(run.get("benchmark_total_return_pct", 0.0) or 0.0) for _, run in runs if run.get("benchmark_total_return_pct") is not None]
    excess_return_pct_values = [float(run.get("excess_return_pct", 0.0) or 0.0) for _, run in runs if run.get("excess_return_pct") is not None]
    annualized_sharpe_values = [float(run.get("annualized_sharpe")) for _, run in runs if run.get("annualized_sharpe") is not None]
    benchmark_sharpe_values = [float(run.get("benchmark_annualized_sharpe")) for _, run in runs if run.get("benchmark_annualized_sharpe") is not None]
    info_ratio_values = [float(run.get("information_ratio")) for _, run in runs if run.get("information_ratio") is not None]

    all_daily_returns: List[float] = []
    all_benchmark_daily_returns: List[float] = []
    for _, run in runs:
        all_daily_returns.extend(float(x) for x in (run.get("daily_returns") or []))
        all_benchmark_daily_returns.extend(float(x) for x in (run.get("benchmark_daily_returns") or []))

    trades = _aggregate_trades(runs)
    avg_trade_pnl = (total_pnl / total_trades) if total_trades else 0.0
    expectancy = avg_trade_pnl
    win_rate = (total_wins / total_trades) if total_trades else 0.0
    profit_factor = None if total_gross_loss == 0 else total_gross_profit / abs(total_gross_loss)

    duration_values = [float(t["duration_sec"]) for t in trades if t.get("duration_sec") is not None]
    avg_duration_min = (sum(duration_values) / len(duration_values) / 60.0) if duration_values else None

    return_pct_values = [float(t.get("return_pct", 0.0) or 0.0) for t in trades]
    avg_return_pct = (sum(return_pct_values) / len(return_pct_values)) if return_pct_values else 0.0

    max_drawdown_worst = min(float(run.get("max_drawdown", 0.0) or 0.0) for _, run in runs)
    sharpe_values = [float(run.get("sharpe_like")) for _, run in runs if run.get("sharpe_like") is not None]
    sharpe_avg = (sum(sharpe_values) / len(sharpe_values)) if sharpe_values else None

    strategy_breakdown = _aggregate_strategy_breakdown(runs)

    wf_selection_pf_values = [
        float(row.get("selection_profit_factor"))
        for row in (payload.get("results") or [])
        if isinstance(row, dict) and row.get("selection_profit_factor") is not None
    ]
    wf_candidates = [
        int(row.get("candidates_evaluated", 0) or 0)
        for row in (payload.get("results") or [])
        if isinstance(row, dict)
    ]

    return {
        "scope": scope,
        "runs_analyzed": len(runs),
        "summary": {
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate": round(win_rate, 4),
            "expectancy": round(expectancy, 4),
            "avg_trade_pnl": round(avg_trade_pnl, 4),
            "avg_trade_return_pct": round(avg_return_pct, 6),
            "gross_profit": round(total_gross_profit, 2),
            "gross_loss": round(total_gross_loss, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "avg_duration_min": round(avg_duration_min, 2) if avg_duration_min is not None else None,
            "gross_exposure_minutes": round(total_exposure_minutes, 2),
            "worst_max_drawdown": round(max_drawdown_worst, 6),
            "avg_sharpe_like": round(sharpe_avg, 4) if sharpe_avg is not None else None,
            "avg_total_return_pct": round(sum(total_return_pct_values) / len(total_return_pct_values), 6) if total_return_pct_values else None,
            "avg_benchmark_return_pct": round(sum(benchmark_return_pct_values) / len(benchmark_return_pct_values), 6) if benchmark_return_pct_values else None,
            "avg_excess_return_pct": round(sum(excess_return_pct_values) / len(excess_return_pct_values), 6) if excess_return_pct_values else None,
            "avg_annualized_sharpe": round(sum(annualized_sharpe_values) / len(annualized_sharpe_values), 4) if annualized_sharpe_values else None,
            "avg_benchmark_annualized_sharpe": round(sum(benchmark_sharpe_values) / len(benchmark_sharpe_values), 4) if benchmark_sharpe_values else None,
            "avg_information_ratio": round(sum(info_ratio_values) / len(info_ratio_values), 4) if info_ratio_values else None,
            "combined_daily_return_count": len(all_daily_returns),
            "combined_benchmark_daily_return_count": len(all_benchmark_daily_returns),
            "avg_walkforward_selection_pf": round(sum(wf_selection_pf_values) / len(wf_selection_pf_values), 4)
            if wf_selection_pf_values
            else None,
            "total_walkforward_candidates_evaluated": sum(wf_candidates) if wf_candidates else None,
        },
        "benchmark_comparison": {
            "runs_with_benchmark": len(benchmark_return_pct_values),
            "avg_benchmark_return_pct": round(sum(benchmark_return_pct_values) / len(benchmark_return_pct_values), 6) if benchmark_return_pct_values else None,
            "avg_excess_return_pct": round(sum(excess_return_pct_values) / len(excess_return_pct_values), 6) if excess_return_pct_values else None,
            "avg_information_ratio": round(sum(info_ratio_values) / len(info_ratio_values), 4) if info_ratio_values else None,
        },
        "walkforward_meta": {
            "self_improve": payload.get("self_improve") if isinstance(payload, dict) else None,
            "adaptive_keep_per_param": payload.get("adaptive_keep_per_param") if isinstance(payload, dict) else None,
            "avg_test_profit_factor": payload.get("avg_test_profit_factor") if isinstance(payload, dict) else None,
            "total_candidates_evaluated": payload.get("total_candidates_evaluated") if isinstance(payload, dict) else None,
        },
        "strategy_breakdown": strategy_breakdown,
        "top_trades": sorted(trades, key=lambda x: float(x.get("pnl", 0.0) or 0.0), reverse=True)[:10],
        "bottom_trades": sorted(trades, key=lambda x: float(x.get("pnl", 0.0) or 0.0))[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio-style performance report from HQC outputs.")
    parser.add_argument("--input", required=True, help="Path to backtest_result.json or walkforward_result.json")
    parser.add_argument("--scope", choices=["train", "test", "both"], default="test")
    parser.add_argument("--output", default="", help="Optional path to write report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_json(Path(args.input))
    scope = args.scope if _is_walkforward(payload) else "backtest"
    report = build_report(payload, scope=scope)
    print(json.dumps(report, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()