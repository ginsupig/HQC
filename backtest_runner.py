from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from core.engine.event_bus import Event, EventBus, EventType
from core.execution.broker_router import AlpacaExecutionRouter
from core.execution.eod_liquidator import EODLiquidationManager
from core.execution.slippage_controller import SlippageController
from intelligence.candidate_ranker import CandidateRanker
from risk.position_sizing.confidence_scaler import DynamicRiskSizer
from strategies.orb.equity_orb import USEquityORB
from strategies.vwap.hunter_state_machine import USEquityVWAPHunter


@dataclass
class BacktestConfig:
    symbol: str = "SPY"
    benchmark_symbol: str = "SPY"
    strategy: str = "both"
    initial_capital: float = 100000.0
    min_rank_score: float = 4.75
    orb_range_minutes: int = 15
    orb_max_trades: int = 2
    orb_cooldown_bars: int = 10
    orb_min_range_pct: float = 0.0025
    orb_breakout_buffer_pct: float = 0.0005
    vwap_min_volume_shares: float = 250000.0
    vwap_tolerance_pct: float = 0.002
    vwap_momentum_threshold_pct: float = 0.005
    vwap_max_daily_trades: int = 3
    vwap_cooldown_bars: int = 8
    vwap_min_stop_pct: float = 0.003
    vwap_max_window_bars: int = 8


class TradeLedger:
    def __init__(self, initial_equity: float) -> None:
        self.initial_equity = float(initial_equity)
        self.equity = float(initial_equity)
        self.positions: Dict[str, Dict[str, object]] = {}
        self.realized_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.equity_curve: List[float] = [self.equity]
        self.closed_trades: List[Dict[str, object]] = []
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.duration_total_sec = 0.0
        self.duration_count = 0
        self.strategy_stats: Dict[str, Dict[str, object]] = {}

    async def on_fill(self, event: Event) -> None:
        payload = event.payload or {}
        status = str(payload.get("status", "")).upper()
        if status in {"CANCELED", "CANCELLED", "REJECTED", "ERROR"}:
            return

        symbol = str(payload.get("asset") or payload.get("symbol") or "").upper()
        action = str(payload.get("action") or payload.get("side") or "").upper()
        strategy = str(payload.get("strategy") or "Unknown")

        try:
            qty = int(float(payload.get("fill_qty", payload.get("filled_qty", 0)) or 0))
            price = float(payload.get("fill_price", payload.get("entry_price", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return

        fill_ts = self._normalize_ts(payload.get("timestamp"))

        if not symbol or qty <= 0 or price <= 0:
            return

        if symbol not in self.positions:
            self.positions[symbol] = {
                "qty": 0.0,
                "avg": 0.0,
                "entry_ts": None,
                "strategy": strategy,
                "side": None,
            }

        pos = self.positions[symbol]
        signed_qty = qty if action in {"BUY", "BUY_TO_OPEN", "BUY_TO_COVER"} else -qty

        if pos["qty"] == 0 or (pos["qty"] > 0 and signed_qty > 0) or (pos["qty"] < 0 and signed_qty < 0):
            total_cost = abs(pos["qty"]) * pos["avg"] + abs(signed_qty) * price
            pos["qty"] += signed_qty
            pos["avg"] = total_cost / abs(pos["qty"]) if pos["qty"] != 0 else 0.0
            if pos["entry_ts"] is None:
                pos["entry_ts"] = fill_ts
                pos["strategy"] = strategy
                pos["side"] = "LONG" if signed_qty > 0 else "SHORT"
            return

        close_qty = min(abs(pos["qty"]), abs(signed_qty))
        direction = 1.0 if pos["qty"] > 0 else -1.0
        pnl = (price - pos["avg"]) * close_qty * direction
        self.realized_pnl += pnl
        self.equity += pnl
        self.equity_curve.append(self.equity)
        self.trades += 1
        if pnl > 0:
            self.wins += 1
            self.gross_profit += pnl
        elif pnl < 0:
            self.losses += 1
            self.gross_loss += pnl

        entry_ts = pos.get("entry_ts")
        duration_sec = None
        if isinstance(entry_ts, int) and isinstance(fill_ts, int):
            duration_sec = max(0, (fill_ts - entry_ts) / 1000.0)

        basis = abs(float(pos["avg"]) * close_qty)
        return_pct = (pnl / basis) if basis > 0 else 0.0
        self.closed_trades.append(
            {
                "symbol": symbol,
                "strategy": str(pos.get("strategy") or strategy),
                "side": str(pos.get("side") or ("LONG" if direction > 0 else "SHORT")),
                "qty": int(close_qty),
                "entry_price": round(float(pos["avg"]), 4),
                "exit_price": round(price, 4),
                "entry_ts": self._ts_to_iso(entry_ts),
                "exit_ts": self._ts_to_iso(fill_ts),
                "duration_sec": round(duration_sec, 2) if duration_sec is not None else None,
                "pnl": round(pnl, 2),
                "return_pct": round(return_pct, 6),
            }
        )

        if duration_sec is not None:
            self.duration_total_sec += float(duration_sec)
            self.duration_count += 1

        strategy_key = str(pos.get("strategy") or strategy)
        stats = self.strategy_stats.setdefault(
            strategy_key,
            {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, "exposure_minutes": 0.0},
        )
        stats["trades"] += 1
        stats["pnl"] += float(pnl)
        if pnl > 0:
            stats["wins"] += 1
        elif pnl < 0:
            stats["losses"] += 1
        if duration_sec is not None:
            stats["exposure_minutes"] += float(duration_sec) / 60.0

        old_qty = pos["qty"]
        pos["qty"] += signed_qty
        if pos["qty"] == 0:
            pos["avg"] = 0.0
            pos["entry_ts"] = None
            pos["side"] = None
        elif (old_qty > 0 > pos["qty"]) or (old_qty < 0 < pos["qty"]):
            pos["avg"] = price
            pos["entry_ts"] = fill_ts
            pos["strategy"] = strategy
            pos["side"] = "LONG" if pos["qty"] > 0 else "SHORT"

    def max_drawdown(self) -> float:
        peak = self.equity_curve[0]
        max_dd = 0.0
        for x in self.equity_curve:
            if x > peak:
                peak = x
            if peak > 0:
                dd = (x - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        return max_dd

    def snapshot(self, start_ts: Optional[pd.Timestamp] = None, end_ts: Optional[pd.Timestamp] = None) -> Dict[str, object]:
        win_rate = (self.wins / self.trades) if self.trades else 0.0
        gross_profit = self.gross_profit
        gross_loss = self.gross_loss
        avg_pnl = (self.realized_pnl / self.trades) if self.trades else 0.0
        expectancy = avg_pnl
        profit_factor = None if gross_loss == 0 else gross_profit / abs(gross_loss)
        avg_duration_min = None
        if self.duration_count:
            avg_duration_min = self.duration_total_sec / self.duration_count / 60.0

        exposure_minutes_total = self.duration_total_sec / 60.0 if self.duration_count else 0.0
        span_minutes = None
        if start_ts is not None and end_ts is not None:
            span_minutes = max(0.0, (end_ts - start_ts).total_seconds() / 60.0)

        strategy_breakdown = {
            strategy: {
                "trades": int(row["trades"]),
                "pnl": float(row["pnl"]),
                "wins": int(row["wins"]),
                "losses": int(row["losses"]),
                "exposure_minutes": float(row["exposure_minutes"]),
            }
            for strategy, row in self.strategy_stats.items()
        }

        for strategy, row in strategy_breakdown.items():
            row["pnl"] = round(float(row["pnl"]), 2)
            row["win_rate"] = round((row["wins"] / row["trades"]) if row["trades"] else 0.0, 4)
            if span_minutes and span_minutes > 0:
                row["exposure_ratio"] = round(float(row["exposure_minutes"]) / span_minutes, 6)
            row["exposure_minutes"] = round(float(row["exposure_minutes"]), 2)

        equity_returns: List[float] = []
        for idx in range(1, len(self.equity_curve)):
            prev_val = self.equity_curve[idx - 1]
            curr_val = self.equity_curve[idx]
            if prev_val > 0:
                equity_returns.append((curr_val / prev_val) - 1.0)
        sharpe_like = None
        if len(equity_returns) >= 2:
            mean_ret = sum(equity_returns) / len(equity_returns)
            variance = sum((x - mean_ret) ** 2 for x in equity_returns) / max(1, len(equity_returns) - 1)
            std_ret = variance ** 0.5
            if std_ret > 0:
                sharpe_like = mean_ret / std_ret

        return {
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(self.equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(win_rate, 4),
            "max_drawdown": round(self.max_drawdown(), 6),
            "expectancy": round(expectancy, 4),
            "avg_trade_pnl": round(avg_pnl, 4),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "avg_duration_min": round(avg_duration_min, 2) if avg_duration_min is not None else None,
            "gross_exposure_minutes": round(exposure_minutes_total, 2),
            "exposure_ratio": round((exposure_minutes_total / span_minutes), 6) if span_minutes and span_minutes > 0 else None,
            "sharpe_like": round(sharpe_like, 4) if sharpe_like is not None else None,
            "strategy_breakdown": strategy_breakdown,
            "trade_log": self.closed_trades,
        }

    @staticmethod
    def _normalize_ts(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            ts = int(float(value))
            return ts if ts > 10_000_000_000 else ts * 1000
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ts_to_iso(value: object) -> Optional[str]:
        ts = TradeLedger._normalize_ts(value)
        if ts is None:
            return None
        return str(pd.to_datetime(ts, unit="ms", utc=True))


def _normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename_map = {
        "datetime": "timestamp",
        "date": "timestamp",
        "time": "timestamp",
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "ticker": "symbol",
    }
    work = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    if "symbol" not in work.columns:
        work["symbol"] = symbol
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work = work[work["symbol"] == symbol.upper()].copy()

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    return work


def _bar_to_ticks(row: pd.Series, symbol: str) -> List[Event]:
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    v = max(1.0, float(row["volume"]) / 4.0)
    ts_ms = int(pd.Timestamp(row["timestamp"]).timestamp() * 1000)

    path = [o, l, h, c] if c >= o else [o, h, l, c]
    ticks: List[Event] = []
    for i, px in enumerate(path):
        ticks.append(
            Event(
                type=EventType.TICK,
                payload={
                    "ticker": symbol,
                    "symbol": symbol,
                    "price": float(px),
                    "volume": float(v),
                    "timestamp": ts_ms + (i * 15000),
                },
            )
        )
    return ticks


def _safe_normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    try:
        return _normalize_bars(df, symbol)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])


def _build_daily_equity(initial_equity: float, closed_trades: List[Dict[str, object]], trading_days: List[pd.Timestamp]) -> List[Dict[str, object]]:
    pnl_by_day: Dict[str, float] = {}
    for trade in closed_trades:
        exit_ts = trade.get("exit_ts")
        if not exit_ts:
            continue
        day_key = str(pd.to_datetime(exit_ts, utc=True).normalize().date())
        pnl_by_day[day_key] = pnl_by_day.get(day_key, 0.0) + float(trade.get("pnl", 0.0) or 0.0)

    equity = float(initial_equity)
    series: List[Dict[str, object]] = []
    for day in trading_days:
        day_key = str(pd.Timestamp(day).date())
        equity += pnl_by_day.get(day_key, 0.0)
        series.append({"date": day_key, "equity": round(equity, 2)})
    return series


def _daily_returns_from_equity(daily_equity: List[Dict[str, object]]) -> List[float]:
    returns: List[float] = []
    for idx in range(1, len(daily_equity)):
        prev_val = float(daily_equity[idx - 1]["equity"])
        curr_val = float(daily_equity[idx]["equity"])
        if prev_val > 0:
            returns.append((curr_val / prev_val) - 1.0)
    return returns


def _annualized_sharpe(daily_returns: List[float]) -> Optional[float]:
    if len(daily_returns) < 2:
        return None
    mean_ret = sum(daily_returns) / len(daily_returns)
    variance = sum((x - mean_ret) ** 2 for x in daily_returns) / max(1, len(daily_returns) - 1)
    std_ret = variance ** 0.5
    if std_ret <= 0:
        return None
    return (mean_ret / std_ret) * (252 ** 0.5)


def _benchmark_metrics(bars: pd.DataFrame) -> Dict[str, object]:
    if bars.empty:
        return {
            "benchmark_total_return_pct": None,
            "benchmark_daily_returns": [],
            "benchmark_daily_close": [],
            "benchmark_annualized_sharpe": None,
        }

    daily = (
        bars.assign(day=bars["timestamp"].dt.normalize())
        .groupby("day", as_index=False)
        .agg(open=("open", "first"), close=("close", "last"))
    )
    total_return_pct = None
    if len(daily) >= 1:
        first_open = float(daily["open"].iloc[0])
        last_close = float(daily["close"].iloc[-1])
        if first_open > 0:
            total_return_pct = (last_close / first_open) - 1.0

    daily_returns: List[float] = []
    for idx in range(1, len(daily)):
        prev_close = float(daily["close"].iloc[idx - 1])
        curr_close = float(daily["close"].iloc[idx])
        if prev_close > 0:
            daily_returns.append((curr_close / prev_close) - 1.0)

    return {
        "benchmark_total_return_pct": round(total_return_pct, 6) if total_return_pct is not None else None,
        "benchmark_daily_returns": [round(x, 6) for x in daily_returns],
        "benchmark_daily_close": [
            {"date": str(pd.Timestamp(row.day).date()), "close": round(float(row.close), 4)}
            for row in daily.itertuples(index=False)
        ],
        "benchmark_annualized_sharpe": round(_annualized_sharpe(daily_returns), 4) if _annualized_sharpe(daily_returns) is not None else None,
    }


async def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, object]:
    symbol = cfg.symbol.upper()
    benchmark_symbol = (cfg.benchmark_symbol or cfg.symbol).upper()
    bars = _normalize_bars(df, symbol)
    benchmark_bars = _safe_normalize_bars(df, benchmark_symbol)

    bus = EventBus()
    await bus.start()

    _orb = None
    _vwap = None
    if cfg.strategy in {"orb", "both"}:
        _orb = USEquityORB(
            target_asset=symbol,
            bus=bus,
            range_minutes=cfg.orb_range_minutes,
            max_trades=cfg.orb_max_trades,
            cooldown_bars=cfg.orb_cooldown_bars,
            min_range_pct=cfg.orb_min_range_pct,
            breakout_buffer_pct=cfg.orb_breakout_buffer_pct,
        )
    if cfg.strategy in {"vwap", "both"}:
        _vwap = USEquityVWAPHunter(
            target_asset=symbol,
            bus=bus,
            min_volume_shares=cfg.vwap_min_volume_shares,
            vwap_tolerance_pct=cfg.vwap_tolerance_pct,
            momentum_threshold_pct=cfg.vwap_momentum_threshold_pct,
            max_daily_trades=cfg.vwap_max_daily_trades,
            cooldown_bars=cfg.vwap_cooldown_bars,
            min_stop_pct=cfg.vwap_min_stop_pct,
            max_window_bars=cfg.vwap_max_window_bars,
        )

    _ranker = CandidateRanker(bus=bus, benchmark=benchmark_symbol, min_score=cfg.min_rank_score)
    _sizer = DynamicRiskSizer(bus=bus, account_equity=cfg.initial_capital)
    _slip = SlippageController(bus=bus)
    router = AlpacaExecutionRouter(
        api_key="SIM_KEY",
        api_secret="SIM_SECRET",
        bus=bus,
        is_paper=True,
        simulate_only=True,
        slippage_controller=_slip,
    )
    eod = EODLiquidationManager(bus=bus)
    ledger = TradeLedger(initial_equity=cfg.initial_capital)

    bus.subscribe(EventType.ORDER_FILL, ledger.on_fill)

    await _slip.start()
    await router.start()

    for _, row in bars.iterrows():
        for tick in _bar_to_ticks(row, symbol):
            bus.publish(tick)
        await asyncio.sleep(0)

    await bus._queue.join()
    await eod.force_liquidate_now(reason="backtest_end")
    await asyncio.sleep(0)
    await bus._queue.join()

    await router.stop()
    await _slip.stop()
    await bus.stop()

    start_ts = bars["timestamp"].iloc[0] if len(bars) else None
    end_ts = bars["timestamp"].iloc[-1] if len(bars) else None
    result = ledger.snapshot(start_ts=start_ts, end_ts=end_ts)
    trading_days = list(pd.Series(bars["timestamp"].dt.normalize().drop_duplicates()).tolist())
    daily_equity = _build_daily_equity(cfg.initial_capital, ledger.closed_trades, trading_days)
    daily_returns = _daily_returns_from_equity(daily_equity)
    annualized_sharpe = _annualized_sharpe(daily_returns)
    total_return_pct = (float(result["final_equity"]) / cfg.initial_capital) - 1.0 if cfg.initial_capital > 0 else 0.0
    bench = _benchmark_metrics(benchmark_bars if not benchmark_bars.empty else bars)
    benchmark_total_return_pct = bench.get("benchmark_total_return_pct")
    excess_return_pct = None
    if benchmark_total_return_pct is not None:
        excess_return_pct = total_return_pct - float(benchmark_total_return_pct)

    info_ratio = None
    benchmark_daily_returns = list(bench.get("benchmark_daily_returns") or [])
    if len(daily_returns) >= 2 and len(benchmark_daily_returns) >= 2:
        paired = list(zip(daily_returns[: len(benchmark_daily_returns)], benchmark_daily_returns[: len(daily_returns)]))
        if len(paired) >= 2:
            excess_daily = [a - b for a, b in paired]
            mean_excess = sum(excess_daily) / len(excess_daily)
            variance = sum((x - mean_excess) ** 2 for x in excess_daily) / max(1, len(excess_daily) - 1)
            std_excess = variance ** 0.5
            if std_excess > 0:
                info_ratio = (mean_excess / std_excess) * (252 ** 0.5)

    result.update(
        {
            "symbol": symbol,
            "benchmark_symbol": benchmark_symbol,
            "strategy": cfg.strategy,
            "parameters": {
                "min_rank_score": cfg.min_rank_score,
                "orb_range_minutes": cfg.orb_range_minutes,
                "orb_max_trades": cfg.orb_max_trades,
                "orb_cooldown_bars": cfg.orb_cooldown_bars,
                "orb_min_range_pct": cfg.orb_min_range_pct,
                "orb_breakout_buffer_pct": cfg.orb_breakout_buffer_pct,
                "vwap_min_volume_shares": cfg.vwap_min_volume_shares,
                "vwap_tolerance_pct": cfg.vwap_tolerance_pct,
                "vwap_momentum_threshold_pct": cfg.vwap_momentum_threshold_pct,
                "vwap_max_daily_trades": cfg.vwap_max_daily_trades,
                "vwap_cooldown_bars": cfg.vwap_cooldown_bars,
                "vwap_min_stop_pct": cfg.vwap_min_stop_pct,
                "vwap_max_window_bars": cfg.vwap_max_window_bars,
            },
            "bars": int(len(bars)),
            "from": str(start_ts) if start_ts is not None else None,
            "to": str(end_ts) if end_ts is not None else None,
            "daily_equity": daily_equity,
            "daily_returns": [round(x, 6) for x in daily_returns],
            "annualized_sharpe": round(annualized_sharpe, 4) if annualized_sharpe is not None else None,
            "total_return_pct": round(total_return_pct, 6),
            "benchmark_total_return_pct": benchmark_total_return_pct,
            "benchmark_daily_returns": benchmark_daily_returns,
            "benchmark_daily_close": bench.get("benchmark_daily_close") or [],
            "benchmark_annualized_sharpe": bench.get("benchmark_annualized_sharpe"),
            "excess_return_pct": round(excess_return_pct, 6) if excess_return_pct is not None else None,
            "information_ratio": round(info_ratio, 4) if info_ratio is not None else None,
        }
    )
    return result


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


async def _main_async(args: argparse.Namespace) -> None:
    df = _load_csv(Path(args.csv))
    cfg = BacktestConfig(
        symbol=args.symbol,
        benchmark_symbol=args.benchmark_symbol,
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
    print(json.dumps(result, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run offline HQC backtest with simulated execution.")
    p.add_argument("--csv", required=True, help="Path to OHLCV csv file.")
    p.add_argument("--symbol", default="SPY", help="Ticker to backtest.")
    p.add_argument("--benchmark-symbol", default="SPY", help="Benchmark ticker to use if present in CSV.")
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
    p.add_argument("--output", default="", help="Optional path to write JSON result.")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main_async(parse_args()))
