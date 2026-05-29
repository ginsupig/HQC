from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
from strategies.gap_fade.overnight_gap_fade import OvernightGapFade
from strategies.mean_reversion.kalman_spread import USEquityKalmanPairsTrader
from strategies.orb.equity_orb import USEquityORB
from strategies.vwap.hunter_state_machine import USEquityVWAPHunter

logger = logging.getLogger("BacktestRunner")


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
    # Backtest-only execution simulation knobs.
    # Live trading uses Alpaca's bracket OTO; the offline harness has to
    # simulate the protective stop and an upper bound on intraday holds.
    sim_max_hold_minutes: int = 240
    sim_stop_buffer_ticks: float = 0.0
    # Transaction cost model. Default values are conservative for an
    # Alpaca paper / live equity account: 1.5 bps slippage each leg
    # (~3 bps round-trip), zero commission per share, and the SEC fee
    # on sells (0.000008 of notional, only charged on the sell side).
    # Override these to stress-test cost assumptions.
    slippage_bps_per_side: float = 1.5
    commission_per_share: float = 0.0
    commission_min_per_trade: float = 0.0
    sec_fee_rate: float = 0.000008
    # Annual short-borrow rate. Liquid large caps ~0.25%; HTB names higher.
    short_borrow_apr: float = 0.0025
    # Overnight gap-fade baseline (used as a known-edge sanity check).
    gap_fade_trigger_pct: float = 0.005
    gap_fade_stop_pct: float = 0.012
    gap_fade_max_trades_per_day: int = 1
    # ML candidate-gate (backtest-only). When use_ml_gate is True the
    # harness fits an MLCandidateGate on the leading ml_train_bars rows
    # of the dataset (in-sample) and uses it to veto rule-based
    # candidates that have a probability disagreeing with their
    # direction. The remaining bars are the OOS test set.
    use_ml_gate: bool = False
    ml_train_bars: int = 5000
    ml_threshold: float = 0.55
    ml_horizon: int = 5
    # Kalman pairs trader (mean-reversion in spread). Requires a second
    # tradable symbol; pass hedge_symbol and either include both symbols'
    # bars in the input CSV or supply a separate --hedge-csv.
    hedge_symbol: str = ""
    pair_entry_z: float = 2.0
    pair_exit_z: float = 0.5
    pair_delta: float = 1e-4
    pair_ve: float = 1e-3
    pair_max_leg_staleness_sec: float = 30.0
    pair_cooldown_seconds: float = 5.0
    pair_nominal_stop_pct: float = 0.02
    pair_target_dollar_notional: float = 10000.0


class TradeLedger:
    def __init__(
        self,
        initial_equity: float,
        slippage_bps_per_side: float = 0.0,
        commission_per_share: float = 0.0,
        commission_min_per_trade: float = 0.0,
        sec_fee_rate: float = 0.0,
        short_borrow_apr: float = 0.0025,
    ) -> None:
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
        # Pre-cost figures so we can show how much edge friction is eating.
        self.gross_realized_pnl = 0.0
        self.total_costs = 0.0
        self.duration_total_sec = 0.0
        self.duration_count = 0
        self.strategy_stats: Dict[str, Dict[str, object]] = {}
        # Transaction cost model. Slippage is applied to the opposite-side
        # leg (the close) as a haircut on close_qty * exit_price; the
        # entry-leg slippage is charged at close time too so a single
        # round-trip is fully accounted in one place.
        self.slippage_bps_per_side = float(slippage_bps_per_side)
        self.commission_per_share = float(commission_per_share)
        self.commission_min_per_trade = float(commission_min_per_trade)
        self.sec_fee_rate = float(sec_fee_rate)
        # Annual short-borrow rate. Charged per second of holding time on the
        # short leg of any round trip. Default 0.25% APR is a reasonable
        # easy-to-borrow rate for liquid large-cap names (JPM, BAC, GOOG,
        # GOOGL, NVDA, AMD, etc). Hard-to-borrow names can be 1%-50%+, in
        # which case the strategy probably shouldn't trade them.
        self.short_borrow_apr = float(short_borrow_apr)

    def _round_trip_cost(
        self,
        close_qty: int,
        entry_price: float,
        exit_price: float,
        side_is_long: bool,
        duration_sec: Optional[float] = None,
    ) -> float:
        """Total dollar friction charged when a round-trip closes."""
        if close_qty <= 0:
            return 0.0
        slip_pct = self.slippage_bps_per_side / 10000.0
        entry_slip = slip_pct * entry_price * close_qty
        exit_slip = slip_pct * exit_price * close_qty
        entry_comm = max(self.commission_per_share * close_qty, self.commission_min_per_trade)
        exit_comm = max(self.commission_per_share * close_qty, self.commission_min_per_trade)
        # SEC fee is on the proceeds of a sale (the sell leg).
        # For a long round-trip the sell is the exit; for a short the sell is the entry.
        sec_notional = exit_price * close_qty if side_is_long else entry_price * close_qty
        sec_fee = self.sec_fee_rate * sec_notional
        # Borrow fee: only charged on the short leg, prorated by hold time.
        # avg_short_notional is approximated by the entry-side price for shorts;
        # for longs the borrow charge is zero.
        borrow_fee = 0.0
        if not side_is_long and duration_sec and duration_sec > 0 and self.short_borrow_apr > 0:
            avg_notional = entry_price * close_qty
            borrow_fee = avg_notional * self.short_borrow_apr * (duration_sec / (365.0 * 24.0 * 3600.0))
        return entry_slip + exit_slip + entry_comm + exit_comm + sec_fee + borrow_fee

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
        gross_pnl = (price - pos["avg"]) * close_qty * direction
        entry_ts = pos.get("entry_ts")
        duration_sec = None
        if isinstance(entry_ts, int) and isinstance(fill_ts, int):
            duration_sec = max(0, (fill_ts - entry_ts) / 1000.0)
        cost = self._round_trip_cost(
            close_qty=int(close_qty),
            entry_price=float(pos["avg"]),
            exit_price=float(price),
            side_is_long=(direction > 0),
            duration_sec=duration_sec,
        )
        pnl = gross_pnl - cost
        self.realized_pnl += pnl
        self.gross_realized_pnl += gross_pnl
        self.total_costs += cost
        self.equity += pnl
        self.equity_curve.append(self.equity)
        self.trades += 1
        if pnl > 0:
            self.wins += 1
            self.gross_profit += pnl
        elif pnl < 0:
            self.losses += 1
            self.gross_loss += pnl

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
                "gross_pnl": round(gross_pnl, 2),
                "cost": round(cost, 2),
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
            "gross_realized_pnl": round(self.gross_realized_pnl, 2),
            "total_costs": round(self.total_costs, 2),
            "cost_drag_pct": round(
                (self.total_costs / self.initial_equity) if self.initial_equity > 0 else 0.0, 6
            ),
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


_BAR_CLOSE_OFFSET_MS = 30_000


def _bar_to_ticks(row: pd.Series, symbol: str) -> List[Event]:
    o = float(row["open"])
    c = float(row["close"])
    v = max(1.0, float(row["volume"]) / 2.0)
    ts_ms = int(pd.Timestamp(row["timestamp"]).timestamp() * 1000)

    path = [(o, ts_ms), (c, ts_ms + _BAR_CLOSE_OFFSET_MS)]
    ticks: List[Event] = []
    for px, tick_ts_ms in path:
        ticks.append(
            Event(
                type=EventType.TICK,
                payload={
                    "ticker": symbol,
                    "symbol": symbol,
                    "price": float(px),
                    "volume": float(v),
                    "timestamp": tick_ts_ms,
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


class _SimulatedExitEngine:
    """
    Backtest-only exit simulator.

    The live system relies on Alpaca's OTO bracket for stop-loss enforcement,
    so the strategies (notably USEquityVWAPHunter) only emit BUY/SELL_SHORT
    entries and never publish their own exit. In the offline harness the
    bracket isn't honored, so without this engine open positions persist
    until the EOD liquidator at 15:55 ET — that turns "intraday" trades
    into multi-day holds and corrupts every PnL/exposure metric.

    Responsibilities:
    - Track each entry fill keyed by (symbol, decision_id).
    - On every TICK, check the live price against the recorded stop_loss
      and against a hard time-stop; emit a flat-out ORDER_CREATE (already
      sized, stage=SIZED so the router routes it directly) if either
      threshold trips.
    - Reconcile exits against fills so re-entries get tracked cleanly.

    This is intentionally a pure-backtest component; in the live system
    the broker enforces the bracket.
    """

    def __init__(self, bus: EventBus, max_hold_minutes: int = 240, stop_buffer_ticks: float = 0.0) -> None:
        self.bus = bus
        self.max_hold_seconds: float = max(60.0, float(max_hold_minutes) * 60.0)
        self.stop_buffer_ticks = float(stop_buffer_ticks)
        # symbol -> dict(qty (signed), avg_price, stop, entry_ts_ms, strategy, decision_id)
        self._positions: Dict[str, Dict[str, object]] = {}
        # decision_id -> stop_loss_price; populated when the SIZED order is
        # routed and consumed when the matching fill arrives. _simulate_fill
        # in the live broker router does not forward stop_loss into the
        # ORDER_FILL payload, so we have to capture it here.
        self._pending_stops: Dict[str, float] = {}
        # Decision IDs we issued for exits. The engine pops the position
        # eagerly when emitting an exit (to suppress duplicate fires on the
        # next tick); when the matching ORDER_FILL comes back the engine
        # would otherwise treat it as a brand-new opposite-direction entry
        # and start tracking a phantom short / long. Skipping fills whose
        # decision_id is in this set keeps the position book consistent.
        self._exit_decision_ids: set[str] = set()
        self.bus.subscribe(EventType.ORDER_CREATE, self.on_order_create)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        self.bus.subscribe(EventType.TICK, self.on_tick)

    async def on_order_create(self, event: Event) -> None:
        payload = event.payload or {}
        # Only capture sized opening intents that carry a real stop_loss.
        if payload.get("stage") != "SIZED":
            return
        if (payload.get("meta") or {}).get("simulated_exit"):
            return
        if (payload.get("meta") or {}).get("eod_liquidation"):
            return
        decision_id = payload.get("decision_id")
        if not decision_id:
            return
        stop = self._coerce_float(payload.get("stop_loss") or payload.get("stop_loss_price"))
        if stop is None:
            return
        self._pending_stops[str(decision_id)] = stop

    async def on_fill(self, event: Event) -> None:
        payload = event.payload or {}
        status = str(payload.get("status", "")).upper()
        if status in {"CANCELED", "CANCELLED", "REJECTED", "ERROR"}:
            return
        decision_id = str(payload.get("decision_id") or "")
        # Fill is the round-trip of an exit we issued; the position was
        # already popped at emit time so accept the close as a no-op.
        if decision_id and decision_id in self._exit_decision_ids:
            self._exit_decision_ids.discard(decision_id)
            return
        action = str(payload.get("action") or payload.get("side") or "").upper()
        symbol = str(payload.get("asset") or payload.get("symbol") or "").upper()
        if not symbol or not action:
            return
        try:
            fill_qty = int(float(payload.get("fill_qty", payload.get("filled_qty", 0)) or 0))
            fill_price = float(payload.get("fill_price", payload.get("entry_price", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return
        if fill_qty <= 0 or fill_price <= 0:
            return

        signed = fill_qty if action in {"BUY", "BUY_TO_OPEN", "BUY_TO_COVER"} else -fill_qty
        pos = self._positions.get(symbol)
        prior_qty = int(pos["qty"]) if pos else 0
        new_qty = prior_qty + signed

        if new_qty == 0:
            self._positions.pop(symbol, None)
            return

        # New or same-direction increment: update average and (re)arm stop.
        if prior_qty == 0 or (prior_qty > 0 and signed > 0) or (prior_qty < 0 and signed < 0):
            stop = self._coerce_float(payload.get("stop_loss") or payload.get("stop_loss_price"))
            decision_id = str(payload.get("decision_id") or "")
            if stop is None and decision_id:
                stop = self._pending_stops.pop(decision_id, None)
            elif decision_id:
                self._pending_stops.pop(decision_id, None)
            entry_ts = self._coerce_int(payload.get("timestamp"))
            if pos is None:
                self._positions[symbol] = {
                    "qty": new_qty,
                    "avg_price": fill_price,
                    "stop": stop,
                    "entry_ts_ms": entry_ts or 0,
                    "strategy": payload.get("strategy") or "Unknown",
                    "decision_id": payload.get("decision_id"),
                }
            else:
                total_cost = abs(prior_qty) * float(pos["avg_price"]) + abs(signed) * fill_price
                pos["qty"] = new_qty
                pos["avg_price"] = total_cost / abs(new_qty)
                # Keep the original stop unless the latest fill provided a tighter one.
                if stop is not None and stop > 0:
                    if pos.get("stop") is None or pos["stop"] in (0, 0.0):
                        pos["stop"] = stop
            return

        # Opposite direction: partially or fully reduce; re-anchor if flipped.
        if (prior_qty > 0 and new_qty < 0) or (prior_qty < 0 and new_qty > 0):
            stop = self._coerce_float(payload.get("stop_loss") or payload.get("stop_loss_price"))
            decision_id = str(payload.get("decision_id") or "")
            if stop is None and decision_id:
                stop = self._pending_stops.pop(decision_id, None)
            elif decision_id:
                self._pending_stops.pop(decision_id, None)
            self._positions[symbol] = {
                "qty": new_qty,
                "avg_price": fill_price,
                "stop": stop,
                "entry_ts_ms": self._coerce_int(payload.get("timestamp")) or 0,
                "strategy": payload.get("strategy") or "Unknown",
                "decision_id": payload.get("decision_id"),
            }
            return

        # Pure reduction (no flip): keep the existing entry context intact.
        if pos is not None:
            pos["qty"] = new_qty

    async def on_tick(self, event: Event) -> None:
        if not self._positions:
            return
        payload = event.payload or {}
        symbol = str(payload.get("ticker") or payload.get("symbol") or "").upper()
        if not symbol or symbol not in self._positions:
            return
        try:
            price = float(payload.get("price", 0.0))
        except (TypeError, ValueError):
            return
        if price <= 0:
            return

        ts_ms = self._coerce_int(payload.get("timestamp")) or 0
        pos = self._positions[symbol]
        qty = int(pos["qty"])
        if qty == 0:
            return

        stop = self._coerce_float(pos.get("stop"))
        entry_ts = int(pos.get("entry_ts_ms") or 0)
        # Stop hit: long below stop, short above stop.
        stop_hit = False
        if stop is not None and stop > 0:
            if qty > 0 and price <= stop:
                stop_hit = True
            elif qty < 0 and price >= stop:
                stop_hit = True

        time_hit = False
        if entry_ts > 0 and ts_ms > 0:
            if (ts_ms - entry_ts) / 1000.0 >= self.max_hold_seconds:
                time_hit = True

        if not stop_hit and not time_hit:
            return

        reason = "stop" if stop_hit else "time_stop"
        exit_action = "SELL" if qty > 0 else "BUY_TO_COVER"
        original_decision = pos.get("decision_id") or f"SIMEXIT-{symbol}-{ts_ms}"
        exit_decision_id = f"{original_decision}-{reason}"
        # Drop the position before publishing so a re-fill in the same tick
        # storm doesn't double-fire the exit, and remember the decision_id
        # so we don't treat the round-trip fill as a new opposite-direction
        # opening trade.
        self._positions.pop(symbol, None)
        self._exit_decision_ids.add(exit_decision_id)
        self.bus.publish(
            Event(
                type=EventType.ORDER_CREATE,
                payload={
                    "asset": symbol,
                    "action": exit_action,
                    "strategy": str(pos.get("strategy") or "SIM_EXIT"),
                    "stage": "SIZED",
                    "shares": abs(qty),
                    "reference_price": round(price, 4),
                    "entry_price": round(price, 4),
                    "timestamp": ts_ms,
                    "stop_loss": round(price, 4),
                    "stop_loss_price": round(price, 4),
                    "status": "READY_FOR_BROKER",
                    "decision_id": exit_decision_id,
                    "meta": {"simulated_exit": True, "reason": reason, "eod_liquidation": True},
                },
            )
        )

    @staticmethod
    def _coerce_float(value: object) -> Optional[float]:
        try:
            v = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    @staticmethod
    def _coerce_int(value: object) -> Optional[int]:
        try:
            return int(float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


async def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, object]:
    symbol = cfg.symbol.upper()
    benchmark_symbol = (cfg.benchmark_symbol or cfg.symbol).upper()
    bars = _normalize_bars(df, symbol)
    benchmark_bars = _safe_normalize_bars(df, benchmark_symbol)

    bus = EventBus()
    await bus.start()

    _orb = None
    _vwap = None
    _gap_fade = None
    _pairs = None
    hedge_bars: Optional[pd.DataFrame] = None
    if cfg.strategy == "pairs":
        if not cfg.hedge_symbol:
            raise ValueError("strategy=pairs requires cfg.hedge_symbol to be set.")
        hedge_bars = _safe_normalize_bars(df, cfg.hedge_symbol.upper())
        if hedge_bars.empty:
            raise ValueError(
                f"No bars found for hedge symbol {cfg.hedge_symbol!r} in the input dataset."
            )
        _pairs = USEquityKalmanPairsTrader(
            asset_y=symbol,
            asset_x=cfg.hedge_symbol.upper(),
            bus=bus,
            delta=cfg.pair_delta,
            ve=cfg.pair_ve,
            entry_z=cfg.pair_entry_z,
            exit_z=cfg.pair_exit_z,
            max_leg_staleness_sec=cfg.pair_max_leg_staleness_sec,
            cooldown_seconds=cfg.pair_cooldown_seconds,
            nominal_stop_pct=cfg.pair_nominal_stop_pct,
            target_dollar_notional=cfg.pair_target_dollar_notional,
        )
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
    if cfg.strategy in {"gap_fade", "all"}:
        _gap_fade = OvernightGapFade(
            target_asset=symbol,
            bus=bus,
            gap_trigger_pct=cfg.gap_fade_trigger_pct,
            stop_pct=cfg.gap_fade_stop_pct,
            max_trades_per_day=cfg.gap_fade_max_trades_per_day,
        )
    if cfg.strategy == "all" and _orb is None:
        _orb = USEquityORB(
            target_asset=symbol,
            bus=bus,
            range_minutes=cfg.orb_range_minutes,
            max_trades=cfg.orb_max_trades,
            cooldown_bars=cfg.orb_cooldown_bars,
            min_range_pct=cfg.orb_min_range_pct,
            breakout_buffer_pct=cfg.orb_breakout_buffer_pct,
        )
    if cfg.strategy == "all" and _vwap is None:
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

    ml_gate = None
    test_bars = bars
    if cfg.use_ml_gate:
        try:
            from intelligence.ml_pipeline.ml_candidate_gate import MLCandidateGate

            train_n = max(0, min(int(cfg.ml_train_bars), max(0, len(bars) - 100)))
            if train_n < 200:
                logger.warning(
                    "use_ml_gate set but only %d train bars available; gate disabled.", train_n
                )
            else:
                train_bars = bars.iloc[:train_n].copy()
                test_bars = bars.iloc[train_n:].copy()
                ml_gate = MLCandidateGate(
                    historical_df=train_bars,
                    target_horizon=cfg.ml_horizon,
                    threshold=cfg.ml_threshold,
                )
                # Extend gate's view to include the test window so live
                # feature computation has enough lookback at the start of OOS.
                ml_gate.update_history(bars)
        except Exception as exc:
            logger.error("Failed to fit ML candidate gate; running without: %s", exc, exc_info=True)
            ml_gate = None
            test_bars = bars

    _ranker = CandidateRanker(
        bus=bus, benchmark=benchmark_symbol, min_score=cfg.min_rank_score, ml_gate=ml_gate
    )
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
    _exit_sim = _SimulatedExitEngine(
        bus=bus,
        max_hold_minutes=cfg.sim_max_hold_minutes,
        stop_buffer_ticks=cfg.sim_stop_buffer_ticks,
    )
    ledger = TradeLedger(
        initial_equity=cfg.initial_capital,
        slippage_bps_per_side=cfg.slippage_bps_per_side,
        commission_per_share=cfg.commission_per_share,
        commission_min_per_trade=cfg.commission_min_per_trade,
        sec_fee_rate=cfg.sec_fee_rate,
        short_borrow_apr=cfg.short_borrow_apr,
    )

    bus.subscribe(EventType.ORDER_FILL, ledger.on_fill)

    await _slip.start()
    await router.start()

    if cfg.strategy == "pairs" and hedge_bars is not None:
        # Interleave primary and hedge bars by timestamp so the Kalman
        # filter sees both legs in chronological order. Restrict the hedge
        # stream to the same time window the primary trades on (matters
        # when use_ml_gate trims a leading chunk for training).
        if len(test_bars) > 0:
            t_lo = test_bars["timestamp"].iloc[0]
            t_hi = test_bars["timestamp"].iloc[-1]
            hedge_window = hedge_bars[
                (hedge_bars["timestamp"] >= t_lo) & (hedge_bars["timestamp"] <= t_hi)
            ]
        else:
            hedge_window = hedge_bars.iloc[0:0]
        replay_df = pd.concat(
            [test_bars.assign(_replay_symbol=symbol), hedge_window.assign(_replay_symbol=cfg.hedge_symbol.upper())],
            ignore_index=True,
        ).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        for _, row in replay_df.iterrows():
            for tick in _bar_to_ticks(row, str(row["_replay_symbol"])):
                bus.publish(tick)
            await asyncio.sleep(0)
    else:
        for _, row in test_bars.iterrows():
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

    metrics_bars = test_bars
    start_ts = metrics_bars["timestamp"].iloc[0] if len(metrics_bars) else None
    end_ts = metrics_bars["timestamp"].iloc[-1] if len(metrics_bars) else None
    result = ledger.snapshot(start_ts=start_ts, end_ts=end_ts)
    trading_days = list(pd.Series(metrics_bars["timestamp"].dt.normalize().drop_duplicates()).tolist())
    daily_equity = _build_daily_equity(cfg.initial_capital, ledger.closed_trades, trading_days)
    daily_returns = _daily_returns_from_equity(daily_equity)
    annualized_sharpe = _annualized_sharpe(daily_returns)
    total_return_pct = (float(result["final_equity"]) / cfg.initial_capital) - 1.0 if cfg.initial_capital > 0 else 0.0
    benchmark_metrics_bars = benchmark_bars
    if not benchmark_metrics_bars.empty and start_ts is not None and end_ts is not None:
        benchmark_metrics_bars = benchmark_metrics_bars[
            (benchmark_metrics_bars["timestamp"] >= start_ts)
            & (benchmark_metrics_bars["timestamp"] <= end_ts)
        ]
    bench = _benchmark_metrics(benchmark_metrics_bars if not benchmark_metrics_bars.empty else metrics_bars)
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
            "bars": int(len(test_bars)),
            "train_bars": int(len(bars) - len(test_bars)) if cfg.use_ml_gate else 0,
            "ml_gate": ml_gate.stats() if ml_gate is not None else None,
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
    if args.strategy == "pairs":
        if not args.hedge_symbol:
            raise SystemExit("--strategy pairs requires --hedge-symbol")
        if args.hedge_csv:
            hedge_df = _load_csv(Path(args.hedge_csv))
            df = pd.concat([df, hedge_df], ignore_index=True)
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
        sim_max_hold_minutes=args.sim_max_hold_minutes,
        sim_stop_buffer_ticks=args.sim_stop_buffer_ticks,
        slippage_bps_per_side=args.slippage_bps_per_side,
        commission_per_share=args.commission_per_share,
        commission_min_per_trade=args.commission_min_per_trade,
        sec_fee_rate=args.sec_fee_rate,
        short_borrow_apr=args.short_borrow_apr,
        use_ml_gate=args.use_ml_gate,
        ml_train_bars=args.ml_train_bars,
        ml_threshold=args.ml_threshold,
        ml_horizon=args.ml_horizon,
        hedge_symbol=args.hedge_symbol,
        pair_entry_z=args.pair_entry_z,
        pair_exit_z=args.pair_exit_z,
        pair_delta=args.pair_delta,
        pair_ve=args.pair_ve,
        pair_max_leg_staleness_sec=args.pair_max_leg_staleness_sec,
        pair_cooldown_seconds=args.pair_cooldown_seconds,
        pair_nominal_stop_pct=args.pair_nominal_stop_pct,
        pair_target_dollar_notional=args.pair_target_dollar_notional,
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
    p.add_argument(
        "--hedge-csv",
        default="",
        help="Path to OHLCV csv for the hedge leg (only when --strategy pairs). "
             "If omitted, both symbols' bars must be present in --csv.",
    )
    p.add_argument(
        "--hedge-symbol",
        default="",
        help="Hedge leg ticker for --strategy pairs (e.g. for an SPY/QQQ pair, --symbol SPY --hedge-symbol QQQ).",
    )
    p.add_argument("--pair-entry-z", type=float, default=2.0)
    p.add_argument("--pair-exit-z", type=float, default=0.5)
    p.add_argument("--pair-delta", type=float, default=1e-4)
    p.add_argument("--pair-ve", type=float, default=1e-3)
    p.add_argument("--pair-max-leg-staleness-sec", type=float, default=30.0)
    p.add_argument("--pair-cooldown-seconds", type=float, default=5.0)
    p.add_argument("--pair-nominal-stop-pct", type=float, default=0.02)
    p.add_argument(
        "--pair-target-dollar-notional",
        type=float,
        default=10000.0,
        help="Per-leg target dollar notional for pairs entries (beta-hedged on x).",
    )
    p.add_argument("--symbol", default="SPY", help="Ticker to backtest.")
    p.add_argument("--benchmark-symbol", default="SPY", help="Benchmark ticker to use if present in CSV.")
    p.add_argument(
        "--strategy",
        choices=["orb", "vwap", "both", "gap_fade", "pairs", "all"],
        default="both",
        help="'gap_fade' runs only the overnight-gap-fade baseline; 'all' runs ORB + VWAP + gap_fade.",
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
    p.add_argument(
        "--slippage-bps-per-side",
        type=float,
        default=1.5,
        help="Per-leg slippage in basis points (default 1.5 -> ~3 bps round-trip).",
    )
    p.add_argument(
        "--commission-per-share",
        type=float,
        default=0.0,
        help="Per-share commission charged on each leg (Alpaca: 0).",
    )
    p.add_argument(
        "--commission-min-per-trade",
        type=float,
        default=0.0,
        help="Minimum dollar commission per leg.",
    )
    p.add_argument(
        "--sec-fee-rate",
        type=float,
        default=0.000008,
        help="SEC fee rate on sell-side notional (default 0.000008 ~ Alpaca live rate).",
    )
    p.add_argument(
        "--short-borrow-apr",
        type=float,
        default=0.0025,
        help="Annual short-borrow rate (default 0.25%% — easy-to-borrow large caps).",
    )
    p.add_argument(
        "--use-ml-gate",
        action="store_true",
        help="Fit a probability gate on the leading ml-train-bars rows and use it as a candidate veto.",
    )
    p.add_argument("--ml-train-bars", type=int, default=5000, help="Bars at the head of the dataset reserved for the ML gate fit.")
    p.add_argument("--ml-threshold", type=float, default=0.55, help="Min P(positive return) to pass a long; 1-threshold ceiling for shorts.")
    p.add_argument("--ml-horizon", type=int, default=5, help="Number of bars ahead the ML gate predicts.")
    p.add_argument("--output", default="", help="Optional path to write JSON result.")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main_async(parse_args()))
