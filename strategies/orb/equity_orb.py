from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from datetime import datetime, time, timedelta, date
from enum import Enum, auto
from typing import Deque, Optional

import pytz

from core.engine.event_bus import EventBus, Event, EventType

logger = logging.getLogger("EquityORB")


class ORBState(Enum):
    PRE_MARKET = auto()
    BUILDING_RANGE = auto()
    ACTIVE = auto()
    DONE_FOR_DAY = auto()


class USEquityORB:
    """
    Optimized US Equity Opening Range Breakout strategy.

    Upgrades:
    - safe session reset and timezone handling
    - duplicate-breakout protection via cooldown
    - standardized payload for downstream ranker/sizer/router
    - cleaner stop placement and range validation
    - supports both long breakout and short breakdown
    """

    def __init__(
        self,
        target_asset: str,
        bus: EventBus,
        range_minutes: int = 15,
        max_trades: int = 2,
        cooldown_bars: int = 10,
        min_range_pct: float = 0.0025,
        breakout_buffer_pct: float = 0.0005,
        breakout_confirmation_ticks: int = 1,
        fakeout_reset_pct: float = 0.0003,
        enable_shorts: bool = True,
    ) -> None:
        self.asset = str(target_asset).upper()
        self.bus = bus
        self.range_minutes = int(range_minutes)
        self.max_trades = int(max_trades)
        self.cooldown_bars = int(cooldown_bars)
        self.min_range_pct = float(min_range_pct)
        self.breakout_buffer_pct = float(breakout_buffer_pct)
        self.breakout_confirmation_ticks = max(1, int(breakout_confirmation_ticks))
        self.fakeout_reset_pct = max(0.0, float(fakeout_reset_pct))
        self.enable_shorts = bool(enable_shorts)

        self.tz = pytz.timezone("US/Eastern")
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        self.range_end_time: Optional[time] = None

        self.state = ORBState.PRE_MARKET
        self.trades_today = 0
        self.range_high: float = float("-inf")
        self.range_low: float = float("inf")
        self.last_date: Optional[date] = None

        self.bar_count = 0
        self.last_trigger_bar = -10_000
        self.long_breakout_streak = 0
        self.short_breakdown_streak = 0
        self._volume_window: Deque[float] = deque(maxlen=20)
        self._min_volume_ratio: float = 0.20

        self.bus.subscribe(EventType.TICK, self.on_tick)

    def _get_est_time(self, timestamp_ms: Optional[int] = None) -> datetime:
        if timestamp_ms is not None:
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=pytz.utc)
        else:
            dt = datetime.now(pytz.utc)
        return dt.astimezone(self.tz)

    def _reset_daily_state(self, current_date: date) -> None:
        self.state = ORBState.PRE_MARKET
        self.trades_today = 0
        self.range_high = float("-inf")
        self.range_low = float("inf")
        self.last_date = current_date
        self.bar_count = 0
        self.last_trigger_bar = -10_000
        self.long_breakout_streak = 0
        self.short_breakdown_streak = 0
        self._volume_window.clear()

        est_open = self.tz.localize(datetime.combine(current_date, self.market_open))
        end_dt = est_open + timedelta(minutes=self.range_minutes)
        self.range_end_time = end_dt.time()

        logger.info(
            "[%s] ORB reset. Range build window: %s -> %s",
            self.asset,
            self.market_open,
            self.range_end_time,
        )

    def _in_cooldown(self) -> bool:
        return (self.bar_count - self.last_trigger_bar) < self.cooldown_bars

    def _range_is_valid(self) -> bool:
        if self.range_high == float("-inf") or self.range_low == float("inf"):
            return False
        if self.range_high <= self.range_low:
            return False

        mid = (self.range_high + self.range_low) / 2.0
        if mid <= 0:
            return False

        range_pct = (self.range_high - self.range_low) / mid
        return range_pct >= self.min_range_pct

    def _long_breakout_level(self) -> float:
        return self.range_high * (1.0 + self.breakout_buffer_pct)

    def _short_breakdown_level(self) -> float:
        return self.range_low * (1.0 - self.breakout_buffer_pct)

    async def on_tick(self, event: Event) -> None:
        payload = event.payload or {}

        ticker = payload.get("ticker") or payload.get("symbol")
        if str(ticker).upper() != self.asset:
            return

        raw_price = payload.get("price")
        if raw_price is None:
            return
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            return

        if price <= 0:
            return

        volume = float(payload.get("volume") or 0.0)

        timestamp_ms = payload.get("timestamp")
        current_dt = self._get_est_time(timestamp_ms)
        current_time = current_dt.time()
        current_date = current_dt.date()

        if self.last_date != current_date:
            self._reset_daily_state(current_date)

        if current_time < self.market_open or current_time >= self.market_close:
            return

        self.bar_count += 1
        if volume > 0:
            self._volume_window.append(volume)

        if self.range_end_time is None:
            return

        if self.market_open <= current_time < self.range_end_time:
            if self.state != ORBState.BUILDING_RANGE:
                self.state = ORBState.BUILDING_RANGE
                logger.info("[%s] Open detected. Building %dm opening range...", self.asset, self.range_minutes)

            if price > self.range_high:
                self.range_high = price
            if price < self.range_low:
                self.range_low = price
            return

        if current_time >= self.range_end_time and self.state == ORBState.BUILDING_RANGE:
            if not self._range_is_valid():
                self.state = ORBState.DONE_FOR_DAY
                logger.warning(
                    "[%s] ORB range invalid or too narrow. high=%.4f low=%.4f. Strategy disabled for session.",
                    self.asset,
                    self.range_high,
                    self.range_low,
                )
                return

            self.state = ORBState.ACTIVE
            logger.info(
                "[%s] ORB established. High=%.2f Low=%.2f LongTrig=%.2f ShortTrig=%.2f",
                self.asset,
                self.range_high,
                self.range_low,
                self._long_breakout_level(),
                self._short_breakdown_level(),
            )

        if self.state == ORBState.ACTIVE and self.trades_today < self.max_trades:
            await self._evaluate_breakout(price, timestamp_ms)

    def _volume_is_sufficient(self) -> bool:
        if len(self._volume_window) < 5:
            return True
        avg = sum(self._volume_window) / len(self._volume_window)
        return avg <= 0 or (self._volume_window[-1] / avg) >= self._min_volume_ratio

    async def _evaluate_breakout(self, current_price: float, timestamp_ms: Optional[int]) -> None:
        if not self._volume_is_sufficient():
            return
        if self._in_cooldown():
            return

        long_trigger = self._long_breakout_level()
        short_trigger = self._short_breakdown_level()

        if current_price >= long_trigger:
            self.long_breakout_streak += 1
            self.short_breakdown_streak = 0
        elif current_price <= short_trigger:
            self.short_breakdown_streak += 1
            self.long_breakout_streak = 0
        else:
            # Reset directional streaks when price reclaims back into range.
            range_reclaim_upper = self.range_high * (1.0 - self.fakeout_reset_pct)
            range_reclaim_lower = self.range_low * (1.0 + self.fakeout_reset_pct)
            if current_price <= range_reclaim_upper:
                self.long_breakout_streak = 0
            if current_price >= range_reclaim_lower:
                self.short_breakdown_streak = 0
            return

        if self.long_breakout_streak >= self.breakout_confirmation_ticks:
            stop_loss = round(self.range_low, 4)
            logger.warning(
                "[%s] BULLISH BREAKOUT @ %.2f | range_high=%.2f trigger=%.2f stop=%.2f conf=%d",
                self.asset,
                current_price,
                self.range_high,
                long_trigger,
                stop_loss,
                self.long_breakout_streak,
            )
            self._emit_order("BUY", current_price, stop_loss, timestamp_ms=timestamp_ms)
            self.long_breakout_streak = 0
            self.short_breakdown_streak = 0
            return

        if self.short_breakdown_streak >= self.breakout_confirmation_ticks:
            if not self.enable_shorts:
                logger.info(
                    "[%s] Bearish breakdown confirmed but shorts disabled. trigger=%.2f conf=%d",
                    self.asset,
                    short_trigger,
                    self.short_breakdown_streak,
                )
                self.short_breakdown_streak = 0
                return

            stop_loss = round(self.range_high, 4)
            logger.warning(
                "[%s] BEARISH BREAKDOWN @ %.2f | range_low=%.2f trigger=%.2f stop=%.2f conf=%d",
                self.asset,
                current_price,
                self.range_low,
                short_trigger,
                stop_loss,
                self.short_breakdown_streak,
            )
            self._emit_order("SELL_SHORT", current_price, stop_loss, timestamp_ms=timestamp_ms)
            self.long_breakout_streak = 0
            self.short_breakdown_streak = 0

    def _emit_order(self, action: str, price: float, stop_loss: float, timestamp_ms: Optional[int] = None) -> None:
        signal_id = str(uuid.uuid4())
        order_event = Event(
            type=EventType.ORDER_CREATE,
            payload={
                "signal_id": signal_id,
                "asset": self.asset,
                "action": action,
                "strategy": f"ORB_{self.range_minutes}m",
                "timestamp": timestamp_ms,
                "reference_price": round(price, 4),
                "entry_price": round(price, 4),
                "stop_loss_price": round(stop_loss, 4),
                "signal_context": {
                    "range_high": round(self.range_high, 4),
                    "range_low": round(self.range_low, 4),
                    "range_minutes": self.range_minutes,
                    "breakout_buffer_pct": self.breakout_buffer_pct,
                    "state": self.state.name,
                },
            },
        )

        self.bus.publish(order_event)
        self.trades_today += 1
        self.last_trigger_bar = self.bar_count

        if self.trades_today >= self.max_trades:
            self.state = ORBState.DONE_FOR_DAY
            logger.info("[%s] Max daily trades reached. DONE_FOR_DAY.", self.asset)


if __name__ == "__main__":
    async def run_orb_test() -> None:
        print("Initializing optimized US Equity ORB Test...")
        bus = EventBus()
        await bus.start()

        _orb = USEquityORB(
            target_asset="SPY",
            bus=bus,
            range_minutes=15,
            max_trades=2,
            cooldown_bars=5,
        )

        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=pytz.utc)

        def mock_tick(price: float, minutes_offset: int) -> Event:
            dt = base_date + timedelta(minutes=minutes_offset)
            return Event(
                type=EventType.TICK,
                payload={
                    "ticker": "SPY",
                    "price": price,
                    "timestamp": int(dt.timestamp() * 1000),
                },
            )

        # build range
        bus.publish(mock_tick(500.0, 1))
        bus.publish(mock_tick(503.0, 5))
        bus.publish(mock_tick(499.5, 10))
        await asyncio.sleep(0.05)

        # breakout
        bus.publish(mock_tick(503.6, 16))
        await asyncio.sleep(0.1)

        await bus.stop()

    asyncio.run(run_orb_test())