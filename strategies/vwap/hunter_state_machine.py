from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from datetime import datetime, time, timezone, timedelta, date
from enum import Enum, auto
from typing import Deque, Optional

import pytz

from core.engine.event_bus import EventBus, Event, EventType

logger = logging.getLogger("EquityVWAPHunter")


class HunterState(Enum):
    SCANNING = auto()
    ARMED = auto()
    WINDOW_OPEN = auto()
    DONE_FOR_DAY = auto()


class USEquityVWAPHunter:
    """
    Optimized intraday VWAP bounce strategy for US equities.

    Upgrades:
    - safer session reset / timestamp handling
    - RVOL-style minimum liquidity gate
    - bounce confirmation with cooldown
    - avoids duplicate triggers around VWAP chop
    - cleaner stop placement and standardized payload
    - FIXED: Add unique signal_id to every ORDER_CREATE to enable deduplication
    """

    def __init__(
        self,
        target_asset: str,
        bus: EventBus,
        min_volume_shares: float = 500000.0,
        vwap_tolerance_pct: float = 0.002,
        momentum_threshold_pct: float = 0.001,
        max_daily_trades: int = 3,
        cooldown_bars: int = 8,
        min_stop_pct: float = 0.003,
        max_window_bars: int = 8,
        rebound_confirmation_pct: float = 0.0005,
        bounce_confirmation_ticks: int = 1,
    ) -> None:
        self.asset = str(target_asset).upper()
        self.bus = bus

        self.min_volume = float(min_volume_shares)
        self.vwap_tolerance = float(vwap_tolerance_pct)
        self.momentum_threshold = float(momentum_threshold_pct)
        self.max_daily_trades = int(max_daily_trades)
        self.cooldown_bars = int(cooldown_bars)
        self.min_stop_pct = float(min_stop_pct)
        self.max_window_bars = int(max_window_bars)
        self.rebound_confirmation = float(rebound_confirmation_pct)
        self.bounce_confirmation_ticks = max(1, int(bounce_confirmation_ticks))

        self.tz = pytz.timezone("US/Eastern")
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        self.last_date: Optional[date] = None

        self.current_state: HunterState = HunterState.SCANNING
        self.trades_today: int = 0
        self.cumulative_pv: float = 0.0
        self.cumulative_volume: float = 0.0
        self.current_vwap: float = 0.0
        self.last_price: float = 0.0

        self.bar_count: int = 0
        self.last_trigger_bar: int = -10_000
        self.window_open_bar: int = -1
        self.window_open_price: float = 0.0
        self.last_signal_id: Optional[str] = None  # --- FIX: Track last signal to prevent re-triggering ---
        self.bounce_streak: int = 0
        self._volume_window: Deque[float] = deque(maxlen=20)
        self._min_volume_ratio: float = 0.20

        self.bus.subscribe(EventType.TICK, self.on_tick)

        logger.info(
            "[%s] USEquityVWAPHunter initialized: vwap_tol=%.4f, momentum=%.4f, "
            "rebound=%.4f, max_window=%d bars, min_vol=%.0f",
            self.asset,
            self.vwap_tolerance,
            self.momentum_threshold,
            self.rebound_confirmation,
            self.max_window_bars,
            self.min_volume,
        )

    def _get_est_time(self, timestamp_ms: Optional[int] = None) -> datetime:
        """Convert timestamp to US/Eastern time."""
        if timestamp_ms is not None:
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=pytz.utc)
        else:
            dt = datetime.now(pytz.utc)
        return dt.astimezone(self.tz)

    def _reset_daily_state(self, current_date: date) -> None:
        """Reset all intraday state at market open."""
        self.current_state = HunterState.SCANNING
        self.trades_today = 0
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        self.current_vwap = 0.0
        self.last_price = 0.0
        self.bar_count = 0
        self.last_trigger_bar = -10_000
        self.window_open_bar = -1
        self.window_open_price = 0.0
        self.last_signal_id = None
        self.bounce_streak = 0
        self._volume_window.clear()
        self.last_date = current_date
        logger.info("[%s] Session reset. VWAP anchor restarted.", self.asset)

    def _update_vwap(self, price: float, volume: float) -> None:
        """Update cumulative VWAP calculation."""
        self.cumulative_pv += price * volume
        self.cumulative_volume += volume
        if self.cumulative_volume > 0:
            self.current_vwap = self.cumulative_pv / self.cumulative_volume

    def _in_cooldown(self) -> bool:
        """Check if we're in cooldown period after last execution."""
        return (self.bar_count - self.last_trigger_bar) < self.cooldown_bars

    def _build_stop(self, price: float) -> float:
        """Calculate stop loss: min of VWAP-based or percentage-based."""
        vwap_stop = self.current_vwap * (1.0 - 0.005)
        pct_stop = price * (1.0 - self.min_stop_pct)
        return round(min(vwap_stop, pct_stop), 4)

    async def on_tick(self, event: Event) -> None:
        """
        Handle incoming tick events.
        
        Flow:
        1. Filter for relevant symbol
        2. Validate price/volume
        3. Check for new trading day (reset state)
        4. Check market hours
        5. Increment bar counter
        6. Update VWAP
        7. Run state machine
        """
        payload = event.payload or {}

        ticker = payload.get("ticker") or payload.get("symbol")
        if str(ticker).upper() != self.asset:
            return

        try:
            price = float(payload.get("price"))
            volume = float(payload.get("volume"))
        except (TypeError, ValueError):
            return

        if price <= 0 or volume < 0:
            return

        timestamp_ms = payload.get("timestamp")
        current_dt = self._get_est_time(timestamp_ms)
        current_time = current_dt.time()
        current_date = current_dt.date()

        if self.last_date != current_date:
            self._reset_daily_state(current_date)

        if current_time < self.market_open or current_time >= self.market_close:
            return

        self.bar_count += 1
        self.last_price = price
        if volume > 0:
            self._volume_window.append(volume)
        self._update_vwap(price, volume)

        if self.current_state != HunterState.DONE_FOR_DAY:
            await self._process_state_machine(price, volume, timestamp_ms)

    async def _process_state_machine(self, price: float, volume: float, timestamp_ms: Optional[int]) -> None:
        """
        Main state machine logic.
        
        States:
        - SCANNING: Waiting for enough volume to be ARMED
        - ARMED: Monitoring for price pullback to VWAP
        - WINDOW_OPEN: Waiting for bounce confirmation (rebound or breakout)
        - DONE_FOR_DAY: Max daily trades reached
        """
        if self.current_vwap <= 0:
            return

        if self.current_state == HunterState.SCANNING:
            if self.cumulative_volume >= self.min_volume:
                self.current_state = HunterState.ARMED
                logger.debug(
                    "[%s] Volume gate passed: %.0f >= %.0f. ARMED.",
                    self.asset,
                    self.cumulative_volume,
                    self.min_volume,
                )
            return

        if self.current_state == HunterState.ARMED:
            upper_bound = self.current_vwap * (1.0 + self.vwap_tolerance)
            lower_bound = self.current_vwap * (1.0 - self.vwap_tolerance)

            if lower_bound <= price <= upper_bound:
                self.current_state = HunterState.WINDOW_OPEN
                self.window_open_bar = self.bar_count
                self.window_open_price = price
                logger.info(
                    "[%s] Pullback into VWAP zone. price=%.2f vwap=%.2f WINDOW_OPEN.",
                    self.asset,
                    price,
                    self.current_vwap,
                )
            return

        if self.current_state == HunterState.WINDOW_OPEN:
            bars_in_window = self.bar_count - self.window_open_bar

            if bars_in_window > self.max_window_bars:
                logger.info(
                    "[%s] VWAP window stale after %d bars. Resetting to SCANNING.",
                    self.asset,
                    bars_in_window,
                )
                self.current_state = HunterState.SCANNING
                return

            breakout_level = self.current_vwap * (1.0 + self.momentum_threshold)
            is_strong_breakout = price >= breakout_level

            rebound_level = self.current_vwap * (1.0 + self.rebound_confirmation)
            is_rebound = price > rebound_level

            if is_strong_breakout or is_rebound:
                if self._in_cooldown():
                    logger.debug(
                        "[%s] Bounce trigger ignored due to cooldown. Bar: %d, Last: %d, Need: %d",
                        self.asset,
                        self.bar_count,
                        self.last_trigger_bar,
                        self.cooldown_bars,
                    )
                    self.bounce_streak = 0
                    return

                if len(self._volume_window) >= 5:
                    avg_vol = sum(self._volume_window) / len(self._volume_window)
                    if avg_vol > 0 and (self._volume_window[-1] / avg_vol) < self._min_volume_ratio:
                        self.bounce_streak = 0
                        return

                self.bounce_streak += 1
                if self.bounce_streak < self.bounce_confirmation_ticks:
                    logger.debug(
                        "[%s] Bounce candidate tick %d/%d — awaiting confirmation.",
                        self.asset,
                        self.bounce_streak,
                        self.bounce_confirmation_ticks,
                    )
                    return

                stop_loss = self._build_stop(price)

                # --- FIX: Generate unique signal ID for deduplication ---
                signal_id = str(uuid.uuid4())
                self.last_signal_id = signal_id
                # --- END FIX ---

                self.bounce_streak = 0
                confirmation_type = "breakout" if is_strong_breakout else "rebound"
                order_event = Event(
                    type=EventType.ORDER_CREATE,
                    payload={
                        "signal_id": signal_id,  # --- FIX: Add signal ID ---
                        "asset": self.asset,
                        "action": "BUY",
                        "strategy": "Equity_VWAP_Hunter",
                        "timestamp": timestamp_ms,
                        "reference_price": round(price, 4),
                        "entry_price": round(price, 4),
                        "stop_loss_price": stop_loss,
                        "signal_context": {
                            "vwap": round(self.current_vwap, 4),
                            "breakout_level": round(breakout_level, 4),
                            "rebound_level": round(rebound_level, 4),
                            "state": self.current_state.name,
                            "cumulative_volume": round(self.cumulative_volume, 2),
                            "bars_in_window": bars_in_window,
                            "confirmation_type": confirmation_type,
                        },
                    },
                )

                logger.warning(
                    "[%s] VWAP bounce confirmed (%s) @ %.2f | VWAP %.2f | stop %.2f | bars %d | signal_id=%s",
                    self.asset,
                    confirmation_type,
                    price,
                    self.current_vwap,
                    stop_loss,
                    bars_in_window,
                    signal_id[:8],  # Log first 8 chars of UUID
                )

                self.bus.publish(order_event)
                self._register_execution()
                return

            breakdown_level = self.current_vwap * (1.0 - self.momentum_threshold)
            if price < breakdown_level:
                logger.info(
                    "[%s] Bounce failed. price=%.2f < breakdown=%.2f. Resetting to SCANNING.",
                    self.asset,
                    price,
                    breakdown_level,
                )
                self.bounce_streak = 0
                self.current_state = HunterState.SCANNING
                return

            # Price is inside window but not yet triggering: reset streak
            # so a lull between ticks doesn't accumulate false confirmations.
            if not (is_strong_breakout or is_rebound):
                self.bounce_streak = 0

            logger.debug(
                "[%s] Window open (bar %d/%d): price=%.2f vwap=%.2f (rebound %.2f, breakout %.2f)",
                self.asset,
                bars_in_window,
                self.max_window_bars,
                price,
                self.current_vwap,
                rebound_level,
                breakout_level,
            )

    def _register_execution(self) -> None:
        """
        Register that a trade was executed.
        Update state and check if daily limit reached.
        """
        self.trades_today += 1
        self.last_trigger_bar = self.bar_count

        if self.trades_today >= self.max_daily_trades:
            self.current_state = HunterState.DONE_FOR_DAY
            logger.info("[%s] Max daily VWAP trades reached (%d). DONE_FOR_DAY.", 
                       self.asset, self.max_daily_trades)
        else:
            self.current_state = HunterState.SCANNING
            logger.info("[%s] Trade registered (%d/%d). Back to SCANNING.", 
                       self.asset, self.trades_today, self.max_daily_trades)


if __name__ == "__main__":
    async def run_vwap_equity_test() -> None:
        print("Initializing optimized US Equity VWAP Hunter Test...")
        bus = EventBus()
        await bus.start()

        hunter = USEquityVWAPHunter(
            target_asset="NVDA",
            bus=bus,
            min_volume_shares=1000.0,
            cooldown_bars=3,
            momentum_threshold_pct=0.001,
            rebound_confirmation_pct=0.0005,
        )

        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)

        def mock_tick(price: float, volume: float, minutes_offset: int) -> Event:
            dt = base_date + timedelta(minutes=minutes_offset)
            return Event(
                type=EventType.TICK,
                payload={
                    "ticker": "NVDA",
                    "price": price,
                    "volume": volume,
                    "timestamp": int(dt.timestamp() * 1000),
                },
            )

        print("\n=== Scenario: Pullback + Rebound ===")
        bus.publish(mock_tick(800.0, 5000.0, 1))
        await asyncio.sleep(0.01)
        bus.publish(mock_tick(810.0, 2000.0, 5))
        await asyncio.sleep(0.01)
        bus.publish(mock_tick(803.0, 1000.0, 15))
        await asyncio.sleep(0.01)
        bus.publish(mock_tick(803.5, 1500.0, 16))
        await asyncio.sleep(0.1)

        await bus.stop()

    asyncio.run(run_vwap_equity_test())