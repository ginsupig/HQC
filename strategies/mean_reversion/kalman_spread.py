from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, time, timezone, timedelta
from typing import Dict, Optional

import numpy as np
import pytz

from core.engine.event_bus import EventBus, Event, EventType

logger = logging.getLogger("KalmanPairsEquity")


class USEquityKalmanPairsTrader:
    """
    Optimized US equity Kalman pairs trader.

    Upgrades:
    - strict RTH time gating
    - stale-price protection across pair legs
    - deterministic absolute-time cooldown logic (replaces volatile tick-count logic)
    - cleaner entry/exit state handling
    - standardized ORDER_CREATE payloads for HQC pipeline
    """

    def __init__(
        self,
        asset_y: str,
        asset_x: str,
        bus: EventBus,
        delta: float = 1e-4,
        ve: float = 1e-3,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_leg_staleness_sec: float = 30.0,
        cooldown_seconds: float = 5.0,  # <-- UPGRADE: Absolute time instead of ticks
        nominal_stop_pct: float = 0.02,
        target_dollar_notional: float = 10000.0,
        tick_event_type: EventType = EventType.TICK,
    ) -> None:
        self.asset_y = str(asset_y).upper()
        self.asset_x = str(asset_x).upper()
        self.bus = bus

        self.delta = float(delta)
        self.ve = float(ve)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.max_leg_staleness_sec = float(max_leg_staleness_sec)
        self.cooldown_seconds = float(cooldown_seconds)
        self.nominal_stop_pct = float(nominal_stop_pct)
        # Per-leg target dollar notional. Used by _compute_entry_sizes to
        # derive integer share counts that are beta-hedged on entry; the
        # exact same counts are used on exit so positions close to zero
        # rather than drifting into qty=1-3 residuals.
        self.target_dollar_notional = float(target_dollar_notional)

        self.position: int = 0  # 1 long spread, -1 short spread, 0 flat
        # Currently held share counts per leg (set on entry, cleared on exit).
        # Sizer bypass requires the strategy to be the sole authority on
        # share counts; closing orders re-emit exactly these values.
        self.current_qty_y: int = 0
        self.current_qty_x: int = 0

        self.beta: float = 0.0
        self.P: float = 0.0
        self.is_initialized: bool = False

        self.latest_prices: Dict[str, float] = {self.asset_y: 0.0, self.asset_x: 0.0}
        self.latest_ts_ms: Dict[str, int] = {self.asset_y: 0, self.asset_x: 0}

        self.tz = pytz.timezone("US/Eastern")
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)

        # <-- UPGRADE: Tracking timestamps rather than arbitrary tick counters
        self.last_signal_ts_ms: int = -10_000_000
        # Latest innovation z-score, updated every processed tick. Exposed
        # for heartbeat logging so an idle-but-healthy bot is observable.
        self.last_z_score: float = 0.0

        # Which event channel drives the filter. Backtest replays bar-cadence
        # ticks as EventType.TICK; the live runtime feeds EventType.BAR_TICK
        # from TickResampler so the live cadence matches the validated
        # backtest. on_tick is identical for either channel.
        self._tick_event_type = tick_event_type
        self.bus.subscribe(tick_event_type, self.on_tick)

    def _get_est_dt(self, timestamp_ms: Optional[int] = None) -> datetime:
        if timestamp_ms is not None:
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=pytz.utc)
        else:
            dt = datetime.now(pytz.utc)
        return dt.astimezone(self.tz)

    def _update_filter(self, price_y: float, price_x: float) -> float:
        """
        1D Kalman update on hedge ratio beta.
        Returns current innovation z-score.
        """
        if not self.is_initialized:
            self.beta = price_y / price_x if price_x > 0 else 0.0
            self.P = 1.0
            self.is_initialized = True
            logger.info(
                "[%s/%s] Kalman initialized. beta=%.4f",
                self.asset_y,
                self.asset_x,
                self.beta,
            )
            return 0.0

        beta_pred = self.beta
        P_pred = self.P + self.delta

        y_pred = price_x * beta_pred
        error = price_y - y_pred

        S = (price_x ** 2) * P_pred + self.ve
        K = (P_pred * price_x / S) if S > 0 else 0.0

        self.beta = beta_pred + K * error
        self.P = (1.0 - K * price_x) * P_pred

        z_score = error / np.sqrt(S) if S > 0 else 0.0
        return float(z_score)

    def _legs_are_fresh(self) -> bool:
        ts_y = self.latest_ts_ms.get(self.asset_y, 0)
        ts_x = self.latest_ts_ms.get(self.asset_x, 0)
        if ts_y <= 0 or ts_x <= 0:
            return False
        return abs(ts_y - ts_x) <= int(self.max_leg_staleness_sec * 1000)

    def _in_cooldown(self, current_ts_ms: int) -> bool:
        # <-- UPGRADE: Absolute deterministic timeout check based on SIP feed
        return (current_ts_ms - self.last_signal_ts_ms) < (self.cooldown_seconds * 1000)

    async def on_tick(self, event: Event) -> None:
        payload = event.payload or {}
        ticker = str(payload.get("ticker") or payload.get("symbol") or "").upper()

        if ticker not in {self.asset_y, self.asset_x}:
            return

        try:
            price = float(payload.get("price"))
        except (TypeError, ValueError):
            return

        if price <= 0:
            return

        timestamp_ms = payload.get("timestamp")
        if timestamp_ms is None:
            timestamp_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            
        timestamp_ms = int(timestamp_ms)
        current_dt = self._get_est_dt(timestamp_ms)
        current_time = current_dt.time()

        # strict RTH only
        if current_time < self.market_open or current_time >= self.market_close:
            return

        self.latest_prices[ticker] = price
        self.latest_ts_ms[ticker] = timestamp_ms

        if self.latest_prices[self.asset_y] <= 0 or self.latest_prices[self.asset_x] <= 0:
            return
        if not self._legs_are_fresh():
            return

        current_z = self._update_filter(
            self.latest_prices[self.asset_y],
            self.latest_prices[self.asset_x],
        )
        # Expose the latest innovation z-score for heartbeat / monitoring.
        self.last_z_score = float(current_z)

        await self._evaluate_signals(current_z, timestamp_ms)

    def _compute_entry_sizes(self, price_y: float, price_x: float) -> tuple[int, int]:
        """
        Beta-hedged integer share counts for one spread entry.

        Sizing rationale: the Kalman filter models y = beta * x. To be
        approximately dollar-neutral on a 1-unit y-price move, we need
        qty_x = qty_y * beta. We pick qty_y to put approximately
        target_dollar_notional dollars on the y-leg, then derive qty_x.

        Returns (qty_y, qty_x). Both are clamped to >= 1 so the strategy
        never emits a zero-share order; if the budget can't cover even 1
        share of either leg at current prices, the caller should suppress
        the trade rather than emit a degenerate one.
        """
        if price_y <= 0 or price_x <= 0:
            return (0, 0)
        beta = self.beta if self.beta > 0 else (price_y / price_x)
        qty_y = max(1, int(self.target_dollar_notional / price_y))
        qty_x = max(1, int(round(qty_y * beta)))
        return (int(qty_y), int(qty_x))

    async def _evaluate_signals(self, z_score: float, current_ts_ms: int) -> None:
        if self._in_cooldown(current_ts_ms):
            return

        price_y = self.latest_prices[self.asset_y]
        price_x = self.latest_prices[self.asset_x]

        # Exit logic first — use the exact share counts that were opened so
        # positions close to zero rather than leaving qty=1-3 residuals.
        if self.position != 0 and abs(z_score) <= self.exit_z:
            qty_y = self.current_qty_y
            qty_x = self.current_qty_x
            if qty_y <= 0 or qty_x <= 0:
                # Safety net: position flag is set but sizes are missing.
                # Don't emit ghost-qty exits; clear state and resync next signal.
                logger.warning(
                    "[%s/%s] Exit requested but stored sizes missing (y=%d x=%d). Clearing state.",
                    self.asset_y,
                    self.asset_x,
                    qty_y,
                    qty_x,
                )
                self.position = 0
                self.current_qty_y = 0
                self.current_qty_x = 0
                return

            logger.info(
                "[%s/%s] Mean reversion achieved. z=%.2f exiting spread position=%d qty_y=%d qty_x=%d",
                self.asset_y,
                self.asset_x,
                z_score,
                self.position,
                qty_y,
                qty_x,
            )

            if self.position == 1:
                # long spread: long Y / short X -> sell Y, cover X
                self._emit_order(self.asset_y, "SELL", price_y, qty_y, hedge_role="lead_exit", timestamp_ms=current_ts_ms)
                self._emit_order(self.asset_x, "BUY_TO_COVER", price_x, qty_x, is_hedge=True, hedge_role="hedge_exit", timestamp_ms=current_ts_ms)
            else:
                # short spread: short Y / long X -> cover Y, sell X
                self._emit_order(self.asset_y, "BUY_TO_COVER", price_y, qty_y, hedge_role="lead_exit", timestamp_ms=current_ts_ms)
                self._emit_order(self.asset_x, "SELL", price_x, qty_x, is_hedge=True, hedge_role="hedge_exit", timestamp_ms=current_ts_ms)

            self.position = 0
            self.current_qty_y = 0
            self.current_qty_x = 0
            self.last_signal_ts_ms = current_ts_ms
            return

        if self.position != 0:
            return

        # Entry logic: compute beta-hedged share counts, store them so the
        # matching exit can re-emit the exact same sizes.
        if z_score > self.entry_z:
            qty_y, qty_x = self._compute_entry_sizes(price_y, price_x)
            if qty_y <= 0 or qty_x <= 0:
                return
            logger.warning(
                "[%s/%s] Spread wide. z=%.2f -> short %s x%d / long %s x%d (beta=%.3f)",
                self.asset_y,
                self.asset_x,
                z_score,
                self.asset_y,
                qty_y,
                self.asset_x,
                qty_x,
                self.beta,
            )
            self._emit_order(self.asset_y, "SELL_SHORT", price_y, qty_y, hedge_role="lead_entry", timestamp_ms=current_ts_ms)
            self._emit_order(self.asset_x, "BUY", price_x, qty_x, is_hedge=True, hedge_role="hedge_entry", timestamp_ms=current_ts_ms)
            self.position = -1
            self.current_qty_y = qty_y
            self.current_qty_x = qty_x
            self.last_signal_ts_ms = current_ts_ms

        elif z_score < -self.entry_z:
            qty_y, qty_x = self._compute_entry_sizes(price_y, price_x)
            if qty_y <= 0 or qty_x <= 0:
                return
            logger.warning(
                "[%s/%s] Spread compressed. z=%.2f -> long %s x%d / short %s x%d (beta=%.3f)",
                self.asset_y,
                self.asset_x,
                z_score,
                self.asset_y,
                qty_y,
                self.asset_x,
                qty_x,
                self.beta,
            )
            self._emit_order(self.asset_y, "BUY", price_y, qty_y, hedge_role="lead_entry", timestamp_ms=current_ts_ms)
            self._emit_order(self.asset_x, "SELL_SHORT", price_x, qty_x, is_hedge=True, hedge_role="hedge_entry", timestamp_ms=current_ts_ms)
            self.position = 1
            self.current_qty_y = qty_y
            self.current_qty_x = qty_x
            self.last_signal_ts_ms = current_ts_ms

    def _emit_order(
        self,
        asset: str,
        action: str,
        price: float,
        shares: int,
        is_hedge: bool = False,
        hedge_role: str = "none",
        timestamp_ms: Optional[int] = None,
    ) -> None:
        stop_loss_distance = price * self.nominal_stop_pct
        # Long entries (BUY) and short-cover exits (BUY_TO_COVER) get a stop
        # below the entry; short entries (SELL_SHORT) and long-flatten exits
        # (SELL) get a stop above the entry. The simple "BUY" in action
        # check captures all four — the broker router skips protective
        # stops on flatten orders anyway, so the value passed in for those
        # cases never gets routed.
        stop_loss = price - stop_loss_distance if "BUY" in action else price + stop_loss_distance

        # Bypass ranker + sizer: emit stage="SIZED" with explicit shares so
        # the strategy is the sole authority on per-leg quantities. The
        # default sizer recomputes share counts from per-share-risk at every
        # call, which produced 1-3-share residuals each round-trip because
        # exit-time prices (and floored max-position caps) differed from
        # entry-time. With explicit shares + sizer skip, the close emits
        # the exact qty that was opened and the position lands at zero.
        signal_id = str(uuid.uuid4())
        order_event = Event(
            type=EventType.ORDER_CREATE,
            payload={
                "signal_id": signal_id,
                "asset": asset,
                "action": action,
                "strategy": f"KalmanPair_{self.asset_y}_{self.asset_x}",
                "stage": "SIZED",
                "approved_by_ranker": True,
                "shares": int(shares),
                "decision_id": signal_id,
                "rank_score": 5.0,
                "rank_components": {
                    "score": 5.0,
                    "rs": 0.0,
                    "rvol": 1.0,
                    "spread_bps": 0.0,
                    "dist_vwap_pct": 0.0,
                    "liquidity_score": 0.5,
                    "hard_veto": False,
                    "reasons": [],
                    "source": "kalman_pairs_bypass",
                },
                "capital_allocated": round(shares * price, 2),
                "risk_dollars": round(shares * stop_loss_distance, 2),
                "timestamp": timestamp_ms,
                "reference_price": round(price, 4),
                "entry_price": round(price, 4),
                "stop_loss": round(stop_loss, 4),
                "stop_loss_price": round(stop_loss, 4),
                "hedge_ratio": round(self.beta if is_hedge else 1.0, 6),
                "status": "READY_FOR_BROKER",
                "signal_context": {
                    "pair_y": self.asset_y,
                    "pair_x": self.asset_x,
                    "pair_position": self.position,
                    "is_hedge": is_hedge,
                    "hedge_role": hedge_role,
                    "beta": round(self.beta, 6),
                    "entry_z": self.entry_z,
                    "exit_z": self.exit_z,
                },
            },
        )
        self.bus.publish(order_event)


if __name__ == "__main__":
    async def run_kalman_equity_test() -> None:
        print("Initializing optimized Kalman Pairs Test...")
        bus = EventBus()
        await bus.start()

        trader = USEquityKalmanPairsTrader(asset_y="KO", asset_x="PEP", bus=bus)

        def mock_tick(asset: str, price: float, hour: int, minute: int, second: int = 0) -> Event:
            dt = datetime(2026, 3, 9, hour, minute, second, tzinfo=timezone.utc)
            return Event(
                type=EventType.TICK,
                payload={
                    "ticker": asset,
                    "price": price,
                    "timestamp": int(dt.timestamp() * 1000),
                },
            )

        print("\n[SYSTEM] Simulating pre-market ticks (ignored)...")
        bus.publish(mock_tick("KO", 60.0, 13, 0))
        bus.publish(mock_tick("PEP", 170.0, 13, 0))
        await asyncio.sleep(0.1)
        print(f"Filter initialized: {trader.is_initialized}")

        print("\n[SYSTEM] Simulating market open...")
        bus.publish(mock_tick("KO", 60.5, 14, 30))
        bus.publish(mock_tick("PEP", 170.5, 14, 30))
        await asyncio.sleep(0.1)
        print(f"Filter initialized: {trader.is_initialized}")

        print("\n[SYSTEM] Simulating divergence during RTH with absolute time pacing...")
        for i in range(10):
            # Advance 2 seconds per tick to respect the 5.0 second cooldown between entries/exits
            bus.publish(mock_tick("KO", 60.5 - (i * 0.2), 15, i, second=i * 2))
            bus.publish(mock_tick("PEP", 170.5 + (i * 0.2), 15, i, second=i * 2))
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.5)
        await bus.stop()

    asyncio.run(run_kalman_equity_test())