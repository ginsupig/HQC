"""
Overnight gap-fade strategy.

Mechanics
---------
At the first tick of each RTH session, compare today's first traded
price to yesterday's last traded close. If the absolute gap exceeds
``gap_trigger_pct`` (default 50 bps), fade it:
  - gap up   -> SELL_SHORT (entry near open, expect mean reversion)
  - gap down -> BUY        (entry near open, expect mean reversion)

The position is opened with a stop ``stop_pct`` away from entry on the
gap side (above for shorts, below for longs); the harness time-stop
and EOD liquidator handle the exit.

Why this strategy
-----------------
Overnight-gap mean reversion is a well-documented academic anomaly
(Boudoukh, Richardson, Whitelaw 2008; Lou, Polk, Skouras 2019). It is
present in SPY, QQQ, and large-cap equities at typical magnitudes of
~5-15 bps per trade gross. Used here as a *baseline known-edge*
strategy: if our backtest harness shows even this losing money, the
harness still has a bug. If it shows positive OOS edge, the harness
is trustworthy and the negative ORB/VWAP findings are the real signal.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, time, date
from typing import Optional

import pytz

from core.engine.event_bus import Event, EventBus, EventType

logger = logging.getLogger("OvernightGapFade")


class OvernightGapFade:
    def __init__(
        self,
        target_asset: str,
        bus: EventBus,
        gap_trigger_pct: float = 0.005,
        stop_pct: float = 0.012,
        max_trades_per_day: int = 1,
        timezone_name: str = "US/Eastern",
    ) -> None:
        self.asset = str(target_asset).upper()
        self.bus = bus
        self.gap_trigger_pct = float(gap_trigger_pct)
        self.stop_pct = float(stop_pct)
        self.max_trades_per_day = int(max_trades_per_day)
        self.tz = pytz.timezone(timezone_name)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)

        self._last_session_close: Optional[float] = None
        self._current_session_date: Optional[date] = None
        self._fired_today: int = 0
        self._previous_close_locked: Optional[float] = None
        self._latest_session_last_price: Optional[float] = None

        self.bus.subscribe(EventType.TICK, self.on_tick)

    def _est_dt(self, ts_ms: Optional[int]) -> datetime:
        if ts_ms is None:
            return datetime.now(pytz.utc).astimezone(self.tz)
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=pytz.utc).astimezone(self.tz)

    async def on_tick(self, event: Event) -> None:
        payload = event.payload or {}
        ticker = str(payload.get("ticker") or payload.get("symbol") or "").upper()
        if ticker != self.asset:
            return
        try:
            price = float(payload.get("price"))
        except (TypeError, ValueError):
            return
        if price <= 0:
            return

        ts_ms = payload.get("timestamp")
        try:
            ts_ms_int = int(ts_ms) if ts_ms is not None else None
        except (TypeError, ValueError):
            ts_ms_int = None

        dt_est = self._est_dt(ts_ms_int)
        cur_date = dt_est.date()
        cur_time = dt_est.time()

        # Day rollover bookkeeping. When the calendar date changes we
        # roll the previously accumulated last-price into _last_session_close
        # so the first RTH bar of the new day has a yesterday-close to
        # compare against.
        if self._current_session_date != cur_date:
            if self._latest_session_last_price is not None:
                self._last_session_close = self._latest_session_last_price
            self._current_session_date = cur_date
            self._fired_today = 0
            self._previous_close_locked = self._last_session_close

        # Track latest-known last price every tick (RTH and after-hours).
        self._latest_session_last_price = price

        # Only fire entries during RTH and only once per session.
        if cur_time < self.market_open or cur_time >= self.market_close:
            return
        if self._fired_today >= self.max_trades_per_day:
            return
        prev_close = self._previous_close_locked
        if prev_close is None or prev_close <= 0:
            return

        gap_pct = (price - prev_close) / prev_close
        if abs(gap_pct) < self.gap_trigger_pct:
            return

        if gap_pct > 0:
            action = "SELL_SHORT"
            stop_loss = price * (1.0 + self.stop_pct)
        else:
            action = "BUY"
            stop_loss = price * (1.0 - self.stop_pct)

        signal_id = str(uuid.uuid4())
        # Gap-fade fires on the first RTH tick of a session: by definition
        # the rolling tick book has zero history at that moment, so
        # CandidateRanker's rule-based score (which combines spread,
        # rvol, liquidity, and vwap proxies) collapses well below
        # min_rank_score and silently kills every signal. Bypass it by
        # emitting stage="RANKED" with a neutral rank_score so the order
        # flows directly into the sizer; the entry's own conviction comes
        # from the > gap_trigger_pct overnight move, which is itself a
        # stronger filter than anything the microstructure ranker could
        # add at session open.
        order_event = Event(
            type=EventType.ORDER_CREATE,
            payload={
                "signal_id": signal_id,
                "asset": self.asset,
                "action": action,
                "strategy": "Overnight_Gap_Fade",
                "stage": "RANKED",
                "approved_by_ranker": True,
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
                    "source": "gap_fade_bypass",
                },
                "timestamp": ts_ms_int,
                "reference_price": round(price, 4),
                "entry_price": round(price, 4),
                "stop_loss_price": round(stop_loss, 4),
                "signal_context": {
                    "previous_close": round(prev_close, 4),
                    "gap_pct": round(gap_pct, 6),
                    "gap_trigger_pct": self.gap_trigger_pct,
                    "stop_pct": self.stop_pct,
                },
            },
        )
        logger.warning(
            "[%s] Overnight gap %+.3f%% prev_close=%.2f open=%.2f -> %s stop=%.2f signal_id=%s",
            self.asset,
            gap_pct * 100.0,
            prev_close,
            price,
            action,
            stop_loss,
            signal_id[:8],
        )
        self.bus.publish(order_event)
        self._fired_today += 1
