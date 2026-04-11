from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, Optional

import pytz


class SessionPhase(Enum):
    CLOSED = "closed"
    PREWARM = "prewarm"
    IN_SESSION = "in_session"
    AFTER_HOURS = "after_hours"


@dataclass(frozen=True)
class SessionWindow:
    trading_date: date
    prewarm_start: datetime
    market_open: datetime
    market_close: datetime
    shutdown_time: datetime


class MarketSessionClock:
    """Simple US-equity session clock for day-cycle orchestration."""

    def __init__(
        self,
        timezone_name: str = "US/Eastern",
        market_open_hour: int = 9,
        market_open_minute: int = 30,
        market_close_hour: int = 16,
        market_close_minute: int = 0,
        prewarm_minutes: int = 5,
        shutdown_delay_minutes: int = 5,
    ) -> None:
        self.tz = pytz.timezone(timezone_name)
        self.market_open_time = time(int(market_open_hour), int(market_open_minute))
        self.market_close_time = time(int(market_close_hour), int(market_close_minute))
        self.prewarm_minutes = int(prewarm_minutes)
        self.shutdown_delay_minutes = int(shutdown_delay_minutes)
        self._window_overrides: Dict[date, SessionWindow] = {}
        self._closed_overrides: set[date] = set()

    def now(self) -> datetime:
        return datetime.now(pytz.utc).astimezone(self.tz)

    def is_trading_day(self, current_dt: datetime | None = None) -> bool:
        current_dt = current_dt or self.now()
        trading_date = current_dt.date()
        return current_dt.weekday() < 5 and trading_date not in self._closed_overrides

    def set_trading_day_override(
        self,
        trading_date: date,
        market_open_hour: int,
        market_open_minute: int,
        market_close_hour: int,
        market_close_minute: int,
    ) -> None:
        market_open = self.tz.localize(
            datetime.combine(trading_date, time(int(market_open_hour), int(market_open_minute)))
        )
        market_close = self.tz.localize(
            datetime.combine(trading_date, time(int(market_close_hour), int(market_close_minute)))
        )
        if market_close <= market_open:
            return

        prewarm_start = market_open - timedelta(minutes=self.prewarm_minutes)
        shutdown_time = market_close + timedelta(minutes=self.shutdown_delay_minutes)
        self._window_overrides[trading_date] = SessionWindow(
            trading_date=trading_date,
            prewarm_start=prewarm_start,
            market_open=market_open,
            market_close=market_close,
            shutdown_time=shutdown_time,
        )
        self._closed_overrides.discard(trading_date)

    def set_market_closed(self, trading_date: date) -> None:
        self._closed_overrides.add(trading_date)
        self._window_overrides.pop(trading_date, None)

    def window_for(self, trading_date: date) -> SessionWindow:
        override = self._window_overrides.get(trading_date)
        if override is not None:
            return override

        market_open = self.tz.localize(datetime.combine(trading_date, self.market_open_time))
        market_close = self.tz.localize(datetime.combine(trading_date, self.market_close_time))
        prewarm_start = market_open - timedelta(minutes=self.prewarm_minutes)
        shutdown_time = market_close + timedelta(minutes=self.shutdown_delay_minutes)
        return SessionWindow(
            trading_date=trading_date,
            prewarm_start=prewarm_start,
            market_open=market_open,
            market_close=market_close,
            shutdown_time=shutdown_time,
        )

    def current_window(self, current_dt: datetime | None = None) -> SessionWindow:
        current_dt = current_dt or self.now()
        return self.window_for(current_dt.date())

    def _trading_window_for_date(self, trading_date: date) -> Optional[SessionWindow]:
        if trading_date.weekday() >= 5:
            return None
        if trading_date in self._closed_overrides:
            return None
        return self.window_for(trading_date)

    def phase(self, current_dt: datetime | None = None) -> SessionPhase:
        current_dt = current_dt or self.now()
        if not self.is_trading_day(current_dt):
            return SessionPhase.CLOSED

        window = self._trading_window_for_date(current_dt.date())
        if window is None:
            return SessionPhase.CLOSED

        if current_dt < window.prewarm_start:
            return SessionPhase.CLOSED
        if current_dt < window.market_open:
            return SessionPhase.PREWARM
        if current_dt <= window.shutdown_time:
            return SessionPhase.IN_SESSION
        return SessionPhase.AFTER_HOURS

    def next_prewarm_start(self, current_dt: datetime | None = None) -> datetime:
        current_dt = current_dt or self.now()
        probe_date = current_dt.date()

        for offset in range(15):
            candidate_date = probe_date + timedelta(days=offset)
            candidate_window = self._trading_window_for_date(candidate_date)
            if candidate_window is None:
                continue
            if offset == 0 and current_dt <= candidate_window.prewarm_start:
                return candidate_window.prewarm_start
            if offset > 0:
                return candidate_window.prewarm_start

        fallback = self.window_for(probe_date + timedelta(days=7))
        return fallback.prewarm_start

    def seconds_until_next_prewarm(self, current_dt: datetime | None = None) -> float:
        current_dt = current_dt or self.now()
        delta = self.next_prewarm_start(current_dt) - current_dt
        return max(0.0, delta.total_seconds())