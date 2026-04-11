import unittest
from datetime import date, datetime

import pytz

from core.engine.session_clock import MarketSessionClock, SessionPhase


class TestSessionClock(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = MarketSessionClock(prewarm_minutes=5, shutdown_delay_minutes=5)
        self.tz = pytz.timezone("US/Eastern")

    def test_phase_transitions_cover_prewarm_session_and_after_hours(self) -> None:
        prewarm = self.tz.localize(datetime(2026, 4, 7, 9, 27))
        in_session = self.tz.localize(datetime(2026, 4, 7, 10, 15))
        after_hours = self.tz.localize(datetime(2026, 4, 7, 16, 6))
        saturday = self.tz.localize(datetime(2026, 4, 11, 10, 0))

        self.assertEqual(self.clock.phase(prewarm), SessionPhase.PREWARM)
        self.assertEqual(self.clock.phase(in_session), SessionPhase.IN_SESSION)
        self.assertEqual(self.clock.phase(after_hours), SessionPhase.AFTER_HOURS)
        self.assertEqual(self.clock.phase(saturday), SessionPhase.CLOSED)

    def test_next_prewarm_start_skips_weekend(self) -> None:
        friday_after_close = self.tz.localize(datetime(2026, 4, 10, 18, 0))
        next_start = self.clock.next_prewarm_start(friday_after_close)

        self.assertEqual(next_start.weekday(), 0)
        self.assertEqual(next_start.hour, 9)
        self.assertEqual(next_start.minute, 25)

    def test_early_close_override_switches_to_after_hours(self) -> None:
        self.clock.set_trading_day_override(
            trading_date=date(2026, 7, 3),
            market_open_hour=9,
            market_open_minute=30,
            market_close_hour=13,
            market_close_minute=0,
        )

        before_shutdown = self.tz.localize(datetime(2026, 7, 3, 13, 4))
        after_shutdown = self.tz.localize(datetime(2026, 7, 3, 13, 6))

        self.assertEqual(self.clock.phase(before_shutdown), SessionPhase.IN_SESSION)
        self.assertEqual(self.clock.phase(after_shutdown), SessionPhase.AFTER_HOURS)

    def test_market_closed_override_forces_closed_phase(self) -> None:
        holiday = date(2026, 12, 25)
        self.clock.set_market_closed(holiday)

        holiday_dt = self.tz.localize(datetime(2026, 12, 25, 10, 0))
        self.assertEqual(self.clock.phase(holiday_dt), SessionPhase.CLOSED)


if __name__ == "__main__":
    unittest.main()