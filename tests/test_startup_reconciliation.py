import unittest

from core.engine.event_bus import EventBus
from core.execution.eod_liquidator import EODLiquidationManager
from risk.virtual_monitor.equity_slope_detector import VirtualEquitySlopeDetector


class TestStartupReconciliation(unittest.TestCase):
    def test_seed_positions_updates_liquidator_and_virtual_equity_state(self) -> None:
        bus = EventBus()
        liquidator = EODLiquidationManager(bus=bus)
        detector = VirtualEquitySlopeDetector(bus=bus, initial_capital=100000.0)

        snapshot = {
            "SPY": {"qty": 10, "avg_entry_price": 500.0, "last_price": 501.0},
            "QQQ": {"qty": -5, "avg_entry_price": 430.0, "last_price": 429.5},
        }

        liquidator.seed_positions(snapshot)
        detector.seed_positions(snapshot, current_equity=101250.0)

        self.assertEqual(liquidator.snapshot(), {"SPY": 10, "QQQ": -5})
        self.assertEqual(detector.snapshot()["open_positions"]["SPY"]["qty"], 10)
        self.assertEqual(detector.snapshot()["open_positions"]["QQQ"]["qty"], -5)
        self.assertEqual(detector.snapshot()["current_equity"], 101250.0)


if __name__ == "__main__":
    unittest.main()