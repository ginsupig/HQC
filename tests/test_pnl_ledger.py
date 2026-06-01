"""Unit tests for risk/portfolio/pnl_ledger.py."""
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from risk.portfolio.pnl_ledger import PnLLedger  # noqa: E402


class TestPnLLedger(unittest.TestCase):
    def test_long_roundtrip_realizes_gain(self):
        led = PnLLedger()
        led.on_fill(order_id="o1", symbol="JPM", action="BUY", cumulative_filled_qty=100, price=100.0)
        self.assertEqual(led.position("JPM"), 100)
        r = led.on_fill(order_id="o2", symbol="JPM", action="SELL", cumulative_filled_qty=100, price=110.0)
        self.assertAlmostEqual(r, 1000.0)  # (110-100)*100
        self.assertAlmostEqual(led.realized_pnl, 1000.0)
        self.assertEqual(led.position("JPM"), 0)

    def test_short_roundtrip_realizes_gain(self):
        led = PnLLedger()
        led.on_fill(order_id="o1", symbol="BAC", action="SELL_SHORT", cumulative_filled_qty=50, price=40.0)
        self.assertEqual(led.position("BAC"), -50)
        r = led.on_fill(order_id="o2", symbol="BAC", action="BUY_TO_COVER", cumulative_filled_qty=50, price=35.0)
        self.assertAlmostEqual(r, 250.0)  # (40-35)*50
        self.assertEqual(led.position("BAC"), 0)

    def test_average_cost_on_add(self):
        led = PnLLedger()
        led.on_fill(order_id="o1", symbol="X", action="BUY", cumulative_filled_qty=100, price=10.0)
        led.on_fill(order_id="o2", symbol="X", action="BUY", cumulative_filled_qty=100, price=20.0)
        # avg cost (10*100+20*100)/200 = 15
        r = led.on_fill(order_id="o3", symbol="X", action="SELL", cumulative_filled_qty=200, price=25.0)
        self.assertAlmostEqual(r, (25 - 15) * 200)

    def test_flip_through_zero(self):
        led = PnLLedger()
        led.on_fill(order_id="o1", symbol="X", action="BUY", cumulative_filled_qty=100, price=10.0)
        # Sell 150: closes 100 long (realize (12-10)*100=200), opens 50 short @ 12
        r = led.on_fill(order_id="o2", symbol="X", action="SELL_SHORT", cumulative_filled_qty=150, price=12.0)
        self.assertAlmostEqual(r, 200.0)
        self.assertEqual(led.position("X"), -50)
        # Cover 50 @ 11 => (12-11)*50 = 50 gain
        r2 = led.on_fill(order_id="o3", symbol="X", action="BUY_TO_COVER", cumulative_filled_qty=50, price=11.0)
        self.assertAlmostEqual(r2, 50.0)

    def test_cumulative_partial_fills_not_double_counted(self):
        led = PnLLedger()
        # Same order fills in two cumulative updates: 40 then 100 total.
        led.on_fill(order_id="o1", symbol="X", action="BUY", cumulative_filled_qty=40, price=10.0)
        led.on_fill(order_id="o1", symbol="X", action="BUY", cumulative_filled_qty=100, price=10.0)
        self.assertEqual(led.position("X"), 100)  # not 140
        # Duplicate/stale update with same cumulative -> no change.
        led.on_fill(order_id="o1", symbol="X", action="BUY", cumulative_filled_qty=100, price=10.0)
        self.assertEqual(led.position("X"), 100)

    def test_net_and_gross_exposure(self):
        led = PnLLedger()
        led.on_fill(order_id="o1", symbol="JPM", action="BUY", cumulative_filled_qty=100, price=100.0)
        led.on_fill(order_id="o2", symbol="BAC", action="SELL_SHORT", cumulative_filled_qty=200, price=40.0)
        self.assertAlmostEqual(led.net_exposure("JPM"), 100 * 100.0)
        self.assertAlmostEqual(led.net_exposure("BAC"), -200 * 40.0)
        self.assertAlmostEqual(led.gross_exposure(), 10000 + 8000)
        # Mark moves exposure.
        self.assertAlmostEqual(led.net_exposure("JPM", mark_price=110.0), 11000.0)

    def test_unknown_action_ignored(self):
        led = PnLLedger()
        r = led.on_fill(order_id="o1", symbol="X", action="WHATEVER", cumulative_filled_qty=10, price=5.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(led.position("X"), 0)


if __name__ == "__main__":
    unittest.main()
