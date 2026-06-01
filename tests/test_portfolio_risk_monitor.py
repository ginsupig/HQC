"""Async tests for risk/portfolio/portfolio_risk_monitor.py against a live EventBus."""
import asyncio
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.engine.event_bus import Event, EventBus, EventType  # noqa: E402
from risk.portfolio.portfolio_risk_monitor import PortfolioRiskMonitor  # noqa: E402


def _fill(symbol, action, qty, price, order_id):
    return Event(
        type=EventType.ORDER_FILL,
        payload={
            "order_id": order_id, "asset": symbol, "action": action,
            "filled_qty": qty, "fill_price": price, "status": "FILLED",
        },
    )


class TestPortfolioRiskMonitor(unittest.IsolatedAsyncioTestCase):
    async def _bus_with_capture(self):
        bus = EventBus()
        await bus.start()
        shutdowns = []

        async def _capture(ev):
            shutdowns.append(ev.payload)

        bus.subscribe(EventType.SYSTEM_SHUTDOWN, _capture)
        return bus, shutdowns

    async def test_daily_pnl_kill_fires(self):
        bus, shutdowns = await self._bus_with_capture()
        PortfolioRiskMonitor(bus=bus, equity=10000, daily_loss_pct_kill=0.05)
        # Open long 100 @ 100, close @ 90 => realized -1000 <= -500 limit.
        bus.publish(_fill("JPM", "BUY", 100, 100.0, "o1"))
        await asyncio.sleep(0.05)
        bus.publish(_fill("JPM", "SELL", 100, 90.0, "o2"))
        await asyncio.sleep(0.1)
        self.assertTrue(shutdowns)
        self.assertEqual(shutdowns[0]["source"], "PortfolioRiskMonitor")
        self.assertIn("daily realized PnL", shutdowns[0]["reason"])
        await bus.stop()

    async def test_no_kill_when_within_limit(self):
        bus, shutdowns = await self._bus_with_capture()
        PortfolioRiskMonitor(bus=bus, equity=10000, daily_loss_pct_kill=0.05)
        bus.publish(_fill("JPM", "BUY", 100, 100.0, "o1"))
        await asyncio.sleep(0.05)
        bus.publish(_fill("JPM", "SELL", 100, 99.0, "o2"))  # -100 only
        await asyncio.sleep(0.1)
        self.assertEqual(shutdowns, [])
        await bus.stop()

    async def test_symbol_exposure_kill(self):
        bus, shutdowns = await self._bus_with_capture()
        # max_symbol_pct 0.05 of 100k = 5k. Buying 100 @ 100 = 10k > 5k.
        PortfolioRiskMonitor(bus=bus, equity=100000, daily_loss_pct_kill=None,
                             max_symbol_pct=0.05)
        bus.publish(_fill("JPM", "BUY", 100, 100.0, "o1"))
        await asyncio.sleep(0.1)
        self.assertTrue(shutdowns)
        self.assertIn("net exposure", shutdowns[0]["reason"])
        await bus.stop()

    async def test_gross_exposure_kill(self):
        bus, shutdowns = await self._bus_with_capture()
        # max_gross_leverage 1.0 of 10k = 10k. Two 8k legs = 16k gross.
        PortfolioRiskMonitor(bus=bus, equity=10000, daily_loss_pct_kill=None,
                             max_symbol_pct=None, max_gross_leverage=1.0)
        bus.publish(_fill("JPM", "BUY", 80, 100.0, "o1"))
        bus.publish(_fill("BAC", "SELL_SHORT", 80, 100.0, "o2"))
        await asyncio.sleep(0.1)
        self.assertTrue(shutdowns)
        self.assertIn("gross exposure", shutdowns[0]["reason"])
        await bus.stop()

    async def test_halt_is_idempotent(self):
        bus, shutdowns = await self._bus_with_capture()
        PortfolioRiskMonitor(bus=bus, equity=10000, daily_loss_pct_kill=0.05)
        bus.publish(_fill("JPM", "BUY", 100, 100.0, "o1"))
        await asyncio.sleep(0.05)
        bus.publish(_fill("JPM", "SELL", 100, 80.0, "o2"))
        await asyncio.sleep(0.05)
        bus.publish(_fill("JPM", "SELL", 100, 70.0, "o3"))
        await asyncio.sleep(0.1)
        self.assertEqual(len(shutdowns), 1)  # only one SYSTEM_SHUTDOWN
        await bus.stop()


if __name__ == "__main__":
    unittest.main()
