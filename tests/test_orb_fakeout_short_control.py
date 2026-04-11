import asyncio
import unittest
from datetime import datetime, timedelta, timezone

from core.engine.event_bus import Event, EventBus, EventType
from strategies.orb.equity_orb import ORBState, USEquityORB


class TestORBFakeoutAndShortControl(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = EventBus()
        await self.bus.start()
        self.captured_orders = []
        self.bus.subscribe(EventType.ORDER_CREATE, self._capture_order)

    async def asyncTearDown(self) -> None:
        await self.bus.stop()

    async def _capture_order(self, event: Event) -> None:
        self.captured_orders.append(event.payload)

    async def _pump_and_wait(self, event: Event) -> None:
        self.bus.publish(event)
        await asyncio.sleep(0.01)
        await self.bus._queue.join()
        await asyncio.sleep(0.03)

    @staticmethod
    def _tick(asset: str, price: float, when: datetime) -> Event:
        return Event(
            type=EventType.TICK,
            payload={
                "ticker": asset,
                "symbol": asset,
                "price": price,
                "volume": 2500.0,
                "timestamp": int(when.timestamp() * 1000),
            },
        )

    async def test_short_breakdown_ignored_when_shorts_disabled(self) -> None:
        asset = "SPY"
        orb = USEquityORB(
            target_asset=asset,
            bus=self.bus,
            range_minutes=15,
            min_range_pct=0.0015,
            breakout_buffer_pct=0.0005,
            enable_shorts=False,
        )

        base = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        for px, minutes in ((501.5, 0), (502.0, 5), (501.0, 10), (501.4, 14), (501.3, 15)):
            await self._pump_and_wait(self._tick(asset, px, base + timedelta(minutes=minutes)))

        self.assertEqual(orb.state, ORBState.ACTIVE)

        await self._pump_and_wait(self._tick(asset, 500.5, base + timedelta(minutes=18)))
        self.assertEqual(len(self.captured_orders), 0)

    async def test_fakeout_needs_two_confirm_ticks(self) -> None:
        asset = "SPY"
        orb = USEquityORB(
            target_asset=asset,
            bus=self.bus,
            range_minutes=15,
            min_range_pct=0.0015,
            breakout_buffer_pct=0.0005,
            breakout_confirmation_ticks=2,
            fakeout_reset_pct=0.0003,
        )

        base = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        for px, minutes in ((501.0, 0), (502.0, 5), (500.0, 10), (501.4, 14), (501.3, 15)):
            await self._pump_and_wait(self._tick(asset, px, base + timedelta(minutes=minutes)))

        self.assertEqual(orb.state, ORBState.ACTIVE)

        # One-tick breakout then reclaim inside range: should not fire.
        await self._pump_and_wait(self._tick(asset, 502.30, base + timedelta(minutes=16)))
        await self._pump_and_wait(self._tick(asset, 501.70, base + timedelta(minutes=17)))
        self.assertEqual(len(self.captured_orders), 0)

        # Two consecutive confirming ticks above trigger: should fire one BUY.
        await self._pump_and_wait(self._tick(asset, 502.35, base + timedelta(minutes=18)))
        await self._pump_and_wait(self._tick(asset, 502.40, base + timedelta(minutes=19)))
        self.assertEqual(len(self.captured_orders), 1)
        self.assertEqual(self.captured_orders[0]["action"], "BUY")


if __name__ == "__main__":
    unittest.main()
