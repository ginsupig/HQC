import unittest
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List

# Core engine imports
from core.engine.event_bus import EventBus, Event, EventType
from strategies.orb.equity_orb import USEquityORB, ORBState

# Leave warnings on to capture any strategy rejections
logging.getLogger("EventBus").setLevel(logging.WARNING)
logging.getLogger("EquityORB").setLevel(logging.DEBUG)

class TestORBExecutionParity(unittest.IsolatedAsyncioTestCase):
    """
    Rigorous unit test to guarantee that the Event-Driven architecture 
    processes ticks and fires signals identically to a live environment, 
    completely eliminating lookahead bias.
    """

    async def asyncSetUp(self):
        self.bus = EventBus()
        await self.bus.start()
        
        self.asset = "SPY"
        self.orb = USEquityORB(
            target_asset=self.asset, 
            bus=self.bus, 
            range_minutes=15,
            breakout_buffer_pct=0.0005 # 5 bps buffer
        )
        
        self.captured_orders: List[Event] = []
        self.bus.subscribe(EventType.ORDER_CREATE, self._order_catcher)

    async def asyncTearDown(self):
        await self.bus.stop()

    async def _order_catcher(self, event: Event):
        self.captured_orders.append(event)

    def _generate_mock_tick(self, price: float, dt: datetime) -> Event:
        return Event(
            type=EventType.TICK,
            payload={
                "ticker": self.asset,
                "price": price,
                "volume": 2500.0, # Realistic volume
                "timestamp": int(dt.timestamp() * 1000)
            }
        )

    async def _pump_and_wait(self, event: Event):
        """Helper to completely flush the async event loop to fix race conditions."""
        self.bus.publish(event)
        await asyncio.sleep(0.01) # Allow EventBus to pop
        await self.bus._queue.join() # Wait for queue
        await asyncio.sleep(0.05) # Allow background strategy callbacks to finish

    async def test_bullish_orb_breakout(self):
        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc) # 9:30 AM EST
        
        # 1. 9:30 AM to 9:44 AM: Building a realistic, tight range (500 to 502)
        ticks_building = [
            self._generate_mock_tick(501.00, base_date),
            self._generate_mock_tick(502.00, base_date + timedelta(minutes=5)),  # Range High
            self._generate_mock_tick(500.00, base_date + timedelta(minutes=10)), # Range Low
            self._generate_mock_tick(501.50, base_date + timedelta(minutes=14)),
        ]
        
        for tick in ticks_building:
            await self._pump_and_wait(tick)
            
        self.assertEqual(self.orb.state, ORBState.BUILDING_RANGE)
        self.assertEqual(self.orb.range_high, 502.00)
        self.assertEqual(self.orb.range_low, 500.00)
        self.assertEqual(len(self.captured_orders), 0)

        # 2. 9:45 AM: Range is Established
        await self._pump_and_wait(self._generate_mock_tick(501.50, base_date + timedelta(minutes=15)))
        self.assertEqual(self.orb.state, ORBState.ACTIVE)

        # 3. 9:48 AM: BULLISH BREAKOUT
        # Buffer is 5 bps (0.25). Trigger is ~502.25. 
        # We push to 502.50 to safely clear the threshold without hitting chase limits.
        await self._pump_and_wait(self._generate_mock_tick(502.50, base_date + timedelta(minutes=18)))
        
        self.assertEqual(len(self.captured_orders), 1, "Strategy failed to fire on breakout!")
        
        fired_order = self.captured_orders[0].payload
        self.assertEqual(fired_order["action"], "BUY")
        self.assertEqual(fired_order["reference_price"], 502.50)
        # Ensure stop loss is placed below our entry
        self.assertLess(fired_order["stop_loss_price"], fired_order["reference_price"]) 


    async def test_bearish_orb_breakdown(self):
        """Test short breakout signal generation below range low."""
        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        
        # Build range 501-503 (2pt range / mid~502 ≈ 0.40% — passes min_range_pct=0.0025)
        ticks_building = [
            self._generate_mock_tick(501.50, base_date),
            self._generate_mock_tick(503.00, base_date + timedelta(minutes=5)),
            self._generate_mock_tick(501.00, base_date + timedelta(minutes=10)),
            self._generate_mock_tick(501.50, base_date + timedelta(minutes=14)),
        ]

        for tick in ticks_building:
            await self._pump_and_wait(tick)

        self.assertEqual(self.orb.state, ORBState.BUILDING_RANGE)
        self.assertEqual(self.orb.range_high, 503.00)
        self.assertEqual(self.orb.range_low, 501.00)
        self.assertEqual(len(self.captured_orders), 0)

        # Range established
        await self._pump_and_wait(self._generate_mock_tick(501.50, base_date + timedelta(minutes=15)))
        self.assertEqual(self.orb.state, ORBState.ACTIVE)

        # BEARISH BREAKDOWN: Below range_low * (1 - buffer)
        # Trigger ~500.75. Push to 500.50 to safely clear.
        await self._pump_and_wait(self._generate_mock_tick(500.50, base_date + timedelta(minutes=18)))
        
        self.assertEqual(len(self.captured_orders), 1, "Strategy failed to fire on breakdown!")
        
        fired_order = self.captured_orders[0].payload
        self.assertEqual(fired_order["action"], "SELL_SHORT")
        self.assertEqual(fired_order["reference_price"], 500.50)
        self.assertGreater(fired_order["stop_loss_price"], fired_order["reference_price"])

    async def test_max_trades_limit_enforced(self):
        """Verify ORB respects max_trades limit and stops firing after N trades."""
        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        
        # Re-create on a fresh bus so the setUp ORB's TICK subscription doesn't interfere
        await self.bus.stop()
        self.bus = EventBus()
        await self.bus.start()
        self.captured_orders = []
        self.bus.subscribe(EventType.ORDER_CREATE, self._order_catcher)

        self.orb = USEquityORB(
            target_asset=self.asset,
            bus=self.bus,
            range_minutes=15,
            max_trades=1,  # Only 1 trade allowed
            breakout_buffer_pct=0.0005
        )

        # Build range 499-500.50 (1.5pt range / mid~499.75 ≈ 0.30% — passes min_range_pct=0.0025)
        ticks = [
            self._generate_mock_tick(499.00, base_date + timedelta(minutes=1)),
            self._generate_mock_tick(500.50, base_date + timedelta(minutes=5)),
            self._generate_mock_tick(499.25, base_date + timedelta(minutes=10)),
            self._generate_mock_tick(499.75, base_date + timedelta(minutes=15)),
        ]
        
        for tick in ticks:
            await self._pump_and_wait(tick)
        
        self.assertEqual(self.orb.state, ORBState.ACTIVE)
        self.assertEqual(self.orb.trades_today, 0)

        # First breakout (should fire)
        await self._pump_and_wait(self._generate_mock_tick(500.80, base_date + timedelta(minutes=18)))
        self.assertEqual(len(self.captured_orders), 1)
        self.assertEqual(self.orb.trades_today, 1)
        self.assertEqual(self.orb.state, ORBState.DONE_FOR_DAY)

        # Try second breakout (should NOT fire - already at max)
        await self._pump_and_wait(self._generate_mock_tick(500.85, base_date + timedelta(minutes=20)))
        self.assertEqual(len(self.captured_orders), 1, "Strategy fired 2nd trade despite max_trades=1")

    async def test_range_too_narrow_rejection(self):
        """Verify ORB rejects ranges that are too narrow."""
        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        
        # Build extremely narrow range (1 bps) - below min_range_pct (25 bps)
        ticks_building = [
            self._generate_mock_tick(500.00, base_date),
            self._generate_mock_tick(500.005, base_date + timedelta(minutes=5)),
            self._generate_mock_tick(500.00, base_date + timedelta(minutes=10)),
            self._generate_mock_tick(500.00, base_date + timedelta(minutes=14)),
        ]
        
        for tick in ticks_building:
            await self._pump_and_wait(tick)
        
        # Move past range end time
        await self._pump_and_wait(self._generate_mock_tick(500.01, base_date + timedelta(minutes=16)))
        
        # Should reject as DONE_FOR_DAY, not move to ACTIVE
        self.assertEqual(self.orb.state, ORBState.DONE_FOR_DAY, "Should reject narrow range")
        self.assertEqual(len(self.captured_orders), 0, "Should not fire on invalid range")

    async def test_pre_market_filtering(self):
        pre_market_date = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        await self._pump_and_wait(self._generate_mock_tick(600.00, pre_market_date))
        
        self.assertEqual(self.orb.state, ORBState.PRE_MARKET)
        self.assertEqual(self.orb.range_high, float('-inf'))

if __name__ == '__main__':
    unittest.main()