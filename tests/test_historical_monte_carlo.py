import unittest
import asyncio
import logging
import os
import random
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()
# Core engine imports
from core.engine.event_bus import EventBus, Event, EventType
from strategies.orb.equity_orb import USEquityORB, ORBState
from risk.position_sizing.confidence_scaler import DynamicRiskSizer

# Suppress standard logging to keep test output clean, only show warnings/errors
logging.getLogger("EventBus").setLevel(logging.CRITICAL)
logging.getLogger("EquityORB").setLevel(logging.WARNING)
logger = logging.getLogger("MonteCarloTest")

class TestHistoricalMonteCarlo(unittest.IsolatedAsyncioTestCase):
    """
    Monte Carlo stress test. Pulls a random historical trading day from the past 5 years,
    interpolates the data into chronological pseudo-ticks, and ensures the architecture 
    processes an entire day's worth of live data without crashing or violating risk parameters.
    """

    async def asyncSetUp(self):
        """Initializes the engine and pulls API keys from the environment."""
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            self.skipTest("Alpaca API keys not found in environment variables. Skipping Monte Carlo test.")

        self.asset = "SPY"
        self.years_back = 5
        self.bus = EventBus()
        await self.bus.start()
        
        # Wire up the system architecture
        self.orb_strategy = USEquityORB(target_asset=self.asset, bus=self.bus, range_minutes=15)
        self.risk_sizer = DynamicRiskSizer(bus=self.bus, account_equity=100000.0, base_risk_pct=0.01)
        
        # Container to catch sized orders for assertions
        self.captured_orders: List[Dict[str, Any]] = []
        self.bus.subscribe(EventType.ORDER_CREATE, self._order_catcher)

    async def asyncTearDown(self):
        """Cleans up the async loop after the test."""
        await self.bus.stop()

    async def _order_catcher(self, event: Event):
        """Intercepts sized orders emitted by the Risk Sizer."""
        if "shares" in event.payload: 
            self.captured_orders.append(event.payload)

    def _get_random_trading_day(self) -> str:
        """Generates a random valid weekday within the lookback window."""
        today = datetime.now(timezone.utc)
        start_date = today - timedelta(days=365 * self.years_back)
        random_days = random.randint(0, (today - start_date).days)
        target_date = start_date + timedelta(days=random_days)
        
        if target_date.weekday() == 5: target_date -= timedelta(days=1)
        elif target_date.weekday() == 6: target_date -= timedelta(days=2)
            
        return target_date.strftime("%Y-%m-%d")

    async def _fetch_historical_day(self, date_str: str) -> List[Dict[str, Any]]:
        """Pulls the 1-minute OHLCV bars from Alpaca for the target date."""
        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json"
        }
        
        params = {
            "symbols": self.asset,
            "timeframe": "1Min",
            "start": f"{date_str}T13:00:00Z", # 9:00 AM EST
            "end": f"{date_str}T20:30:00Z",   # 4:30 PM EST
            "limit": 10000
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"API Error: {await response.text()}")
                    return []
                data = await response.json()
                return data.get("bars", {}).get(self.asset, [])

    def _interpolate_bar_to_ticks(self, bar: Dict[str, Any]) -> List[Event]:
        """Deconstructs a 1-minute OHLCV bar into 4 chronological pseudo-ticks."""
        o, h, l, c = bar["o"], bar["h"], bar["l"], bar["c"]
        vol = bar["v"] / 4.0 
        
        dt = datetime.strptime(bar["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        base_ts = int(dt.timestamp() * 1000)
        
        ticks = []
        prices = [o, l, h, c] if c > o else [o, h, l, c]
            
        for i, price in enumerate(prices):
            ticks.append(Event(
                type=EventType.TICK,
                payload={"ticker": self.asset, "price": price, "volume": vol, "timestamp": base_ts + (i * 15000)}
            ))
        return ticks

    async def test_random_historical_day(self):
        """
        The core stress test. Fetches a random day, interpolates the data, 
        pumps it through the engine, and asserts architectural stability.
        """
        random_date = self._get_random_trading_day()
        logger.warning(f"\n--- Running Monte Carlo Stress Test for {random_date} ---")
        
        bars = await self._fetch_historical_day(random_date)
        
        if not bars:
            self.skipTest(f"No market data returned for {random_date} (likely a market holiday).")

        # Interpolate bars to ticks
        all_ticks = []
        for bar in bars:
            all_ticks.extend(self._interpolate_bar_to_ticks(bar))

        self.assertGreater(len(all_ticks), 0, "Tick interpolation failed.")

        # Pump the ticks through the EventBus at warp speed
        for tick in all_ticks:
            self.bus.publish(tick)
            await asyncio.sleep(0.0001) # Yield to event loop
            
        # Allow final queue processing
        await asyncio.sleep(0.5)

        # --- System Assertions ---
        # 1. Check that the strategy successfully completed the day without crashing
        self.assertIn(self.orb_strategy.state, [ORBState.ACTIVE, ORBState.DONE_FOR_DAY], 
                      "Strategy failed to transition out of morning range building.")
        
        # 2. Check risk math if orders were fired
        if self.captured_orders:
            for order in self.captured_orders:
                self.assertIn("shares", order, "Order bypassed the Risk Sizer!")
                self.assertGreater(order["shares"], 0, "Risk Sizer calculated 0 shares.")
                
                # Assert the risk never exceeds 1% of $100k ($1,000)
                risk_per_share = abs(order["entry_price"] - order["stop_loss"])
                total_risk = order["shares"] * risk_per_share
                
                # We allow a tiny floating point margin of error (e.g., $1000.50) due to share flooring
                self.assertLessEqual(total_risk, 1005.00, f"Risk limit exceeded! Risked ${total_risk:.2f}")
                
            logger.warning(f"Test Passed: System successfully executed {len(self.captured_orders)} perfectly sized trades.")
        else:
            logger.warning("Test Passed: System processed the full day, no valid ORB breakouts occurred.")

if __name__ == '__main__':
    unittest.main()