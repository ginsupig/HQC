"""
Enhanced HQC test with REALISTIC market open timing.
"""
import asyncio
import random
from datetime import datetime, time, timezone, timedelta
from core.engine.event_bus import EventBus, Event, EventType
from core.engine.state_machine import GlobalStateMachine
from strategies.orb.equity_orb import USEquityORB
from strategies.vwap.hunter_state_machine import USEquityVWAPHunter
from intelligence.candidate_ranker import CandidateRanker
from risk.position_sizing.confidence_scaler import DynamicRiskSizer
import pytz

async def mock_market_feed_realistic(bus: EventBus, symbols: list, duration_sec: int = 120):
    """
    Generate realistic mock ticks with:
    - Timestamps during US market hours (9:30 AM - 4:00 PM EST)
    - Proper timezone handling
    - Market structure: range building → breakout
    """
    prices = {s: random.uniform(300, 500) for s in symbols}
    
    # Start at 9:30 AM EST (market open)
    tz = pytz.timezone("US/Eastern")
    today = datetime.now(tz).date()
    market_open_est = tz.localize(datetime.combine(today, time(9, 30, 0)))
    
    # Convert to UTC for timestamp calculation
    market_open_utc = market_open_est.astimezone(pytz.utc)
    
    start_time = market_open_utc
    tick_count = 0
    
    print("\n" + "="*70)
    print("MOCK FEED STARTING - SIMULATING REALISTIC MARKET OPEN")
    print("="*70)
    print(f"Market Open Time: {market_open_est.strftime('%I:%M %p %Z')}")
    print(f"UTC Equivalent: {market_open_utc.strftime('%I:%M %p %Z')}")
    print("="*70 + "\n")
    
    elapsed = 0.0
    
    while elapsed < duration_sec:
        elapsed_min = elapsed / 60.0
        
        # Calculate current market time
        current_utc = start_time + timedelta(seconds=elapsed)
        current_est = current_utc.astimezone(tz)
        
        for symbol in symbols:
            # Simulate market structure:
            # 0-15 min: tight range building (small moves)
            # 15-20 min: potential breakout (directional)
            # 20+ min: trend continuation
            
            if elapsed_min < 15:
                # Range building phase - tight consolidation (5-50 bps)
                move = random.uniform(-0.0005, 0.0005)
            elif elapsed_min < 20:
                # Breakout phase - directional move (10-200 bps)
                move = random.uniform(-0.001, 0.002)
            else:
                # Trend continuation (5-100 bps)
                move = random.uniform(-0.0005, 0.001)
            
            prices[symbol] *= (1 + move)
            tick_count += 1
            
            # Use UTC timestamp in milliseconds (as required by EventBus)
            tick_timestamp_ms = int(current_utc.timestamp() * 1000)
            
            tick = Event(
                type=EventType.TICK,
                payload={
                    "ticker": symbol,
                    "symbol": symbol,
                    "price": round(prices[symbol], 2),
                    "volume": random.uniform(500, 2500),
                    "timestamp": tick_timestamp_ms,  # UTC ms
                }
            )
            bus.publish(tick)
        
        # Log progress every 30 seconds of simulated time
        if int(elapsed) % 30 == 0 and int(elapsed) > 0:
            print(f"[{elapsed_min:.1f}m | {current_est.strftime('%H:%M:%S %Z')}] "
                  f"Published {tick_count} ticks | "
                  f"Prices: {', '.join([f'{s}=${prices[s]:.2f}' for s in symbols[:2]])}")
        
        # Advance time by small increment (faster simulation)
        elapsed += 0.5  # 0.5 second increments
        await asyncio.sleep(0.01)  # Small sleep to allow event processing
    
    print(f"\n✅ Mock feed completed ({tick_count} total ticks in {duration_sec}s)")

async def test_hqc_with_debug():
    """Test with detailed event logging."""
    print("\nStarting HQC with REALISTIC Mock Market Feed (DEBUG MODE)...\n")
    
    bus = EventBus()
    await bus.start()
    
    state_machine = GlobalStateMachine(bus)
    
    # Initialize strategies
    orb_spy = USEquityORB(target_asset="SPY", bus=bus, range_minutes=15, max_trades=2)
    orb_qqq = USEquityORB(target_asset="QQQ", bus=bus, range_minutes=15, max_trades=2)
    orb_tsla = USEquityORB(target_asset="TSLA", bus=bus, range_minutes=15, max_trades=2)
    
    vwap_spy = USEquityVWAPHunter(target_asset="SPY", bus=bus, min_volume_shares=1000.0)
    vwap_qqq = USEquityVWAPHunter(target_asset="QQQ", bus=bus, min_volume_shares=1000.0)
    vwap_tsla = USEquityVWAPHunter(target_asset="TSLA", bus=bus, min_volume_shares=1000.0)
    
    # Initialize processing pipeline
    ranker = CandidateRanker(bus, min_score=2.0)  # Very permissive for testing
    sizer = DynamicRiskSizer(bus, account_equity=100000.0)
    
    # Counters
    tick_count = 0
    raw_order_count = 0
    ranked_order_count = 0
    sized_order_count = 0
    
    # Event handlers
    async def on_tick(event: Event):
        nonlocal tick_count
        tick_count += 1
    
    async def on_raw_order(event: Event):
        """Capture raw ORDER_CREATE events from strategies."""
        nonlocal raw_order_count
        payload = event.payload or {}
        
        # Only count strategy intents (no stage or stage="RANKED" means pre-ranker)
        if payload.get("stage") not in {"RANKED", "SIZED", "ROUTED"}:
            raw_order_count += 1
            asset = payload.get("asset")
            action = payload.get("action")
            strategy = payload.get("strategy", "Unknown")
            price = payload.get("reference_price", "N/A")
            print(f"  [RAW ORDER] {strategy:20s} {asset} {action:12s} @ ${price}")
    
    async def on_ranked_order(event: Event):
        """Capture ranked ORDER_CREATE events."""
        nonlocal ranked_order_count
        payload = event.payload or {}
        
        if payload.get("stage") == "RANKED":
            ranked_order_count += 1
            approved = payload.get("approved_by_ranker", False)
            score = payload.get("rank_score", 0.0)
            asset = payload.get("asset")
            status = "✅ APPROVED" if approved else "❌ REJECTED"
            print(f"  [RANKED] {asset:6s} Score={score:6.2f} {status}")
    
    async def on_sized_order(event: Event):
        """Capture sized ORDER_CREATE events."""
        nonlocal sized_order_count
        payload = event.payload or {}
        
        if payload.get("stage") == "SIZED" and payload.get("shares"):
            sized_order_count += 1
            asset = payload.get("asset")
            shares = payload.get("shares")
            risk = payload.get("risk_dollars", 0.0)
            print(f"  [SIZED] {asset:6s} x{shares:4d} shares | Risk=${risk:8.2f}")
    
    # Subscribe to all events
    bus.subscribe(EventType.TICK, on_tick)
    bus.subscribe(EventType.ORDER_CREATE, on_raw_order)
    bus.subscribe(EventType.ORDER_CREATE, on_ranked_order)
    bus.subscribe(EventType.ORDER_CREATE, on_sized_order)
    
    try:
        # Run mock feed for 120 seconds (covers 9:30 AM - 11:32 AM EST)
        # This gives enough time for ORB range formation (15m) + breakout (5m)
        feed_task = asyncio.create_task(
            mock_market_feed_realistic(bus, ["SPY", "QQQ", "TSLA"], duration_sec=120)
        )
        await feed_task
        
        # Allow final event processing
        await asyncio.sleep(1.0)
        
        # Final summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Ticks Published:        {tick_count:6d}")
        print(f"Raw Orders (Strategies): {raw_order_count:6d}")
        print(f"Ranked Orders:          {ranked_order_count:6d}")
        print(f"Sized Orders:           {sized_order_count:6d}")
        print("="*70)
        
        # Diagnosis
        if tick_count == 0:
            print("❌ ERROR: No ticks were published!")
        elif raw_order_count == 0:
            print("❌ ERROR: Strategies did not fire any orders!")
            print("   → Possible causes:")
            print("     1. Timestamps not in RTH (9:30 AM - 4:00 PM EST)")
            print("     2. ORB range not formed (need 15m minimum)")
            print("     3. Price movement too small to trigger breakout")
            print("     4. Strategy thresholds too strict")
        elif ranked_order_count == 0:
            print("❌ ERROR: Ranker rejected all orders!")
            print("   → Check: min_score threshold, VWAP alignment, liquidity")
        elif sized_order_count == 0:
            print("❌ ERROR: Sizer rejected all ranked orders!")
            print("   → Check: capital available, risk per trade, position limits")
        else:
            print(f"✅ SUCCESS: Full pipeline executed!")
            print(f"   {sized_order_count} ready-to-execute orders generated")
        
        return sized_order_count > 0
        
    finally:
        await bus.stop()

if __name__ == "__main__":
    result = asyncio.run(test_hqc_with_debug())
    exit(0 if result else 1)