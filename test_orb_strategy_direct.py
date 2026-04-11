"""
Direct test of ORB strategy WITHOUT async event bus complexity.
Tests the core strategy logic synchronously.
"""
import asyncio
from datetime import datetime, time, timezone, timedelta
from core.engine.event_bus import EventBus, Event, EventType
from strategies.orb.equity_orb import USEquityORB, ORBState
import pytz

async def direct_orb_test():
    """Test ORB strategy with controlled tick injection."""
    
    print("\n" + "="*70)
    print("DIRECT ORB STRATEGY TEST")
    print("="*70 + "\n")
    
    bus = EventBus()
    await bus.start()
    
    # Create ORB strategy for SPY
    orb = USEquityORB(
        target_asset="SPY",
        bus=bus,
        range_minutes=15,
        max_trades=2,
        min_range_pct=0.0025,  # 25 bps minimum range
        breakout_buffer_pct=0.0005,  # 5 bps buffer
    )
    
    # Capture orders
    orders_fired = []
    async def capture_order(event: Event):
        orders_fired.append(event.payload)
    
    bus.subscribe(EventType.ORDER_CREATE, capture_order)
    
    # Setup market time (9:30 AM EDT)
    tz = pytz.timezone("US/Eastern")
    today = datetime.now(tz).date()
    market_open_est = tz.localize(datetime.combine(today, time(9, 30, 0)))
    market_open_utc = market_open_est.astimezone(pytz.utc)
    
    print(f"Starting test at: {market_open_est.strftime('%I:%M %p %Z')}")
    print(f"Strategy State: {orb.state.name}")
    print("\n" + "-"*70)
    print("PHASE 1: BUILDING RANGE (9:30 - 9:45)")
    print("-"*70 + "\n")
    
        # Simulate 15 minutes of range building
    # Range: 500 - 503 (300 bps = 0.6% range, well above 25 bps minimum)
    base_price = 500.00
    range_low = 500.00
    range_high = 503.00  # Changed from 502.00 → 503.00 for wider range
    
    # Generate ticks every 10 seconds for 15 minutes = 90 ticks
    for minute in range(0, 15):
        for second in [0, 10, 20, 30, 40, 50]:
            elapsed_sec = minute * 60 + second
            
            # Create realistic price oscillations within range
            # Use a more volatile pattern to ensure we hit both range extremes
            oscillation = ((elapsed_sec % 120) / 120.0)  # 0 to 1 cycle over 2 minutes
            if oscillation < 0.5:
                # First half: move from low to high
                tick_price = range_low + (range_high - range_low) * (oscillation * 2)
            else:
                # Second half: move from high to low
                tick_price = range_high - (range_high - range_low) * ((oscillation - 0.5) * 2)
            
            tick_price = round(tick_price, 2)
            
            # Calculate timestamp
            current_time = market_open_utc + timedelta(seconds=elapsed_sec)
            ts_ms = int(current_time.timestamp() * 1000)
            
            # Publish tick
            tick = Event(
                type=EventType.TICK,
                payload={
                    "ticker": "SPY",
                    "symbol": "SPY",
                    "price": tick_price,
                    "volume": 500.0,
                    "timestamp": ts_ms,
                }
            )
            bus.publish(tick)
            
            # Check state periodically
            if second == 0:
                range_pct = 0
                if orb.range_high > orb.range_low:
                    range_pct = ((orb.range_high - orb.range_low) / ((orb.range_high + orb.range_low) / 2)) * 100
                
                print(f"  [{minute:2d}m] Price ${tick_price:.2f} | "
                      f"Range: ${orb.range_low:.2f}-${orb.range_high:.2f} ({range_pct:.2f}%) | "
                      f"State: {orb.state.name}")
            
            await asyncio.sleep(0.01)  # Small delay for event processing
    
    await asyncio.sleep(0.1)  # Allow final event processing
    
    print(f"\nAfter range building:")
    print(f"  Range High: ${orb.range_high:.2f}")
    print(f"  Range Low: ${orb.range_low:.2f}")
    print(f"  Range Size: {((orb.range_high - orb.range_low) / ((orb.range_high + orb.range_low) / 2)) * 100:.2f}%")
    print(f"  State: {orb.state.name}")
    print(f"  Orders Fired: {len(orders_fired)}")
    
    # Now trigger breakout
    print("\n" + "-"*70)
    print("PHASE 2: BREAKOUT (9:46 - 10:00)")
    print("-"*70 + "\n")
    
    # Calculate breakout level
    breakout_level = orb.range_high * (1 + orb.breakout_buffer_pct)
    print(f"Breakout trigger level: ${breakout_level:.4f}\n")
    
    # Send breakout ticks
    for second in range(900, 960, 10):  # 9:45 - 10:00
        tick_price = breakout_level + (0.50 * ((second - 900) / 60))  # Gradual rise
        tick_price = round(tick_price, 2)
        
        current_time = market_open_utc + timedelta(seconds=second)
        ts_ms = int(current_time.timestamp() * 1000)
        
        tick = Event(
            type=EventType.TICK,
            payload={
                "ticker": "SPY",
                "symbol": "SPY",
                "price": tick_price,
                "volume": 500.0,
                "timestamp": ts_ms,
            }
        )
        bus.publish(tick)
        
        minute = (second // 60) % 60
        sec = second % 60
        print(f"  [{minute:2d}:{sec:02d}] Price ${tick_price:.2f} | State: {orb.state.name} | Orders: {len(orders_fired)}")
        
        await asyncio.sleep(0.05)
    
    await asyncio.sleep(0.2)  # Final processing
    
    # Results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Final State: {orb.state.name}")
    print(f"Orders Fired: {len(orders_fired)}")
    
    if orders_fired:
        print("\nOrders Details:")
        for i, order in enumerate(orders_fired, 1):
            print(f"  [{i}] {order.get('action')} {order.get('asset')} @ ${order.get('reference_price', 'N/A')}")
    else:
        print("\n❌ NO ORDERS FIRED!")
        print("\nDiagnostics:")
        print(f"  - Range formed: {orb.range_high > orb.range_low}")
        print(f"  - Range is valid: {orb._range_is_valid() if hasattr(orb, '_range_is_valid') else 'N/A'}")
        print(f"  - State progression: PRE_MARKET → BUILDING_RANGE → ACTIVE → DONE/triggered")
        print(f"  - Current state stuck at: {orb.state.name}")
    
    print("="*70 + "\n")
    
    await bus.stop()
    return len(orders_fired) > 0

if __name__ == "__main__":
    result = asyncio.run(direct_orb_test())
    exit(0 if result else 1)