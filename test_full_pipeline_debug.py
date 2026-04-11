"""
Test with detailed ranker debugging to see why orders are rejected.
"""
import asyncio
from datetime import datetime, time, timezone, timedelta
from core.engine.event_bus import EventBus, Event, EventType
from strategies.orb.equity_orb import USEquityORB
from intelligence.candidate_ranker import CandidateRanker
from risk.position_sizing.confidence_scaler import DynamicRiskSizer
import pytz

async def full_pipeline_debug_test():
    """Test with ranker debugging."""
    
    print("\n" + "="*70)
    print("FULL HQC PIPELINE TEST - RANKER DEBUG")
    print("="*70 + "\n")
    
    bus = EventBus()
    await bus.start()
    
    # Initialize pipeline
    orb = USEquityORB(
        target_asset="SPY",
        bus=bus,
        range_minutes=15,
        max_trades=2,
    )
    
    ranker = CandidateRanker(
        bus=bus,
        benchmark="SPY",
        min_score=0.0,  # Accept everything
        max_spread_bps=100.0,  # Very permissive
        max_dist_vwap_pct=0.05,  # Very permissive
    )
    
    sizer = DynamicRiskSizer(
        bus=bus,
        account_equity=100000.0,
        base_risk_pct=0.01,
    )
    
    # Counters
    raw_count = 0
    ranked_count = 0
    sized_count = 0
    
    async def capture_orders(event: Event):
        nonlocal raw_count, ranked_count, sized_count
        payload = event.payload or {}
        stage = payload.get("stage", "RAW")
        
        if stage not in {"RANKED", "SIZED", "ROUTED"}:
            raw_count += 1
            print(f"\n[1️⃣  RAW ORDER]")
            print(f"    Asset: {payload.get('asset')}")
            print(f"    Action: {payload.get('action')}")
            print(f"    Price: ${payload.get('reference_price', 'N/A')}")
            print(f"    Strategy: {payload.get('strategy', 'Unknown')}")
            print(f"    Stage: {payload.get('stage', 'NONE')}")
        
        elif stage == "RANKED":
            ranked_count += 1
            approved = payload.get("approved_by_ranker", False)
            score = payload.get("rank_score", 0.0)
            components = payload.get("rank_components", {})
            
            print(f"\n[2️⃣  RANKED ORDER]")
            print(f"    Approved: {approved}")
            print(f"    Score: {score:.4f}")
            print(f"    Components:")
            print(f"      RS: {components.get('rs', 0):.6f}")
            print(f"      RVOL: {components.get('rvol', 0):.4f}")
            print(f"      Spread BPS: {components.get('spread_bps', 0):.4f}")
            print(f"      Dist VWAP %: {components.get('dist_vwap_pct', 0):.6f}")
            print(f"      Liquidity Score: {components.get('liquidity_score', 0):.4f}")
            
            if not approved:
                reasons = components.get("reasons", [])
                print(f"    ❌ Rejection Reasons: {reasons}")
        
        elif stage == "SIZED" and payload.get("shares"):
            sized_count += 1
            print(f"\n[3️⃣  SIZED ORDER]")
            print(f"    Shares: {payload.get('shares')}")
            print(f"    Risk: ${payload.get('risk_dollars', 'N/A'):.2f}")
    
    bus.subscribe(EventType.ORDER_CREATE, capture_orders)
    
    # Setup market time
    tz = pytz.timezone("US/Eastern")
    today = datetime.now(tz).date()
    market_open_est = tz.localize(datetime.combine(today, time(9, 30, 0)))
    market_open_utc = market_open_est.astimezone(pytz.utc)
    
    print(f"Market Open: {market_open_est.strftime('%I:%M %p %Z')}\n")
    
    # Generate ticks
    print("Generating ticks...")
    for minute in range(0, 15):
        for second in [0, 10, 20, 30, 40, 50]:
            elapsed_sec = minute * 60 + second
            oscillation = ((elapsed_sec % 120) / 120.0)
            range_low, range_high = 500.0, 503.0
            
            if oscillation < 0.5:
                price = range_low + (range_high - range_low) * (oscillation * 2)
            else:
                price = range_high - (range_high - range_low) * ((oscillation - 0.5) * 2)
            
            price = round(price, 2)
            ts_ms = int((market_open_utc + timedelta(seconds=elapsed_sec)).timestamp() * 1000)
            
            tick = Event(
                type=EventType.TICK,
                payload={
                    "ticker": "SPY",
                    "price": price,
                    "volume": 1000.0,
                    "timestamp": ts_ms,
                }
            )
            bus.publish(tick)
            await asyncio.sleep(0.01)
    
    print("✓ Range formation complete\n")
    print("Generating breakout...")
    
    for second in range(900, 930, 10):
        price = 503.0 + (0.5 * ((second - 900) / 30))
        price = round(price, 2)
        ts_ms = int((market_open_utc + timedelta(seconds=second)).timestamp() * 1000)
        
        tick = Event(
            type=EventType.TICK,
            payload={
                "ticker": "SPY",
                "price": price,
                "volume": 1500.0,
                "timestamp": ts_ms,
            }
        )
        bus.publish(tick)
        await asyncio.sleep(0.05)
    
    await asyncio.sleep(0.5)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nRaw Orders:    {raw_count}")
    print(f"Ranked Orders: {ranked_count}")
    print(f"Sized Orders:  {sized_count}")
    
    if raw_count > 0 and ranked_count == 0:
        print("\n⚠️  ISSUE: Raw orders created but ranker rejected them!")
        print("\nPossible causes:")
        print("  1. Ranker's on_order_create not being called")
        print("  2. Ranker filtering on 'stage' attribute")
        print("  3. Ranker filtering on 'shares' attribute")
    elif sized_count == 0 and ranked_count > 0:
        print("\n⚠️  ISSUE: Ranker approved but sizer rejected!")
    
    print("\n" + "="*70 + "\n")
    
    await bus.stop()
    return sized_count > 0

if __name__ == "__main__":
    result = asyncio.run(full_pipeline_debug_test())
    exit(0 if result else 1)