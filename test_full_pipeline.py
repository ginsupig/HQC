"""
Test the complete HQC pipeline: Strategy → Ranker → Sizer → Ready for Execution
"""
import asyncio
from datetime import datetime, time, timezone, timedelta
from core.engine.event_bus import EventBus, Event, EventType
from strategies.orb.equity_orb import USEquityORB
from intelligence.candidate_ranker import CandidateRanker
from risk.position_sizing.confidence_scaler import DynamicRiskSizer
import pytz

async def full_pipeline_test():
    """Test strategy → ranker → sizer pipeline."""
    
    print("\n" + "="*70)
    print("FULL HQC PIPELINE TEST")
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
        min_score=2.0,  # Very permissive for testing
    )
    
    sizer = DynamicRiskSizer(
        bus=bus,
        account_equity=100000.0,
        base_risk_pct=0.01,
    )
    
    # Capture all order stages
    raw_orders = []
    ranked_orders = []
    sized_orders = []
    
    async def capture_orders(event: Event):
        payload = event.payload or {}
        stage = payload.get("stage", "RAW")
        
        if stage not in {"RANKED", "SIZED", "ROUTED"}:
            raw_orders.append(payload)
            print(f"\n[RAW ORDER] {payload.get('asset')} {payload.get('action')}")
            print(f"  Price: ${payload.get('reference_price', 'N/A')}")
            print(f"  Strategy: {payload.get('strategy', 'Unknown')}")
        
        elif stage == "RANKED":
            ranked_orders.append(payload)
            approved = payload.get("approved_by_ranker", False)
            score = payload.get("rank_score", 0.0)
            print(f"\n[RANKED] Approved={approved}, Score={score:.2f}")
            if not approved:
                reasons = payload.get("rank_components", {}).get("reasons", [])
                print(f"  Rejection reasons: {reasons}")
        
        elif stage == "SIZED" and payload.get("shares"):
            sized_orders.append(payload)
            shares = payload.get("shares")
            risk = payload.get("risk_dollars", 0.0)
            mult = payload.get("effective_risk_mult", 0.0)
            print(f"\n[SIZED] Ready for execution!")
            print(f"  Shares: {shares}")
            print(f"  Risk: ${risk:.2f}")
            print(f"  Risk Multiplier: {mult:.4f}")
    
    bus.subscribe(EventType.ORDER_CREATE, capture_orders)
    
    # Setup market time
    tz = pytz.timezone("US/Eastern")
    today = datetime.now(tz).date()
    market_open_est = tz.localize(datetime.combine(today, time(9, 30, 0)))
    market_open_utc = market_open_est.astimezone(pytz.utc)
    
    print(f"Market Open: {market_open_est.strftime('%I:%M %p %Z')}\n")
    print("-"*70)
    print("PHASE 1: RANGE FORMATION (9:30 - 9:45)")
    print("-"*70)
    
    # Generate range-building ticks
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
    
    print("✓ Range formation complete")
    await asyncio.sleep(0.1)
    
    print("\n" + "-"*70)
    print("PHASE 2: BREAKOUT TRIGGER (9:46 - 9:50)")
    print("-"*70)
    
    # Send breakout ticks
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
    print("PIPELINE RESULTS")
    print("="*70)
    print(f"\nRaw Orders (from strategies):    {len(raw_orders)}")
    print(f"Ranked Orders (approved):       {len(ranked_orders)}")
    print(f"Sized Orders (ready to trade):  {len(sized_orders)}")
    
    if sized_orders:
        print("\n✅ PIPELINE SUCCESS!")
        print(f"\n{len(sized_orders)} order(s) ready for execution on Alpaca:\n")
        for i, order in enumerate(sized_orders, 1):
            print(f"[{i}] {order.get('asset')} {order.get('action')}")
            print(f"    Entry: ${order.get('entry_price', 'N/A')}")
            print(f"    Stop:  ${order.get('stop_loss_price', 'N/A')}")
            print(f"    Size:  {order.get('shares')} shares")
            print(f"    Risk:  ${order.get('risk_dollars', 'N/A')}")
    else:
        print("\n❌ Pipeline failed!")
        print(f"Raw: {len(raw_orders)}, Ranked: {len(ranked_orders)}, Sized: {len(sized_orders)}")
    
    print("\n" + "="*70 + "\n")
    
    await bus.stop()
    return len(sized_orders) > 0

if __name__ == "__main__":
    result = asyncio.run(full_pipeline_test())
    exit(0 if result else 1)