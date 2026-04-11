"""
Test HQC with mock market data instead of Tradier.
"""
import asyncio
import random
from datetime import datetime, timedelta, timezone
from core.engine.event_bus import EventBus, Event, EventType
from core.engine.state_machine import GlobalStateMachine
from core.execution.broker_router import AlpacaExecutionRouter
from intelligence.candidate_ranker import CandidateRanker
from risk.position_sizing.confidence_scaler import DynamicRiskSizer

async def mock_market_feed(bus: EventBus, symbols: list, duration_sec: int = 60):
    """Generate realistic mock ticks for testing."""
    prices = {s: random.uniform(100, 500) for s in symbols}
    start_time = datetime.now(timezone.utc)
    
    while (datetime.now(timezone.utc) - start_time).total_seconds() < duration_sec:
        for symbol in symbols:
            # Random walk price movement
            prices[symbol] *= (1 + random.uniform(-0.001, 0.001))
            
            tick = Event(
                type=EventType.TICK,
                payload={
                    "ticker": symbol,
                    "price": round(prices[symbol], 2),
                    "volume": random.uniform(1000, 5000),
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }
            )
            bus.publish(tick)
        
        await asyncio.sleep(0.1)
    
    print("✅ Mock feed completed")

async def test_hqc_with_mock():
    print("Starting HQC with Mock Market Feed...")
    
    bus = EventBus()
    await bus.start()
    
    state_machine = GlobalStateMachine(bus)
    ranker = CandidateRanker(bus, min_score=0.0)  # Accept all signals
    sizer = DynamicRiskSizer(bus, account_equity=100000.0)
    
    # Start mock feed for 60 seconds
    feed_task = asyncio.create_task(mock_market_feed(bus, ["SPY", "QQQ", "TSLA"], duration_sec=60))
    
    # Monitor state transitions
    transition_count = 0
    async def monitor_states(event: Event):
        nonlocal transition_count
        transition_count += 1
    
    bus.subscribe(EventType.ORDER_CREATE, monitor_states)
    
    try:
        await asyncio.sleep(2)  # Let system run
        await feed_task
        
        print(f"✅ System ran successfully for 60 seconds")
        print(f"✅ Generated {transition_count} ORDER_CREATE events")
        
    finally:
        await bus.stop()

if __name__ == "__main__":
    asyncio.run(test_hqc_with_mock())