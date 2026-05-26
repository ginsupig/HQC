"""
D2 — EventBus handler exception isolation.

Covers the live-safety guarantee that PairsRiskMonitor depends on:
when ONE handler raises (sync or async), the bus survives AND every
OTHER handler subscribed to the same event continues to receive
that event and all subsequent events.

Failure mode this defends against: a single buggy handler silently
starving the kill switch. Without isolation, a sync exception inside
one callback would break the worker's for-loop and skip all
following subscribers for the same event.
"""
from __future__ import annotations

import asyncio
from typing import List

import pytest

from core.engine.event_bus import Event, EventBus, EventType


def _run(coro):
    """Test helper: run an async test on a fresh event loop."""
    return asyncio.run(coro)


@pytest.mark.asyncio
async def test_sync_raising_handler_does_not_block_peer():
    """If handler A raises synchronously, handler B subscribed to the
    SAME event still receives that event and subsequent events."""
    bus = EventBus()
    await bus.start()
    try:
        received: List[Event] = []

        def bad_handler(event):  # sync (non-coroutine) handler that raises
            raise RuntimeError("intentional sync failure")

        async def good_handler(event):
            received.append(event)

        # Order matters: bad subscribed first so the for-loop would hit
        # it before good without the isolation patch.
        bus.subscribe(EventType.TICK, bad_handler)
        bus.subscribe(EventType.TICK, good_handler)

        bus.publish(Event(type=EventType.TICK, payload={"n": 1}))
        bus.publish(Event(type=EventType.TICK, payload={"n": 2}))
        # Let the worker drain + tasks complete.
        await asyncio.sleep(0.1)
        await bus._queue.join()
        await asyncio.sleep(0.05)

        payloads = [e.payload.get("n") for e in received]
        assert payloads == [1, 2], (
            f"good_handler should have received both events despite bad_handler raising; got {payloads}"
        )
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_async_raising_handler_does_not_block_peer():
    """If handler A's coroutine raises mid-body, handler B still
    receives that event and subsequent events. The bus's
    _done_callback should log the failure without disrupting peers."""
    bus = EventBus()
    await bus.start()
    try:
        received: List[Event] = []

        async def async_raiser(event):
            await asyncio.sleep(0)  # yield once
            raise ValueError("intentional async failure")

        async def good_handler(event):
            received.append(event)

        bus.subscribe(EventType.TICK, async_raiser)
        bus.subscribe(EventType.TICK, good_handler)

        bus.publish(Event(type=EventType.TICK, payload={"n": 1}))
        bus.publish(Event(type=EventType.TICK, payload={"n": 2}))
        await asyncio.sleep(0.1)
        await bus._queue.join()
        await asyncio.sleep(0.05)

        payloads = [e.payload.get("n") for e in received]
        assert payloads == [1, 2], (
            f"good_handler should still receive after async raiser; got {payloads}"
        )
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_bus_survives_many_consecutive_handler_failures():
    """Stress: a chronically broken handler must not degrade the bus
    over many events. The worker must keep processing and peers must
    keep receiving."""
    bus = EventBus()
    await bus.start()
    try:
        received: List[int] = []

        def always_bad(event):
            raise RuntimeError("every time")

        async def counter(event):
            received.append(event.payload["n"])

        bus.subscribe(EventType.TICK, always_bad)
        bus.subscribe(EventType.TICK, counter)

        for n in range(50):
            bus.publish(Event(type=EventType.TICK, payload={"n": n}))
        await asyncio.sleep(0.2)
        await bus._queue.join()
        await asyncio.sleep(0.05)

        assert received == list(range(50)), (
            f"counter should record all 50 events despite always_bad raising on every one; "
            f"got len={len(received)} first/last={received[:3]}...{received[-3:] if received else []}"
        )
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_independent_event_types_unaffected():
    """A handler failing on TICK must not block handlers for
    unrelated event types like ORDER_FILL."""
    bus = EventBus()
    await bus.start()
    try:
        order_received: List[Event] = []

        def tick_raiser(event):
            raise RuntimeError("tick handler is broken")

        async def order_handler(event):
            order_received.append(event)

        bus.subscribe(EventType.TICK, tick_raiser)
        bus.subscribe(EventType.ORDER_FILL, order_handler)

        bus.publish(Event(type=EventType.TICK, payload={"n": 1}))
        bus.publish(Event(type=EventType.ORDER_FILL, payload={"id": "x1"}))
        bus.publish(Event(type=EventType.TICK, payload={"n": 2}))
        bus.publish(Event(type=EventType.ORDER_FILL, payload={"id": "x2"}))
        await asyncio.sleep(0.1)
        await bus._queue.join()
        await asyncio.sleep(0.05)

        ids = [e.payload.get("id") for e in order_received]
        assert ids == ["x1", "x2"], f"order_handler should still receive ORDER_FILLs; got {ids}"
    finally:
        await bus.stop()
