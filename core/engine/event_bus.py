from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional
import time

logger = logging.getLogger("EventBus")


class EventType(Enum):
    TICK = auto()
    ORDER_CREATE = auto()
    ORDER_FILL = auto()
    REGIME_CHANGE = auto()
    MODEL_UPDATE = auto()
    RISK_ALERT = auto()
    SYSTEM_HEALTH = auto()
    EQUITY_UPDATE = auto()

    # --- PATCH: missing in state_machine.py references ---
    SYSTEM_SHUTDOWN = auto()

    # --- PATCH: new market microstructure channels ---
    BOOK_SNAPSHOT = auto()
    MICROSTRUCTURE_ALERT = auto()

    # Bar-cadence ticks emitted by TickResampler. Strategies validated on
    # 1-minute bar data (e.g. the Kalman pairs trader) subscribe to this
    # instead of raw TICK so their live tick cadence matches the backtest.
    BAR_TICK = auto()


@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any]

    @staticmethod
    def validate_timestamp(ts_value) -> int:
        """Normalizes and validates a timestamp to milliseconds."""
        if ts_value is None:
            return int(time.time() * 1000)
        
        # Already in milliseconds
        if isinstance(ts_value, int) and ts_value > 1_000_000_000_000:
            return ts_value
        
        # Seconds to milliseconds
        if isinstance(ts_value, (int, float)) and ts_value < 1_000_000_000_000:
            return int(ts_value * 1000)
        
        # ISO string format
        if isinstance(ts_value, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except Exception as e:
                raise ValueError(f"Cannot parse timestamp string: {ts_value}") from e
        
        raise ValueError(f"Invalid timestamp type: {type(ts_value)}")


class EventBus:
    """
    Asynchronous Publish-Subscribe Event Router for HQC.

    Improvements:
    - adds missing SYSTEM_SHUTDOWN channel
    - adds BOOK_SNAPSHOT and MICROSTRUCTURE_ALERT channels
    - keeps strong references to callback tasks so they are not GC-pruned
    - done callbacks surface async subscriber exceptions to logs
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable[[Event], Coroutine[Any, Any, None]]]] = {
            event_type: [] for event_type in EventType
        }
        queue_maxsize = int(os.getenv("HQC_EVENTBUS_QUEUE_MAXSIZE", "500000"))
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max(1000, queue_maxsize))
        self._running: bool = False
        self._worker_task: Optional[asyncio.Task] = None
        self._inflight_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker(), name="event_bus_worker")
            logger.info("EventBus started successfully.")

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._inflight_tasks:
            pending = list(self._inflight_tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        logger.info("EventBus shut down.")

    def subscribe(self, event_type: EventType, callback: Callable[[Event], Coroutine[Any, Any, None]]) -> None:
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def publish(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.error("EventBus queue full. Dropping event: %s", event.type.name)

    async def publish_async(self, event: Event) -> None:
        await self._queue.put(event)

    def _track_task(self, task: asyncio.Task) -> None:
        self._inflight_tasks.add(task)

        def _done_callback(t: asyncio.Task) -> None:
            self._inflight_tasks.discard(t)
            try:
                exc = t.exception()
                if exc is not None:
                    logger.error("Subscriber task failed: %s", exc, exc_info=True)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Failed checking subscriber task result: %s", e, exc_info=True)

        task.add_done_callback(_done_callback)

    async def _worker(self) -> None:
        while self._running:
            try:
                event = await self._queue.get()
                callbacks = self._subscribers.get(event.type, [])

                for callback in callbacks:
                    task = asyncio.create_task(callback(event))
                    self._track_task(task)

                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("EventBus worker encountered an error: %s", e, exc_info=True)