from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Awaitable, Callable

import aiohttp

from core.engine.event_bus import EventBus, Event, EventType
from core.feedback.unified_logger import UnifiedFeedbackLogger

logger = logging.getLogger("ResilientFallbackRouter")


class CircuitState(Enum):
    CLOSED = auto()      # Primary route healthy
    OPEN = auto()        # Primary route bypassed
    HALF_OPEN = auto()   # Testing primary recovery


class ResilientFallbackManager:
    """
    Resilient execution wrapper with retry, backoff, circuit breaking, and fallback routing.

    Behavior:
    - retries primary route with exponential backoff
    - trips circuit after repeated failures
    - routes directly to fallback while circuit is OPEN
    - emits unified outcome records for degradation / failure
    - only emits SYSTEM_SHUTDOWN on total routing failure, not merely on degradation
    """

    def __init__(
        self,
        bus: EventBus,
        max_retries: int = 3,
        base_backoff_sec: float = 0.5,
        circuit_trip_threshold: int = 5,
        recovery_timeout_sec: float = 60.0,
        system_name: str = "HQC",
        arm: str = "equities",
        env: str = "paper",
    ) -> None:
        self.bus = bus
        self.max_retries = int(max_retries)
        self.base_backoff = float(base_backoff_sec)
        self.trip_threshold = int(circuit_trip_threshold)
        self.recovery_timeout_sec = float(recovery_timeout_sec)

        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time: float = 0.0

        self.feedback = UnifiedFeedbackLogger(
            root="state/feedback",
            system_name=system_name,
            arm=arm,
            env=env,
        )

    async def execute_with_resilience(
        self,
        order_event: Event,
        primary_route: Callable[[Event], Awaitable[bool]],
        fallback_route: Callable[[Event], Awaitable[bool]],
    ) -> bool:
        """
        Execute an order via primary route with retry/backoff, then fail over if needed.
        """
        payload = order_event.payload or {}
        asset = payload.get("asset", "UNKNOWN")
        decision_id = payload.get("decision_id")
        strategy = payload.get("strategy")
        side = payload.get("action")
        qty = payload.get("shares")

        if self.state == CircuitState.OPEN:
            time_since_failure = asyncio.get_event_loop().time() - self.last_failure_time
            if time_since_failure > self.recovery_timeout_sec:
                logger.info("Circuit HALF_OPEN: attempting primary route recovery test.")
                self.state = CircuitState.HALF_OPEN
            else:
                logger.warning("Circuit OPEN: bypassing primary route for %s.", asset)
                self.feedback.write_outcome(
                    {
                        "decision_id": decision_id,
                        "status": "degraded_bypass_primary",
                        "symbol": asset,
                        "strategy": strategy,
                        "side": side,
                        "qty": qty,
                        "meta": {
                            "circuit_state": self.state.name,
                            "reason": "primary route bypassed",
                        },
                    }
                )
                return await self._execute_fallback(order_event, fallback_route)

        for attempt in range(1, self.max_retries + 1):
            try:
                success = await asyncio.wait_for(primary_route(order_event), timeout=2.5)

                if success:
                    self._record_success()
                    self.feedback.write_outcome(
                        {
                            "decision_id": decision_id,
                            "status": "primary_route_success",
                            "symbol": asset,
                            "strategy": strategy,
                            "side": side,
                            "qty": qty,
                            "meta": {
                                "attempt": attempt,
                                "circuit_state": self.state.name,
                            },
                        }
                    )
                    return True

                logger.warning(
                    "Primary route returned False for %s on attempt %d/%d.",
                    asset,
                    attempt,
                    self.max_retries,
                )

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.error(
                    "Primary route network failure for %s on attempt %d/%d: %s",
                    asset,
                    attempt,
                    self.max_retries,
                    e,
                )
            except Exception as e:
                logger.critical(
                    "Unexpected primary route failure for %s on attempt %d/%d: %s",
                    asset,
                    attempt,
                    self.max_retries,
                    e,
                    exc_info=True,
                )

            if attempt < self.max_retries:
                sleep_time = self.base_backoff * (2 ** (attempt - 1))
                logger.debug("Primary route backoff %.2fs before retry.", sleep_time)
                await asyncio.sleep(sleep_time)

        self._record_failure()

        logger.error(
            "Primary route exhausted all %d retries for %s. Initiating fallback.",
            self.max_retries,
            asset,
        )
        self.feedback.write_outcome(
            {
                "decision_id": decision_id,
                "status": "primary_route_exhausted",
                "symbol": asset,
                "strategy": strategy,
                "side": side,
                "qty": qty,
                "meta": {
                    "circuit_state": self.state.name,
                    "consecutive_failures": self.consecutive_failures,
                },
            }
        )

        return await self._execute_fallback(order_event, fallback_route)

    async def _execute_fallback(
        self,
        order_event: Event,
        fallback_route: Callable[[Event], Awaitable[bool]],
    ) -> bool:
        payload = order_event.payload or {}
        asset = payload.get("asset", "UNKNOWN")
        decision_id = payload.get("decision_id")
        strategy = payload.get("strategy")
        side = payload.get("action")
        qty = payload.get("shares")

        try:
            success = await asyncio.wait_for(fallback_route(order_event), timeout=3.0)

            if success:
                logger.info("Fallback route successfully handled order for %s.", asset)
                self.feedback.write_outcome(
                    {
                        "decision_id": decision_id,
                        "status": "fallback_route_success",
                        "symbol": asset,
                        "strategy": strategy,
                        "side": side,
                        "qty": qty,
                        "meta": {
                            "circuit_state": self.state.name,
                        },
                    }
                )
                return True

            logger.error("Fallback route explicitly rejected the order for %s.", asset)
            self.feedback.write_outcome(
                {
                    "decision_id": decision_id,
                    "status": "fallback_route_rejected",
                    "symbol": asset,
                    "strategy": strategy,
                    "side": side,
                    "qty": qty,
                }
            )

        except Exception as e:
            logger.critical("Fallback route failed for %s: %s", asset, e, exc_info=True)
            self.feedback.write_outcome(
                {
                    "decision_id": decision_id,
                    "status": "fallback_route_error",
                    "symbol": asset,
                    "strategy": strategy,
                    "side": side,
                    "qty": qty,
                    "meta": {"exception": str(e)},
                }
            )

        self._trigger_system_halt(order_event)
        return False

    def _record_success(self) -> None:
        if self.consecutive_failures > 0 or self.state != CircuitState.CLOSED:
            logger.info("Primary route stability restored. Circuit CLOSED.")
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.consecutive_failures >= self.trip_threshold and self.state != CircuitState.OPEN:
            logger.critical(
                "Primary route failure threshold reached (%d). Circuit OPEN.",
                self.trip_threshold,
            )
            self.state = CircuitState.OPEN

            # Degradation notice only; do not halt the full system here.
            self.feedback.write_outcome(
                {
                    "status": "primary_circuit_open",
                    "meta": {
                        "consecutive_failures": self.consecutive_failures,
                        "trip_threshold": self.trip_threshold,
                    },
                }
            )

    def _trigger_system_halt(self, failed_event: Event) -> None:
        payload = failed_event.payload or {}
        asset = payload.get("asset", "UNKNOWN")
        decision_id = payload.get("decision_id")
        strategy = payload.get("strategy")
        side = payload.get("action")
        qty = payload.get("shares")

        logger.critical("ORPHANED ORDER RISK: both routing tiers failed for %s.", asset)

        self.feedback.write_outcome(
            {
                "decision_id": decision_id,
                "status": "routing_total_failure",
                "symbol": asset,
                "strategy": strategy,
                "side": side,
                "qty": qty,
                "meta": {
                    "reason": "both primary and fallback routing failed",
                },
            }
        )

        halt_event = Event(
            type=EventType.SYSTEM_SHUTDOWN,
            payload={
                "reason": f"Execution routing completely failed for {asset}. Manual intervention required.",
            },
        )
        self.bus.publish(halt_event)


if __name__ == "__main__":
    async def run_resilience_test() -> None:
        print("Initializing Resilient Routing Test...")

        bus = EventBus()
        await bus.start()

        fallback_manager = ResilientFallbackManager(
            bus,
            max_retries=3,
            base_backoff_sec=0.2,
            circuit_trip_threshold=2,
        )

        test_order = Event(
            type=EventType.ORDER_CREATE,
            payload={"asset": "TSLA", "action": "BUY", "shares": 50, "strategy": "TEST"},
        )

        async def dead_primary_api(event: Event) -> bool:
            print("[PRIMARY] Attempting connection...")
            await asyncio.sleep(5.0)
            return True

        async def backup_paper_api(event: Event) -> bool:
            print("[FALLBACK] Securing order to safe paper ledger...")
            await asyncio.sleep(0.1)
            return True

        print("\n--- Testing Exhaustion and Fallback ---")
        await fallback_manager.execute_with_resilience(test_order, dead_primary_api, backup_paper_api)

        print("\n--- Testing Circuit Breaker Trip ---")
        await fallback_manager.execute_with_resilience(test_order, dead_primary_api, backup_paper_api)

        print("\n--- Testing Open Circuit Fast-Fail ---")
        await fallback_manager.execute_with_resilience(test_order, dead_primary_api, backup_paper_api)

        await asyncio.sleep(0.5)
        await bus.stop()

    asyncio.run(run_resilience_test())