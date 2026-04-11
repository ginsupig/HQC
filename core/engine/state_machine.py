from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Dict, List

from core.engine.event_bus import EventBus, Event, EventType
from core.feedback.unified_logger import UnifiedFeedbackLogger

logger = logging.getLogger("SystemStateMachine")


class SystemState(Enum):
    """
    Global operational state of the trading system.
    """
    INITIALIZING = auto()
    WARMING_UP = auto()
    LIVE_TRADING = auto()
    VIRTUAL_TRADING = auto()
    HALTED = auto()


class GlobalStateMachine:
    """
    Master state machine governing whether live capital may be deployed.

    Hardened behavior:
    - does not emit SYSTEM_SHUTDOWN for ordinary state changes
    - only halts on true shutdown / catastrophic events
    - tolerates mixed REGIME_CHANGE payloads used elsewhere in HQC
    - records state transitions to unified feedback logs
    - prevents shutdown rebroadcast recursion cleanly
    - supports optional manual-only reentry into LIVE_TRADING
    """

    SELF_ORIGIN = "GlobalStateMachine"

    def __init__(
        self,
        bus: EventBus,
        system_name: str = "HQC",
        arm: str = "equities",
        env: str = "paper",
        soft_drawdown_pct: float = -0.05,
        hard_drawdown_pct: float = -0.10,
        live_reentry_slope_threshold: float = 0.02,
        allow_live_reentry: bool = True,
    ) -> None:
        self.bus = bus
        self.current_state: SystemState = SystemState.INITIALIZING
        self.state_history: List[Dict[str, Any]] = []

        self.soft_drawdown_pct = float(soft_drawdown_pct)
        self.hard_drawdown_pct = float(hard_drawdown_pct)
        self.live_reentry_slope_threshold = float(live_reentry_slope_threshold)
        self.allow_live_reentry = bool(allow_live_reentry)

        self.feedback = UnifiedFeedbackLogger(
            root="state/feedback",
            system_name=system_name,
            arm=arm,
            env=env,
        )

        self._valid_transitions: Dict[SystemState, List[SystemState]] = {
            SystemState.INITIALIZING: [SystemState.WARMING_UP, SystemState.HALTED],
            SystemState.WARMING_UP: [SystemState.LIVE_TRADING, SystemState.VIRTUAL_TRADING, SystemState.HALTED],
            SystemState.LIVE_TRADING: [SystemState.VIRTUAL_TRADING, SystemState.HALTED],
            SystemState.VIRTUAL_TRADING: [SystemState.LIVE_TRADING, SystemState.HALTED],
            SystemState.HALTED: [],
        }

        self.bus.subscribe(EventType.REGIME_CHANGE, self.on_regime_change)
        self.bus.subscribe(EventType.EQUITY_UPDATE, self.on_equity_update)
        self.bus.subscribe(EventType.SYSTEM_SHUTDOWN, self.on_shutdown_event)

    def _can_transition(self, target_state: SystemState) -> bool:
        return target_state in self._valid_transitions.get(self.current_state, [])

    def transition_to(self, new_state: SystemState, reason: str) -> bool:
        if self.current_state == new_state:
            return True

        if not self._can_transition(new_state):
            logger.error(
                "ILLEGAL STATE TRANSITION ATTEMPTED: %s -> %s. Reason: %s",
                self.current_state.name,
                new_state.name,
                reason,
            )
            return False

        old_state = self.current_state
        self.current_state = new_state

        record = {
            "from": old_state.name,
            "to": new_state.name,
            "reason": reason,
        }
        self.state_history.append(record)

        logger.warning("[STATE CHANGE] %s -> %s | Reason: %s", old_state.name, new_state.name, reason)

        self.feedback.write_health(
            state=new_state.name,
            status="running" if new_state != SystemState.HALTED else "halted",
            feed_ok=1,
            router_ok=1 if new_state != SystemState.HALTED else 0,
            strategies_loaded=0,
            extra={
                "action": "GLOBAL_STATE_UPDATE",
                "previous_state": old_state.name,
                "current_state": new_state.name,
                "reason": reason,
            },
        )

        if new_state == SystemState.HALTED:
            self.bus.publish(
                Event(
                    type=EventType.SYSTEM_SHUTDOWN,
                    payload={
                        "origin": self.SELF_ORIGIN,
                        "reason": reason,
                        "previous_state": old_state.name,
                        "current_state": new_state.name,
                    },
                )
            )
        else:
            self.bus.publish(
                Event(
                    type=EventType.REGIME_CHANGE,
                    payload={
                        "action": "GLOBAL_STATE_UPDATE",
                        "previous_state": old_state.name,
                        "current_state": new_state.name,
                        "reason": reason,
                    },
                )
            )

        return True

    async def on_regime_change(self, event: Event) -> None:
        payload = event.payload or {}

        # Ignore our own rebroadcasted state updates.
        if payload.get("action") == "GLOBAL_STATE_UPDATE":
            return

        new_regime = (
            payload.get("regime")
            or payload.get("regime_label")
            or payload.get("current_regime")
            or payload.get("label")
        )
        if not new_regime:
            return

        regime_text = str(new_regime).lower()

        if any(x in regime_text for x in ("shock", "hostile", "panic", "crash", "high volatility")):
            if self.current_state == SystemState.LIVE_TRADING:
                self.transition_to(
                    SystemState.VIRTUAL_TRADING,
                    reason=f"Hostile market regime detected: {new_regime}",
                )
            return

        if any(x in regime_text for x in ("trend", "range", "range bound", "calm", "low volatility")):
            # Re-engagement is mainly handled by equity slope, so this path stays permissive.
            return

    async def on_equity_update(self, event: Event) -> None:
        payload = event.payload or {}

        drawdown_pct = self._to_float(payload.get("drawdown_pct"), 0.0)
        virtual_slope = self._to_float(payload.get("virtual_slope"), 0.0)

        # Hard circuit breaker
        if drawdown_pct <= self.hard_drawdown_pct and self.current_state != SystemState.HALTED:
            self.transition_to(
                SystemState.HALTED,
                reason=f"Catastrophic drawdown threshold reached: {drawdown_pct * 100:.2f}%",
            )
            return

        # Soft circuit breaker
        if drawdown_pct <= self.soft_drawdown_pct and self.current_state == SystemState.LIVE_TRADING:
            self.transition_to(
                SystemState.VIRTUAL_TRADING,
                reason=f"Soft drawdown threshold reached: {drawdown_pct * 100:.2f}%",
            )
            return

        # Re-engagement
        if (
            self.allow_live_reentry
            and self.current_state == SystemState.VIRTUAL_TRADING
            and virtual_slope > self.live_reentry_slope_threshold
        ):
            self.transition_to(
                SystemState.LIVE_TRADING,
                reason=(
                    f"Virtual equity slope recovered: {virtual_slope * 100:.2f}%. "
                    "Resuming live execution."
                ),
            )

    async def on_shutdown_event(self, event: Event) -> None:
        payload = event.payload or {}
        reason = payload.get("reason", "Manual or system exception")
        origin = payload.get("origin")

        # If we already halted, do nothing.
        if self.current_state == SystemState.HALTED:
            return

        # If this shutdown was emitted by our own transition_to(HALTED), just settle state.
        if (
            origin == self.SELF_ORIGIN
            or payload.get("current_state") == SystemState.HALTED.name
        ):
            self.current_state = SystemState.HALTED
            logger.warning("[STATE SYNC] Confirmed self-originated HALTED state.")
            return

        self.transition_to(SystemState.HALTED, reason=reason)

    def is_live(self) -> bool:
        return self.current_state == SystemState.LIVE_TRADING

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state.name,
            "soft_drawdown_pct": self.soft_drawdown_pct,
            "hard_drawdown_pct": self.hard_drawdown_pct,
            "live_reentry_slope_threshold": self.live_reentry_slope_threshold,
            "allow_live_reentry": self.allow_live_reentry,
            "state_history": list(self.state_history),
        }

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default