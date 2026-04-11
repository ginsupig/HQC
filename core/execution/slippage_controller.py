from __future__ import annotations

import asyncio
import heapq
import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from core.engine.event_bus import EventBus, Event, EventType

logger = logging.getLogger("SlippageController")


class OrderStatus(Enum):
    PENDING = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELED = auto()


class ActiveOrder:
    def __init__(
        self,
        order_id: str,
        asset: str,
        side: str,
        total_qty: int,
        expected_price: float,
        decision_id: Optional[str] = None,
        strategy: str = "Unknown",
    ):
        self.order_id = str(order_id)
        self.asset = str(asset).upper()
        self.side = str(side).upper()
        self.total_qty = int(total_qty)
        self.expected_price = float(expected_price) if expected_price is not None else 0.0
        self.decision_id = decision_id
        self.strategy = strategy

        self.filled_qty = 0
        self.cumulative_fill_cost = 0.0
        self.status = OrderStatus.PENDING

    @property
    def remaining_qty(self) -> int:
        return max(0, self.total_qty - self.filled_qty)

    @property
    def vwap_fill_price(self) -> float:
        return self.cumulative_fill_cost / self.filled_qty if self.filled_qty > 0 else 0.0


class SlippageController:
    """
    Slippage / timeout guard for live orders.

    Upgrades:
    - Decoupled timeout monitoring from the live tick feed.
    - Continuous async background loop for deterministic hanging-order cancellation.
    """

    def __init__(
        self,
        bus: EventBus,
        max_slippage_bps: float = 10.0,
        max_hanging_time_sec: float = 5.0,
    ):
        self.bus = bus
        self.max_slippage_bps = float(max_slippage_bps)
        self.max_hanging_time_sec = float(max_hanging_time_sec)

        self.active_orders: Dict[str, ActiveOrder] = {}
        self.order_timestamps: Dict[str, float] = {}
        self._order_time_heap: List[Tuple[float, str]] = []

        self._running: bool = False
        self._monitor_task: Optional[asyncio.Task] = None

        self.bus.subscribe(EventType.ORDER_FILL, self.on_order_update)
        self.bus.subscribe(EventType.TICK, self.on_tick_drift_monitor)

    async def start(self) -> None:
        """Starts the independent background timeout monitor."""
        if not self._running:
            self._running = True
            self._monitor_task = asyncio.create_task(self._hanging_order_monitor())
            logger.info("Slippage Controller started background timeout monitor.")

    async def stop(self) -> None:
        """Gracefully stops the background monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Slippage Controller background monitor stopped.")

    def register_new_order(
        self,
        order_id: str,
        asset: str,
        side: str,
        shares: int,
        expected_price: float,
        decision_id: Optional[str] = None,
        strategy: str = "Unknown",
    ) -> None:
        self.active_orders[str(order_id)] = ActiveOrder(
            order_id=order_id,
            asset=asset,
            side=side,
            total_qty=shares,
            expected_price=expected_price,
            decision_id=decision_id,
            strategy=strategy,
        )
        order_ts = asyncio.get_event_loop().time()
        normalized_order_id = str(order_id)
        self.order_timestamps[normalized_order_id] = order_ts
        heapq.heappush(self._order_time_heap, (order_ts, normalized_order_id))
        logger.debug(
            "Tracking new order %s %s %s x%s @ %.4f",
            order_id,
            asset,
            side,
            shares,
            expected_price,
        )

    def restore_open_order(
        self,
        order_id: str,
        asset: str,
        side: str,
        shares: int,
        expected_price: float,
        decision_id: Optional[str] = None,
        strategy: str = "RECONCILED_ORDER",
        filled_qty: int = 0,
        filled_avg_price: float = 0.0,
        age_sec: Optional[float] = None,
    ) -> None:
        self.register_new_order(
            order_id=order_id,
            asset=asset,
            side=side,
            shares=shares,
            expected_price=expected_price,
            decision_id=decision_id,
            strategy=strategy,
        )

        order = self.active_orders.get(str(order_id))
        if order is None:
            return

        restored_filled_qty = max(0, min(int(filled_qty), order.total_qty))
        order.filled_qty = restored_filled_qty
        if restored_filled_qty > 0 and filled_avg_price > 0:
            order.cumulative_fill_cost = float(filled_avg_price) * restored_filled_qty
            order.status = OrderStatus.PARTIALLY_FILLED

        timestamp = asyncio.get_event_loop().time()
        if age_sec is not None:
            timestamp -= max(0.0, float(age_sec))
        normalized_order_id = str(order_id)
        self.order_timestamps[normalized_order_id] = timestamp
        heapq.heappush(self._order_time_heap, (timestamp, normalized_order_id))

    async def _hanging_order_monitor(self) -> None:
        """
        Independent background loop evaluating hanging times globally.
        Ensures execution halts trigger cancellations even if tick data stops.
        """
        while self._running:
            await asyncio.sleep(1.0)  # Check every second

            if not self._order_time_heap:
                continue

            current_time = asyncio.get_event_loop().time()
            while self._order_time_heap:
                oldest_ts, order_id = self._order_time_heap[0]
                time_alive = current_time - oldest_ts
                if time_alive <= self.max_hanging_time_sec:
                    break

                heapq.heappop(self._order_time_heap)

                current_ts = self.order_timestamps.get(order_id)
                if current_ts is None:
                    continue
                if current_ts != oldest_ts:
                    continue

                order = self.active_orders.get(order_id)
                if order is None:
                    continue

                logger.warning(
                    "[TIMEOUT CANCEL] %s timeout %.2fs > %.2fs (Asset: %s)",
                    order_id,
                    time_alive,
                    self.max_hanging_time_sec,
                    order.asset,
                )
                self._dispatch_cancel(order_id, reason="Time-in-flight limit exceeded")

    async def on_order_update(self, event: Event) -> None:
        payload = event.payload or {}

        # Ignore self-generated cancel control messages
        if payload.get("action") == "CANCEL_ORDER":
            return

        order_id = payload.get("order_id") or payload.get("exchange_order_id")
        if not order_id:
            return

        order_id = str(order_id)
        if order_id not in self.active_orders:
            return

        order = self.active_orders[order_id]

        fill_qty = self._to_int(payload.get("fill_qty", payload.get("filled_qty", 0)), 0)
        fill_price = self._to_float(payload.get("fill_price", payload.get("entry_price", 0.0)), 0.0)
        status_str = str(payload.get("status", "UNKNOWN")).upper()

        if fill_qty > 0 and fill_price > 0:
            cumulative_fill = min(fill_qty, order.total_qty)
            incremental_fill = max(0, cumulative_fill - order.filled_qty)
            if incremental_fill > 0:
                order.filled_qty += incremental_fill
                order.cumulative_fill_cost += fill_price * incremental_fill
                order.status = OrderStatus.PARTIALLY_FILLED

        if status_str in {"FILLED", "ACCEPTED", "SUBMITTED"} and order.remaining_qty <= 0:
            order.status = OrderStatus.FILLED
            self._cleanup_order(order_id)
        elif status_str in {"CANCELED", "CANCELLED", "REJECTED", "ERROR"}:
            order.status = OrderStatus.CANCELED
            self._cleanup_order(order_id)

    async def on_tick_drift_monitor(self, event: Event) -> None:
        """
        Strictly monitors price slippage dynamically against incoming ticks.
        """
        if not self.active_orders:
            return

        payload = event.payload or {}
        tick_asset = payload.get("ticker") or payload.get("symbol")
        current_price = payload.get("price")

        if not tick_asset or current_price is None:
            return

        tick_asset = str(tick_asset).upper()
        current_price = float(current_price)

        orders_to_cancel: List[str] = []

        for order_id, order in list(self.active_orders.items()):
            if order.asset != tick_asset:
                continue

            if order.expected_price <= 0:
                continue

            drift_pct = abs(current_price - order.expected_price) / order.expected_price
            drift_bps = drift_pct * 10000.0

            slipping_away = (
                (order.side.startswith("BUY") and current_price > order.expected_price)
                or (order.side.startswith("SELL") and current_price < order.expected_price)
            )

            if slipping_away and drift_bps > self.max_slippage_bps:
                logger.warning(
                    "[SLIPPAGE CANCEL] %s drift %.2fbps > %.2fbps asset=%s side=%s expected=%.4f current=%.4f",
                    order_id,
                    drift_bps,
                    self.max_slippage_bps,
                    order.asset,
                    order.side,
                    order.expected_price,
                    current_price,
                )
                orders_to_cancel.append(order_id)

        for order_id in orders_to_cancel:
            self._dispatch_cancel(order_id, reason="Price slippage limits exceeded")

    def _dispatch_cancel(self, order_id: str, reason: str = "Slippage/Timeout limits exceeded") -> None:
        if order_id not in self.active_orders:
            return

        order = self.active_orders[order_id]

        self.bus.publish(
            Event(
                type=EventType.ORDER_FILL,
                payload={
                    "action": "CANCEL_ORDER",
                    "order_id": order_id,
                    "exchange_order_id": order_id,
                    "asset": order.asset,
                    "symbol": order.asset,
                    "side": order.side,
                    "decision_id": order.decision_id,
                    "strategy": order.strategy,
                    "reason": reason,
                    "status": "CANCELED",
                },
            )
        )

        self._cleanup_order(order_id)

    def _cleanup_order(self, order_id: str) -> None:
        self.active_orders.pop(str(order_id), None)
        self.order_timestamps.pop(str(order_id), None)

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default