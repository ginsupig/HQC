from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp

from core.engine.event_bus import EventBus, Event, EventType
from core.feedback.unified_logger import UnifiedFeedbackLogger

logger = logging.getLogger("AlpacaRouter")


class AlpacaExecutionRouter:
    """
    Alpaca execution router.

    Behavior:
    - consumes ORDER_CREATE with stage == "SIZED"
    - submits market OTO orders with stop loss
    - supports short/cover intents
    - registers accepted orders with slippage controller
    - emits ORDER_FILL lifecycle acknowledgements
    - writes unified outcomes.jsonl records for MegaMind ingestion
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        bus: EventBus,
        is_paper: bool = True,
        simulate_only: bool = True,
        resilient_manager: Optional[Any] = None,
        slippage_controller: Optional[Any] = None,
        request_timeout_sec: float = 8.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.bus = bus
        self.simulate_only = bool(simulate_only)
        self.resilient_manager = resilient_manager
        self.slippage_controller = slippage_controller
        self.timeout = aiohttp.ClientTimeout(total=request_timeout_sec)

        subdomain = "paper-api" if is_paper else "api"
        self.base_url = f"https://{subdomain}.alpaca.markets/v2"
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

        self.session: Optional[aiohttp.ClientSession] = None
        self._stopping = False
        self._order_poll_tasks: Dict[str, asyncio.Task] = {}
        self._reported_order_state: Dict[str, tuple[str, float]] = {}
        self._tracked_order_ids: set[str] = set()

        # Short-lived cache of /account buying_power so a burst of routed
        # orders does not generate one HTTP round-trip per submission.
        self._buying_power_cache: Optional[float] = None
        self._buying_power_cache_at: float = 0.0
        self._buying_power_ttl_sec: float = 1.0
        self.feedback = UnifiedFeedbackLogger(
            root="state/feedback",
            system_name="HQC",
            arm="equities",
            env="paper" if is_paper else "live",
        )

        self.bus.subscribe(EventType.ORDER_CREATE, self.on_order_routed)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_execution_control)

    async def start(self) -> None:
        self._stopping = False
        if self.simulate_only:
            logger.warning("Alpaca Execution Router running in SIMULATE_ONLY mode. No broker API calls will be made.")
            return
        if self._looks_like_placeholder(self.api_key) or self._looks_like_placeholder(self.api_secret):
            raise ValueError(
                "AlpacaExecutionRouter cannot start with placeholder credentials when simulate_only=False."
            )
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout)
            logger.info("Alpaca Execution Router initialized. Target: %s", self.base_url)

    async def reconcile_startup_state(self) -> Dict[str, Any]:
        if self.simulate_only:
            return {
                "account_equity": None,
                "cash": None,
                "buying_power": None,
                "positions": {},
                "open_orders": [],
            }
        if not self.session:
            raise RuntimeError("Execution router session not started.")

        account_payload = await self._get_json("/account")
        positions_payload = await self._get_json("/positions")
        orders_payload = await self._get_json(
            "/orders",
            params={"status": "open", "nested": "true", "direction": "desc"},
        )

        positions = self._normalize_positions(positions_payload)
        open_orders = self._normalize_open_orders(orders_payload)

        for order in open_orders:
            self.restore_open_order_tracking(order)

        return {
            "account_equity": self._to_float(account_payload.get("equity"), 0.0),
            "cash": self._to_float(account_payload.get("cash"), 0.0),
            "buying_power": self._to_float(account_payload.get("buying_power"), 0.0),
            "positions": positions,
            "open_orders": open_orders,
        }

    async def stop(self) -> None:
        self._stopping = True
        for task in list(self._order_poll_tasks.values()):
            task.cancel()
        if self._order_poll_tasks:
            await asyncio.gather(*self._order_poll_tasks.values(), return_exceptions=True)
        self._order_poll_tasks.clear()
        self._reported_order_state.clear()
        self._tracked_order_ids.clear()

        if self.simulate_only:
            return
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Alpaca Execution Router shut down.")

    async def on_order_routed(self, event: Event) -> None:
        payload = dict(event.payload or {})

        if payload.get("stage") != "SIZED":
            return

        asset = payload.get("asset")
        action = str(payload.get("action", "")).upper()
        shares = int(payload.get("shares", 0) or 0)
        strategy = payload.get("strategy", "Unknown")
        decision_id = payload.get("decision_id")
        force_exit = bool((payload.get("meta") or {}).get("eod_liquidation"))

        entry_price = self._to_float(payload.get("entry_price", payload.get("reference_price")), 0.0)
        stop_loss = self._to_float(payload.get("stop_loss", payload.get("stop_loss_price")), 0.0)

        if not asset or not action or shares <= 0 or entry_price <= 0:
            logger.error("Malformed sized order payload. asset=%s action=%s shares=%s", asset, action, shares)
            self.feedback.write_outcome(
                {
                    "decision_id": decision_id,
                    "status": "rejected",
                    "symbol": asset,
                    "strategy": strategy,
                    "side": action,
                    "qty": shares,
                    "entry_price": entry_price,
                    "meta": {"reason": "malformed sized payload"},
                }
            )
            return

        if self._stopping:
            logger.warning("Router is stopping. Ignoring new routed order for %s %s.", asset, action)
            return

        # Fetch latest buying power from Alpaca before submitting (skipped when session not ready)
        buying_power = None
        if not self.simulate_only and self.session and not self.session.closed:
            try:
                buying_power = await self._cached_buying_power()
            except Exception as exc:
                if self._stopping:
                    # Session closed during shutdown; in-flight event arrived too late — drop silently.
                    logger.debug("Router stopping; dropping order for %s during shutdown.", asset)
                    return
                logger.error("Failed to fetch buying power from Alpaca: %s", exc, exc_info=True)
                self.feedback.write_outcome(
                    {
                        "decision_id": decision_id,
                        "status": "rejected",
                        "symbol": asset,
                        "strategy": strategy,
                        "side": action,
                        "qty": shares,
                        "entry_price": entry_price,
                        "meta": {"reason": "failed to fetch buying power", "exception": str(exc)},
                    }
                )
                return

        # Check if order exceeds buying power
        order_value = shares * entry_price
        if buying_power is not None and order_value > buying_power:
            logger.error("Order value %.2f exceeds available buying power %.2f. Rejecting order.", order_value, buying_power)
            self.feedback.write_outcome(
                {
                    "decision_id": decision_id,
                    "status": "rejected",
                    "symbol": asset,
                    "strategy": strategy,
                    "side": action,
                    "qty": shares,
                    "entry_price": entry_price,
                    "meta": {"reason": "insufficient buying power", "order_value": order_value, "buying_power": buying_power},
                }
            )
            return

        side, position_intent = self._alpaca_side(action)

        order_data: Dict[str, Any] = {
            "symbol": str(asset).upper(),
            "qty": str(shares),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }

        requires_protective_stop = self._requires_protective_stop(action=action, force_exit=force_exit)
        if requires_protective_stop:
            if stop_loss <= 0:
                logger.error(
                    "Malformed sized order payload. Missing stop_loss for opening trade. asset=%s action=%s shares=%s",
                    asset,
                    action,
                    shares,
                )
                self.feedback.write_outcome(
                    {
                        "decision_id": decision_id,
                        "status": "rejected",
                        "symbol": asset,
                        "strategy": strategy,
                        "side": action,
                        "qty": shares,
                        "entry_price": entry_price,
                        "meta": {"reason": "missing stop_loss for opening trade"},
                    }
                )
                return
            order_data["order_class"] = "oto"
            order_data["stop_loss"] = {"stop_price": str(round(stop_loss, 2))}

        if position_intent:
            order_data["position_intent"] = position_intent

        logger.info(
            "[ROUTING] %s %s x%d %s | entry=%.2f stop=%.2f score=%.2f",
            strategy,
            action,
            shares,
            asset,
            entry_price,
            stop_loss,
            float(payload.get("rank_score", 0.0) or 0.0),
        )

        if self.simulate_only:
            await self._simulate_fill(payload)
            return

        if self.resilient_manager is not None:
            base_event = Event(type=EventType.ORDER_CREATE, payload=payload)

            async def _primary(_: Event) -> bool:
                return await self._submit_order(order_data, payload)

            async def _fallback(_: Event) -> bool:
                await self._simulate_fill(payload)
                return True

            await self.resilient_manager.execute_with_resilience(
                order_event=base_event,
                primary_route=_primary,
                fallback_route=_fallback,
            )
            return

        await self._submit_order(order_data, payload)

    async def on_execution_control(self, event: Event) -> None:
        payload = event.payload or {}
        if payload.get("action") != "CANCEL_ORDER":
            return

        order_id = payload.get("order_id") or payload.get("exchange_order_id")
        if not order_id or not self.session:
            return

        # Check if already terminal
        last_state = self._reported_order_state.get(str(order_id))
        if last_state is not None:
            status = last_state[0].lower()
            if status in {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}:
                logger.info("[CANCEL IGNORED] %s already terminal (%s)", order_id, status)
                return

        endpoint = f"{self.base_url}/orders/{order_id}"

        try:
            async with self.session.delete(endpoint) as response:
                text = await response.text()

                if response.status in (200, 204):
                    logger.warning("[CANCEL OK] %s", order_id)
                    self._emit_order_update(
                        parsed={"id": order_id, "status": "canceled", "filled_qty": 0},
                        order_data={"symbol": str(payload.get("asset") or payload.get("symbol") or "").upper()},
                        original_payload=payload,
                    )
                    self.feedback.write_outcome(
                        {
                            "decision_id": payload.get("decision_id"),
                            "order_id": order_id,
                            "status": "cancelled",
                            "symbol": payload.get("asset") or payload.get("symbol"),
                            "strategy": payload.get("strategy"),
                            "side": payload.get("side") or payload.get("action"),
                            "meta": {"reason": payload.get("reason", "cancel request")},
                        }
                    )
                else:
                    logger.error("[CANCEL FAIL] %s status=%s body=%s", order_id, response.status, text)
                    self.feedback.write_outcome(
                        {
                            "decision_id": payload.get("decision_id"),
                            "order_id": order_id,
                            "status": "cancel_error",
                            "symbol": payload.get("asset") or payload.get("symbol"),
                            "strategy": payload.get("strategy"),
                            "side": payload.get("side") or payload.get("action"),
                            "meta": {
                                "http_status": response.status,
                                "body": text,
                            },
                        }
                    )
        except Exception as exc:
            logger.error("Cancel failure for %s: %s", order_id, exc, exc_info=True)
            self.feedback.write_outcome(
                {
                    "decision_id": payload.get("decision_id"),
                    "order_id": order_id,
                    "status": "cancel_error",
                    "symbol": payload.get("asset") or payload.get("symbol"),
                    "strategy": payload.get("strategy"),
                    "side": payload.get("side") or payload.get("action"),
                    "meta": {"exception": str(exc)},
                }
            )

    async def _submit_order(self, order_data: Dict[str, Any], original_payload: Dict[str, Any]) -> bool:
        if not self.session or self.session.closed:
            logger.error("HTTP session not started. Cannot route order.")
            return False

        endpoint = f"{self.base_url}/orders"

        try:
            async with self.session.post(endpoint, json=order_data) as response:
                response_text = await response.text()

                if response.status not in (200, 201):
                    logger.error("[EXCHANGE REJECT] status=%s body=%s", response.status, response_text)
                    self.feedback.write_outcome(
                        {
                            "decision_id": original_payload.get("decision_id"),
                            "status": "rejected",
                            "symbol": order_data["symbol"],
                            "strategy": original_payload.get("strategy"),
                            "side": original_payload.get("action"),
                            "qty": int(original_payload.get("shares", 0) or 0),
                            "entry_price": self._to_float(original_payload.get("entry_price"), 0.0),
                            "meta": {
                                "http_status": response.status,
                                "body": response_text,
                            },
                        }
                    )
                    return False

                parsed = json.loads(response_text)
                order_id = parsed.get("id")
                broker_status = parsed.get("status", "submitted")
                filled_qty = self._to_float(parsed.get("filled_qty", 0), 0.0)
                total_qty = int(original_payload.get("shares", 0) or 0)

                logger.info("[EXCHANGE ACK] order_id=%s symbol=%s", order_id, order_data["symbol"])

                if self.slippage_controller is not None and order_id and filled_qty < total_qty:
                    self.slippage_controller.register_new_order(
                        order_id=order_id,
                        asset=order_data["symbol"],
                        side=original_payload.get("action", "BUY"),
                        shares=int(original_payload.get("shares", 0) or 0),
                        expected_price=float(original_payload.get("entry_price", 0.0) or 0.0),
                        decision_id=original_payload.get("decision_id"),
                        strategy=original_payload.get("strategy", "Unknown"),
                    )
                    self._tracked_order_ids.add(str(order_id))

                self._emit_order_update(parsed=parsed, order_data=order_data, original_payload=original_payload)

                if order_id and not self._is_terminal_status(broker_status):
                    self._start_order_poll(order_id, order_data, original_payload)
                return True

        except asyncio.TimeoutError:
            logger.error("Alpaca API timeout while routing %s order.", order_data["symbol"])
            self.feedback.write_outcome(
                {
                    "decision_id": original_payload.get("decision_id"),
                    "status": "timeout",
                    "symbol": order_data["symbol"],
                    "strategy": original_payload.get("strategy"),
                    "side": original_payload.get("action"),
                    "qty": int(original_payload.get("shares", 0) or 0),
                    "entry_price": self._to_float(original_payload.get("entry_price"), 0.0),
                    "meta": {"reason": "broker timeout"},
                }
            )
            return False
        except Exception as exc:
            logger.error("Critical execution failure: %s", exc, exc_info=True)
            self.feedback.write_outcome(
                {
                    "decision_id": original_payload.get("decision_id"),
                    "status": "error",
                    "symbol": order_data["symbol"],
                    "strategy": original_payload.get("strategy"),
                    "side": original_payload.get("action"),
                    "qty": int(original_payload.get("shares", 0) or 0),
                    "entry_price": self._to_float(original_payload.get("entry_price"), 0.0),
                    "meta": {"exception": str(exc)},
                }
            )
            return False

    async def _simulate_fill(self, original_payload: Dict[str, Any]) -> None:
        """Simulate immediate fill for safe testing without broker side effects."""
        symbol = str(original_payload.get("asset", "")).upper()
        shares = int(original_payload.get("shares", 0) or 0)
        entry_price = self._to_float(original_payload.get("entry_price", original_payload.get("reference_price")), 0.0)
        if not symbol or shares <= 0 or entry_price <= 0:
            return

        decision_id = original_payload.get("decision_id")
        strategy = original_payload.get("strategy")
        action = original_payload.get("action")
        order_id = f"SIM-{decision_id or 'NOID'}"

        self.feedback.write_outcome(
            {
                "decision_id": decision_id,
                "order_id": order_id,
                "status": "simulated_filled",
                "symbol": symbol,
                "strategy": strategy,
                "side": action,
                "qty": shares,
                "filled_qty": shares,
                "entry_price": entry_price,
                "fill_price": entry_price,
                "meta": {"mode": "simulate_only"},
            }
        )

        self.bus.publish(
            Event(
                type=EventType.ORDER_FILL,
                payload={
                    "order_id": order_id,
                    "exchange_order_id": order_id,
                    "asset": symbol,
                    "symbol": symbol,
                    "action": action,
                    "side": action,
                    "fill_qty": shares,
                    "filled_qty": shares,
                    "fill_price": entry_price,
                    "entry_price": entry_price,
                    "timestamp": original_payload.get("timestamp"),
                    "status": "FILLED",
                    "strategy": strategy,
                    "decision_id": decision_id,
                },
            )
        )

    def restore_open_order_tracking(self, order: Dict[str, Any]) -> None:
        order_id = str(order.get("id") or "")
        if not order_id:
            return

        total_qty = int(float(order.get("qty", 0) or 0))
        filled_qty = int(float(order.get("filled_qty", 0) or 0))
        remaining_qty = max(0, total_qty - filled_qty)
        if remaining_qty <= 0:
            return

        side = str(order.get("side") or "buy").upper()
        symbol = str(order.get("symbol") or "").upper()
        strategy = str(order.get("strategy") or "RECONCILED_ORDER")
        expected_price = self._first_positive_float(
            order.get("limit_price"),
            order.get("stop_price"),
            order.get("filled_avg_price"),
            order.get("avg_entry_price"),
            0.0,
        )
        age_sec = self._age_seconds_from_iso(order.get("submitted_at"))

        if self.slippage_controller is not None:
            self.slippage_controller.restore_open_order(
                order_id=order_id,
                asset=symbol,
                side=side,
                shares=total_qty,
                expected_price=expected_price,
                decision_id=order.get("decision_id"),
                strategy=strategy,
                filled_qty=filled_qty,
                filled_avg_price=self._to_float(order.get("filled_avg_price"), 0.0),
                age_sec=age_sec,
            )

        self._tracked_order_ids.add(order_id)
        self._reported_order_state[order_id] = (str(order.get("status", "accepted")).lower(), float(filled_qty))
        self._start_order_poll(
            order_id,
            order_data={"symbol": symbol},
            original_payload={
                "action": side,
                "side": side,
                "shares": total_qty,
                "entry_price": expected_price,
                "strategy": strategy,
                "decision_id": order.get("decision_id") or f"RECONCILED-{order_id}",
                "timestamp": order.get("submitted_at"),
            },
        )

    def _emit_order_update(
        self,
        *,
        parsed: Dict[str, Any],
        order_data: Dict[str, Any],
        original_payload: Dict[str, Any],
    ) -> None:
        order_id = parsed.get("id") or parsed.get("order_id")
        broker_status = str(parsed.get("status", "submitted") or "submitted")
        filled_qty = self._to_float(parsed.get("filled_qty", 0), 0.0)
        if order_id and not self._should_emit_state(str(order_id), broker_status, filled_qty):
            return

        fill_price = self._extract_fill_price(parsed, original_payload)
        self.feedback.write_outcome(
            {
                "decision_id": original_payload.get("decision_id"),
                "order_id": order_id,
                "status": broker_status,
                "symbol": order_data["symbol"],
                "strategy": original_payload.get("strategy"),
                "side": original_payload.get("action") or original_payload.get("side"),
                "qty": int(original_payload.get("shares", 0) or 0),
                "filled_qty": filled_qty,
                "entry_price": self._to_float(original_payload.get("entry_price"), 0.0),
                "fill_price": fill_price,
                "meta": {"raw_status": broker_status},
            }
        )

        self.bus.publish(
            Event(
                type=EventType.ORDER_FILL,
                payload={
                    "order_id": order_id,
                    "exchange_order_id": order_id,
                    "asset": order_data["symbol"],
                    "symbol": order_data["symbol"],
                    "action": original_payload.get("action") or original_payload.get("side"),
                    "side": original_payload.get("action") or original_payload.get("side"),
                    "fill_qty": filled_qty,
                    "filled_qty": filled_qty,
                    "fill_price": fill_price,
                    "entry_price": self._to_float(original_payload.get("entry_price"), 0.0),
                    "timestamp": original_payload.get("timestamp"),
                    "status": broker_status.upper(),
                    "strategy": original_payload.get("strategy"),
                    "decision_id": original_payload.get("decision_id"),
                },
            )
        )

        if order_id and self._is_terminal_status(broker_status):
            self._cleanup_order_tracking(str(order_id))

    def _start_order_poll(
        self,
        order_id: str,
        order_data: Dict[str, Any],
        original_payload: Dict[str, Any],
    ) -> None:
        order_id = str(order_id)
        existing = self._order_poll_tasks.get(order_id)
        if existing is not None and not existing.done():
            return

        self._order_poll_tasks[order_id] = asyncio.create_task(
            self._poll_order_until_terminal(order_id, order_data, original_payload),
            name=f"alpaca_order_poll_{order_id}",
        )

    async def _poll_order_until_terminal(
        self,
        order_id: str,
        order_data: Dict[str, Any],
        original_payload: Dict[str, Any],
    ) -> None:
        if not self.session:
            return

        endpoint = f"{self.base_url}/orders/{order_id}"
        try:
            while self.session and not self.session.closed:
                await asyncio.sleep(0.75)

                async with self.session.get(endpoint) as response:
                    if response.status == 404:
                        logger.warning("Order %s no longer available on broker.", order_id)
                        self._cleanup_order_tracking(order_id)
                        return
                    if response.status != 200:
                        logger.debug("Order poll returned %s for %s", response.status, order_id)
                        continue

                    response_text = await response.text()
                    parsed = json.loads(response_text)
                    self._emit_order_update(parsed=parsed, order_data=order_data, original_payload=original_payload)

                    if self._is_terminal_status(parsed.get("status")):
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Order poll failed for %s: %s", order_id, exc, exc_info=False)
        finally:
            self._cleanup_order_tracking(order_id)

    def _cleanup_order_tracking(self, order_id: str) -> None:
        self._reported_order_state.pop(str(order_id), None)
        self._tracked_order_ids.discard(str(order_id))

        task = self._order_poll_tasks.pop(str(order_id), None)
        current = asyncio.current_task()
        if task is not None and task is not current and not task.done():
            task.cancel()

    def _should_emit_state(self, order_id: str, broker_status: str, filled_qty: float) -> bool:
        new_state = (str(broker_status or "").lower(), float(filled_qty or 0.0))
        previous = self._reported_order_state.get(order_id)
        if previous == new_state:
            return False
        self._reported_order_state[order_id] = new_state
        return True

    def _extract_fill_price(self, parsed: Dict[str, Any], original_payload: Dict[str, Any]) -> Optional[float]:
        fill_price = self._to_float(parsed.get("filled_avg_price"), 0.0)
        if fill_price > 0:
            return fill_price
        if self._to_float(parsed.get("filled_qty"), 0.0) > 0:
            return self._to_float(original_payload.get("entry_price"), 0.0)
        return None

    @staticmethod
    def _is_terminal_status(status: Any) -> bool:
        return str(status or "").strip().lower() in {
            "filled",
            "canceled",
            "cancelled",
            "rejected",
            "expired",
            "done_for_day",
        }

    async def _cached_buying_power(self) -> float:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if (
            self._buying_power_cache is not None
            and (now - self._buying_power_cache_at) < self._buying_power_ttl_sec
        ):
            return self._buying_power_cache
        account_payload = await self._get_json("/account")
        value = self._to_float(account_payload.get("buying_power"), 0.0)
        self._buying_power_cache = value
        self._buying_power_cache_at = now
        return value

    async def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.session:
            raise RuntimeError("Execution router session not started.")

        endpoint = f"{self.base_url}{path}"
        async with self.session.get(endpoint, params=params) as response:
            response_text = await response.text()
            if response.status != 200:
                raise RuntimeError(f"Alpaca GET {path} failed: status={response.status} body={response_text[:500]}")
            return json.loads(response_text)

    def _normalize_positions(self, payload: Any) -> Dict[str, Dict[str, float | int]]:
        positions: Dict[str, Dict[str, float | int]] = {}
        if not isinstance(payload, list):
            return positions

        for item in payload:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").upper()
            qty = int(float(item.get("qty", 0) or 0))
            if not symbol or qty == 0:
                continue
            positions[symbol] = {
                "qty": qty,
                "avg_entry_price": self._to_float(item.get("avg_entry_price"), 0.0),
                "last_price": self._to_float(item.get("current_price"), 0.0),
                "market_value": self._to_float(item.get("market_value"), 0.0),
            }
        return positions

    def _normalize_open_orders(self, payload: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        if not isinstance(payload, list):
            return normalized

        for item in payload:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "id": item.get("id"),
                    "symbol": str(item.get("symbol") or "").upper(),
                    "side": str(item.get("side") or "buy").upper(),
                    "qty": item.get("qty", 0),
                    "filled_qty": item.get("filled_qty", 0),
                    "status": item.get("status", "accepted"),
                    "limit_price": item.get("limit_price"),
                    "stop_price": item.get("stop_price"),
                    "filled_avg_price": item.get("filled_avg_price"),
                    "submitted_at": item.get("submitted_at"),
                }
            )
        return normalized

    @staticmethod
    def _first_positive_float(*values: Any) -> float:
        for value in values:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return 0.0

    @staticmethod
    def _age_seconds_from_iso(value: Any) -> Optional[float]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            now_utc = datetime.now(timezone.utc)
            return max(0.0, (now_utc - parsed.astimezone(timezone.utc)).total_seconds())
        except Exception:
            return None

    @staticmethod
    def _alpaca_side(action: str) -> tuple[str, Optional[str]]:
        action = action.upper()

        if action == "BUY":
            return "buy", "buy_to_open"
        if action == "SELL":
            return "sell", "sell_to_close"
        if action == "SELL_SHORT":
            return "sell", "sell_to_open"
        if action == "BUY_TO_COVER":
            return "buy", "buy_to_close"

        return ("buy" if "BUY" in action else "sell"), None

    @staticmethod
    def _requires_protective_stop(action: str, force_exit: bool = False) -> bool:
        """Opening risk orders should carry stop protection; forced exits should not."""
        if force_exit:
            return False

        action = str(action or "").upper()
        return action in {"BUY", "SELL_SHORT", "BUY_TO_OPEN", "SELL_TO_OPEN"}

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _looks_like_placeholder(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return True
        upper = text.upper()
        placeholders = {
            "YOUR_PAPER_KEY",
            "YOUR_PAPER_SECRET",
            "YOUR_API_KEY",
            "YOUR_API_SECRET",
            "CHANGE_ME",
        }
        return (
            upper in placeholders
            or "YOUR_" in upper
            or "PLACEHOLDER" in upper
            or upper.startswith("PK_YOUR")
        )