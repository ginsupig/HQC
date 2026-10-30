import asyncio
import json
import unittest
from typing import Any, Dict, List, Optional

from core.engine.event_bus import Event, EventBus, EventType
from core.execution.broker_router import AlpacaExecutionRouter
from core.execution.eod_liquidator import EODLiquidationManager


class _Response:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return json.dumps(self._body)

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeAlpaca:
    """Paper account with one short position held by an OTO stop leg."""

    def __init__(self, positions: Dict[str, int], legs: Dict[str, str]) -> None:
        self.positions = dict(positions)
        self.open_legs = dict(legs)  # order id -> symbol
        self.closed = False
        self.calls: List[tuple] = []

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> _Response:
        self.calls.append(("GET", url.split("/v2", 1)[1], dict(params or {})))
        if url.endswith("/positions"):
            return _Response(200, [{"symbol": s, "qty": str(q)} for s, q in self.positions.items()])
        if url.endswith("/orders"):
            symbol = (params or {}).get("symbols")
            parents = [
                {"id": "parent", "status": "filled", "legs": [{"id": oid, "status": "new"}]}
                for oid, s in self.open_legs.items()
                if s == symbol
            ]
            return _Response(200, [{"id": leg["legs"][0]["id"], "legs": []} for leg in parents])
        return _Response(200, {"id": url.rsplit("/", 1)[1], "status": "filled", "filled_qty": "0"})

    def delete(self, url: str) -> _Response:
        path = url.split("/v2", 1)[1]
        self.calls.append(("DELETE", path, {}))
        if path.startswith("/orders/"):
            self.open_legs.pop(path.rsplit("/", 1)[1], None)
            return _Response(204, {})
        symbol = path.rsplit("/", 1)[1]
        if symbol not in self.positions:
            return _Response(404, {"message": "position does not exist"})
        if any(s == symbol for s in self.open_legs.values()):
            return _Response(403, {"message": "insufficient qty available for order"})
        qty = self.positions.pop(symbol)
        return _Response(200, {"id": f"close-{symbol}", "symbol": symbol, "side": "buy" if qty < 0 else "sell",
                               "qty": str(abs(qty)), "status": "filled", "filled_qty": str(abs(qty)),
                               "filled_avg_price": "60.5"})


def _router(bus: EventBus, broker: _FakeAlpaca, **kwargs: Any) -> AlpacaExecutionRouter:
    router = AlpacaExecutionRouter(api_key="key", api_secret="secret", bus=bus, simulate_only=False, is_paper=True, **kwargs)
    router.session = broker  # type: ignore[assignment]
    return router


class TestFlattenExits(unittest.IsolatedAsyncioTestCase):
    async def test_flatten_cancels_the_stop_leg_before_closing(self) -> None:
        broker = _FakeAlpaca({"USB": -65}, {"leg-1": "USB"})
        router = _router(EventBus(), broker)

        self.assertTrue(await router.flatten_symbol("USB"))

        deletes = [path for method, path, _ in broker.calls if method == "DELETE"]
        self.assertEqual(deletes, ["/orders/leg-1", "/positions/USB"])
        self.assertEqual(broker.positions, {})

    async def test_flatten_of_a_flat_symbol_is_a_no_op(self) -> None:
        broker = _FakeAlpaca({}, {})
        self.assertTrue(await _router(EventBus(), broker).flatten_symbol("PNC"))

    async def test_exit_with_wrong_side_closes_what_the_broker_holds(self) -> None:
        # The trader thinks it is long USB and asks to SELL; the account is short.
        bus = EventBus()
        fills: List[Dict[str, Any]] = []

        async def on_fill(event: Event) -> None:
            fills.append(event.payload)

        bus.subscribe(EventType.ORDER_FILL, on_fill)
        await bus.start()
        try:
            broker = _FakeAlpaca({"USB": -65}, {})
            router = _router(bus, broker, flatten_on_exit=True)
            await router.on_order_routed(Event(type=EventType.ORDER_CREATE, payload={
                "asset": "USB", "action": "SELL", "strategy": "KalmanPair_PNC_USB", "stage": "SIZED",
                "shares": 65, "entry_price": 60.7, "stop_loss": 61.9,
            }))
            await asyncio.sleep(0.05)
        finally:
            await bus.stop()

        self.assertEqual(broker.positions, {})
        self.assertNotIn(("POST",), [c[:1] for c in broker.calls])
        self.assertEqual(fills[-1]["action"], "BUY_TO_COVER")
        self.assertEqual(fills[-1]["fill_qty"], 65.0)

    async def test_entries_still_route_as_oto_with_flatten_on_exit(self) -> None:
        router = _router(EventBus(), _FakeAlpaca({}, {}), flatten_on_exit=True)
        router._cached_buying_power = lambda: _value(1e6)  # type: ignore[method-assign]
        captured: List[Dict[str, Any]] = []

        async def capture_submit(order_data: Dict[str, Any], original_payload: Dict[str, Any]) -> bool:
            captured.append(order_data)
            return True

        router._submit_order = capture_submit  # type: ignore[method-assign]
        await router.on_order_routed(Event(type=EventType.ORDER_CREATE, payload={
            "asset": "PNC", "action": "SELL_SHORT", "strategy": "KalmanPair_PNC_USB", "stage": "SIZED",
            "shares": 17, "entry_price": 232.44, "stop_loss": 237.09,
        }))

        self.assertEqual(captured[0]["order_class"], "oto")

    async def test_eod_flattens_from_the_broker_not_fill_counts(self) -> None:
        broker = _FakeAlpaca({"PNC": -17, "USB": -65}, {"leg-a": "PNC", "leg-b": "USB"})
        router = _router(EventBus(), broker)
        eod = EODLiquidationManager(bus=EventBus(), flatten=lambda: router.flatten_all({"PNC", "USB"}))
        eod.positions = {"PNC": -15, "USB": 99}  # what today's fill counting believed

        await eod.force_liquidate_now(reason="scheduled_eod")

        self.assertEqual(broker.positions, {})
        self.assertEqual(broker.open_legs, {})
        self.assertEqual(eod.snapshot(), {})


async def _value(value: float) -> float:
    return value


if __name__ == "__main__":
    unittest.main()
