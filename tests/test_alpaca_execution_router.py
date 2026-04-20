import asyncio
import unittest
from typing import Any, Dict, List

from core.engine.event_bus import EventBus, EventType
from core.engine.event_bus import Event
from core.execution.broker_router import AlpacaExecutionRouter


class TestAlpacaExecutionRouter(unittest.IsolatedAsyncioTestCase):
    def test_alpaca_side_mapping_supports_long_and_short_intents(self) -> None:
        self.assertEqual(AlpacaExecutionRouter._alpaca_side("BUY"), ("buy", "buy_to_open"))
        self.assertEqual(AlpacaExecutionRouter._alpaca_side("SELL"), ("sell", "sell_to_close"))
        self.assertEqual(AlpacaExecutionRouter._alpaca_side("SELL_SHORT"), ("sell", "sell_to_open"))
        self.assertEqual(AlpacaExecutionRouter._alpaca_side("BUY_TO_COVER"), ("buy", "buy_to_close"))

    def test_terminal_status_detection_covers_alpaca_end_states(self) -> None:
        self.assertTrue(AlpacaExecutionRouter._is_terminal_status("filled"))
        self.assertTrue(AlpacaExecutionRouter._is_terminal_status("canceled"))
        self.assertTrue(AlpacaExecutionRouter._is_terminal_status("rejected"))
        self.assertFalse(AlpacaExecutionRouter._is_terminal_status("accepted"))

    def test_protective_stop_required_only_for_opening_risk_orders(self) -> None:
        self.assertTrue(AlpacaExecutionRouter._requires_protective_stop("BUY"))
        self.assertTrue(AlpacaExecutionRouter._requires_protective_stop("SELL_SHORT"))
        self.assertFalse(AlpacaExecutionRouter._requires_protective_stop("SELL"))
        self.assertFalse(AlpacaExecutionRouter._requires_protective_stop("BUY_TO_COVER"))
        self.assertFalse(AlpacaExecutionRouter._requires_protective_stop("SELL", force_exit=True))


    async def test_router_rejects_placeholder_credentials_when_real_routing_enabled(self) -> None:
        bus = EventBus()
        router = AlpacaExecutionRouter(
            api_key="YOUR_PAPER_KEY",
            api_secret="YOUR_PAPER_SECRET",
            bus=bus,
            simulate_only=False,
            is_paper=True,
        )

        with self.assertRaises(ValueError):
            await router.start()


    async def test_simulated_short_fill_emits_short_order_fill_event(self) -> None:
        bus = EventBus()
        captured = []

        async def on_fill(event):
            captured.append(event.payload)

        bus.subscribe(EventType.ORDER_FILL, on_fill)
        await bus.start()

        try:
            router = AlpacaExecutionRouter(
                api_key="key",
                api_secret="secret",
                bus=bus,
                simulate_only=True,
                is_paper=True,
            )

            await router._simulate_fill(
                {
                    "decision_id": "test-short-1",
                    "asset": "SPY",
                    "action": "SELL_SHORT",
                    "shares": 7,
                    "entry_price": 501.25,
                    "strategy": "test",
                    "timestamp": 1710000000000,
                }
            )

            await asyncio.sleep(0.05)

            self.assertEqual(len(captured), 1)
            payload = captured[0]
            self.assertEqual(payload["action"], "SELL_SHORT")
            self.assertEqual(payload["side"], "SELL_SHORT")
            self.assertEqual(payload["fill_qty"], 7)
            self.assertEqual(payload["filled_qty"], 7)
            self.assertEqual(payload["status"], "FILLED")
        finally:
            await bus.stop()

    async def test_eod_liquidation_routes_plain_market_without_stop_payload(self) -> None:
        bus = EventBus()
        router = AlpacaExecutionRouter(
            api_key="key",
            api_secret="secret",
            bus=bus,
            simulate_only=False,
            is_paper=True,
        )

        captured_orders: List[Dict[str, Any]] = []

        async def capture_submit(order_data: Dict[str, Any], original_payload: Dict[str, Any]) -> bool:
            captured_orders.append(dict(order_data))
            return True

        router._submit_order = capture_submit  # type: ignore[method-assign]

        await router.on_order_routed(
            Event(
                type=EventType.ORDER_CREATE,
                payload={
                    "asset": "SPY",
                    "action": "SELL",
                    "strategy": "EOD_LIQUIDATOR",
                    "stage": "SIZED",
                    "shares": 10,
                    "entry_price": 685.67,
                    "stop_loss": 685.67,
                    "meta": {"eod_liquidation": True},
                },
            )
        )

        self.assertEqual(len(captured_orders), 1)
        order_data = captured_orders[0]
        self.assertEqual(order_data["symbol"], "SPY")
        self.assertEqual(order_data["side"], "sell")
        self.assertNotIn("order_class", order_data)
        self.assertNotIn("stop_loss", order_data)

    async def test_opening_buy_routes_with_oto_and_stop_payload(self) -> None:
        bus = EventBus()
        router = AlpacaExecutionRouter(
            api_key="key",
            api_secret="secret",
            bus=bus,
            simulate_only=False,
            is_paper=True,
        )

        captured_orders: List[Dict[str, Any]] = []

        async def capture_submit(order_data: Dict[str, Any], original_payload: Dict[str, Any]) -> bool:
            captured_orders.append(dict(order_data))
            return True

        router._submit_order = capture_submit  # type: ignore[method-assign]

        await router.on_order_routed(
            Event(
                type=EventType.ORDER_CREATE,
                payload={
                    "asset": "TSLA",
                    "action": "BUY",
                    "strategy": "ORB_15m",
                    "stage": "SIZED",
                    "shares": 5,
                    "entry_price": 350.35,
                    "stop_loss": 348.66,
                },
            )
        )

        self.assertEqual(len(captured_orders), 1)
        order_data = captured_orders[0]
        self.assertEqual(order_data["symbol"], "TSLA")
        self.assertEqual(order_data["side"], "buy")
        self.assertEqual(order_data["order_class"], "oto")
        self.assertIn("stop_loss", order_data)


if __name__ == "__main__":
    unittest.main()
