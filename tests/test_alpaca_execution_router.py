import asyncio
import unittest

from core.engine.event_bus import EventBus, EventType
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


if __name__ == "__main__":
    unittest.main()
