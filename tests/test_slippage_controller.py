import unittest

from core.engine.event_bus import Event, EventBus, EventType
from core.execution.slippage_controller import SlippageController


class TestSlippageController(unittest.IsolatedAsyncioTestCase):
    async def test_cumulative_fill_updates_do_not_double_count(self) -> None:
        bus = EventBus()
        controller = SlippageController(bus=bus)

        controller.register_new_order(
            order_id="order-1",
            asset="SPY",
            side="BUY",
            shares=5,
            expected_price=500.0,
        )

        await controller.on_order_update(
            Event(
                type=EventType.ORDER_FILL,
                payload={
                    "order_id": "order-1",
                    "fill_qty": 2,
                    "fill_price": 500.0,
                    "status": "PARTIALLY_FILLED",
                },
            )
        )
        self.assertEqual(controller.active_orders["order-1"].filled_qty, 2)

        await controller.on_order_update(
            Event(
                type=EventType.ORDER_FILL,
                payload={
                    "order_id": "order-1",
                    "fill_qty": 3,
                    "fill_price": 500.5,
                    "status": "PARTIALLY_FILLED",
                },
            )
        )
        self.assertEqual(controller.active_orders["order-1"].filled_qty, 3)

        await controller.on_order_update(
            Event(
                type=EventType.ORDER_FILL,
                payload={
                    "order_id": "order-1",
                    "fill_qty": 5,
                    "fill_price": 501.0,
                    "status": "FILLED",
                },
            )
        )
        self.assertNotIn("order-1", controller.active_orders)


if __name__ == "__main__":
    unittest.main()