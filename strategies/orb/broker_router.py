import aiohttp
import asyncio
import logging
import json
from typing import Dict, Any

# Core engine imports
from core.engine.event_bus import EventBus, Event, EventType

logger = logging.getLogger("AlpacaRouter")

class AlpacaExecutionRouter:
    """
    High-performance, non-blocking execution router for Alpaca.
    Converts strictly sized internal ORDER_ROUTED events into native exchange 
    API calls, utilizing Advanced Trade Routing (Bracket/OTO orders) for zero-latency risk management.
    """

    def __init__(self, api_key: str, api_secret: str, bus: EventBus, is_paper: bool = True):
        """
        Args:
            api_key (str): Alpaca API Key ID.
            api_secret (str): Alpaca API Secret Key.
            bus (EventBus): The live asynchronous event router.
            is_paper (bool): Toggles between paper and live trading endpoints.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.bus = bus
        
        # Endpoint selection
        subdomain = "paper-api" if is_paper else "api"
        self.base_url = f"https://{subdomain}.alpaca.markets/v2"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }

        # In a full system, Risk emits an ORDER_ROUTED or similar final event.
        # We assume EventType.ORDER_CREATE is what Risk outputs for this module to catch.
        self.bus.subscribe(EventType.ORDER_CREATE, self.on_order_routed)
        self.session: aiohttp.ClientSession | None = None

    async def start(self):
        """Initializes the persistent aiohttp session for connection pooling."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
            logger.info(f"Alpaca Execution Router initialized. Target: {self.base_url}")

    async def stop(self):
        """Gracefully closes the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Alpaca Execution Router shut down.")

    async def on_order_routed(self, event: Event) -> None:
        """
        Intercepts risk-approved order events, formats the Alpaca API payload, 
        and executes the HTTP POST request asynchronously.
        """
        payload = event.payload
        
        # Verify this order has been processed by Risk (has 'shares' rather than empty intent)
        if "shares" not in payload:
            return

        asset = payload.get("asset")
        action = payload.get("action")
        shares = payload.get("shares")
        stop_loss = payload.get("stop_loss")
        strategy = payload.get("strategy", "Unknown")

        # Map internal actions to Alpaca's required 'side' parameter
        alpaca_side = "buy" if action == "BUY" else "sell"

        # Construct Advanced Bracket Order (OTO - One Triggers Other)
        order_data = {
            "symbol": asset,
            "qty": str(shares),
            "side": alpaca_side,
            "type": "market",
            "time_in_force": "day", # Standard for intraday ORB
            "order_class": "oto",   # Attach the stop loss natively to the entry
            "stop_loss": {
                "stop_price": str(round(stop_loss, 2))
            }
        }

        logger.info(f"[ROUTING] Sending {strategy} order to Alpaca: {alpaca_side.upper()} {shares} {asset} | SL: {stop_loss:.2f}")
        
        await self._submit_order(order_data, event)

    async def _submit_order(self, order_data: Dict[str, Any], original_event: Event) -> None:
        """Executes the actual API request and emits the resulting state back to the bus."""
        if not self.session:
            logger.error("HTTP Session not started. Cannot route order.")
            return

        endpoint = f"{self.base_url}/orders"

        try:
            async with self.session.post(endpoint, json=order_data) as response:
                response_text = await response.text()
                
                if response.status in (200, 201):
                    parsed_response = json.loads(response_text)
                    order_id = parsed_response.get("id")
                    logger.info(f"[EXCHANGE ACK] Order {order_id} successfully submitted.")
                    
                    # Emit success back to the state machine
                    fill_event = Event(
                        type=EventType.ORDER_FILL,
                        payload={
                            "asset": order_data["symbol"],
                            "exchange_order_id": order_id,
                            "status": "SUBMITTED",
                            "strategy": original_event.payload.get("strategy")
                        }
                    )
                    self.bus.publish(fill_event)
                else:
                    logger.error(f"[EXCHANGE REJECT] Status {response.status}: {response_text}")
                    # Handle rejection (e.g., insufficient buying power, short restricted)
                    
        except asyncio.TimeoutError:
            logger.error(f"Alpaca API timeout while routing {order_data['symbol']} order.")
        except Exception as e:
            logger.error(f"Critical execution failure: {e}", exc_info=True)

# --- Integration / Boot Example ---
if __name__ == "__main__":
    async def run_execution_test():
        # Boot the asynchronous bus
        bus = EventBus()
        await bus.start()
        
        # Initialize the Router with dummy keys for paper trading
        router = AlpacaExecutionRouter(
            api_key="YOUR_PAPER_KEY", 
            api_secret="YOUR_PAPER_SECRET", 
            bus=bus, 
            is_paper=True
        )
        await router.start()
        
        # Simulate an order approved and sized by the Risk module
        risk_approved_event = Event(
            type=EventType.ORDER_CREATE,
            payload={
                "asset": "SPY",
                "action": "BUY",
                "shares": 15,
                "entry_price": 505.00,
                "stop_loss": 503.50, # 1.50 risk per share. System routes this natively to Alpaca.
                "strategy": "ORB_15m"
            }
        )
        
        print("Publishing risk-approved order to the bus...")
        bus.publish(risk_approved_event)
        
        # Allow time for the async POST request to fire
        await asyncio.sleep(2.0)
        
        # Teardown
        await router.stop()
        await bus.stop()

    # asyncio.run(run_execution_test())
    print("Execution router ready. Awaiting valid Alpaca API keys.")