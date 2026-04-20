from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, List, Optional

import websockets

from core.engine.event_bus import Event, EventBus, EventType

logger = logging.getLogger("AlpacaDataFeed")


class AlpacaWebsocketManager:
    """
    Live Alpaca market-data websocket manager for US equities.

    Behavior:
    - authenticates against Alpaca's stock data stream
    - subscribes to trade ticks for the requested symbols
    - blocks start() until the first connection is established
    - reconnects with bounded exponential backoff
    - normalizes timestamps to milliseconds for the event bus
    """

    BASE_WSS_URL = "wss://stream.data.alpaca.markets/v2"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        bus: EventBus,
        feed: str = "iex",
        connect_timeout_sec: float = 300.0,
        reconnect_base_sec: float = 1.0,
        reconnect_max_sec: float = 30.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
        self.bus = bus
        self.feed = str(feed or "iex").strip().lower()
        if self.feed not in {"iex", "sip"}:
            raise ValueError(f"Unsupported ALPACA_DATA_FEED={self.feed}. Expected 'iex' or 'sip'.")

        self.wss_url = f"{self.BASE_WSS_URL}/{self.feed}"
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.reconnect_base_sec = float(reconnect_base_sec)
        self.reconnect_max_sec = float(reconnect_max_sec)

        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()
        self._connected_once = False

    # Sentinel returned when Alpaca rejects with 406 (connection limit).
    _AUTH_LIMIT_EXCEEDED = object()

    async def _authenticate(self):
        """Returns True on success, _AUTH_LIMIT_EXCEEDED on 406, False on bad credentials."""
        auth_payload = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret,
        }
        await self.connection.send(json.dumps(auth_payload))
        response = await asyncio.wait_for(self.connection.recv(), timeout=10.0)
        response_data = self._decode_message(response)

        for event in response_data:
            if event.get("T") == "success" and event.get("msg") == "authenticated":
                logger.info("Authenticated with Alpaca %s market-data feed.", self.feed.upper())
                return True
            if event.get("T") == "error":
                if event.get("code") == 406:
                    logger.warning(
                        "Alpaca market-data: connection limit exceeded (406). "
                        "A previous session is still open on Alpaca's side. Will retry after delay."
                    )
                    return self._AUTH_LIMIT_EXCEEDED
                logger.error("Alpaca market-data auth failed: %s", event)
                return False

        logger.error("Unexpected authentication response: %s", response_data)
        return False

    async def _subscribe(self) -> None:
        sub_payload = {
            "action": "subscribe",
            "trades": self.symbols,
        }
        await self.connection.send(json.dumps(sub_payload))
        logger.info("Subscribed to Alpaca trades for %s", self.symbols)

    async def _listen(self) -> None:
        retry_count = 0

        while self._running:
            self.connection = None
            try:
                logger.info("Connecting to Alpaca %s market-data feed...", self.feed.upper())
                async with websockets.connect(
                    self.wss_url,
                    close_timeout=5.0,
                    ping_interval=20,
                    ping_timeout=10,
                ) as websocket:
                    self.connection = websocket
                    welcome = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    logger.debug("Alpaca welcome message: %s", welcome)

                    auth_result = await self._authenticate()
                    if auth_result is self._AUTH_LIMIT_EXCEEDED:
                        # Previous session still alive on Alpaca's side; back off exponentially.
                        limit_wait = min(30.0 * (2 ** retry_count), 120.0)
                        logger.info(
                            "Waiting %.0fs for Alpaca connection slot to free up (attempt %d)...",
                            limit_wait,
                            retry_count + 1,
                        )
                        await asyncio.sleep(limit_wait)
                        raise RuntimeError("Alpaca connection limit exceeded (406)")
                    if not auth_result:
                        raise RuntimeError("Alpaca websocket authentication failed")

                    await self._subscribe()
                    retry_count = 0

                    if not self._connected_once:
                        self._connected.set()
                        self._connected_once = True

                    async for raw_message in websocket:
                        if not self._running:
                            break

                        try:
                            data = self._decode_message(raw_message)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON Alpaca message: %s", raw_message)
                            continue

                        for event in data:
                            await self._handle_stream_event(event)

            except asyncio.CancelledError:
                break
            except websockets.ConnectionClosed as exc:
                logger.warning("Alpaca market-data websocket closed: %s", exc)
            except Exception as exc:
                logger.warning("Alpaca market-data stream error: %s", exc, exc_info=False)

            if not self._running:
                break

            retry_count += 1
            delay_sec = min(self.reconnect_base_sec * (2 ** (retry_count - 1)), self.reconnect_max_sec)
            logger.info("Reconnecting Alpaca market-data feed in %.1fs", delay_sec)
            await asyncio.sleep(delay_sec)

        logger.info("Alpaca market-data listen loop exited. running=%s", self._running)

    async def _handle_stream_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("T")

        if event_type == "t":
            symbol = str(event.get("S") or "").upper()
            price = self._to_float(event.get("p"), 0.0)
            size = self._to_float(event.get("s"), 0.0)
            if not symbol or price <= 0 or size <= 0:
                return

            self.bus.publish(
                Event(
                    type=EventType.TICK,
                    payload={
                        "ticker": symbol,
                        "symbol": symbol,
                        "price": price,
                        "volume": size,
                        "timestamp": self._normalize_ts_ms(event.get("t")),
                        "conditions": event.get("c") or [],
                        "exchange": event.get("x"),
                        "feed": self.feed,
                    },
                )
            )
            return

        if event_type == "subscription":
            logger.info("Alpaca subscription acknowledged: %s", event)
            return

        if event_type == "error":
            logger.error("Alpaca market-data error event: %s", event)
            return

        if event_type == "success":
            logger.debug("Alpaca stream success event: %s", event)

    async def start(self) -> None:
        if self._running:
            logger.warning("Alpaca market-data feed already running.")
            return

        if self._looks_like_placeholder(self.api_key) or self._looks_like_placeholder(self.api_secret):
            raise ValueError("Alpaca market-data feed requires real API credentials.")

        self._running = True
        self._connected.clear()
        self._connected_once = False
        self._ws_task = asyncio.create_task(self._listen(), name="alpaca_market_data_feed")

        try:
            await asyncio.wait_for(self._connected.wait(), timeout=self.connect_timeout_sec)
            logger.info("Alpaca market-data feed connection established successfully.")
        except asyncio.TimeoutError:
            self._running = False
            if self._ws_task is not None:
                self._ws_task.cancel()
            raise RuntimeError("Alpaca market-data feed connection timeout") from None

    async def stop(self) -> None:
        self._running = False
        if self.connection:
            await self.connection.close()
        if self._ws_task is not None:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        logger.info("Alpaca market-data feed terminated.")

    @staticmethod
    def _decode_message(message: str) -> list[dict[str, Any]]:
        data = json.loads(message)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    @staticmethod
    def _normalize_ts_ms(value: Any) -> int:
        if value is None:
            return int(time.time() * 1000)

        if isinstance(value, (int, float)):
            raw = int(value)
            if raw > 1_000_000_000_000_000:
                return raw // 1_000_000
            if raw > 1_000_000_000_000:
                return raw
            return raw * 1000

        if isinstance(value, str):
            try:
                numeric = int(float(value))
                return AlpacaWebsocketManager._normalize_ts_ms(numeric)
            except (TypeError, ValueError):
                pass
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except Exception:
                return int(time.time() * 1000)

        return int(time.time() * 1000)

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
        return (
            "YOUR_" in upper
            or "PLACEHOLDER" in upper
            or upper.startswith("PK_YOUR")
            or upper in {"YOUR_PAPER_KEY", "YOUR_PAPER_SECRET", "YOUR_API_KEY", "YOUR_API_SECRET", "CHANGE_ME"}
        )


USEquityWebsocketManager = AlpacaWebsocketManager


if __name__ == "__main__":
    async def dummy_strategy_node(event: Event):
        payload = event.payload
        print(
            f"[ENGINE RECEIVED] {payload['ticker']} | {payload['volume']} shares @ ${payload['price']:.2f} | Cond: {payload['conditions']}"
        )

    async def run_live_feed():
        print("Initializing Alpaca stock market live feed...")

        live_bus = EventBus()
        await live_bus.start()
        live_bus.subscribe(EventType.TICK, dummy_strategy_node)

        api_key = "YOUR_ALPACA_API_KEY"
        api_secret = "YOUR_ALPACA_API_SECRET"

        feed = AlpacaWebsocketManager(
            api_key=api_key,
            api_secret=api_secret,
            symbols=["SPY", "TSLA", "NVDA"],
            bus=live_bus,
        )

        feed_task = asyncio.create_task(feed.start())
        await asyncio.sleep(10)
        await feed.stop()
        await feed_task
        await live_bus.stop()

    print("Code is fully run-ready for US Equities. Insert API keys to stream live.")