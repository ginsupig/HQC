from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

import aiohttp
import websockets

from core.engine.event_bus import Event, EventBus, EventType

logger = logging.getLogger("TradierDataFeed")


class TradierWebsocketManager:
    """
    Live, low-latency WebSocket manager for US Equities via Tradier Pro.
    Connects to the Tradier consolidated SIP stream to feed tick-level 
    data directly into the EventBus.
    """

    SESSION_URL = "https://api.tradier.com/v1/markets/events/session"
    WSS_URL = "wss://ws.tradier.com/v1/markets/events"

    def __init__(self, api_token: str, symbols: List[str], bus: EventBus) -> None:
        self.api_token = api_token
        self.symbols = [str(s).upper() for s in symbols]
        self.bus = bus
        self._running = False
        self.session_id: Optional[str] = None
        self._ws_task: Optional[asyncio.Task] = None

    async def _create_session(self) -> bool:
        """Tradier requires generating a short-lived session token before connecting to the WS."""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.SESSION_URL, headers=headers) as response:
                    if response.status != 200:
                        logger.error("Failed to create Tradier session: %s", await response.text())
                        return False

                    data = await response.json()
                    self.session_id = data.get("stream", {}).get("sessionid")
                    return bool(self.session_id)
        except Exception as e:
            logger.error("Error connecting to Tradier Session API: %s", e)
            return False

    # MODIFY the _listen() method with improved reconnect logic

    async def _listen(self) -> None:
        """
        Continuous listening loop with exponential backoff reconnection.
        Handles session expiration and transient WS failures.
        """
        # --- PATCH: Add reconnect strategy ---
        retry_count = 0
        max_retries = 10
        backoff_sec = 1.0
        max_backoff_sec = 30.0
        # --- END PATCH ---
        
        while self._running:
            if not self.session_id:
                success = await self._create_session()
                if not success:
                    # --- PATCH: Exponential backoff ---
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(
                            "Failed to create Tradier session after %d retries. Halting feed.",
                            max_retries,
                        )
                        self._running = False
                        break
                    
                    wait_sec = min(backoff_sec * (2 ** (retry_count - 1)), max_backoff_sec)
                    logger.warning(
                        "Retrying Tradier session creation in %.1f seconds... (attempt %d/%d)",
                        wait_sec,
                        retry_count,
                        max_retries,
                    )
                    await asyncio.sleep(wait_sec)
                    # --- END PATCH ---
                    continue
                else:
                    retry_count = 0  # --- PATCH: Reset retry counter on success ---
                    backoff_sec = 1.0

            try:
                # SSL is required for wss://
                async with websockets.connect(self.WSS_URL, ssl=True) as ws:
                    payload = {
                        "symbols": self.symbols,
                        "sessionid": self.session_id,
                        "filter": ["trade"],  # We only want executed trades for tick generation
                        "linebreak": True,
                    }
                    await ws.send(json.dumps(payload))
                    logger.info("Connected to Tradier SIP feed. Subscribed to %s", self.symbols)

                    async for msg in ws:
                        if not self._running:
                            break

                        try:
                            data = json.loads(msg)

                            # Map Tradier Payload to HQC standard TICK event
                            if data.get("type") == "trade":
                                # --- PATCH: Normalize timestamp to milliseconds ---
                                ts_raw = data.get("date", 0)
                                if isinstance(ts_raw, (int, float)):
                                    # Tradier provides unix timestamp in seconds
                                    ts_ms = int(ts_raw * 1000)
                                else:
                                    ts_ms = int(ts_raw) if ts_raw else 0
                                # --- END PATCH ---
                                
                                tick_event = Event(
                                    type=EventType.TICK,
                                    payload={
                                        "ticker": data.get("symbol"),
                                        "price": float(data.get("price", 0.0)),
                                        "volume": float(data.get("size", 0.0)),
                                        "timestamp": ts_ms,  # --- PATCH: Now normalized ---
                                    },
                                )
                                self.bus.publish(tick_event)

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.debug("Error parsing Tradier message: %s", e)

            except websockets.ConnectionClosed:
                logger.warning("Tradier WS connection dropped (session may have expired). Reconnecting...")
                self.session_id = None
                await asyncio.sleep(1.0)
            except asyncio.TimeoutError:
                logger.warning("Tradier WS timeout. Reconnecting...")
                self.session_id = None
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.warning("Unexpected Tradier WS error: %s. Reconnecting...", e)
                self.session_id = None
                await asyncio.sleep(2.0)

    async def start(self) -> None:
        if not self._running:
            self._running = True
            logger.info("Initializing Tradier Pro Market Data Feed...")
            self._ws_task = asyncio.create_task(self._listen(), name="tradier_feed")

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        logger.info("Tradier Data Feed shutting down.")