from __future__ import annotations

from datetime import datetime, time, date
from typing import Dict, Optional

import pytz

from core.engine.event_bus import Event, EventBus, EventType


class EODLiquidationManager:
    """Tracks filled inventory and forces flat positions near market close."""

    def __init__(
        self,
        bus: EventBus,
        liquidate_hour: int = 15,
        liquidate_minute: int = 55,
        timezone_name: str = "US/Eastern",
    ) -> None:
        self.bus = bus
        self.tz = pytz.timezone(timezone_name)
        self.liquidate_time = time(liquidate_hour, liquidate_minute)
        self.positions: Dict[str, int] = {}
        self.last_prices: Dict[str, float] = {}
        self.last_timestamps: Dict[str, int] = {}
        self._last_liquidated_date: Optional[date] = None

        self.bus.subscribe(EventType.TICK, self.on_tick)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_order_fill)

    async def on_tick(self, event: Event) -> None:
        payload = event.payload or {}
        symbol = str(payload.get("ticker") or payload.get("symbol") or "").upper()
        if symbol:
            try:
                px = float(payload.get("price", 0.0))
                if px > 0:
                    self.last_prices[symbol] = px
            except (TypeError, ValueError):
                pass

        ts_raw = payload.get("timestamp")
        if ts_raw is None:
            return

        try:
            ts_ms = int(ts_raw)
            if ts_ms < 10_000_000_000:
                ts_ms *= 1000
            dt_est = datetime.fromtimestamp(ts_ms / 1000.0, tz=pytz.utc).astimezone(self.tz)
            if symbol:
                self.last_timestamps[symbol] = ts_ms
        except Exception:
            return

        if dt_est.time() >= self.liquidate_time:
            if self._last_liquidated_date == dt_est.date():
                return
            await self.force_liquidate_now(reason="scheduled_eod")
            self._last_liquidated_date = dt_est.date()

    async def on_order_fill(self, event: Event) -> None:
        payload = event.payload or {}
        status = str(payload.get("status", "")).upper()
        if status in {"CANCELED", "CANCELLED", "REJECTED", "ERROR"}:
            return

        symbol = str(payload.get("asset") or payload.get("symbol") or "").upper()
        if not symbol:
            return

        try:
            fill_qty = int(float(payload.get("fill_qty", payload.get("filled_qty", 0)) or 0))
            fill_price = float(payload.get("fill_price", payload.get("entry_price", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return

        if fill_qty <= 0:
            return

        action = str(payload.get("action") or payload.get("side") or "").upper()
        delta = 0
        if action in {"BUY", "BUY_TO_OPEN", "BUY_TO_COVER"}:
            delta = fill_qty
        elif action in {"SELL", "SELL_TO_OPEN", "SELL_SHORT"}:
            delta = -fill_qty

        if delta == 0:
            return

        self.positions[symbol] = int(self.positions.get(symbol, 0) + delta)
        if fill_price > 0:
            self.last_prices[symbol] = fill_price

    async def force_liquidate_now(self, reason: str = "manual") -> None:
        for symbol, qty in list(self.positions.items()):
            if qty == 0:
                continue

            last_px = float(self.last_prices.get(symbol, 0.0) or 0.0)
            if last_px <= 0:
                continue

            action = "SELL" if qty > 0 else "BUY_TO_COVER"
            shares = abs(int(qty))

            self.bus.publish(
                Event(
                    type=EventType.ORDER_CREATE,
                    payload={
                        "asset": symbol,
                        "action": action,
                        "strategy": "EOD_LIQUIDATOR",
                        "stage": "SIZED",
                        "shares": shares,
                        "reference_price": round(last_px, 4),
                        "entry_price": round(last_px, 4),
                        "timestamp": self.last_timestamps.get(symbol),
                        "stop_loss": round(last_px, 4),
                        "stop_loss_price": round(last_px, 4),
                        "status": "READY_FOR_BROKER",
                        "decision_id": f"EOD-{symbol}-{reason}",
                        "meta": {"eod_liquidation": True, "reason": reason},
                    },
                )
            )

        self.positions.clear()

    def seed_positions(self, positions: Dict[str, Dict[str, float | int]]) -> None:
        self.positions.clear()
        for symbol, payload in positions.items():
            normalized_symbol = str(symbol).upper()
            qty = int(float(payload.get("qty", 0) or 0))
            if qty == 0:
                continue
            self.positions[normalized_symbol] = qty

            last_price = float(payload.get("last_price", payload.get("avg_entry_price", 0.0)) or 0.0)
            if last_price > 0:
                self.last_prices[normalized_symbol] = last_price

            timestamp = payload.get("timestamp")
            if timestamp is not None:
                try:
                    self.last_timestamps[normalized_symbol] = int(float(timestamp))
                except (TypeError, ValueError):
                    pass

    def snapshot(self) -> Dict[str, int]:
        return {k: v for k, v in self.positions.items() if v != 0}
