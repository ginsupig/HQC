"""
Tick-to-bar resampler.

The Kalman pairs strategy was validated by walk-forward on Alpaca
1-minute OHLCV history, which backtest_runner replays as 4 synthetic
ticks per bar (open, low/high, high/low, close — see
backtest_runner._bar_to_ticks). The live Alpaca websocket feed,
however, emits every raw trade — hundreds per minute for liquid
names. The Kalman filter adds process noise once per tick, so feeding
it raw ticks makes it adapt ~100x faster than it did in the backtest:
beta tracks the instantaneous price ratio, innovations collapse, and
the z-score never reaches the entry threshold. The strategy looks
alive but never trades.

TickResampler closes that gap. It subscribes to raw EventType.TICK,
buckets trades into fixed-length bars per symbol, and on each bar
boundary emits EventType.BAR_TICK events in exactly the shape
_bar_to_ticks produces (4 ticks per bar, O/L/H/C path, volume/4,
15s-spaced synthetic timestamps). Strategies that need bar cadence
subscribe to BAR_TICK; the live tick stream they see is then
identical in cadence and shape to the validated backtest.

A bar is finalized when the first trade of the *next* bar arrives,
so emission lags real time by up to one bar — acceptable for a
strategy whose holds run minutes to hours.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

from core.engine.event_bus import Event, EventBus, EventType

logger = logging.getLogger("TickResampler")


class TickResampler:
    def __init__(
        self,
        bus: EventBus,
        symbols: Iterable[str],
        bar_seconds: int = 60,
        ticks_per_bar: int = 4,
    ) -> None:
        self.bus = bus
        self.symbols = {str(s).upper() for s in symbols}
        self.bar_ms = int(bar_seconds) * 1000
        self.ticks_per_bar = max(1, int(ticks_per_bar))
        # symbol -> {key, o, h, l, c, v}
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._bars_emitted = 0
        self.bus.subscribe(EventType.TICK, self.on_tick)

    async def on_tick(self, event: Event) -> None:
        payload = event.payload or {}
        symbol = str(payload.get("ticker") or payload.get("symbol") or "").upper()
        if symbol not in self.symbols:
            return
        try:
            price = float(payload.get("price"))
            volume = float(payload.get("volume", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        if price <= 0:
            return
        ts_ms = self._coerce_ts(payload.get("timestamp"))
        if ts_ms is None:
            return

        bar_key = ts_ms // self.bar_ms
        bucket = self._buckets.get(symbol)

        if bucket is None:
            self._buckets[symbol] = self._new_bucket(bar_key, price, volume)
            return

        if bar_key > bucket["key"]:
            # Previous bar is complete — emit it, then open a fresh bucket.
            self._emit_bar(symbol, bucket)
            self._buckets[symbol] = self._new_bucket(bar_key, price, volume)
            return

        if bar_key < bucket["key"]:
            # Out-of-order / stale tick — drop rather than corrupt the bar.
            return

        # Same bar: fold the trade in.
        if price > bucket["h"]:
            bucket["h"] = price
        if price < bucket["l"]:
            bucket["l"] = price
        bucket["c"] = price
        bucket["v"] += volume

    def _emit_bar(self, symbol: str, bucket: Dict[str, float]) -> None:
        """Emit one finalized bar as `ticks_per_bar` BAR_TICK events,
        replicating backtest_runner._bar_to_ticks exactly."""
        o, h, l, c = bucket["o"], bucket["h"], bucket["l"], bucket["c"]
        per_tick_volume = max(1.0, bucket["v"] / self.ticks_per_bar)
        base_ts = int(bucket["key"]) * self.bar_ms
        step = self.bar_ms // self.ticks_per_bar
        # Same intrabar path heuristic as _bar_to_ticks: up bars walk
        # open->low->high->close, down bars open->high->low->close.
        path = [o, l, h, c] if c >= o else [o, h, l, c]
        for i, px in enumerate(path):
            self.bus.publish(
                Event(
                    type=EventType.BAR_TICK,
                    payload={
                        "ticker": symbol,
                        "symbol": symbol,
                        "price": float(px),
                        "volume": float(per_tick_volume),
                        "timestamp": base_ts + (i * step),
                    },
                )
            )
        self._bars_emitted += 1

    @staticmethod
    def _new_bucket(key: int, price: float, volume: float) -> Dict[str, float]:
        return {"key": float(key), "o": price, "h": price, "l": price, "c": price, "v": volume}

    @staticmethod
    def _coerce_ts(value: object) -> Optional[int]:
        try:
            ts = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        # Normalize seconds -> milliseconds if a 10-digit epoch slips through.
        if ts < 10_000_000_000:
            ts *= 1000
        return ts

    def stats(self) -> Dict[str, int]:
        return {"bars_emitted": self._bars_emitted, "open_buckets": len(self._buckets)}
