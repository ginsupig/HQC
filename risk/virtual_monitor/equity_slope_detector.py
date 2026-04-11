from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from core.engine.event_bus import EventBus, Event, EventType
from core.feedback.unified_logger import UnifiedFeedbackLogger

logger = logging.getLogger("EquitySlopeDetector")


@dataclass
class VirtualPosition:
    """
    Internal inventory record for realized PnL accounting.
    qty:
      > 0 long
      < 0 short
      = 0 flat
    """
    asset: str
    qty: int = 0
    avg_price: float = 0.0


class VirtualEquitySlopeDetector:
    """
    Continuous background equity-curve monitor.

    Tracks:
    - realized PnL from ORDER_FILL lifecycle events
    - virtual equity
    - high-water mark and drawdown
    - trailing normalized OLS slope of equity history

    Emits:
    - EQUITY_UPDATE event for governance / state machine
    - outcomes.jsonl snapshots for MegaMind-compatible monitoring
    """

    def __init__(
        self,
        bus: EventBus,
        initial_capital: float = 100000.0,
        slope_lookback: int = 20,
        system_name: str = "HQC",
        arm: str = "equities",
        env: str = "paper",
    ) -> None:
        self.bus = bus
        self.initial_capital = float(initial_capital)
        self.current_equity = float(initial_capital)
        self.high_water_mark = float(initial_capital)
        self.slope_lookback = int(slope_lookback)

        self.positions: Dict[str, VirtualPosition] = {}
        self.equity_history: Deque[float] = deque(maxlen=self.slope_lookback)
        self.equity_history.append(self.initial_capital)

        self.feedback = UnifiedFeedbackLogger(
            root="state/feedback",
            system_name=system_name,
            arm=arm,
            env=env,
        )

        self.bus.subscribe(EventType.ORDER_FILL, self.on_order_fill)

    def _calculate_realized_pnl(
        self,
        asset: str,
        action: str,
        fill_qty: int,
        fill_price: float,
    ) -> float:
        """
        Maintain an internal ledger and compute realized PnL for the fill.

        Action interpretation:
        - BUY / BUY_TO_OPEN increase long or reduce short
        - SELL / SELL_TO_OPEN / SELL_SHORT increase short or reduce long
        - BUY_TO_COVER reduces short
        """
        asset = str(asset).upper()
        action = str(action).upper()
        fill_qty = int(fill_qty)
        fill_price = float(fill_price)

        if asset not in self.positions:
            self.positions[asset] = VirtualPosition(asset=asset)

        pos = self.positions[asset]
        realized_pnl = 0.0

        if action in {"BUY", "BUY_TO_OPEN", "BUY_TO_COVER"}:
            trade_qty = fill_qty
        elif action in {"SELL", "SELL_TO_OPEN", "SELL_SHORT"}:
            trade_qty = -fill_qty
        else:
            logger.warning("Unknown fill action for %s: %s", asset, action)
            return 0.0

        # Same-direction increase or brand-new position
        if pos.qty == 0 or (pos.qty > 0 and trade_qty > 0) or (pos.qty < 0 and trade_qty < 0):
            total_cost = (abs(pos.qty) * pos.avg_price) + (abs(trade_qty) * fill_price)
            pos.qty += trade_qty
            pos.avg_price = total_cost / abs(pos.qty) if pos.qty != 0 else 0.0
            return 0.0

        # Opposite-direction trade: reduce or flip position
        closing_qty = min(abs(pos.qty), abs(trade_qty))
        direction = 1 if pos.qty > 0 else -1
        realized_pnl = (fill_price - pos.avg_price) * closing_qty * direction

        old_qty = pos.qty
        pos.qty += trade_qty

        # Fully flat
        if pos.qty == 0:
            pos.avg_price = 0.0
            return realized_pnl

        # Flipped through zero: reset avg price to new trade price
        if (old_qty > 0 and pos.qty < 0) or (old_qty < 0 and pos.qty > 0):
            pos.avg_price = fill_price

        return realized_pnl

    async def on_order_fill(self, event: Event) -> None:
        """
        Intercept execution fills, calculate PnL, update virtual equity,
        and emit governance signals.
        """
        payload = event.payload or {}

        asset = payload.get("asset") or payload.get("symbol")
        action = payload.get("action") or payload.get("side") or "UNKNOWN"
        fill_qty = payload.get("fill_qty", payload.get("filled_qty", 0))
        fill_price = payload.get("fill_price", payload.get("entry_price", 0.0))
        decision_id = payload.get("decision_id")
        strategy = payload.get("strategy", "Unknown")
        order_id = payload.get("order_id")

        try:
            fill_qty = int(float(fill_qty))
            fill_price = float(fill_price)
        except (TypeError, ValueError):
            return

        if not asset or fill_qty == 0 or fill_price <= 0:
            return

        realized_pnl = self._calculate_realized_pnl(
            asset=str(asset).upper(),
            action=str(action).upper(),
            fill_qty=fill_qty,
            fill_price=fill_price,
        )

        # Entry / inventory-adjustment only: no realized PnL yet
        if realized_pnl == 0.0:
            return

        self.current_equity += realized_pnl
        self.equity_history.append(self.current_equity)

        if self.current_equity > self.high_water_mark:
            self.high_water_mark = self.current_equity

        drawdown_pct = (
            (self.current_equity - self.high_water_mark) / self.high_water_mark
            if self.high_water_mark > 0
            else 0.0
        )
        slope_pct = self._calculate_normalized_slope()

        logger.info(
            "[VIRTUAL EQUITY] Equity=$%.2f RealizedPnL=$%.2f Drawdown=%.2f%% TrailingSlope=%.3f%%",
            self.current_equity,
            realized_pnl,
            drawdown_pct * 100.0,
            slope_pct * 100.0,
        )

        self.feedback.write_outcome(
            {
                "decision_id": decision_id,
                "order_id": order_id,
                "status": "virtual_equity_update",
                "symbol": str(asset).upper(),
                "strategy": strategy,
                "side": str(action).upper(),
                "qty": fill_qty,
                "filled_qty": fill_qty,
                "fill_price": fill_price,
                "gross_pnl": round(realized_pnl, 2),
                "net_pnl": round(realized_pnl, 2),
                "meta": {
                    "virtual_equity": round(self.current_equity, 2),
                    "drawdown_pct": round(drawdown_pct, 6),
                    "virtual_slope": round(slope_pct, 6),
                },
            }
        )

        update_event = Event(
            type=EventType.EQUITY_UPDATE,
            payload={
                "current_equity": self.current_equity,
                "drawdown_pct": drawdown_pct,
                "virtual_slope": slope_pct,
            },
        )
        self.bus.publish(update_event)

    def _calculate_normalized_slope(self) -> float:
        """
        OLS slope of trailing equity history, normalized by average equity.
        """
        history_array = np.asarray(self.equity_history, dtype=float)

        if len(history_array) < 5:
            return 0.0

        x = np.arange(len(history_array), dtype=float)
        y = history_array

        try:
            raw_slope, _ = np.polyfit(x, y, 1)
        except Exception:
            return 0.0

        mean_equity = float(np.mean(history_array))
        if mean_equity == 0:
            return 0.0

        return float(raw_slope / mean_equity)

    def snapshot(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "current_equity": self.current_equity,
            "high_water_mark": self.high_water_mark,
            "drawdown_pct": (
                (self.current_equity - self.high_water_mark) / self.high_water_mark
                if self.high_water_mark > 0
                else 0.0
            ),
            "virtual_slope": self._calculate_normalized_slope(),
            "open_positions": {
                sym: {"qty": pos.qty, "avg_price": pos.avg_price}
                for sym, pos in self.positions.items()
                if pos.qty != 0
            },
        }

    def seed_positions(self, positions: Dict[str, Dict[str, float | int]], current_equity: Optional[float] = None) -> None:
        self.positions.clear()
        for symbol, payload in positions.items():
            normalized_symbol = str(symbol).upper()
            qty = int(float(payload.get("qty", 0) or 0))
            if qty == 0:
                continue
            avg_entry_price = float(payload.get("avg_entry_price", 0.0) or 0.0)
            self.positions[normalized_symbol] = VirtualPosition(
                asset=normalized_symbol,
                qty=qty,
                avg_price=avg_entry_price,
            )

        if current_equity is not None and current_equity > 0:
            self.current_equity = float(current_equity)
            self.high_water_mark = max(self.high_water_mark, self.current_equity)
            self.equity_history.clear()
            self.equity_history.append(self.current_equity)


if __name__ == "__main__":
    async def run_virtual_equity_test() -> None:
        print("Initializing Virtual Equity Slope Detector Test...")

        bus = EventBus()
        await bus.start()

        detector = VirtualEquitySlopeDetector(
            bus,
            initial_capital=100000.0,
            slope_lookback=10,
            system_name="HQC",
            arm="equities",
            env="paper",
        )

        def mock_trade(action: str, qty: int, price: float) -> Event:
            return Event(
                type=EventType.ORDER_FILL,
                payload={
                    "asset": "SPY",
                    "action": action,
                    "fill_qty": qty,
                    "fill_price": price,
                    "strategy": "TEST",
                },
            )

        print("\n[SYSTEM] Simulating a winning streak...")
        bus.publish(mock_trade("BUY", 100, 500.0))
        bus.publish(mock_trade("SELL", 100, 510.0))

        bus.publish(mock_trade("BUY", 100, 510.0))
        bus.publish(mock_trade("SELL", 100, 515.0))

        await asyncio.sleep(0.1)
        print(f"Current Virtual Slope: {detector._calculate_normalized_slope() * 100:.3f}%")

        print("\n[SYSTEM] Simulating a losing streak...")
        bus.publish(mock_trade("BUY", 100, 515.0))
        bus.publish(mock_trade("SELL", 100, 490.0))

        bus.publish(mock_trade("BUY", 100, 490.0))
        bus.publish(mock_trade("SELL", 100, 470.0))

        await asyncio.sleep(0.1)

        print("\nSnapshot:")
        print(detector.snapshot())

        await bus.stop()

    asyncio.run(run_virtual_equity_test())