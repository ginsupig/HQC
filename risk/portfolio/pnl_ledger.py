"""Signed average-cost PnL + position ledger, fed from ORDER_FILL events.

The deployed runtime previously had no realized-PnL accounting — the daily-loss
kill switch in main_pairs was a documented stub. This ledger fills that gap with
the standard signed average-cost method:

  - Each symbol holds a signed position ``qty`` (>0 long, <0 short) and an
    ``avg_cost``.
  - A fill in the same direction as the position updates the average cost.
  - A fill in the opposite direction realizes PnL on the closed quantity
    (``(price - avg_cost)`` for longs, ``(avg_cost - price)`` for shorts) and,
    if it over-closes, opens the remainder at the fill price.

It is robust to the broker router's fill semantics: ``filled_qty`` arrives as a
**cumulative** quantity per order (partial fills emit repeated updates), so the
ledger tracks the last cumulative quantity seen per ``order_id`` and applies only
the delta — repeated or out-of-order updates can't double-count.

Pure and synchronous: no event-bus coupling (that lives in
PortfolioRiskMonitor), so it is trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

_BUY_ACTIONS = {"BUY", "BUY_TO_COVER"}
_SELL_ACTIONS = {"SELL", "SELL_SHORT"}


@dataclass
class _Position:
    qty: float = 0.0       # signed
    avg_cost: float = 0.0


@dataclass
class PnLLedger:
    realized_pnl: float = 0.0
    _positions: Dict[str, _Position] = field(default_factory=dict)
    _order_cum_filled: Dict[str, float] = field(default_factory=dict)
    _last_price: Dict[str, float] = field(default_factory=dict)

    # --------------------------------------------------------------------- #
    @staticmethod
    def signed_delta(action: str, qty: float) -> float:
        a = (action or "").upper()
        if a in _BUY_ACTIONS:
            return float(qty)
        if a in _SELL_ACTIONS:
            return -float(qty)
        return 0.0

    def on_fill(
        self,
        *,
        order_id: Optional[str],
        symbol: str,
        action: str,
        cumulative_filled_qty: float,
        price: float,
    ) -> float:
        """Apply a fill update. ``cumulative_filled_qty`` is the order's total
        filled quantity so far (not the increment). Returns realized PnL from
        this update (0.0 if it opened/added or was a duplicate)."""
        symbol = symbol.upper()
        price = float(price)
        if price > 0:
            self._last_price[symbol] = price

        new_cum = float(cumulative_filled_qty or 0.0)
        key = order_id or f"{symbol}:{action}:noid"
        prev_cum = self._order_cum_filled.get(key, 0.0)
        delta = new_cum - prev_cum
        # Record the high-water cumulative qty even on no-op so a later
        # in-order update still computes the right delta.
        self._order_cum_filled[key] = max(prev_cum, new_cum)
        if delta <= 0 or price <= 0:
            return 0.0

        signed = self.signed_delta(action, delta)
        if signed == 0.0:
            return 0.0
        realized = self._apply(symbol, signed, price)
        self.realized_pnl += realized
        return realized

    def _apply(self, symbol: str, signed_delta: float, price: float) -> float:
        pos = self._positions.setdefault(symbol, _Position())
        realized = 0.0

        opening_or_adding = pos.qty == 0 or (pos.qty > 0) == (signed_delta > 0)
        if opening_or_adding:
            total = abs(pos.qty) + abs(signed_delta)
            pos.avg_cost = (
                (pos.avg_cost * abs(pos.qty) + price * abs(signed_delta)) / total
                if total > 0 else price
            )
            pos.qty += signed_delta
        else:
            closing = min(abs(pos.qty), abs(signed_delta))
            direction = 1.0 if pos.qty > 0 else -1.0
            realized = closing * (price - pos.avg_cost) * direction
            pos.qty += signed_delta
            if abs(signed_delta) > closing:
                # Flipped through zero: remainder opens a new position at price.
                pos.avg_cost = price
            elif abs(pos.qty) < 1e-9:
                pos.qty = 0.0
                pos.avg_cost = 0.0
        return realized

    # --------------------------------------------------------------------- #
    def position(self, symbol: str) -> float:
        pos = self._positions.get(symbol.upper())
        return pos.qty if pos else 0.0

    def mark(self, symbol: str, price: float) -> None:
        if price > 0:
            self._last_price[symbol.upper()] = float(price)

    def net_exposure(self, symbol: str, mark_price: Optional[float] = None) -> float:
        """Signed dollar exposure for one symbol (qty * mark). Falls back to the
        last seen price, then avg_cost, when no mark is supplied."""
        symbol = symbol.upper()
        pos = self._positions.get(symbol)
        if not pos or pos.qty == 0.0:
            return 0.0
        price = mark_price or self._last_price.get(symbol) or pos.avg_cost
        return pos.qty * float(price)

    def symbol_exposures(self) -> Dict[str, float]:
        return {s: self.net_exposure(s) for s, p in self._positions.items() if p.qty != 0.0}

    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.symbol_exposures().values())
