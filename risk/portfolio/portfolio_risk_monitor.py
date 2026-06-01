"""PortfolioRiskMonitor — basket-level kill switches built on PnLLedger.

Complements main_pairs.PairsRiskMonitor (which owns the per-pair beta-drift
kill). This monitor owns the *portfolio* safeguards that only matter once more
than one pair trades at once:

  - **Daily realized-PnL kill** — halts the basket when the day's realized PnL
    falls below ``-daily_loss_pct_kill * equity``. This replaces the documented
    stub in main_pairs with a real ledger-backed number.
  - **Per-symbol net-exposure kill** — halts if any symbol's signed dollar
    exposure exceeds ``max_symbol_pct * equity``. The BasketAllocator makes this
    statically impossible at steady state; this catches drift from partial
    fills, leg imbalances, or a missed exit.
  - **Gross-exposure kill** — halts if total gross exposure exceeds
    ``max_gross_leverage * equity``.

Each cap is optional: pass ``None`` to disable that check. The daily-PnL kill is
the one that runs even when portfolio allocation is off.

A halt publishes SYSTEM_SHUTDOWN once (idempotent); the runtime's existing
shutdown watcher flattens and stops, exactly as for the beta-drift kill.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from core.engine.event_bus import Event, EventBus, EventType
from risk.portfolio.pnl_ledger import PnLLedger

logger = logging.getLogger("PortfolioRiskMonitor")

_CLOSED_STATUSES = {"CANCELED", "CANCELLED", "REJECTED", "ERROR", "NEW", "PENDING", "ACCEPTED"}


class PortfolioRiskMonitor:
    def __init__(
        self,
        bus: EventBus,
        equity: float,
        daily_loss_pct_kill: Optional[float] = 0.02,
        max_symbol_pct: Optional[float] = None,
        max_gross_leverage: Optional[float] = None,
        check_interval_sec: float = 5.0,
        ledger: Optional[PnLLedger] = None,
    ) -> None:
        self.bus = bus
        self.equity = float(equity)
        self.daily_loss_pct_kill = daily_loss_pct_kill
        self.max_symbol_pct = max_symbol_pct
        self.max_gross_leverage = max_gross_leverage
        self.check_interval_sec = float(check_interval_sec)
        self.ledger = ledger or PnLLedger()

        self._halted = False
        self._current_day = datetime.now(timezone.utc).date()
        self._day_start_realized = self.ledger.realized_pnl
        self._last_tick_check = 0.0

        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        self.bus.subscribe(EventType.TICK, self.on_tick)
        self.bus.subscribe(EventType.BAR_TICK, self.on_tick)

    # ------------------------------------------------------------------ #
    @property
    def daily_realized_pnl(self) -> float:
        return self.ledger.realized_pnl - self._day_start_realized

    def _roll_day_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._current_day:
            self._current_day = today
            self._day_start_realized = self.ledger.realized_pnl
            logger.info("[portfolio] new trading day %s; daily PnL counter reset.", today)

    async def on_fill(self, event: Event) -> None:
        if self._halted:
            return
        payload = event.payload or {}
        status = str(payload.get("status", "")).upper()
        if status in _CLOSED_STATUSES:
            return
        symbol = str(payload.get("asset") or payload.get("symbol") or "").upper()
        if not symbol:
            return
        action = str(payload.get("action") or payload.get("side") or "").upper()
        try:
            filled = float(payload.get("filled_qty", payload.get("fill_qty", 0)) or 0)
            price = float(payload.get("fill_price", payload.get("entry_price", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return
        if filled <= 0 or price <= 0:
            return

        self._roll_day_if_needed()
        self.ledger.on_fill(
            order_id=str(payload.get("order_id") or payload.get("decision_id") or ""),
            symbol=symbol,
            action=action,
            cumulative_filled_qty=filled,
            price=price,
        )
        self._check_and_maybe_halt()

    async def on_tick(self, event: Event) -> None:
        if self._halted:
            return
        payload = event.payload or {}
        ticker = str(payload.get("ticker") or payload.get("symbol") or "").upper()
        try:
            price = float(payload.get("price"))
        except (TypeError, ValueError):
            return
        if ticker and price > 0:
            self.ledger.mark(ticker, price)

        loop = asyncio.get_running_loop()
        now = loop.time()
        if (now - self._last_tick_check) < self.check_interval_sec:
            return
        self._last_tick_check = now
        self._roll_day_if_needed()
        self._check_and_maybe_halt()

    # ------------------------------------------------------------------ #
    def _check_and_maybe_halt(self) -> None:
        if self._halted or self.equity <= 0:
            return

        if self.daily_loss_pct_kill is not None:
            loss_limit = -abs(self.daily_loss_pct_kill) * self.equity
            if self.daily_realized_pnl <= loss_limit:
                self._halt(
                    f"daily realized PnL ${self.daily_realized_pnl:,.0f} <= limit "
                    f"${loss_limit:,.0f} ({self.daily_loss_pct_kill:.1%} of equity)"
                )
                return

        if self.max_symbol_pct is not None:
            symbol_limit = self.max_symbol_pct * self.equity
            for sym, exposure in self.ledger.symbol_exposures().items():
                if abs(exposure) > symbol_limit + 1e-6:
                    self._halt(
                        f"symbol {sym} net exposure ${exposure:,.0f} exceeds "
                        f"${symbol_limit:,.0f} ({self.max_symbol_pct:.0%} of equity)"
                    )
                    return

        if self.max_gross_leverage is not None:
            gross_limit = self.max_gross_leverage * self.equity
            gross = self.ledger.gross_exposure()
            if gross > gross_limit + 1e-6:
                self._halt(
                    f"gross exposure ${gross:,.0f} exceeds ${gross_limit:,.0f} "
                    f"({self.max_gross_leverage:g}x equity)"
                )

    def _halt(self, reason: str) -> None:
        if self._halted:
            return
        self._halted = True
        logger.error("[KILL SWITCH] portfolio: %s", reason)
        self.bus.publish(
            Event(
                type=EventType.SYSTEM_SHUTDOWN,
                payload={"source": "PortfolioRiskMonitor", "reason": reason},
            )
        )
