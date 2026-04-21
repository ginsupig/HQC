from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from core.engine.event_bus import Event, EventBus, EventType

logger = logging.getLogger("RiskPositionSizer")


class DynamicRiskSizer:
    """
    Adaptive, rank-aware position sizer.

    Pipeline contract:
    - consumes ORDER_CREATE events with stage == "RANKED"
    - requires ranker approval
    - computes effective dollar risk using two-way multipliers
    - republishes ORDER_CREATE with stage == "SIZED"

    Sizing philosophy:
    - neutral baseline from account_equity * base_risk_pct
    - size DOWN in bad regime / weak liquidity / poor sector / crowded portfolio
    - size UP modestly in favorable regime / strong sector / strong rank / clean liquidity
    - always bounded by hard risk and position caps
    """

    def __init__(
        self,
        bus: EventBus,
        account_equity: float,
        base_risk_pct: float = 0.01,
        max_position_pct: float = 0.20,
        min_effective_risk_mult: float = 0.35,
        max_effective_risk_mult: float = 1.35,
        max_risk_pct_per_trade: float = 0.015,
        min_trade_dollars: float = 100.0,
        min_share_quantity: int = 1,
        max_concurrent_positions: int = 5,
    ) -> None:
        self.bus = bus
        self.account_equity = float(account_equity)
        self.base_risk_pct = float(base_risk_pct)
        self.max_position_pct = float(max_position_pct)

        self.min_effective_risk_mult = float(min_effective_risk_mult)
        self.max_effective_risk_mult = float(max_effective_risk_mult)
        self.max_risk_pct_per_trade = float(max_risk_pct_per_trade)

        self.min_trade_dollars = float(min_trade_dollars)
        self.min_share_quantity = int(min_share_quantity)
        self.max_concurrent_positions = int(max_concurrent_positions)

        self._open_positions: Dict[str, int] = {}

        self.bus.subscribe(EventType.ORDER_CREATE, self.on_order_create)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_order_fill)

    def update_equity(self, new_equity: float) -> None:
        self.account_equity = float(new_equity)
        logger.debug("Account equity updated to $%.2f", self.account_equity)

    def seed_positions(self, positions: Dict[str, int]) -> None:
        self._open_positions = {k: v for k, v in positions.items() if v > 0}
        logger.info("[SIZER] Seeded %d open positions from reconciliation.", len(self._open_positions))

    async def on_order_fill(self, event: Event) -> None:
        payload = event.payload or {}
        asset = str(payload.get("asset") or payload.get("symbol") or "").upper()
        action = str(payload.get("action") or payload.get("side") or "").upper()
        filled_qty = float(payload.get("filled_qty") or payload.get("fill_qty") or 0.0)
        if not asset or filled_qty <= 0:
            return
        if action in {"BUY", "BUY_TO_OPEN"}:
            self._open_positions[asset] = int(self._open_positions.get(asset, 0) + filled_qty)
        elif action in {"SELL", "SELL_TO_CLOSE", "BUY_TO_COVER"}:
            remaining = int(self._open_positions.get(asset, 0) - filled_qty)
            if remaining <= 0:
                self._open_positions.pop(asset, None)
            else:
                self._open_positions[asset] = remaining

    async def on_order_create(self, event: Event) -> None:
        payload = dict(event.payload or {})

        # Only size ranker-approved intents.
        if payload.get("stage") != "RANKED":
            return
        if not payload.get("approved_by_ranker", False):
            return
        if payload.get("shares") is not None:
            return

        asset = str(payload.get("asset", "")).upper()
        action = str(payload.get("action", "")).upper()
        strategy = payload.get("strategy", "Unknown")

        entry_price = self._to_float(payload.get("reference_price"))
        stop_loss_price = self._to_float(payload.get("stop_loss_price"))
        rank_score = self._to_float(payload.get("rank_score"), 0.0)

        if not asset or not action or entry_price is None or stop_loss_price is None:
            logger.error("Malformed ranked order payload from %s. Dropping execution.", strategy)
            return

        is_opening = action in {"BUY", "BUY_TO_OPEN", "SELL_SHORT", "SELL_TO_OPEN"}
        if is_opening and asset not in self._open_positions and len(self._open_positions) >= self.max_concurrent_positions:
            logger.warning(
                "[SIZER] Concurrent position limit reached (%d/%d). Dropping %s %s.",
                len(self._open_positions),
                self.max_concurrent_positions,
                asset,
                action,
            )
            return

        if entry_price <= 0 or stop_loss_price <= 0:
            logger.error("Invalid entry/stop values for %s. Trade aborted.", asset)
            return

        per_share_risk = abs(entry_price - stop_loss_price)
        if per_share_risk <= 0:
            logger.error("Invalid stop distance for %s. Trade aborted.", asset)
            return

        # Pull optional context from payload / rank components.
        rank_components: Dict[str, Any] = payload.get("rank_components", {}) or {}
        liquidity_score = self._first_float(
            payload,
            rank_components,
            keys=("liquidity_score",),
            default=0.50,
        )
        spread_bps = self._first_float(
            payload,
            rank_components,
            keys=("spread_bps",),
            default=0.0,
        )
        regime_label = str(
            payload.get("regime")
            or payload.get("regime_label")
            or rank_components.get("regime")
            or rank_components.get("regime_label")
            or ""
        ).strip().lower()
        sector_strength = self._first_float(
            payload,
            rank_components,
            keys=("sector_strength", "sector_score", "sector_mult_input"),
            default=0.0,
        )
        correlation_load = self._first_float(
            payload,
            rank_components,
            keys=("correlation_load", "portfolio_correlation", "crowding_score"),
            default=0.0,
        )
        rs_value = self._first_float(
            payload,
            rank_components,
            keys=("rs",),
            default=0.0,
        )
        rvol_value = self._first_float(
            payload,
            rank_components,
            keys=("rvol",),
            default=1.0,
        )

        regime_mult = self._regime_multiplier(regime_label)
        sector_mult = self._sector_multiplier(sector_strength)
        rank_mult = self._rank_multiplier(rank_score)
        liquidity_mult = self._liquidity_multiplier(liquidity_score, spread_bps)
        rs_mult = self._relative_strength_multiplier(action, rs_value)
        rvol_mult = self._relative_volume_multiplier(rvol_value)
        correlation_mult = self._correlation_multiplier(correlation_load)

        raw_effective_mult = (
            regime_mult
            * sector_mult
            * rank_mult
            * liquidity_mult
            * rs_mult
            * rvol_mult
            * correlation_mult
        )
        effective_mult = self._clamp(
            raw_effective_mult,
            self.min_effective_risk_mult,
            self.max_effective_risk_mult,
        )

        base_dollar_risk = self.account_equity * self.base_risk_pct
        max_dollar_risk = self.account_equity * self.max_risk_pct_per_trade
        effective_dollar_risk = min(base_dollar_risk * effective_mult, max_dollar_risk)

        raw_shares = effective_dollar_risk / per_share_risk
        max_position_dollars = self.account_equity * self.max_position_pct
        max_position_shares = max(1, math.floor(max_position_dollars / entry_price))
        final_shares = min(math.floor(raw_shares), max_position_shares)

        if final_shares < self.min_share_quantity:
            logger.warning(
                "Risk budget too small for %s under current stop distance. shares=%s risk_mult=%.3f",
                asset,
                final_shares,
                effective_mult,
            )
            return

        capital_allocated = final_shares * entry_price
        if capital_allocated < self.min_trade_dollars:
            logger.warning(
                "Trade value below minimum threshold for %s. allocated=$%.2f < $%.2f",
                asset,
                capital_allocated,
                self.min_trade_dollars,
            )
            return

        risk_dollars = final_shares * per_share_risk

        sized_payload = {
            **payload,
            "stage": "SIZED",
            "shares": final_shares,
            "entry_price": entry_price,
            "stop_loss": stop_loss_price,
            "stop_loss_price": stop_loss_price,
            "capital_allocated": round(capital_allocated, 2),
            "risk_dollars": round(risk_dollars, 2),
            "base_dollar_risk": round(base_dollar_risk, 2),
            "effective_dollar_risk": round(effective_dollar_risk, 2),
            "effective_risk_mult": round(effective_mult, 4),
            "sizing_components": {
                "regime_mult": round(regime_mult, 4),
                "sector_mult": round(sector_mult, 4),
                "rank_mult": round(rank_mult, 4),
                "liquidity_mult": round(liquidity_mult, 4),
                "rs_mult": round(rs_mult, 4),
                "rvol_mult": round(rvol_mult, 4),
                "correlation_mult": round(correlation_mult, 4),
                "raw_effective_mult": round(raw_effective_mult, 4),
            },
            "status": "READY_FOR_BROKER",
        }

        logger.info(
            "[SIZED] %s %s x%d entry=%.2f stop=%.2f risk=$%.2f mult=%.3f rank=%.2f regime=%.3f sector=%.3f liq=%.3f rs=%.3f rvol=%.3f corr=%.3f",
            asset,
            action,
            final_shares,
            entry_price,
            stop_loss_price,
            risk_dollars,
            effective_mult,
            rank_score,
            regime_mult,
            sector_mult,
            liquidity_mult,
            rs_mult,
            rvol_mult,
            correlation_mult,
        )

        self.bus.publish(
            Event(
                type=EventType.ORDER_CREATE,
                payload=sized_payload,
            )
        )

    def _regime_multiplier(self, regime_label: str) -> float:
        if not regime_label:
            return 1.00

        label = regime_label.lower()

        if any(x in label for x in ("shock", "panic", "crash", "high volatility")):
            return 0.55
        if any(x in label for x in ("range", "range bound", "chop", "sideways")):
            return 0.90
        if any(x in label for x in ("trend", "directional", "breakout")):
            return 1.15
        if any(x in label for x in ("low volatility", "calm")):
            return 1.05
        return 1.00

    def _sector_multiplier(self, sector_strength: float) -> float:
        # Expected rough range: -1 .. +1
        return self._clamp(1.00 + (sector_strength * 0.15), 0.80, 1.15)

    def _rank_multiplier(self, rank_score: float) -> float:
        # Neutral around ~5.0, modest uplift/downgrade only.
        centered = rank_score - 5.0
        return self._clamp(1.00 + (centered * 0.04), 0.85, 1.20)

    def _liquidity_multiplier(self, liquidity_score: float, spread_bps: float) -> float:
        base = self._clamp(0.75 + (liquidity_score * 0.40), 0.70, 1.10)

        # Small additional spread penalty if tape is sloppy.
        if spread_bps >= 25:
            base *= 0.80
        elif spread_bps >= 18:
            base *= 0.90
        elif spread_bps <= 5:
            base *= 1.03

        return self._clamp(base, 0.70, 1.10)

    def _relative_strength_multiplier(self, action: str, rs_value: float) -> float:
        action = action.upper()

        aligned_rs = rs_value
        if action in {"SELL", "SELL_SHORT", "SELL_TO_OPEN"}:
            aligned_rs = -rs_value

        return self._clamp(1.00 + (aligned_rs * 25.0), 0.85, 1.10)

    def _relative_volume_multiplier(self, rvol_value: float) -> float:
        if rvol_value <= 0:
            return 0.85
        if rvol_value < 0.8:
            return 0.85
        if rvol_value < 1.0:
            return 0.93
        if rvol_value < 1.5:
            return 1.00
        if rvol_value < 2.5:
            return 1.08
        return 1.12

    def _correlation_multiplier(self, correlation_load: float) -> float:
        # Expected rough range: 0 .. 1 where higher means more crowding/correlation.
        return self._clamp(1.00 - (correlation_load * 0.35), 0.65, 1.00)

    @staticmethod
    def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _first_float(
        self,
        payload: Dict[str, Any],
        nested: Dict[str, Any],
        keys: tuple[str, ...],
        default: float,
    ) -> float:
        for key in keys:
            if key in payload:
                parsed = self._to_float(payload.get(key), None)
                if parsed is not None:
                    return parsed
            if key in nested:
                parsed = self._to_float(nested.get(key), None)
                if parsed is not None:
                    return parsed
        return float(default)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))