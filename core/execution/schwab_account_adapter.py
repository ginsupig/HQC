from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from core.execution.schwab_router import (
    BrokerAccountSnapshot,
    BrokerOrderResult,
    SchwabBrokerClientProtocol,
)

logger = logging.getLogger("SchwabAccountAdapter")


class SchwabAccountAdapter(SchwabBrokerClientProtocol):
    """
    Adapter boundary between HQC and your real Schwab API/session code.

    Replace the TODO sections with your approved Schwab auth + request layer.

    Expected integration points:
    - hash/resolve account number
    - bearer token refresh
    - GET account balances/positions
    - POST order
    - DELETE cancel order
    """

    def __init__(
        self,
        *,
        account_hash: str,
        http_client: Any,
        default_preview: bool = False,
    ) -> None:
        self.account_hash = str(account_hash)
        self.http = http_client
        self.default_preview = bool(default_preview)
        self._started = False

    async def start(self) -> None:
        self._started = True
        logger.info("SchwabAccountAdapter started for account_hash=%s", self.account_hash)

    async def stop(self) -> None:
        self._started = False
        logger.info("SchwabAccountAdapter stopped.")

    async def get_account_snapshot(self) -> BrokerAccountSnapshot:
        self._ensure_started()

        # TODO:
        # resp = await self.http.get_account(account_hash=self.account_hash)
        # Parse Schwab balances into:
        #   equity
        #   buying_power
        #   cash
        #
        # This fallback shape keeps the router/test path stable even before
        # the exact Schwab parser is dropped in.
        resp: Dict[str, Any] = await self.http.get_account(account_hash=self.account_hash)

        balances = resp.get("balances", {}) or {}
        equity = self._pick_float(balances, ["liquidationValue", "equity", "netLiquidationValue"], 0.0)
        buying_power = self._pick_float(balances, ["buyingPower", "dayTradingBuyingPower", "availableFunds"], 0.0)
        cash = self._pick_float(balances, ["cashBalance", "cashAvailableForTrading"], 0.0)

        return BrokerAccountSnapshot(
            account_id=self.account_hash,
            equity=equity,
            buying_power=buying_power,
            cash=cash,
            raw=resp,
        )

    async def place_equity_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        order_type: str,
        tif: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        preview: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> BrokerOrderResult:
        self._ensure_started()

        preview = bool(preview or self.default_preview)

        # IMPORTANT:
        # Do not guess the final Schwab payload in the router.
        # Build it here, where account-specific Schwab order wiring belongs.
        payload = self._build_equity_order_payload(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            tif=tif,
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=client_order_id,
            preview=preview,
            meta=meta or {},
        )

        try:
            raw = await self.http.place_order(
                account_hash=self.account_hash,
                payload=payload,
                preview=preview,
            )
        except Exception as exc:
            return BrokerOrderResult(
                ok=False,
                status="error",
                error=str(exc),
            )

        # TODO:
        # Replace with exact Schwab response parsing once your real HTTP layer is wired.
        order_id = (
            raw.get("orderId")
            or raw.get("id")
            or raw.get("order_id")
            or client_order_id
        )
        status = str(raw.get("status", "submitted")).lower()

        return BrokerOrderResult(
            ok=True,
            order_id=str(order_id) if order_id is not None else None,
            status=status,
            raw=raw,
            filled_qty=self._to_float(raw.get("filledQuantity", raw.get("filled_qty", 0.0)), 0.0),
            fill_price=self._to_float(raw.get("price", raw.get("fill_price", 0.0)), 0.0),
        )

    async def cancel_order(
        self,
        *,
        order_id: str,
    ) -> BrokerOrderResult:
        self._ensure_started()

        try:
            raw = await self.http.cancel_order(
                account_hash=self.account_hash,
                order_id=str(order_id),
            )
            return BrokerOrderResult(
                ok=True,
                order_id=str(order_id),
                status="cancelled",
                raw=raw if isinstance(raw, dict) else {"raw": raw},
            )
        except Exception as exc:
            return BrokerOrderResult(
                ok=False,
                order_id=str(order_id),
                status="cancel_error",
                error=str(exc),
            )

    def _build_equity_order_payload(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        order_type: str,
        tif: str,
        limit_price: Optional[float],
        stop_price: Optional[float],
        client_order_id: Optional[str],
        preview: bool,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Conservative broker-neutral internal shape.

        Your actual Schwab HTTP client can either:
        1. accept this shape directly and translate it internally, or
        2. replace this builder with exact Schwab schema once confirmed.
        """
        return {
            "session": "NORMAL",
            "duration": tif.upper(),
            "orderType": order_type.upper(),
            "orderStrategyType": "SINGLE",
            "clientOrderId": client_order_id,
            "preview": preview,
            "orderLegCollection": [
                {
                    "instruction": self._map_instruction(side),
                    "quantity": int(qty),
                    "instrument": {
                        "symbol": str(symbol).upper(),
                        "assetType": "EQUITY",
                    },
                }
            ],
            "price": round(float(limit_price), 4) if limit_price is not None else None,
            "stopPrice": round(float(stop_price), 4) if stop_price is not None else None,
            "meta": meta,
        }

    def _map_instruction(self, side: str) -> str:
        side = str(side).upper()
        mapping = {
            "BUY": "BUY",
            "SELL": "SELL",
            "SELL_SHORT": "SELL_SHORT",
            "BUY_TO_COVER": "BUY_TO_COVER",
        }
        if side not in mapping:
            raise ValueError(f"Unsupported side: {side}")
        return mapping[side]

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("SchwabAccountAdapter not started")

    @staticmethod
    def _pick_float(src: Dict[str, Any], keys: list[str], default: float) -> float:
        for key in keys:
            try:
                if key in src and src[key] is not None:
                    return float(src[key])
            except Exception:
                continue
        return default

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default