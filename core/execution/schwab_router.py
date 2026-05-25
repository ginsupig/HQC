from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import abc


@dataclass
class BrokerAccountSnapshot:
    account_id: str
    equity: float
    buying_power: float
    cash: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerOrderResult:
    ok: bool
    order_id: Optional[str] = None
    status: str = ""
    error: Optional[str] = None
    filled_qty: float = 0.0
    fill_price: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


class SchwabBrokerClientProtocol(abc.ABC):
    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def get_account_snapshot(self) -> BrokerAccountSnapshot: ...

    @abc.abstractmethod
    async def place_equity_order(self, **kwargs: Any) -> BrokerOrderResult: ...

    @abc.abstractmethod
    async def cancel_order(self, *, order_id: str) -> BrokerOrderResult: ...
