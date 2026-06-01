"""Portfolio-level risk for a multi-pair Kalman basket.

Three pieces, deliberately separated so each is independently testable:

  - allocator.BasketAllocator  — sizes each pair's per-leg notional to fit the
    account (equity x max_gross_leverage) and enforces a per-symbol exposure
    cap *statically*, which — because each pair trades a fixed notional — is a
    hard pre-trade guarantee rather than an after-the-fact halt.
  - pnl_ledger.PnLLedger        — signed average-cost realized-PnL + live
    position/exposure book, fed from ORDER_FILL events (handles cumulative
    partial fills).
  - portfolio_risk_monitor.PortfolioRiskMonitor — wires the ledger to the event
    bus and halts the basket on a real daily-PnL loss, a per-symbol net-exposure
    breach, or a gross-exposure breach.
"""
from risk.portfolio.allocator import (  # noqa: F401
    AllocationResult,
    BasketAllocator,
    PairSpec,
    allocate,
)
from risk.portfolio.pnl_ledger import PnLLedger  # noqa: F401
from risk.portfolio.portfolio_risk_monitor import PortfolioRiskMonitor  # noqa: F401
