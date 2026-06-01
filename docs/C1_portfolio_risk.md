# C1 — Portfolio risk for a multi-pair basket

Trading more than one pair at once introduces exposure problems the single-pair
runtime never had: aggregate gross leverage, symbols shared across pairs, and a
basket-wide daily loss. `risk/portfolio/` adds three independently-testable
pieces that address them, wired into `main_pairs.py` behind a `portfolio:`
config block (off by default — a one-pair deployment is unchanged).

```
config portfolio: block
        │
        ▼
BasketAllocator  ── sizes each pair's per-leg notional (pre-trade, static)
        │
        ▼
USEquityKalmanPairsTrader (×N)  ── trade at the allocated notional
        │ ORDER_FILL
        ▼
PortfolioRiskMonitor ── PnLLedger → daily-PnL / per-symbol / gross kill switches
```

## 1. BasketAllocator (`allocator.py`)

Sizes each pair's per-leg dollar notional as:

```
notional_i = min(
    ceiling_i,                            # validated target_dollar_notional — never exceeded
    gross_budget_i / 2,                   # fit equity x max_gross_leverage
    equity x max_symbol_pct / count[s],   # tightest shared-symbol cap
)
```

Because each pair trades a **fixed** notional, this is a *static, pre-trade
guarantee*, not an after-the-fact halt:

- `sum(2 x notional_i) <= equity x max_gross_leverage`  (gross)
- for every symbol `s`, `sum_{pairs containing s} notional_i <= equity x max_symbol_pct`

The `ceiling` is the per-pair `target_dollar_notional` the pair was *validated*
at, so allocation only ever scales **down** — a pair is never traded larger than
its walk-forward evidence supports. The result records which constraint bound
each pair and warns if the basket is too large for the account (a pair sized
below `min_notional`).

## 2. PnLLedger (`pnl_ledger.py`)

A signed average-cost realized-PnL + position book, fed from `ORDER_FILL`
events. Replaces the documented daily-PnL **stub** with a real number.

- Same-direction fills update average cost; opposite-direction fills realize PnL
  on the closed quantity and open any remainder after a flip.
- `filled_qty` is **cumulative** per order (partial fills repeat), so the ledger
  tracks the last cumulative quantity per `order_id` and applies only the delta —
  repeated/stale updates can't double-count.
- Pure and synchronous (no bus coupling) → trivially unit-tested.

## 3. PortfolioRiskMonitor (`portfolio_risk_monitor.py`)

Wires the ledger to the `EventBus` and publishes `SYSTEM_SHUTDOWN` (once,
idempotent) on any breach. Complements `PairsRiskMonitor`'s per-pair beta-drift
kill.

| Kill switch | Trigger | Active when |
|---|---|---|
| Daily realized-PnL | day's realized PnL `<= -daily_loss_pct_kill x equity` | always |
| Per-symbol net exposure | any symbol `> max_symbol_pct x equity` | `portfolio.enabled` |
| Gross exposure | total gross `> max_gross_leverage x equity` | `portfolio.enabled` |

The allocator makes the per-symbol and gross caps statically satisfiable; the
monitor catches *drift* from partial fills, leg imbalances, or a missed exit.

## Config

```yaml
# config/pairs.yaml
portfolio:
  enabled: true
  equity: 450000          # buying power (<=0 -> risk.initial_capital)
  max_gross_leverage: 3   # comfortable at 2-3x
  max_symbol_pct: 0.5     # cap any single symbol at 50% of equity
  min_notional: 100
```

With `enabled: false` (default), notionals come straight from each pair's
`target_dollar_notional` and only the daily-PnL kill runs — i.e. exactly the
prior single-pair behaviour, now with a real PnL kill instead of a stub.

## Tests

- `tests/test_portfolio_allocator.py` — gross/symbol/ceiling binding, leverage
  relaxation, shared-symbol cap, invariants under a mixed basket, too-small
  account, zero-equity / empty safety.
- `tests/test_pnl_ledger.py` — long/short round trips, average cost, flip
  through zero, cumulative partial fills, net/gross exposure.
- `tests/test_portfolio_risk_monitor.py` — daily-PnL / per-symbol / gross kills
  against a live `EventBus`, within-limit no-op, idempotent halt.
