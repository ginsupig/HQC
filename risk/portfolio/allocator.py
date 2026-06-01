"""Basket allocator — per-pair sizing with a static shared-symbol guarantee.

A multi-pair basket has two exposure problems the single-pair runtime never
faced:

  1. **Gross.** N pairs at a fixed per-leg notional put ``sum(2 * notional)``
     dollars of gross exposure on the account. Eight pairs at $10k/leg is $160k
     gross — fine on a 3x account, not on a 1x one.
  2. **Shared symbols.** If JPM appears in both JPM/BAC and JPM/WFC, the basket
     can stack up to ``notional_JPM/BAC + notional_JPM/WFC`` of JPM on one side.

Because each pair trades a *fixed* notional, both can be solved **at allocation
time** rather than policed after the fact. The allocator picks each pair's
per-leg notional as::

    notional_i = min(
        ceiling_i,                              # never exceed the validated size
        gross_budget_i / 2,                     # fit the account's gross leverage
        equity * max_symbol_pct / count[s],     # for the tightest symbol s in i
    )

which guarantees, with no runtime intervention:

  - ``sum(2 * notional_i) <= equity * max_gross_leverage``     (gross)
  - for every symbol s, ``sum_{i contains s} notional_i <= equity * max_symbol_pct``

The ``ceiling`` is the per-pair ``target_dollar_notional`` the pair was
*validated* at — the allocator only ever scales **down** from it, never up, so a
pair is never traded larger than its walk-forward evidence supports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PairSpec:
    """One pair the allocator must size. ``ceiling_notional`` is the validated
    per-leg target_dollar_notional; ``weight`` lets an operator tilt the gross
    budget toward higher-conviction pairs (default equal-weight)."""

    y: str
    x: str
    ceiling_notional: float
    weight: float = 1.0

    @property
    def label(self) -> str:
        return f"{self.y.upper()}/{self.x.upper()}"

    @property
    def symbols(self) -> Tuple[str, str]:
        return (self.y.upper(), self.x.upper())


@dataclass
class AllocationResult:
    notionals: Dict[str, float]          # label -> per-leg dollar notional
    binding: Dict[str, str]              # label -> which constraint bound it
    per_symbol_notional: Dict[str, float]
    gross: float
    gross_limit: float
    warnings: List[str] = field(default_factory=list)

    def as_table(self) -> str:
        rows = ["  pair            notional   binding"]
        for label, notional in self.notionals.items():
            rows.append(f"  {label:<14} ${notional:>9,.0f}  {self.binding.get(label, '')}")
        rows.append(f"  -- gross ${self.gross:,.0f} / limit ${self.gross_limit:,.0f}")
        return "\n".join(rows)


def allocate(
    pairs: List[PairSpec],
    equity: float,
    max_gross_leverage: float,
    max_symbol_pct: float = 1.0,
    min_notional: float = 100.0,
) -> AllocationResult:
    """Size each pair's per-leg notional. See module docstring for guarantees."""
    notionals: Dict[str, float] = {}
    binding: Dict[str, str] = {}
    warnings: List[str] = []

    if not pairs:
        return AllocationResult({}, {}, {}, 0.0, max(0.0, equity * max_gross_leverage), [])

    if equity <= 0:
        warnings.append(f"equity={equity} <= 0; all notionals set to 0.")
        return AllocationResult(
            {p.label: 0.0 for p in pairs},
            {p.label: "equity<=0" for p in pairs},
            {}, 0.0, 0.0, warnings,
        )

    gross_budget = equity * max_gross_leverage
    symbol_budget = equity * max_symbol_pct

    # How many pairs touch each symbol (for the per-symbol cap split).
    symbol_count: Dict[str, int] = {}
    for p in pairs:
        for s in p.symbols:
            symbol_count[s] = symbol_count.get(s, 0) + 1

    total_weight = sum(max(0.0, p.weight) for p in pairs) or float(len(pairs))

    for p in pairs:
        w = max(0.0, p.weight) if total_weight else 1.0
        # Gross share for this pair, converted to a per-leg notional (2 legs).
        gross_notional = (w / total_weight) * gross_budget / 2.0
        # Tightest per-symbol cap among this pair's two legs.
        symbol_cap = min(symbol_budget / symbol_count[s] for s in p.symbols)

        candidates = {
            "ceiling": float(p.ceiling_notional),
            "gross": gross_notional,
            "symbol": symbol_cap,
        }
        chosen_key = min(candidates, key=candidates.get)
        notional = max(0.0, candidates[chosen_key])
        notionals[p.label] = notional
        binding[p.label] = chosen_key

        if notional < min_notional:
            warnings.append(
                f"{p.label}: allocated ${notional:,.0f} < min ${min_notional:,.0f} "
                f"(bound by {chosen_key}). Basket likely too large for the account — "
                f"reduce pairs, raise max_gross_leverage, or raise max_symbol_pct."
            )

    per_symbol: Dict[str, float] = {}
    for p in pairs:
        for s in p.symbols:
            per_symbol[s] = per_symbol.get(s, 0.0) + notionals[p.label]

    gross = sum(2.0 * n for n in notionals.values())

    # Defensive post-checks: the construction guarantees these, but assert them
    # so a future refactor that breaks the invariant fails loudly in tests.
    over_symbol = {s: v for s, v in per_symbol.items() if v > symbol_budget + 1e-6}
    if over_symbol:
        warnings.append(f"INVARIANT VIOLATION: per-symbol over cap: {over_symbol}")
    if gross > gross_budget + 1e-6:
        warnings.append(f"INVARIANT VIOLATION: gross ${gross:,.0f} > budget ${gross_budget:,.0f}")

    return AllocationResult(
        notionals=notionals,
        binding=binding,
        per_symbol_notional=per_symbol,
        gross=gross,
        gross_limit=gross_budget,
        warnings=warnings,
    )


class BasketAllocator:
    """Thin OO wrapper around :func:`allocate` for callers that prefer to hold
    configured limits and re-run allocation as the basket changes."""

    def __init__(
        self,
        equity: float,
        max_gross_leverage: float,
        max_symbol_pct: float = 1.0,
        min_notional: float = 100.0,
    ) -> None:
        self.equity = float(equity)
        self.max_gross_leverage = float(max_gross_leverage)
        self.max_symbol_pct = float(max_symbol_pct)
        self.min_notional = float(min_notional)

    def allocate(self, pairs: List[PairSpec]) -> AllocationResult:
        return allocate(
            pairs,
            equity=self.equity,
            max_gross_leverage=self.max_gross_leverage,
            max_symbol_pct=self.max_symbol_pct,
            min_notional=self.min_notional,
        )
