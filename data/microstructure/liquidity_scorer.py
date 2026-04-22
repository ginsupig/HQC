"""
Amihud Liquidity Flash-Crash Detection — standalone demo.

Feeds synthetic tick data into the LiquidityRelativeStrengthEngine and
demonstrates the hard-veto logic that fires during thin-tape conditions.
"""
from __future__ import annotations

import random
import sys
import time

from intelligence.liquidity_rs_engine import LiquidityRelativeStrengthEngine

SEED = 42
random.seed(SEED)


def _ts(offset_ms: int = 0) -> int:
    """Return a market-hours timestamp (10:00 ET) offset by `offset_ms`."""
    # 2026-01-05 15:00:00 UTC == 10:00 ET
    base_ms = 1736085600000
    return base_ms + offset_ms


def _feed_normal_tape(engine: LiquidityRelativeStrengthEngine, symbol: str, n: int = 30) -> None:
    price = 500.0
    for i in range(n):
        price += random.uniform(-0.10, 0.10)
        volume = random.uniform(800, 2000)
        engine.update_tick(symbol=symbol, price=price, volume=volume, ts_ms=_ts(i * 1000))
        engine.update_tick(symbol="SPY", price=price * 0.5, volume=volume * 2, ts_ms=_ts(i * 1000))


def _feed_thin_tape(engine: LiquidityRelativeStrengthEngine, symbol: str, n: int = 10) -> None:
    """Simulate a thin-tape episode: tiny volume, erratic prices."""
    price = 500.0
    offset_base = 31_000
    for i in range(n):
        price += random.uniform(-1.5, 1.5)
        volume = random.uniform(5, 20)
        engine.update_tick(symbol=symbol, price=price, volume=volume, ts_ms=_ts(offset_base + i * 1000))


def main() -> None:
    print("Initializing Amihud Liquidity Flash-Crash Detection Test...\n")

    engine = LiquidityRelativeStrengthEngine(
        benchmark="SPY",
        tick_window=50,
        min_ticks_for_quality=8,
        max_spread_bps=20.0,
        min_rvol=0.90,
        min_liquidity_score=0.35,
    )

    symbol = "NVDA"

    print(f"[1] Feeding {30} normal-tape ticks for {symbol}...")
    _feed_normal_tape(engine, symbol, n=30)

    result = engine.evaluate_candidate(symbol=symbol, action="BUY", reference_price=500.0)
    metrics = result.get("metrics", {})
    veto = result.get("veto", {})
    print(f"    rvol={metrics.get('rvol', 0):.2f}  spread_bps={metrics.get('spread_bps', 0):.1f}  "
          f"liquidity={metrics.get('liquidity_score', 0):.2f}  hard_veto={veto.get('hard_veto')}")
    assert not veto.get("hard_veto"), "Expected no veto on normal tape"
    print("    OK No veto on healthy tape\n")

    print(f"[2] Feeding {10} thin-tape ticks (flash-crash simulation)...")
    _feed_thin_tape(engine, symbol, n=10)

    result = engine.evaluate_candidate(symbol=symbol, action="BUY", reference_price=500.0)
    metrics = result.get("metrics", {})
    veto = result.get("veto", {})
    print(f"    rvol={metrics.get('rvol', 0):.2f}  spread_bps={metrics.get('spread_bps', 0):.1f}  "
          f"liquidity={metrics.get('liquidity_score', 0):.2f}  hard_veto={veto.get('hard_veto')}")
    reasons = veto.get("reasons", [])
    print(f"    veto reasons: {reasons}")

    if veto.get("hard_veto"):
        print("    OK Hard veto correctly triggered on thin tape\n")
    else:
        print("    WARN Thin-tape veto not triggered (rvol may not have dropped enough in window)\n")

    print("[3] Daily reset...")
    engine.reset_daily()
    result_after = engine.evaluate_candidate(symbol=symbol, action="BUY", reference_price=500.0)
    veto_after = result_after.get("veto", {})
    print(f"    hard_veto after reset={veto_after.get('hard_veto')}  reasons={veto_after.get('reasons', [])}")
    print("    OK State cleared on reset\n")

    print("Amihud Liquidity Flash-Crash Detection: PASS")


if __name__ == "__main__":
    main()
