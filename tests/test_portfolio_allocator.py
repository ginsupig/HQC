"""Unit tests for risk/portfolio/allocator.py."""
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from risk.portfolio.allocator import PairSpec, allocate  # noqa: E402


class TestAllocator(unittest.TestCase):
    def test_equal_weight_fits_gross_budget(self):
        pairs = [
            PairSpec("JPM", "BAC", 10000),
            PairSpec("V", "MA", 10000),
            PairSpec("USB", "PNC", 10000),
            PairSpec("XOM", "CVX", 10000),
        ]
        # 4 pairs, equity 100k, 1x gross => budget 100k => per pair gross 25k =>
        # per-leg notional 12.5k, but ceiling 10k binds.
        res = allocate(pairs, equity=100000, max_gross_leverage=1.0, max_symbol_pct=1.0)
        self.assertTrue(all(v == 10000 for v in res.notionals.values()))
        self.assertTrue(all(b == "ceiling" for b in res.binding.values()))
        self.assertEqual(res.gross, 80000)  # 4 * 2 * 10k
        self.assertEqual(res.warnings, [])

    def test_gross_constraint_scales_down(self):
        pairs = [PairSpec("A", "B", 10000), PairSpec("C", "D", 10000)]
        # equity 10k, 1x => budget 10k => per pair gross 5k => per-leg 2.5k.
        res = allocate(pairs, equity=10000, max_gross_leverage=1.0, max_symbol_pct=1.0)
        self.assertAlmostEqual(res.notionals["A/B"], 2500)
        self.assertEqual(res.binding["A/B"], "gross")
        self.assertLessEqual(res.gross, 10000 + 1e-6)

    def test_leverage_relaxes_gross(self):
        pairs = [PairSpec("A", "B", 10000), PairSpec("C", "D", 10000)]
        # 450k buying power at 3x => 1.35M budget; ceilings bind.
        res = allocate(pairs, equity=450000, max_gross_leverage=3.0, max_symbol_pct=1.0)
        self.assertTrue(all(v == 10000 for v in res.notionals.values()))

    def test_shared_symbol_cap_enforced(self):
        # JPM appears in two pairs. Cap per symbol = 10% of 100k = 10k aggregate.
        pairs = [PairSpec("JPM", "BAC", 50000), PairSpec("JPM", "WFC", 50000)]
        res = allocate(pairs, equity=100000, max_gross_leverage=10.0, max_symbol_pct=0.10)
        # JPM in 2 pairs => each capped at 10k/2 = 5k.
        self.assertAlmostEqual(res.notionals["JPM/BAC"], 5000)
        self.assertAlmostEqual(res.notionals["JPM/WFC"], 5000)
        self.assertEqual(res.binding["JPM/BAC"], "symbol")
        # Aggregate JPM exposure == 10k == cap, not exceeded.
        self.assertLessEqual(res.per_symbol_notional["JPM"], 100000 * 0.10 + 1e-6)

    def test_invariants_hold_under_random_mix(self):
        pairs = [
            PairSpec("JPM", "BAC", 12000, weight=2.0),
            PairSpec("JPM", "WFC", 8000),
            PairSpec("V", "MA", 15000),
            PairSpec("KO", "PEP", 6000),
        ]
        res = allocate(pairs, equity=200000, max_gross_leverage=2.0, max_symbol_pct=0.25)
        symbol_budget = 200000 * 0.25
        for sym, exp in res.per_symbol_notional.items():
            self.assertLessEqual(exp, symbol_budget + 1e-6, f"{sym} over symbol cap")
        self.assertLessEqual(res.gross, 200000 * 2.0 + 1e-6)
        self.assertTrue(all(res.notionals[p.label] <= p.ceiling_notional + 1e-6 for p in pairs))
        self.assertFalse([w for w in res.warnings if "INVARIANT" in w])

    def test_too_small_account_warns(self):
        pairs = [PairSpec(f"Y{i}", f"X{i}", 10000) for i in range(20)]
        # 20 pairs, $2k equity, 1x => per-leg $50 < $100 min => warn.
        res = allocate(pairs, equity=2000, max_gross_leverage=1.0, min_notional=100.0)
        self.assertTrue(any("too large for the account" in w for w in res.warnings))

    def test_zero_equity_safe(self):
        res = allocate([PairSpec("A", "B", 10000)], equity=0, max_gross_leverage=2.0)
        self.assertEqual(res.notionals["A/B"], 0.0)

    def test_empty_pairs(self):
        res = allocate([], equity=100000, max_gross_leverage=1.0)
        self.assertEqual(res.notionals, {})


if __name__ == "__main__":
    unittest.main()
