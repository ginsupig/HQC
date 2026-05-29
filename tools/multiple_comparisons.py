"""
Multiple-comparisons correction for a family of walk-forward pair tests.

Background: walk-forward + bootstrap CI + Newey-West HAC (the harness in
analyze_walkforward.py) gives a per-test p-value under the null
"this pair has mean OOS return <= 0". When we evaluate multiple pairs
to pick which to deploy, single-test p < 0.05 is insufficient — we
expect ~5% of pairs to "pass" by chance even if no pair has edge.

This tool applies two standard corrections to a family of (pair, p-value)
results:

  - Bonferroni: per-test threshold = alpha / family_size. Conservative;
    controls family-wise error rate (FWER) under arbitrary dependence.
  - Benjamini-Hochberg (BH-FDR): less conservative; controls the expected
    proportion of false discoveries among rejected nulls at level q.
    Rank-1 threshold is identical to Bonferroni; thresholds loosen for
    higher ranks.

Usage:
  python tools/multiple_comparisons.py --family family.csv --alpha 0.05

Where family.csv has columns: pair,raw_p (one row per independent test).

Pre-register the family BEFORE running new walk-forwards (roadmap B4).
This tool is also used retrospectively for the existing family (A1).
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TestResult:
    pair: str
    raw_p: float


@dataclass
class CorrectedVerdict:
    pair: str
    raw_p: float
    bonferroni_threshold: float
    bonferroni_pass: bool
    bh_rank: int
    bh_threshold: float
    bh_pass: bool

    def verdict(self) -> str:
        if self.bonferroni_pass:
            return "EDGE+ (survives Bonferroni)"
        if self.bh_pass:
            return "EDGE+ (BH-FDR only; weaker)"
        return "no edge after correction"


def apply_corrections(
    tests: List[TestResult],
    alpha: float = 0.05,
    family_size: Optional[int] = None,
) -> List[CorrectedVerdict]:
    """Apply Bonferroni and Benjamini-Hochberg corrections to a family.

    Returns one CorrectedVerdict per input test, in input order.
    """
    n_tests = len(tests)
    if n_tests == 0:
        return []
    n = family_size or n_tests
    if n < n_tests:
        raise ValueError(f"family_size={n} cannot be smaller than number of tests={n_tests}")
    bonf_threshold = alpha / n

    # BH-FDR: sort by p ascending, compute per-rank thresholds, find the
    # largest rank k for which p_(k) <= (k/n) * alpha, then reject all
    # hypotheses up to and including rank k.
    sorted_idx = sorted(range(n_tests), key=lambda i: tests[i].raw_p)
    bh_thresholds = [((rank + 1) / n) * alpha for rank in range(n_tests)]
    largest_passing_rank = -1
    for rank, idx in enumerate(sorted_idx):
        if tests[idx].raw_p <= bh_thresholds[rank]:
            largest_passing_rank = rank
    bh_passes = set(sorted_idx[: largest_passing_rank + 1]) if largest_passing_rank >= 0 else set()

    # Map sorted rank back to per-test
    rank_by_idx = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}

    out: List[CorrectedVerdict] = []
    for i, t in enumerate(tests):
        out.append(
            CorrectedVerdict(
                pair=t.pair,
                raw_p=t.raw_p,
                bonferroni_threshold=bonf_threshold,
                bonferroni_pass=(t.raw_p <= bonf_threshold),
                bh_rank=rank_by_idx[i],
                bh_threshold=bh_thresholds[rank_by_idx[i] - 1],
                bh_pass=(i in bh_passes),
            )
        )
    return out


def _load_family(path: Path) -> List[TestResult]:
    out: List[TestResult] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = (row.get("pair") or "").strip()
            try:
                p = float(row.get("raw_p"))
            except (TypeError, ValueError):
                continue
            if not pair or not math.isfinite(p) or p < 0 or p > 1:
                continue
            out.append(TestResult(pair=pair, raw_p=p))
    return out


def _format_table(results: List[CorrectedVerdict], alpha: float, family_size: Optional[int] = None) -> str:
    n = family_size or len(results)
    lines = []
    header = f"{'pair':<18} {'raw_p':>8} {'Bonf_thr':>10} {'Bonf?':>6} {'BH_rank':>8} {'BH_thr':>9} {'BH?':>4}  verdict"
    sep = "-" * len(header)
    lines.append("=" * len(header))
    lines.append(f"Multiple-comparisons correction  (family_size={n}, alpha={alpha})")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)
    # Print in original order, then summary.
    for r in results:
        lines.append(
            f"{r.pair:<18} {r.raw_p:>8.4f} {r.bonferroni_threshold:>10.5f} "
            f"{'PASS' if r.bonferroni_pass else 'fail':>6} "
            f"{r.bh_rank:>8} {r.bh_threshold:>9.5f} "
            f"{'PASS' if r.bh_pass else 'fail':>4}  {r.verdict()}"
        )
    lines.append(sep)
    n_bonf = sum(1 for r in results if r.bonferroni_pass)
    n_bh = sum(1 for r in results if r.bh_pass)
    lines.append(f"{n_bonf}/{n} survive Bonferroni;  {n_bh}/{n} survive BH-FDR")
    lines.append("=" * len(header))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", type=Path, required=True, help="CSV with columns: pair,raw_p")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--family-size", type=int, default=0,
                   help="Optional pre-registered family size override. Must be >= row count.")
    args = p.parse_args()
    tests = _load_family(args.family)
    if not tests:
        raise SystemExit(f"No valid (pair, raw_p) rows in {args.family}")
    results = apply_corrections(
        tests,
        alpha=args.alpha,
        family_size=args.family_size or None,
    )
    print(_format_table(results, alpha=args.alpha, family_size=args.family_size or None))


if __name__ == "__main__":
    main()
