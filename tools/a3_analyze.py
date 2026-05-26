"""
A3 — Analyze the cost-stress surface.

Reads the CSV emitted by tools/a3_cost_stress.py and produces:

  1. Per-pair ASCII heatmap of mean OOS return per (slippage, borrow)
     cell with EDGE+ marks (CI_lo > 0 AND raw_p <= Bonferroni
     threshold at family_size = grid cells per pair).
  2. Breakeven boundary: cheapest cost level at which the verdict
     flips from EDGE+ to fail along each axis from the deployed cell.
  3. Fragility flag: deployed = (1.5 bps slippage, 0.25% borrow).
     If a +1 bp slippage or +0.5% borrow APR change pushes the pair
     below the EDGE+ bar, mark FRAGILE -- the pair should not be sized
     up and the live cost model needs continuous monitoring.

Usage:
  python tools/a3_analyze.py --input a3_jpm_bac.csv
  python tools/a3_analyze.py --input a3_basket.csv  # CSV may contain multiple pairs

Significance bar: family_size defaults to the number of cells PER pair
(typically 9). Bonferroni-corrected per-cell threshold = alpha /
family_size = 0.0056 at default alpha=0.05.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Deployed cost assumptions (must match config / TradeLedger defaults).
DEPLOYED_SLIPPAGE_BPS = 1.5
DEPLOYED_BORROW_APR = 0.0025
# Fragility thresholds from the roadmap: deployment is fragile if a small
# adverse cost shock crosses the EDGE+ boundary.
FRAGILE_SLIPPAGE_BUMP_BPS = 1.0     # +1 bp slippage
FRAGILE_BORROW_BUMP_APR = 0.005     # +0.5% APR borrow


def _load(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["slippage_bps_per_side"] = float(row["slippage_bps_per_side"])
                row["short_borrow_apr"] = float(row["short_borrow_apr"])
                row["mean_pct"] = float(row["mean_pct"])
                row["ci_lo"] = float(row["ci_lo"])
                row["ci_hi"] = float(row["ci_hi"])
                row["raw_p"] = float(row["raw_p"]) if row.get("raw_p") not in (None, "") else float("nan")
                row["total_pnl"] = float(row["total_pnl"])
                row["total_costs"] = float(row["total_costs"])
            except (TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def _is_edge(row: dict, bonf_threshold: float) -> bool:
    return row["ci_lo"] > 0 and math.isfinite(row["raw_p"]) and row["raw_p"] <= bonf_threshold


def _mark(row: dict, bonf_threshold: float) -> str:
    if _is_edge(row, bonf_threshold):
        return "B"
    if math.isfinite(row["raw_p"]) and row["raw_p"] < 0.05 and row["ci_lo"] > 0:
        return "r"  # raw p only (no correction)
    return "."


def _analyze_pair(pair_label: str, pair_rows: List[dict], alpha: float) -> None:
    if not pair_rows:
        return
    family_size = len(pair_rows)
    bonf = alpha / family_size

    slip_vals = sorted({r["slippage_bps_per_side"] for r in pair_rows})
    borrow_vals = sorted({r["short_borrow_apr"] for r in pair_rows})
    cells: Dict[Tuple[float, float], dict] = {
        (r["slippage_bps_per_side"], r["short_borrow_apr"]): r for r in pair_rows
    }

    print()
    print("=" * 86)
    print(f"A3 cost-stress surface for {pair_label}")
    print(f"  cells={family_size}  alpha={alpha}  Bonferroni per-cell threshold={bonf:.5f}")
    print("=" * 86)

    # Heatmap.
    header_borrow = "          borrow:  " + "  ".join(f"{b*100:>7.2f}%" for b in borrow_vals)
    print(header_borrow)
    print("-" * len(header_borrow))
    for s in slip_vals:
        row_cells = []
        for b in borrow_vals:
            r = cells.get((s, b))
            if r is None:
                row_cells.append(f"{'n/a':>8}")
                continue
            mark = _mark(r, bonf)
            star = "*" if (abs(s - DEPLOYED_SLIPPAGE_BPS) < 1e-9 and abs(b - DEPLOYED_BORROW_APR) < 1e-9) else " "
            row_cells.append(f"{r['mean_pct']:+6.3f}{mark}{star}")
        print(f"  slip {s:>4.1f}bps  | " + " ".join(row_cells))
    print()
    print("  Legend: B=passes Bonferroni, r=raw p<0.05 only, .=fail; `*`=deployed cost cell")

    # Deployed cell verdict.
    deployed = cells.get((DEPLOYED_SLIPPAGE_BPS, DEPLOYED_BORROW_APR))
    if deployed is None:
        print(f"\n  ! deployed cell ({DEPLOYED_SLIPPAGE_BPS} bps, {DEPLOYED_BORROW_APR*100:.2f}% APR) NOT in grid")
        return
    deployed_edge = _is_edge(deployed, bonf)
    print(f"\n  Deployed cell: mean={deployed['mean_pct']:+.3f}%  "
          f"CI=[{deployed['ci_lo']:+.3f}%, {deployed['ci_hi']:+.3f}%]  "
          f"raw_p={deployed['raw_p']:.4f}  edge+={deployed_edge}")

    if not deployed_edge:
        print(f"\n  >>> {pair_label} FAILS Bonferroni at the deployed cost level. "
              f"This pair shouldn't be running at all.")
        return

    # Breakeven boundaries: walk each axis from deployed in increasing direction.
    print("\n  Breakeven boundaries:")
    # Slippage axis (fix borrow at deployed).
    slip_breakeven = None
    for s in [s for s in slip_vals if s > DEPLOYED_SLIPPAGE_BPS]:
        r = cells.get((s, DEPLOYED_BORROW_APR))
        if r and not _is_edge(r, bonf):
            slip_breakeven = s
            break
    if slip_breakeven is None:
        max_tested = max(slip_vals)
        if max_tested > DEPLOYED_SLIPPAGE_BPS:
            print(f"    slippage : still EDGE+ at the worst tested ({max_tested} bps/side). "
                  f"Edge survives at least +{max_tested - DEPLOYED_SLIPPAGE_BPS:.1f} bps.")
        else:
            print(f"    slippage : no harder cost cell tested above {DEPLOYED_SLIPPAGE_BPS} bps.")
    else:
        gap = slip_breakeven - DEPLOYED_SLIPPAGE_BPS
        flag = " <-- FRAGILE" if gap <= FRAGILE_SLIPPAGE_BUMP_BPS else ""
        print(f"    slippage : EDGE+ fails at {slip_breakeven} bps/side "
              f"(+{gap:.1f} bps above deployed){flag}")

    # Borrow axis (fix slippage at deployed).
    borrow_breakeven = None
    for b in [b for b in borrow_vals if b > DEPLOYED_BORROW_APR]:
        r = cells.get((DEPLOYED_SLIPPAGE_BPS, b))
        if r and not _is_edge(r, bonf):
            borrow_breakeven = b
            break
    if borrow_breakeven is None:
        max_tested = max(borrow_vals)
        if max_tested > DEPLOYED_BORROW_APR:
            print(f"    borrow   : still EDGE+ at the worst tested ({max_tested*100:.2f}% APR). "
                  f"Edge survives at least +{(max_tested - DEPLOYED_BORROW_APR)*100:.2f}% APR.")
        else:
            print(f"    borrow   : no harder cost cell tested above {DEPLOYED_BORROW_APR*100:.2f}% APR.")
    else:
        gap = borrow_breakeven - DEPLOYED_BORROW_APR
        flag = " <-- FRAGILE" if gap <= FRAGILE_BORROW_BUMP_APR else ""
        print(f"    borrow   : EDGE+ fails at {borrow_breakeven*100:.2f}% APR "
              f"(+{gap*100:.2f}% above deployed){flag}")

    # Overall fragility verdict.
    fragile_slip = slip_breakeven is not None and (slip_breakeven - DEPLOYED_SLIPPAGE_BPS) <= FRAGILE_SLIPPAGE_BUMP_BPS
    fragile_borrow = borrow_breakeven is not None and (borrow_breakeven - DEPLOYED_BORROW_APR) <= FRAGILE_BORROW_BUMP_APR
    print()
    if fragile_slip or fragile_borrow:
        causes = []
        if fragile_slip:
            causes.append("slippage")
        if fragile_borrow:
            causes.append("borrow")
        print(f"  VERDICT: {pair_label} is COST-FRAGILE on " + " and ".join(causes) + ".")
        print(f"           Do not size up. Monitor live cost realization continuously.")
    else:
        print(f"  VERDICT: {pair_label} has comfortable cost margin at deployed assumptions.")
        print(f"           Edge survives the next test grid step in both axes.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    rows = _load(args.input)
    if not rows:
        raise SystemExit(f"No usable rows in {args.input}")
    # Group by pair so a multi-pair CSV is handled.
    by_pair: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_pair[(r.get("pair") or "").strip() or "(unnamed)"].append(r)

    for pair_label in sorted(by_pair.keys()):
        _analyze_pair(pair_label, by_pair[pair_label], args.alpha)


if __name__ == "__main__":
    main()
