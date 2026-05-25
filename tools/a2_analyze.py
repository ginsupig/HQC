"""
A2 — Analyze the parameter-sensitivity surface.

Reads the CSV emitted by tools/a2_parameter_sensitivity.py and produces:

  1. Per-(delta, ve) slice ASCII heatmaps: mean OOS return per cell,
     marking each cell with its EDGE+ status under the family-corrected
     bar that A1 established.
  2. A plateau verdict: identifies the connected EDGE+ region in
     (entry_z, exit_z) space that contains the deployed cell
     (default 1.5, 0.4) for each (delta, ve) slice. Reports the
     region's size relative to the surrounding grid.

The roadmap's DoD: "the deployed params sit inside a contiguous EDGE+
region of sufficient width." This analyzer turns that into a number.

Significance bar used by default:
  - Bootstrap 95% CI lower bound > 0  AND
  - Bonferroni-corrected raw_p < 0.05 / family_size
    where family_size = len(non-degenerate grid cells)

The family-size correction here is across the parameter grid (not the
pair family). Two distinct multiple-comparison families:
  - across pairs (A1, n=6 or 8)
  - across parameter cells (this script's --family-size)
A cell that survives BOTH is robust enough to claim EDGE+ at the
post-A2 bar. The analyzer reports raw, BH-FDR, and Bonferroni
verdicts so the operator can choose the strictness.

Usage:
  python tools/a2_analyze.py --input a2_jpm_bac_full.csv \\
      --deployed-entry-z 1.5 --deployed-exit-z 0.4 \\
      --deployed-delta 1e-4 --deployed-ve 1e-3
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


def _load(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["entry_z"] = float(row["entry_z"])
                row["exit_z"] = float(row["exit_z"])
                row["delta"] = float(row["delta"])
                row["ve"] = float(row["ve"])
                row["n_windows"] = int(row["n_windows"])
                row["mean_pct"] = float(row["mean_pct"])
                row["ci_lo"] = float(row["ci_lo"])
                row["ci_hi"] = float(row["ci_hi"])
                row["raw_p"] = float(row["raw_p"]) if row.get("raw_p") not in (None, "") else float("nan")
                row["total_pnl"] = float(row["total_pnl"])
            except (TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def _bh_fdr_mask(p_values: List[float], alpha: float) -> List[bool]:
    """Benjamini-Hochberg: returns a parallel mask of which p-values pass."""
    n = len(p_values)
    if n == 0:
        return []
    idx_sorted = sorted(range(n), key=lambda i: p_values[i])
    largest_passing = -1
    for rank, i in enumerate(idx_sorted):
        threshold = ((rank + 1) / n) * alpha
        if p_values[i] <= threshold:
            largest_passing = rank
    passing = set(idx_sorted[: largest_passing + 1]) if largest_passing >= 0 else set()
    return [i in passing for i in range(n)]


def _verdict_for_row(
    row: dict,
    bonf_threshold: float,
    bh_pass: bool,
) -> str:
    """One-character cell mark for the heatmap."""
    ci_lo = row["ci_lo"]
    p = row["raw_p"]
    if ci_lo <= 0 or not math.isfinite(p):
        return "."  # cannot reject null
    if p <= bonf_threshold:
        return "B"  # passes Bonferroni
    if bh_pass:
        return "F"  # passes BH-FDR only
    if p < 0.05:
        return "r"  # raw p only (no family correction)
    return "."


def _connected_region(
    cells: Dict[Tuple[float, float], dict],
    entry_z_values: List[float],
    exit_z_values: List[float],
    seed: Tuple[float, float],
    is_edge: Dict[Tuple[float, float], bool],
) -> List[Tuple[float, float]]:
    """4-neighbour connected component of EDGE+ cells starting at `seed`,
    moving only between adjacent grid coordinates."""
    if seed not in cells or not is_edge.get(seed, False):
        return []
    ez_idx = {v: i for i, v in enumerate(entry_z_values)}
    xz_idx = {v: i for i, v in enumerate(exit_z_values)}
    seen = {seed}
    stack = [seed]
    region = []
    while stack:
        cur = stack.pop()
        region.append(cur)
        ce, cx = cur
        for de, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ne_idx = ez_idx.get(ce, -1) + de
            nx_idx = xz_idx.get(cx, -1) + dx
            if not (0 <= ne_idx < len(entry_z_values)):
                continue
            if not (0 <= nx_idx < len(exit_z_values)):
                continue
            nb = (entry_z_values[ne_idx], exit_z_values[nx_idx])
            if nb in seen:
                continue
            if nb in cells and is_edge.get(nb, False):
                seen.add(nb)
                stack.append(nb)
    return region


def _print_slice(
    rows_slice: List[dict],
    delta: float,
    ve: float,
    bonf_threshold: float,
    bh_mask_by_idx: Dict[int, bool],
    deployed: Optional[Tuple[float, float]],
) -> Tuple[List[float], List[float], Dict[Tuple[float, float], dict], Dict[Tuple[float, float], bool]]:
    entry_z_values = sorted({r["entry_z"] for r in rows_slice})
    exit_z_values = sorted({r["exit_z"] for r in rows_slice})
    cells: Dict[Tuple[float, float], dict] = {}
    is_edge: Dict[Tuple[float, float], bool] = {}
    for r in rows_slice:
        key = (r["entry_z"], r["exit_z"])
        cells[key] = r
        # EDGE+ definition: CI lower bound > 0 AND Bonferroni-corrected p < threshold.
        # We require the strictest of the available criteria.
        edge = (r["ci_lo"] > 0) and math.isfinite(r["raw_p"]) and r["raw_p"] <= bonf_threshold
        is_edge[key] = edge

    print(f"\n=== slice delta={delta:.0e}  ve={ve:.0e} ===")
    print(f"  (deployed marker `*` at entry_z={deployed[0] if deployed else '-'}, exit_z={deployed[1] if deployed else '-'})")
    header = "entry_z \\\\ exit_z " + "  ".join(f"{x:>8.2f}" for x in exit_z_values)
    print(header)
    print("-" * len(header))
    for ez in entry_z_values:
        cells_row = []
        for xz in exit_z_values:
            r = cells.get((ez, xz))
            if r is None:
                cells_row.append(f"{'n/a':>8}")
                continue
            mark = _verdict_for_row(r, bonf_threshold, bh_mask_by_idx.get(id(r), False))
            star = "*" if deployed and abs(ez - deployed[0]) < 1e-9 and abs(xz - deployed[1]) < 1e-9 else " "
            cells_row.append(f"{r['mean_pct']:+6.3f}{mark}{star}")
        print(f"{ez:>14.2f}  " + " ".join(cells_row))
    print("  Legend: B=passes Bonferroni, F=passes BH-FDR only, r=raw p<0.05 only, .=fail; `*`=deployed cell")
    return entry_z_values, exit_z_values, cells, is_edge


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--deployed-entry-z", type=float, default=1.5)
    p.add_argument("--deployed-exit-z", type=float, default=0.4)
    p.add_argument("--deployed-delta", type=float, default=1e-4)
    p.add_argument("--deployed-ve", type=float, default=1e-3)
    p.add_argument("--family-size", type=int, default=0,
                   help="Override Bonferroni family size. Default = count of grid cells.")
    args = p.parse_args()

    rows = _load(args.input)
    if not rows:
        raise SystemExit(f"No usable rows in {args.input}")

    family_size = args.family_size or len(rows)
    bonf_threshold = args.alpha / family_size

    # Compute BH-FDR mask once across the whole grid.
    ps = [r["raw_p"] if math.isfinite(r["raw_p"]) else 1.0 for r in rows]
    bh = _bh_fdr_mask(ps, args.alpha)
    bh_mask_by_idx = {id(r): bh[i] for i, r in enumerate(rows)}

    # Group by (delta, ve).
    by_slice: Dict[Tuple[float, float], List[dict]] = defaultdict(list)
    for r in rows:
        by_slice[(r["delta"], r["ve"])].append(r)

    print("=" * 78)
    print("A2 parameter-sensitivity surface")
    print(f"  rows={len(rows)} grid cells")
    print(f"  family_size={family_size}, alpha={args.alpha}")
    print(f"  Bonferroni per-cell threshold = {bonf_threshold:.6f}")
    print(f"  Significance bar: CI_lo > 0  AND  raw_p <= Bonferroni threshold")
    print("=" * 78)

    deployed = (args.deployed_entry_z, args.deployed_exit_z)
    deployed_slice_key = (args.deployed_delta, args.deployed_ve)

    plateau_summaries = []
    for (delta, ve) in sorted(by_slice.keys()):
        ez_list, xz_list, cells, is_edge = _print_slice(
            by_slice[(delta, ve)],
            delta, ve,
            bonf_threshold,
            bh_mask_by_idx,
            deployed,
        )
        if deployed in cells:
            region = _connected_region(cells, ez_list, xz_list, deployed, is_edge)
            total = sum(1 for k in cells if is_edge.get(k, False))
            plateau_summaries.append({
                "delta": delta,
                "ve": ve,
                "deployed_passes": is_edge.get(deployed, False),
                "region_size": len(region),
                "total_edge_cells_in_slice": total,
                "slice_cells_total": len(cells),
                "region_cells": region,
            })

    print("\n" + "=" * 78)
    print("PLATEAU ANALYSIS")
    print("=" * 78)
    print(
        "For each (delta, ve) slice, the size of the EDGE+ connected region "
        "containing the deployed cell. A plateau is >=4 connected EDGE+ cells "
        "INCLUDING the deployed cell. A spike is 1 (only the deployed cell)."
    )
    for s in plateau_summaries:
        marker = "  <-- deployed slice" if (s["delta"], s["ve"]) == deployed_slice_key else ""
        verdict = (
            "PLATEAU" if s["deployed_passes"] and s["region_size"] >= 4
            else "SPIKE" if s["deployed_passes"] and s["region_size"] < 4
            else "deployed cell fails Bonferroni"
        )
        print(
            f"  delta={s['delta']:.0e} ve={s['ve']:.0e}  "
            f"deployed_edge+={s['deployed_passes']}  "
            f"region_size={s['region_size']}/{s['slice_cells_total']}  "
            f"total_edge_in_slice={s['total_edge_cells_in_slice']}/{s['slice_cells_total']}  "
            f"-> {verdict}{marker}"
        )

    deployed_summary = next(
        (s for s in plateau_summaries if (s["delta"], s["ve"]) == deployed_slice_key),
        None,
    )
    print("\n" + "=" * 78)
    print("OVERALL VERDICT (deployed (delta, ve) slice)")
    print("=" * 78)
    if deployed_summary is None:
        print(f"  deployed cell {deployed_slice_key} not in grid output. Cannot conclude.")
    elif not deployed_summary["deployed_passes"]:
        print(
            f"  Deployed cell at entry_z={deployed[0]}, exit_z={deployed[1]}, "
            f"delta={deployed_slice_key[0]:.0e}, ve={deployed_slice_key[1]:.0e} "
            f"FAILS Bonferroni at the per-cell threshold.\n"
            f"  Recommendation: DEMOTE JPM/BAC; the surviving edge is not even at the validated parameters."
        )
    elif deployed_summary["region_size"] >= 4:
        print(
            f"  Deployed cell PASSES Bonferroni and sits in a connected EDGE+ region of "
            f"{deployed_summary['region_size']} cells.\n"
            f"  Recommendation: PLATEAU confirmed -- JPM/BAC stays deployed; edge is robust "
            f"to parameter perturbation."
        )
    else:
        print(
            f"  Deployed cell PASSES Bonferroni but its connected EDGE+ region is only "
            f"{deployed_summary['region_size']} cell(s).\n"
            f"  Recommendation: SPIKE -- edge appears curve-fit to a single parameter node. "
            f"DEMOTE JPM/BAC or re-validate after collecting more OOS data."
        )


if __name__ == "__main__":
    main()
