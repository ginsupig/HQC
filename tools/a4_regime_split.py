"""
A4 — Regime-conditioned walk-forward analyzer.

Reads a walkforward_pairs.py / walkforward_basket.py result JSON,
tags each window with the dominant SPY regime over its test period
(via tools.regime_tagger.RegimeTagger), and computes per-regime OOS
statistics with bootstrap CI + Newey-West HAC p-value.

Per the roadmap (workstream A, task A4): "Output a rule: 'JPM/BAC
trades only in regime X' if the edge concentrates." Bank pairs are
rate-regime sensitive, so concentration is the expected pattern.

Significance bar: bootstrap 95% CI lower bound > 0 AND
Bonferroni-corrected p < 0.05 across the regimes that have enough
samples (default min n=4 per regime). Default family_size =
number of regimes evaluated.

Usage:
  python tools/a4_regime_split.py \\
      --walkforward result_wf_pair_jpm_bac_2y.json \\
      --spy-csv data/alpaca/spy_730d_1m.csv \\
      --output a4_jpm_bac_regime.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyze_walkforward import _bootstrap_mean_ci, _one_sample_p_greater_than_zero  # noqa: E402
from tools.regime_tagger import RegimeTagger  # noqa: E402


def _parse_date(s: str) -> date_type:
    return date_type.fromisoformat(s.strip().split("T")[0].split(" ")[0])


def _iter_pair_windows(payload: dict) -> List[Tuple[str, dict]]:
    """Walk through (pair_label, window_block) for every window in a
    walkforward result. Handles both walkforward_pairs.py and
    walkforward_basket.py shapes."""
    out: List[Tuple[str, dict]] = []
    results = payload.get("results") or []
    for sym_block in results:
        label = str(sym_block.get("symbol") or "(unnamed)")
        for w in sym_block.get("results") or []:
            out.append((label, w))
    return out


def _per_regime_split(
    pair_label: str,
    windows: List[dict],
    tagger: RegimeTagger,
    min_n: int,
    alpha: float,
) -> List[Dict]:
    """Returns one summary row per regime with at least min_n windows.
    Each row has: pair, regime, n, mean_pct, ci_lo, ci_hi, raw_p, edge."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for w in windows:
        tr = w.get("test_result") or {}
        ts = tr.get("total_return_pct")
        if not isinstance(ts, (int, float)):
            continue
        try:
            start = _parse_date(str(w.get("test_start")))
            end = _parse_date(str(w.get("test_end")))
        except (TypeError, ValueError):
            continue
        regime = tagger.dominant_label_for_period(start, end)
        if regime is None:
            continue
        buckets[regime].append(float(ts))

    eligible_regimes = [r for r, vs in buckets.items() if len(vs) >= min_n]
    family_size = max(1, len(eligible_regimes))
    bonf_threshold = alpha / family_size

    rows: List[Dict] = []
    for regime, vs in sorted(buckets.items()):
        n = len(vs)
        mean, lo, hi = _bootstrap_mean_ci(vs, n_boot=10000, alpha=alpha)
        p = _one_sample_p_greater_than_zero(vs) if len(vs) >= 3 else float("nan")
        eligible = n >= min_n
        edge = bool(eligible and lo > 0 and math.isfinite(p) and p <= bonf_threshold)
        rows.append(
            dict(
                pair=pair_label,
                regime=regime,
                n=n,
                mean_pct=round(mean * 100.0, 6),
                ci_lo=round(lo * 100.0, 6),
                ci_hi=round(hi * 100.0, 6),
                raw_p=round(p, 6) if math.isfinite(p) else "",
                bonf_threshold=round(bonf_threshold, 6),
                edge_plus=edge,
                eligible=eligible,
            )
        )
    return rows


def _emit_deployment_rule(pair_label: str, rows: List[Dict]) -> str:
    """If exactly one regime is EDGE+ and the others are clearly not,
    emit a deployment rule that gates trading to that regime."""
    edges = [r for r in rows if r["edge_plus"]]
    if not edges:
        return f"  {pair_label}: no regime is EDGE+ after Bonferroni; no concentration rule emitted."
    if len(edges) == len(rows):
        return f"  {pair_label}: edge present across every regime; no gating needed."
    if len(edges) == 1:
        only = edges[0]
        return (
            f"  {pair_label}: edge concentrates in regime '{only['regime']}' "
            f"(n={only['n']}, mean=+{only['mean_pct']:.3f}%, p={only['raw_p']}).\n"
            f"          DEPLOYMENT RULE: trade {pair_label} only when current SPY regime is '{only['regime']}'.\n"
            f"          Live: PairsRiskMonitor should suspend the pair when the live regime differs."
        )
    # 2+ EDGE+ regimes but not all → partial concentration
    names = ", ".join(f"'{r['regime']}'" for r in edges)
    return (
        f"  {pair_label}: edge concentrates in {len(edges)} regimes ({names}).\n"
        f"          DEPLOYMENT RULE: trade {pair_label} only when current SPY regime is one of: {names}."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walkforward", required=True, type=Path,
                   help="walkforward_pairs.py / walkforward_basket.py result JSON.")
    p.add_argument("--spy-csv", required=True, type=Path,
                   help="Benchmark OHLCV CSV (typically SPY). Used to compute "
                        "EMA20 trend and 20-day realized-vol terciles.")
    p.add_argument("--spy-symbol", default="SPY",
                   help="Symbol filter inside --spy-csv if it's multi-symbol.")
    p.add_argument("--ema-span", type=int, default=20)
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--min-n", type=int, default=4,
                   help="Minimum windows per regime to count as eligible.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", type=Path, default=None,
                   help="Optional CSV with the per-regime rows.")
    args = p.parse_args()

    tagger = RegimeTagger.from_csv(
        args.spy_csv,
        ema_span=args.ema_span,
        vol_window=args.vol_window,
        symbol=args.spy_symbol,
    )
    print(
        f"[A4] tagger ready: {len(tagger.label_by_date)} trading days, "
        f"regimes={list(tagger.regimes())}, "
        f"vol terciles (low/mid, mid/high)=({tagger.vol_terciles[0]:.4f}, {tagger.vol_terciles[1]:.4f})"
    )

    payload = json.loads(args.walkforward.read_text(encoding="utf-8"))
    by_pair: Dict[str, List[dict]] = defaultdict(list)
    for label, window in _iter_pair_windows(payload):
        by_pair[label].append(window)
    if not by_pair:
        raise SystemExit(f"No windows found in {args.walkforward}")

    all_rows: List[Dict] = []
    print()
    print("=" * 92)
    print("A4 per-regime walk-forward split")
    print(f"  family_size = #eligible regimes (min_n={args.min_n}); Bonferroni at alpha={args.alpha}")
    print("=" * 92)

    for pair_label in sorted(by_pair.keys()):
        rows = _per_regime_split(pair_label, by_pair[pair_label], tagger, args.min_n, args.alpha)
        all_rows.extend(rows)
        print(f"\n  {pair_label}  (windows={sum(r['n'] for r in rows)})")
        header = f"    {'regime':<10} {'n':>3}  {'mean%':>8}  {'95%CI':>22}  {'raw_p':>8}  {'Bonf<=':>8}  edge+"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for r in rows:
            ci = f"[{r['ci_lo']:+.3f}%, {r['ci_hi']:+.3f}%]"
            star = "  EDGE+" if r["edge_plus"] else ("  (insufficient n)" if not r["eligible"] else "")
            p_str = f"{r['raw_p']:.4f}" if isinstance(r["raw_p"], float) else "n/a"
            print(
                f"    {r['regime']:<10} {r['n']:>3}  {r['mean_pct']:+7.3f}%  {ci:>22}  "
                f"{p_str:>8}  {r['bonf_threshold']:.4f}{star}"
            )

    print()
    print("=" * 92)
    print("DEPLOYMENT RULES")
    print("=" * 92)
    for pair_label in sorted(by_pair.keys()):
        rows = [r for r in all_rows if r["pair"] == pair_label]
        print(_emit_deployment_rule(pair_label, rows))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["pair", "regime", "n", "mean_pct", "ci_lo", "ci_hi",
                            "raw_p", "bonf_threshold", "edge_plus", "eligible"],
            )
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
