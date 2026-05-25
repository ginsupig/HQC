"""
Compute statistical significance of walk-forward results.

Consumes the JSON written by walkforward_basket.py and reports for
each symbol:
- mean per-window OOS return
- bootstrap 95% confidence interval on the mean (10k resamples)
- one-sample t-test p-value for "mean > 0" with HAC-style adjustment
  for serial correlation (Newey-West, lag = floor(n^(1/4)))
- pooled across-symbol mean and CI

The point: stop reading point estimates as truth. avgPF=8.23 with a
95% CI of [-2, 17] is not edge, it's noise. If 0 is inside the CI,
we cannot reject the null that the strategy has zero edge OOS.

Usage:
    python analyze_walkforward.py --input result_walkforward_basket.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _bootstrap_mean_ci(samples: List[float], n_boot: int = 10000, alpha: float = 0.05, rng: np.random.Generator | None = None) -> Tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return (float(arr.mean()), lo, hi)


def _newey_west_se(samples: List[float], lag: int | None = None) -> float:
    """Newey-West HAC standard error of the sample mean."""
    arr = np.asarray(samples, dtype=float)
    n = arr.size
    if n < 3:
        return float("nan")
    if lag is None:
        lag = max(1, int(math.floor(n ** (1.0 / 4.0))))
    mean = arr.mean()
    dev = arr - mean
    # Long-run variance: gamma_0 + 2 sum_{k=1..lag} (1 - k/(lag+1)) gamma_k
    long_run_var = float(np.dot(dev, dev) / n)
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(dev[k:], dev[:-k]) / n)
        weight = 1.0 - (k / (lag + 1.0))
        long_run_var += 2.0 * weight * gamma_k
    long_run_var = max(long_run_var, 1e-12)
    return math.sqrt(long_run_var / n)


def _one_sample_p_greater_than_zero(samples: List[float]) -> float:
    """Approximate p-value for H1: mean > 0 using HAC SE and a t/Z approximation."""
    arr = np.asarray(samples, dtype=float)
    n = arr.size
    if n < 3:
        return float("nan")
    mean = float(arr.mean())
    se = _newey_west_se(samples)
    if not math.isfinite(se) or se <= 0.0:
        return float("nan")
    t_stat = mean / se
    # Use the normal approximation (n is small but Newey-West already
    # accounts for autocorrelation; survival of N(0,1) at t).
    return 0.5 * math.erfc(t_stat / math.sqrt(2.0))


def _per_window_returns(symbol_block: Dict[str, object]) -> List[float]:
    """Pull the OOS test-window returns from one symbol's results."""
    out: List[float] = []
    for w in symbol_block.get("results", []) or []:
        tr = w.get("test_result") or {}
        ret = tr.get("total_return_pct")
        if isinstance(ret, (int, float)):
            out.append(float(ret))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="walkforward_basket.py JSON output.")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples for mean CI.")
    parser.add_argument("--alpha", type=float, default=0.05, help="CI tail (default 0.05 -> 95% CI).")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    symbol_results = payload.get("results") or []

    rng = np.random.default_rng(42)
    header = (
        f"{'symbol':<8} {'n':>3} {'mean%':>8} {'95%CI_lo':>10} "
        f"{'95%CI_hi':>10} {'p(>0)':>9} {'verdict':<14}"
    )
    print()
    print("=" * len(header))
    print(f"Walk-forward OOS return-per-window stats  ({1.0 - args.alpha:.0%} CI)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    all_pooled: List[float] = []
    rejected = 0
    eligible = 0
    for sym_block in symbol_results:
        sym = str(sym_block.get("symbol") or "?")[:7]
        rets = _per_window_returns(sym_block)
        if not rets:
            print(f"{sym:<8} {0:>3} {'-':>8} {'-':>10} {'-':>10} {'-':>9} {'no data':<14}")
            continue
        all_pooled.extend(rets)
        n = len(rets)
        mean, lo, hi = _bootstrap_mean_ci(rets, n_boot=args.bootstrap, alpha=args.alpha, rng=rng)
        p_pos = _one_sample_p_greater_than_zero(rets)

        # Verdict thresholds: clearly positive if 95% CI excludes 0 on the
        # left AND p < 0.05; clearly negative if it excludes 0 on the
        # right; otherwise "no edge".
        if lo > 0 and (math.isnan(p_pos) or p_pos < 0.05):
            verdict = "EDGE+"
            rejected += 1
        elif hi < 0:
            verdict = "EDGE- (lose)"
        else:
            verdict = "no edge"
        eligible += 1

        p_str = "n/a" if math.isnan(p_pos) else f"{p_pos:.3f}"
        print(
            f"{sym:<8} {n:>3} {mean*100:>7.3f}% {lo*100:>9.3f}% {hi*100:>9.3f}% "
            f"{p_str:>9} {verdict:<14}"
        )

    print("-" * len(header))
    if all_pooled:
        n = len(all_pooled)
        mean, lo, hi = _bootstrap_mean_ci(all_pooled, n_boot=args.bootstrap, alpha=args.alpha, rng=rng)
        p_pos = _one_sample_p_greater_than_zero(all_pooled)
        p_str = "n/a" if math.isnan(p_pos) else f"{p_pos:.3f}"
        verdict = "POOLED EDGE+" if (lo > 0) else ("POOLED EDGE-" if hi < 0 else "POOLED no edge")
        print(
            f"{'POOLED':<8} {n:>3} {mean*100:>7.3f}% {lo*100:>9.3f}% {hi*100:>9.3f}% "
            f"{p_str:>9} {verdict:<14}"
        )
    print("=" * len(header))
    if eligible:
        print(
            f"{rejected}/{eligible} symbols reject the null (mean OOS return > 0 at "
            f"{1.0 - args.alpha:.0%} confidence)."
        )


if __name__ == "__main__":
    main()
