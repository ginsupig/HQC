"""
B2 — Cointegration + half-life pre-filter on B1 candidate pairs.

Roadmap reference: Workstream B, task B2. Cheap but more discriminating than
B1's correlation filter. Drops pairs whose y-on-x residual is not stationary
(Engle-Granger fails) or whose residual mean-reverts too slowly to be tradable
on the harness's walk-forward windows. Surviving pairs feed B3.

Statistical rules (frozen for this screen, not configurable post-hoc):
  - Engle-Granger cointegration test on log(y) vs log(x) via
    statsmodels.tsa.stattools.coint. Default trend = 'c' (constant only,
    no time trend), so the residual is tested for stationarity around a
    constant mean. A pair PASSES the cointegration leg only if
    p < --coint-p-max (default 0.05).
  - Half-life of mean reversion from an OU fit
    (Δresid_t = -theta * resid_{t-1} + eps; half-life = ln(2)/theta).
    A pair PASSES the tradability leg only if half_life <= --max-half-life-days
    (default 60). Pairs with longer half-lives nominally cointegrate but
    mean-revert too slowly to be tradable on 30/10 walk-forward windows.
  - Hurst exponent of the residual is REPORTED as a diagnostic (rescaled-
    range estimator) but is NOT used as a gate. Cointegration already tests
    residual stationarity and half-life captures reversion speed — gating
    on Hurst on top adds noise without adding signal. Operators should
    read it alongside coint_p / half_life, not in isolation.

A pair is screen_pass = 1 only if ALL of:
  coint_p < coint_p_max  AND  half_life <= max_half_life_days

What B2 does NOT do:
  - It does NOT score the pair's PnL. That is B3's job (walk-forward).
  - It does NOT decide deployment. screen_pass is a "worth running B3 on"
    flag, not a significance claim. A1 still owes the family-corrected
    p-value at family = K survivors.
  - It does NOT re-fetch data. B2 consumes B1's CSV; if a B1 row references
    a missing CSV the pair is reported and skipped.

Usage:
  python tools/b2_candidate_screen.py \\
      --input b1_financials.csv \\
      --data-dir data/alpaca \\
      --as-of 2026-05-26 \\
      --coint-p-max 0.05 \\
      --max-half-life-days 60 \\
      --output b2_financials.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from statsmodels.tsa.stattools import coint

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the daily-close loader from B1 — same regular-session filter, same
# CSV-pattern discovery. Tools should share, not duplicate.
from tools.b1_universe_clustering import _find_csv, _load_daily_closes  # noqa: E402


# --------------------------------------------------------------------------- #
# Hurst exponent via rescaled-range analysis
# --------------------------------------------------------------------------- #
def _hurst_rs(series: np.ndarray, min_chunks: int = 8) -> Optional[float]:
    """Estimate the Hurst exponent of ``series`` via the rescaled-range method.

    H < 0.5 → mean-reverting; H == 0.5 → random walk; H > 0.5 → trending.

    Returns None if the series is too short for at least ``min_chunks`` chunks.
    """
    n = len(series)
    if n < 2 * min_chunks:
        return None
    # Use log-spaced chunk sizes between (n/min_chunks) and 8.
    max_chunk = n // min_chunks
    min_chunk = 8
    if max_chunk < min_chunk:
        return None
    # log-spaced unique integers
    chunk_sizes = sorted(set(
        int(round(s))
        for s in np.geomspace(min_chunk, max_chunk, num=10)
    ))
    if len(chunk_sizes) < 2:
        return None

    log_rs: List[float] = []
    log_n: List[float] = []
    for size in chunk_sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_vals: List[float] = []
        for k in range(n_chunks):
            chunk = series[k * size:(k + 1) * size]
            mean = chunk.mean()
            dev = chunk - mean
            cumdev = np.cumsum(dev)
            r = cumdev.max() - cumdev.min()
            s = chunk.std(ddof=1)
            if s > 0 and r > 0:
                rs_vals.append(r / s)
        if rs_vals:
            log_rs.append(math.log(float(np.mean(rs_vals))))
            log_n.append(math.log(size))
    if len(log_rs) < 2:
        return None
    # H = slope of log(R/S) vs log(n)
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(slope)


# --------------------------------------------------------------------------- #
# Half-life via Ornstein-Uhlenbeck fit: Δresid_t = -theta * resid_{t-1} + eps
# --------------------------------------------------------------------------- #
def _half_life(residual: np.ndarray) -> Optional[float]:
    """OU half-life of mean reversion in periods (days here). None on failure."""
    if len(residual) < 3:
        return None
    r_lag = residual[:-1]
    r_diff = np.diff(residual)
    # OLS: r_diff = -theta * r_lag + epsilon
    A = np.column_stack([r_lag, np.ones_like(r_lag)])
    try:
        coef, *_ = np.linalg.lstsq(A, r_diff, rcond=None)
    except np.linalg.LinAlgError:
        return None
    theta = -coef[0]
    if theta <= 0 or not math.isfinite(theta):
        # Not mean-reverting (or numerically zero)
        return None
    return float(math.log(2.0) / theta)


# --------------------------------------------------------------------------- #
# OLS beta + residual (Engle-Granger first step)
# --------------------------------------------------------------------------- #
def _ols_residual(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Return (alpha, beta, residual) from OLS y = alpha + beta * x + eps."""
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - (alpha + beta * x)
    return alpha, beta, resid


# --------------------------------------------------------------------------- #
# Per-pair scoring
# --------------------------------------------------------------------------- #
def _aligned_levels(
    a: Dict[dt.date, float], b: Dict[dt.date, float], as_of: dt.date
) -> Tuple[List[dt.date], np.ndarray, np.ndarray]:
    """Return (dates, y_levels, x_levels) for the intersection of dates strictly
    before ``as_of`` — log prices used downstream."""
    shared = sorted(d for d in a.keys() & b.keys() if d < as_of)
    y = np.array([math.log(a[d]) for d in shared], dtype=float)
    x = np.array([math.log(b[d]) for d in shared], dtype=float)
    return shared, y, x


def _score_pair(
    y_closes: Dict[dt.date, float],
    x_closes: Dict[dt.date, float],
    as_of: dt.date,
) -> dict:
    dates, y, x = _aligned_levels(y_closes, x_closes, as_of)
    n = len(dates)
    out = {
        "n_overlap_days": n,
        "coint_t": "",
        "coint_p": "",
        "beta": "",
        "alpha": "",
        "hurst": "",
        "half_life_days": "",
        "screen_pass": "0",
        "fail_reason": "",
    }
    if n < 60:
        out["fail_reason"] = "insufficient_history"
        return out
    # Engle-Granger via statsmodels (regresses y on x with constant, then ADF
    # on residual, uses MacKinnon p-value tables for cointegration).
    try:
        t_stat, p_val, _ = coint(y, x, trend="c")
    except Exception as e:  # extremely rare — usually rank-deficient inputs
        out["fail_reason"] = f"coint_error:{type(e).__name__}"
        return out
    out["coint_t"] = f"{t_stat:.6f}"
    out["coint_p"] = f"{p_val:.6f}"
    # OLS for beta + residual we'll measure Hurst / half-life on
    alpha, beta, resid = _ols_residual(y, x)
    out["alpha"] = f"{alpha:.6f}"
    out["beta"] = f"{beta:.6f}"
    hurst = _hurst_rs(resid)
    if hurst is not None:
        out["hurst"] = f"{hurst:.6f}"
    half_life = _half_life(resid)
    if half_life is not None:
        out["half_life_days"] = f"{half_life:.3f}"
    return out


def _apply_screen(
    scored: dict,
    coint_p_max: float,
    max_half_life_days: float,
) -> dict:
    """Set screen_pass + fail_reason on a scored row.

    Gates: coint_p < coint_p_max AND 0 < half_life <= max_half_life_days.
    Hurst is reported but NOT used as a gate (see module docstring).
    """
    if scored.get("fail_reason"):
        return scored

    failures: List[str] = []
    coint_p = float(scored["coint_p"]) if scored["coint_p"] else float("nan")
    hl = float(scored["half_life_days"]) if scored["half_life_days"] else float("nan")

    if not math.isfinite(coint_p) or coint_p >= coint_p_max:
        failures.append(f"coint_p>={coint_p_max}")
    if not math.isfinite(hl) or hl > max_half_life_days:
        failures.append(f"half_life>{max_half_life_days}")

    if failures:
        scored["fail_reason"] = ";".join(failures)
        scored["screen_pass"] = "0"
    else:
        scored["screen_pass"] = "1"
    return scored


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="B1 output CSV (b1_*.csv).")
    p.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data" / "alpaca",
                   help="Directory of <ticker>_*_1m.csv files.")
    p.add_argument("--as-of", type=str, default=None,
                   help="YYYY-MM-DD; only sessions strictly before are used. "
                        "Defaults to as_of column from B1 input (or today).")
    p.add_argument("--coint-p-max", type=float, default=0.05)
    p.add_argument("--max-half-life-days", type=float, default=60.0)
    p.add_argument("--eligible-only", action="store_true",
                   help="Only score pairs B1 flagged deployment_eligible=1. "
                        "Default: score every pair B1 produced.")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    # Read B1 rows
    with open(args.input, newline="", encoding="utf-8") as f:
        b1_rows = list(csv.DictReader(f))
    if not b1_rows:
        raise SystemExit(f"No rows in {args.input}")

    if args.as_of is None:
        # Prefer the as_of recorded in B1's output for reproducibility.
        as_of_str = b1_rows[0].get("as_of") or dt.date.today().isoformat()
        as_of = dt.date.fromisoformat(as_of_str)
    else:
        as_of = dt.date.fromisoformat(args.as_of)
    print(f"B2: input={args.input}  as_of={as_of}  rows={len(b1_rows)}")

    # Load closes lazily per ticker (cache so we don't reread on every pair)
    closes_cache: Dict[str, Optional[Dict[dt.date, float]]] = {}

    def _get_closes(ticker: str) -> Optional[Dict[dt.date, float]]:
        if ticker not in closes_cache:
            path = _find_csv(args.data_dir, ticker)
            closes_cache[ticker] = _load_daily_closes(path) if path else None
        return closes_cache[ticker]

    scored_rows: List[dict] = []
    pass_count = 0
    skipped_missing = 0
    skipped_filter = 0

    for r in b1_rows:
        if args.eligible_only and r.get("deployment_eligible") != "1":
            skipped_filter += 1
            continue
        y_tk, x_tk = r["y"], r["x"]
        y_closes = _get_closes(y_tk)
        x_closes = _get_closes(x_tk)
        base = {
            "y": y_tk,
            "x": x_tk,
            "sector": r.get("sector", ""),
            "as_of": as_of.isoformat(),
            "universe": r.get("universe", ""),
            "b1_corr": r.get("corr", ""),
            "b1_overlap_days": r.get("n_overlap_days", ""),
        }
        if y_closes is None or x_closes is None:
            base.update({
                "n_overlap_days": "0",
                "coint_t": "", "coint_p": "",
                "beta": "", "alpha": "",
                "hurst": "", "half_life_days": "",
                "screen_pass": "0",
                "fail_reason": "missing_csv",
            })
            scored_rows.append(base)
            skipped_missing += 1
            continue
        scored = _score_pair(y_closes, x_closes, as_of)
        scored = _apply_screen(
            scored,
            coint_p_max=args.coint_p_max,
            max_half_life_days=args.max_half_life_days,
        )
        base.update(scored)
        if base["screen_pass"] == "1":
            pass_count += 1
        scored_rows.append(base)

    # Sort: passes first, then by coint p ascending (strongest cointegration first)
    def _sort_key(row: dict):
        coint_p = float(row["coint_p"]) if row["coint_p"] else 1.0
        return (0 if row["screen_pass"] == "1" else 1, coint_p)

    scored_rows.sort(key=_sort_key)

    fieldnames = [
        "y", "x", "sector", "n_overlap_days",
        "coint_t", "coint_p", "alpha", "beta",
        "hurst", "half_life_days",
        "screen_pass", "fail_reason",
        "as_of", "universe", "b1_corr", "b1_overlap_days",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in scored_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"  scored={len(scored_rows)}  pass={pass_count}  "
          f"missing_csv={skipped_missing}  skipped_by_filter={skipped_filter}")
    print(f"  thresholds: coint_p<{args.coint_p_max}  "
          f"half_life<={args.max_half_life_days}d  (hurst reported but not gated)")
    print(f"  wrote {args.output}")
    print(f"  next step: feed {args.output} to B3 (walkforward_basket.py)")


if __name__ == "__main__":
    main()
