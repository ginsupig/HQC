"""
B1 — Universe + within-sector clustering for pair-discovery.

Roadmap reference: Workstream B, task B1. Produces a CSV of candidate (y, x)
pairs that B2 (cointegration + Hurst screen) will further filter, which in
turn feeds B3 (batch walk-forward).

What B1 does:
  1. Load a curated universe from ``config/b1_universes.yaml`` (or any YAML in
     the same shape).
  2. For each ticker in the universe, load its 1-minute CSV from --data-dir
     (filename pattern: ``<lower>_<days>d_1m.csv``; tolerates any --days).
  3. Resample to daily close. Compute daily log returns.
  4. Within each sector, compute the pairwise return correlation matrix.
  5. Emit every (y, x) ordered pair with y < x alphabetically, sector,
     n_overlap_days, corr, and a "deployment_eligible" flag based on minimum
     correlation and minimum overlap thresholds.
  6. Anti-look-ahead by default: only sessions strictly before --as-of are used
     so a B1 run can be redone on the same as-of date and produce the same
     candidate list. The as-of date is recorded in the output.

What B1 does NOT do:
  - It does NOT score cointegration. That is B2's job (cointegration is
    expensive and worth doing only after the cheap correlation filter has
    cut the candidate pool).
  - It does NOT decide deployment. The deployment_eligible column is a
    correlation-threshold filter, not a significance claim.
  - It does NOT fetch data. Missing tickers are reported and skipped; use
    ``python fetch_alpaca.py`` to backfill.

Usage:
  python tools/b1_universe_clustering.py \\
      --universe large_cap_financials \\
      --data-dir data/alpaca \\
      --as-of 2026-05-26 \\
      --min-corr 0.55 \\
      --min-overlap-days 252 \\
      --output b1_candidates_financials.csv

  python tools/b1_universe_clustering.py --universe sector_etfs --output b1_candidates_etfs.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Universe loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UniverseEntry:
    ticker: str
    sector: str


def load_universe(yaml_path: Path, universe_name: str) -> List[UniverseEntry]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if universe_name not in doc:
        available = ", ".join(sorted(doc.keys()))
        raise SystemExit(
            f"Universe '{universe_name}' not found in {yaml_path}. "
            f"Available: {available}"
        )
    entries: List[UniverseEntry] = []
    for row in doc[universe_name]:
        entries.append(UniverseEntry(ticker=row["ticker"].upper(), sector=row["sector"]))
    if not entries:
        raise SystemExit(f"Universe '{universe_name}' is empty.")
    return entries


# --------------------------------------------------------------------------- #
# CSV ingestion + daily-close derivation (no pandas dependency)
# --------------------------------------------------------------------------- #
def _find_csv(data_dir: Path, ticker: str) -> Optional[Path]:
    """Find a CSV for ``ticker`` in ``data_dir``. Pattern: ``<lower>_*_1m.csv``.

    If multiple match (different lookback windows), return the lexicographically
    last (which is usually the longest window because '730d' > '180d').
    """
    pattern = f"{ticker.lower()}_*_1m.csv"
    matches = sorted(data_dir.glob(pattern))
    return matches[-1] if matches else None


def _load_daily_closes(csv_path: Path) -> Dict[dt.date, float]:
    """Read a 1-minute OHLC CSV and return {trade_date: last_close_of_day}.

    Tolerates either ISO timestamps or epoch-ms in the ``timestamp`` column,
    and either ``close`` or ``c`` for the close price column. Bars whose
    timestamp falls outside 09:30-16:00 ET are ignored so daily close reflects
    the regular session.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    session_open = dt.time(9, 30)
    session_close = dt.time(16, 0)
    closes: Dict[dt.date, float] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ts_col = "timestamp" if "timestamp" in (reader.fieldnames or []) else "t"
        close_col = "close" if "close" in (reader.fieldnames or []) else "c"
        for row in reader:
            ts_raw = row.get(ts_col)
            if ts_raw is None:
                continue
            try:
                if ts_raw.isdigit():
                    ts = dt.datetime.fromtimestamp(int(ts_raw) / 1000.0, tz=dt.timezone.utc)
                else:
                    ts = dt.datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            ts_et = ts.astimezone(et)
            if not (session_open <= ts_et.time() < session_close):
                continue
            try:
                price = float(row[close_col])
            except (TypeError, ValueError, KeyError):
                continue
            # last-write-wins → on rolling through a session, the loop ends on
            # the final regular-session bar of the day, which is what we want.
            closes[ts_et.date()] = price
    return closes


# --------------------------------------------------------------------------- #
# Correlation on aligned daily log returns
# --------------------------------------------------------------------------- #
def _aligned_returns(
    a: Dict[dt.date, float], b: Dict[dt.date, float], as_of: dt.date
) -> Tuple[List[float], List[float]]:
    """Return (log_ret_a, log_ret_b) on the intersection of dates strictly
    before ``as_of``. Day-over-day log returns from sorted shared dates."""
    shared = sorted(d for d in a.keys() & b.keys() if d < as_of)
    if len(shared) < 2:
        return [], []
    ra: List[float] = []
    rb: List[float] = []
    for i in range(1, len(shared)):
        prev, cur = shared[i - 1], shared[i]
        pa_prev, pa_cur = a[prev], a[cur]
        pb_prev, pb_cur = b[prev], b[cur]
        if pa_prev <= 0 or pb_prev <= 0 or pa_cur <= 0 or pb_cur <= 0:
            continue
        ra.append(math.log(pa_cur / pa_prev))
        rb.append(math.log(pb_cur / pb_prev))
    return ra, rb


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2 or n != len(y):
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _within_sector_pairs(entries: List[UniverseEntry]) -> Iterable[Tuple[UniverseEntry, UniverseEntry]]:
    """All ordered (y, x) pairs with y.ticker < x.ticker alphabetically AND
    same sector. Cross-sector pairs are intentionally excluded — cross-sector
    cointegration tends to overfit at the scales this harness operates on."""
    by_sector: Dict[str, List[UniverseEntry]] = {}
    for e in entries:
        by_sector.setdefault(e.sector, []).append(e)
    for sector, group in by_sector.items():
        group_sorted = sorted(group, key=lambda e: e.ticker)
        for i in range(len(group_sorted)):
            for j in range(i + 1, len(group_sorted)):
                yield group_sorted[i], group_sorted[j]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", required=True,
                   help="Name of the universe in --universes-yaml (e.g. large_cap_financials).")
    p.add_argument("--universes-yaml", type=Path,
                   default=_REPO_ROOT / "config" / "b1_universes.yaml",
                   help="Path to the universes YAML.")
    p.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data" / "alpaca",
                   help="Directory of <ticker>_*_1m.csv files.")
    p.add_argument("--as-of", type=str, default=None,
                   help="Date (YYYY-MM-DD) — only sessions strictly BEFORE this date "
                        "are used. Defaults to today (UTC) so a run on a given day "
                        "uses prior-session data only and is reproducible.")
    p.add_argument("--min-corr", type=float, default=0.55,
                   help="Minimum Pearson correlation on daily log returns for "
                        "'deployment_eligible' flag.")
    p.add_argument("--min-overlap-days", type=int, default=252,
                   help="Minimum overlapping regular-session days for "
                        "'deployment_eligible' flag (252 ≈ 1 trading year).")
    p.add_argument("--output", required=True, type=Path,
                   help="CSV output path. Overwrites if present.")
    args = p.parse_args()

    if args.as_of is None:
        as_of = dt.date.today()
    else:
        as_of = dt.date.fromisoformat(args.as_of)

    entries = load_universe(args.universes_yaml, args.universe)
    print(f"B1: universe={args.universe} entries={len(entries)} as_of={as_of}")

    # Load daily closes once per ticker; report missing.
    closes_by_ticker: Dict[str, Dict[dt.date, float]] = {}
    missing: List[str] = []
    for e in entries:
        csv_path = _find_csv(args.data_dir, e.ticker)
        if csv_path is None:
            missing.append(e.ticker)
            continue
        closes_by_ticker[e.ticker] = _load_daily_closes(csv_path)
    if missing:
        print(f"  WARN: missing CSVs (skipped): {', '.join(missing)}")
    if not closes_by_ticker:
        raise SystemExit("No tickers loaded — check --data-dir and CSV filenames.")

    available_entries = [e for e in entries if e.ticker in closes_by_ticker]

    # Score within-sector pairs.
    rows: List[dict] = []
    eligible = 0
    for y, x in _within_sector_pairs(available_entries):
        ra, rb = _aligned_returns(closes_by_ticker[y.ticker], closes_by_ticker[x.ticker], as_of)
        n = len(ra)
        corr = _pearson(ra, rb) if n >= 2 else None
        is_eligible = (
            corr is not None
            and corr >= args.min_corr
            and n >= args.min_overlap_days
        )
        if is_eligible:
            eligible += 1
        rows.append({
            "y": y.ticker,
            "x": x.ticker,
            "sector": y.sector,
            "n_overlap_days": n,
            "corr": "" if corr is None else f"{corr:.6f}",
            "deployment_eligible": "1" if is_eligible else "0",
            "as_of": as_of.isoformat(),
            "universe": args.universe,
        })

    # Sort: eligible first, then by correlation descending.
    rows.sort(key=lambda r: (
        0 if r["deployment_eligible"] == "1" else 1,
        -float(r["corr"]) if r["corr"] else 0.0,
    ))

    fieldnames = ["y", "x", "sector", "n_overlap_days", "corr",
                  "deployment_eligible", "as_of", "universe"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"  wrote {len(rows)} candidate pairs to {args.output}")
    print(f"  deployment_eligible (corr>={args.min_corr}, overlap>={args.min_overlap_days}): {eligible}")
    print(f"  next step: feed --output to B2 (tools/b2_candidate_screen.py)")


if __name__ == "__main__":
    main()
