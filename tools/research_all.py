"""
research_all — one-shot pair discovery across every sector universe.

Fetches data and runs the full discovery pipeline (B1 correlation -> B2
cointegration/half-life -> B3 batch walk-forward) for each universe in
``config/b1_universes.yaml``, then aggregates every B3 survivor into a single
ranked CSV so you can see the strongest candidates across all sectors at once.

This is the *discovery* front-end. It does NOT deploy anything and its ranking
is diagnostic only — the deploy gate remains the campaign
(``tools/pairs_candidate_campaign.py``) plus A1's family-corrected bar. Use this
to find candidates worth adding to ``config/pairs_research.yaml``; promote only
what survives the campaign.

Pipeline per universe:
    B1  tools/b1_universe_clustering.py  --universe <name>     -> b1_<name>.csv
    B2  tools/b2_candidate_screen.py     --input b1_<name>.csv -> b2_<name>.csv
    B3  tools/b3_batch_walkforward.py    --input b2_<name>.csv -> b3_<name>.csv

Usage:
    # Everything (fetch 730d for every ticker, then B1->B2->B3 per universe):
    python tools/research_all.py

    # Specific universes, reuse already-downloaded data:
    python tools/research_all.py --universes large_cap_financials,semiconductors --skip-fetch

    # Tune the screens:
    python tools/research_all.py --min-corr 0.6 --coint-p-max 0.05 --max-half-life-days 45 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent

# Universes that aren't useful pair-discovery pools by default: the deployed
# sanity pair and the broad index ETFs (SPY/QQQ as a "pair" is just a beta bet).
_DEFAULT_EXCLUDE = ("deployed", "index_etfs")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without running any subprocess)
# --------------------------------------------------------------------------- #
def select_universes(
    universes_yaml: dict,
    selected: Optional[List[str]],
    exclude: List[str],
) -> List[str]:
    """Resolve which universe names to process. ``selected`` None/['all'] means
    every key in the YAML; ``exclude`` is always removed."""
    all_names = list(universes_yaml.keys())
    if not selected or selected == ["all"]:
        names = all_names
    else:
        names = [n for n in selected if n in universes_yaml]
    excl = set(exclude)
    return [n for n in names if n not in excl]


def ticker_union(universes_yaml: dict, names: List[str]) -> List[str]:
    """Sorted unique tickers across the named universes."""
    tickers: set[str] = set()
    for name in names:
        for entry in universes_yaml.get(name, []) or []:
            tk = str((entry or {}).get("ticker", "")).strip().upper()
            if tk:
                tickers.add(tk)
    return sorted(tickers)


def _to_float(v: object) -> Optional[float]:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def aggregate_survivors(
    b3_csv_by_universe: Dict[str, Path],
    alpha: float = 0.05,
) -> List[dict]:
    """Read every B3 output CSV and return one combined, ranked list of rows.

    Adds a ``universe`` column and a diagnostic ``candidate`` flag
    (ci_lo > 0 AND raw_p < alpha — UNCORRECTED, for triage only). Sorted by
    raw_p ascending then mean_pct descending; rows without a usable p sort last.
    """
    rows: List[dict] = []
    for universe, path in b3_csv_by_universe.items():
        if not Path(path).exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("y") or not r.get("x"):
                    continue
                raw_p = _to_float(r.get("raw_p"))
                ci_lo = _to_float(r.get("ci_lo"))
                mean_pct = _to_float(r.get("mean_pct"))
                rows.append({
                    "universe": universe,
                    "y": r["y"], "x": r["x"],
                    "n_windows": r.get("n_windows", ""),
                    "total_trades": r.get("total_trades", ""),
                    "mean_pct": "" if mean_pct is None else f"{mean_pct:.6f}",
                    "ci_lo": "" if ci_lo is None else f"{ci_lo:.6f}",
                    "ci_hi": r.get("ci_hi", ""),
                    "raw_p": "" if raw_p is None else f"{raw_p:.6f}",
                    "candidate": "1" if (ci_lo is not None and ci_lo > 0
                                         and raw_p is not None and raw_p < alpha) else "0",
                    "total_pnl": r.get("total_pnl", ""),
                    "error": r.get("error", ""),
                })

    def _key(row: dict) -> Tuple[float, float]:
        p = _to_float(row["raw_p"])
        m = _to_float(row["mean_pct"]) or 0.0
        return (p if p is not None else 9.0, -m)

    rows.sort(key=_key)
    return rows


# --------------------------------------------------------------------------- #
# Subprocess orchestration
# --------------------------------------------------------------------------- #
def _run(cmd: List[str], log_path: Path, dry_run: bool) -> bool:
    """Run a step, tee output to log_path. Returns True on success (non-fatal:
    a failed step is logged and skipped so other universes still run)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(cmd)}", flush=True)
    if dry_run:
        log_path.write_text("DRY RUN:\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return True
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, text=True, capture_output=True, check=False)
    log_path.write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        print(f"    FAILED (rc={proc.returncode}); see {log_path}. last: {' | '.join(tail)}", flush=True)
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universes-yaml", type=Path, default=_REPO_ROOT / "config" / "b1_universes.yaml")
    p.add_argument("--universes", default="all",
                   help="Comma-separated universe names, or 'all' (default).")
    p.add_argument("--exclude", default=",".join(_DEFAULT_EXCLUDE),
                   help=f"Comma-separated universes to skip (default: {','.join(_DEFAULT_EXCLUDE)}).")
    p.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data" / "alpaca")
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "state" / "research_all")
    # Fetch
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--feed", default="iex", choices=["iex", "sip"])
    p.add_argument("--skip-fetch", action="store_true", help="Reuse CSVs already in --data-dir.")
    # B1 / B2 knobs
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD anti-look-ahead date (default: today).")
    p.add_argument("--min-corr", type=float, default=0.55)
    p.add_argument("--min-overlap-days", type=int, default=252)
    p.add_argument("--coint-p-max", type=float, default=0.05)
    p.add_argument("--max-half-life-days", type=float, default=60.0)
    # B3
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="Pass --resume to B3 (skip already-scored pairs).")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    args = p.parse_args(argv)

    if not args.universes_yaml.exists():
        print(f"ERROR: universes YAML not found: {args.universes_yaml}", file=sys.stderr)
        return 2
    universes_yaml = yaml.safe_load(args.universes_yaml.read_text(encoding="utf-8")) or {}

    selected = [s.strip() for s in args.universes.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]
    names = select_universes(universes_yaml, selected, exclude)
    if not names:
        print("ERROR: no universes selected after exclusions.", file=sys.stderr)
        return 2

    tickers = ticker_union(universes_yaml, names)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"research_all: {len(names)} universes, {len(tickers)} unique tickers -> {args.out_dir}")
    print(f"  universes: {', '.join(names)}")

    # 1) Fetch all tickers once.
    if not args.skip_fetch:
        print(f"\n[fetch] {len(tickers)} tickers, {args.days}d, feed={args.feed}")
        ok = _run(
            [sys.executable, str(_REPO_ROOT / "fetch_alpaca.py"),
             "--symbols", ",".join(tickers), "--days", str(args.days),
             "--feed", args.feed, "--out", str(args.data_dir)],
            args.out_dir / "fetch.log", args.dry_run,
        )
        if not ok:
            print("  fetch failed — aborting (no point running screens on missing data).", file=sys.stderr)
            return 1
    else:
        print("\n[fetch] skipped (--skip-fetch)")

    # 2) B1 -> B2 -> B3 per universe.
    b3_by_universe: Dict[str, Path] = {}
    for name in names:
        uni_dir = args.out_dir / name
        uni_dir.mkdir(parents=True, exist_ok=True)
        b1_csv, b2_csv, b3_csv = uni_dir / "b1.csv", uni_dir / "b2.csv", uni_dir / "b3.csv"
        print(f"\n[{name}]")

        b1_cmd = [sys.executable, str(_TOOLS_DIR / "b1_universe_clustering.py"),
                  "--universe", name, "--universes-yaml", str(args.universes_yaml),
                  "--data-dir", str(args.data_dir), "--min-corr", str(args.min_corr),
                  "--min-overlap-days", str(args.min_overlap_days), "--output", str(b1_csv)]
        if args.as_of:
            b1_cmd += ["--as-of", args.as_of]
        if not _run(b1_cmd, uni_dir / "b1.log", args.dry_run):
            continue

        b2_cmd = [sys.executable, str(_TOOLS_DIR / "b2_candidate_screen.py"),
                  "--input", str(b1_csv), "--data-dir", str(args.data_dir),
                  "--coint-p-max", str(args.coint_p_max),
                  "--max-half-life-days", str(args.max_half_life_days), "--output", str(b2_csv)]
        if args.as_of:
            b2_cmd += ["--as-of", args.as_of]
        if not _run(b2_cmd, uni_dir / "b2.log", args.dry_run):
            continue

        b3_cmd = [sys.executable, str(_TOOLS_DIR / "b3_batch_walkforward.py"),
                  "--input", str(b2_csv), "--data-dir", str(args.data_dir),
                  "--workers", str(args.workers), "--output", str(b3_csv)]
        if args.resume:
            b3_cmd += ["--resume"]
        if not _run(b3_cmd, uni_dir / "b3.log", args.dry_run):
            continue
        b3_by_universe[name] = b3_csv

    # 3) Aggregate survivors across universes.
    if args.dry_run:
        print("\nDry-run complete (no data processed).")
        return 0

    rows = aggregate_survivors(b3_by_universe, alpha=args.alpha)
    survivors_csv = args.out_dir / "survivors.csv"
    fieldnames = ["universe", "y", "x", "n_windows", "total_trades",
                  "mean_pct", "ci_lo", "ci_hi", "raw_p", "candidate", "total_pnl", "error"]
    with open(survivors_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    candidates = [r for r in rows if r["candidate"] == "1"]
    print("\n" + "=" * 72)
    print(f"research_all done. {len(rows)} pairs scored across {len(b3_by_universe)} universes.")
    print(f"  {len(candidates)} diagnostic candidates (ci_lo>0 & raw_p<{args.alpha}, UNCORRECTED).")
    print(f"  full ranking: {survivors_csv}")
    print("=" * 72)
    print(f"{'universe':<24} {'pair':<12} {'raw_p':>8} {'mean%':>8} {'cand':>5}")
    print("-" * 60)
    for r in rows[:20]:
        rp = r["raw_p"] or "n/a"
        mp = f"{float(r['mean_pct']):+.3f}" if r["mean_pct"] else "n/a"
        print(f"{r['universe']:<24} {r['y'] + '/' + r['x']:<12} {rp:>8} {mp:>8} {r['candidate']:>5}")
    print("\nNext: add the strongest into config/pairs_research.yaml (bump "
          "pre_registered_family_size), then run the campaign for the deploy gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
