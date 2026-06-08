"""
Show trades from the live/paper feedback log.

main_pairs.py (and main.py) write a per-fill/outcome record to
``state/feedback/outcomes.jsonl`` via UnifiedFeedbackLogger. This tool reads
that log, filters to a single day (default: today, UTC), and prints a per-row
blotter plus daily totals (fills, closed trades, gross/net PnL, win rate).

It does NOT talk to the broker — it reflects what the running system recorded.
The Alpaca paper/live account remains the source of truth for fills; use this
for a quick local read of what the bot did.

Usage:
    # today (UTC)
    python tools/show_trades.py

    # a specific day, and also list entry decisions
    python tools/show_trades.py --date 2026-06-08 --decisions

    # everything in the log
    python tools/show_trades.py --all
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(path: Path) -> List[dict]:
    """Read a .jsonl file, skipping blank/corrupt lines. Missing file -> []."""
    if not path.exists():
        return []
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def filter_by_date(rows: List[dict], date_str: Optional[str]) -> List[dict]:
    """Keep rows whose UTC ts (ISO 'YYYY-MM-DD...') falls on date_str.

    date_str None -> return all rows unchanged.
    """
    if date_str is None:
        return list(rows)
    return [r for r in rows if str(r.get("ts", "")).startswith(date_str)]


def _f(v) -> Optional[float]:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def summarize(outcomes: List[dict]) -> Dict[str, float]:
    """Daily totals. 'Closed trades' = outcome rows with a net_pnl value."""
    closed = [r for r in outcomes if _f(r.get("net_pnl")) is not None]
    nets = [_f(r.get("net_pnl")) for r in closed]
    gross = [_f(r.get("gross_pnl")) for r in closed if _f(r.get("gross_pnl")) is not None]
    wins = sum(1 for n in nets if n is not None and n > 0)
    return {
        "fills": len(outcomes),
        "closed": len(closed),
        "gross_pnl": sum(g for g in gross),
        "net_pnl": sum(n for n in nets if n is not None),
        "wins": wins,
        "win_rate": (wins / len(closed)) if closed else 0.0,
    }


def _hhmmss(ts: str) -> str:
    # ts is ISO-8601 UTC like '2026-06-08T14:30:00+00:00'; show the clock part.
    return ts[11:19] if "T" in ts and len(ts) >= 19 else ts


def format_blotter(outcomes: List[dict]) -> str:
    if not outcomes:
        return "  (no outcome rows)"
    header = f"  {'time(UTC)':9}  {'symbol':6} {'side':10} {'status':9} {'fill_qty':>8} {'fill_px':>10} {'net_pnl':>10} {'hold_s':>7}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for r in outcomes:
        net = _f(r.get("net_pnl"))
        px = _f(r.get("fill_price"))
        hold = _f(r.get("hold_seconds"))
        lines.append(
            f"  {_hhmmss(str(r.get('ts','')))!s:9}  "
            f"{str(r.get('symbol') or '-'):6} "
            f"{str(r.get('side') or '-'):10} "
            f"{str(r.get('status') or '-'):9} "
            f"{(r.get('filled_qty') if r.get('filled_qty') is not None else '-')!s:>8} "
            f"{('-' if px is None else f'{px:.2f}'):>10} "
            f"{('-' if net is None else f'{net:+.2f}'):>10} "
            f"{('-' if hold is None else f'{hold:.0f}'):>7}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="state/feedback",
                   help="Feedback log directory (default: state/feedback).")
    p.add_argument("--date", default=None,
                   help="UTC day to show as YYYY-MM-DD (default: today).")
    p.add_argument("--all", action="store_true",
                   help="Show every row regardless of date.")
    p.add_argument("--decisions", action="store_true",
                   help="Also list entry decisions for the day.")
    args = p.parse_args(argv)

    root = Path(args.root)
    date_str = None if args.all else (args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    outcomes = filter_by_date(load_jsonl(root / "outcomes.jsonl"), date_str)
    label = "all dates" if date_str is None else date_str

    print(f"Trades for {label}  (source: {root}/outcomes.jsonl)")
    print(format_blotter(outcomes))

    s = summarize(outcomes)
    print()
    print(f"  fills/legs        : {s['fills']}")
    print(f"  closed trades     : {s['closed']}")
    print(f"  gross PnL         : {s['gross_pnl']:+.2f}")
    print(f"  net PnL           : {s['net_pnl']:+.2f}")
    print(f"  wins / win rate   : {s['wins']} / {s['win_rate']*100:.1f}%")

    if args.decisions:
        decisions = filter_by_date(load_jsonl(root / "decisions.jsonl"), date_str)
        print()
        print(f"Decisions for {label}: {len(decisions)}")
        for d in decisions:
            print(f"  {_hhmmss(str(d.get('ts','')))}  {d.get('symbol') or '-':6} "
                  f"{str(d.get('side') or '-'):10} approved={d.get('approved')} score={d.get('score')}")

    if not outcomes and date_str is not None:
        print()
        print("  No trades recorded for this day. If the bot is running, it may not have")
        print("  fired yet (z-score never crossed entry), or check --date / --root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
