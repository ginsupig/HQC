"""
Fetch Alpaca historical 1-minute bars for one or more symbols and write
one CSV per symbol into ``data_dir`` (default: ``data/alpaca``).

Usage:
    python fetch_alpaca.py --symbols SPY
    python fetch_alpaca.py --symbols SPY,AAPL,NVDA,TSLA,AMD,COIN,META --days 183
    python fetch_alpaca.py --symbols SPY --feed iex --out data/alpaca

Reads ALPACA_API_KEY / ALPACA_API_SECRET from .env or the environment.
The IEX feed is free; the SIP feed requires a paid Alpaca data plan.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

URL = "https://data.alpaca.markets/v2/stocks/bars"


def _fetch_one(
    symbol: str,
    start: datetime,
    end: datetime,
    feed: str,
    headers: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    page_token: str | None = None
    while True:
        params: dict = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "end": end.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "limit": 10000,
            "feed": feed,
            "adjustment": "raw",
        }
        if page_token:
            params["page_token"] = page_token
        r = requests.get(URL, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            print(f"[{symbol}] error: {r.status_code} {r.text[:300]}", file=sys.stderr)
            sys.exit(1)
        data = r.json()
        bars = (data.get("bars") or {}).get(symbol) or []
        rows.extend(bars)
        page_token = data.get("next_page_token")
        print(
            f"[{symbol}] fetched {len(bars):>6} (total {len(rows):>7}) "
            f"next={'yes' if page_token else 'no'}",
            file=sys.stderr,
        )
        if not page_token:
            break
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])

    df = pd.DataFrame(rows).rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["symbol"] = symbol
    ts = pd.to_datetime(df["timestamp"], utc=True)
    et = ts.dt.tz_convert("US/Eastern")
    rth_mask = (et.dt.weekday < 5) & (
        ((et.dt.hour == 9) & (et.dt.minute >= 30))
        | ((et.dt.hour > 9) & (et.dt.hour < 16))
    )
    return df[rth_mask][["timestamp", "open", "high", "low", "close", "volume", "symbol"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default="SPY",
        help="Comma-separated tickers (e.g. SPY,AAPL,NVDA).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=183,
        help="History window in calendar days ending ~now (default: 183).",
    )
    parser.add_argument(
        "--feed",
        default="iex",
        choices=["iex", "sip"],
        help="Alpaca data feed; sip requires a paid plan.",
    )
    parser.add_argument(
        "--out",
        default="data/alpaca",
        help="Directory to write per-symbol CSVs into.",
    )
    args = parser.parse_args()

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        print(
            "ALPACA_API_KEY / ALPACA_API_SECRET not set; put them in .env or the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=args.days)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"fetching {len(symbols)} symbols ({args.days}d, feed={args.feed}) -> {out_dir}", file=sys.stderr)

    for sym in symbols:
        df = _fetch_one(sym, start, end, args.feed, headers)
        out = out_dir / f"{sym.lower()}_{args.days}d_1m.csv"
        df.to_csv(out, index=False)
        print(f"[{sym}] wrote {len(df)} RTH bars -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
