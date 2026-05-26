"""
Regime tagger for walk-forward windows.

Two crude but defensible regime axes from the roadmap (workstream A4):

  - SPY EMA20 trend  : daily SPY close vs its 20-day EMA. If close >
    EMA, label "up"; else "down". Captures the trend / mean-reversion
    distinction that bank pairs in particular are sensitive to (banks
    tend to underperform in down-trend rate-cutting cycles).

  - Realized-vol tercile : 20-day rolling std of SPY daily returns,
    bucketed into thirds across the dataset. Proxy for VIX without
    requiring a separate VIX data fetch. Bucket labels: "low", "mid",
    "high".

Each trading day gets a combined label like "up_low" or "down_high"
(2 trend states x 3 vol buckets = 6 regimes). Per-window aggregation
uses the dominant regime across the window's RTH dates.

This is intentionally simple. A more sophisticated regime model
(HMM, factor-rotation) is out of scope for A4; the question A4
answers is whether the EDGE+ verdict on JPM/BAC concentrates in
a particular slice of market conditions, and a coarse split is
enough to surface that concentration if it exists.

Usage as a library:
    from tools.regime_tagger import RegimeTagger
    tagger = RegimeTagger.from_csv(
        Path("data/alpaca/spy_730d_1m.csv"),
        ema_span=20,
        vol_window=20,
    )
    label = tagger.label_for_date(date(2024, 9, 12))
    label = tagger.dominant_label_for_period(date(2024, 9, 12), date(2024, 9, 23))
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("RegimeTagger")


@dataclass
class RegimeTagger:
    """Per-trading-date regime labels derived from a single benchmark CSV
    (usually SPY)."""

    label_by_date: Dict[date, str]
    trend_by_date: Dict[date, str]
    vol_bucket_by_date: Dict[date, str]
    daily_close: pd.Series   # date-indexed for diagnostics
    ema: pd.Series
    realized_vol: pd.Series
    vol_terciles: Tuple[float, float]   # (low/mid boundary, mid/high boundary)
    ema_span: int
    vol_window: int

    @staticmethod
    def from_csv(
        path: Path,
        ema_span: int = 20,
        vol_window: int = 20,
        symbol: str = "SPY",
    ) -> "RegimeTagger":
        """Build a tagger from an OHLCV bar CSV at any cadence.
        Resamples to daily close (last bar of the trading day in ET)."""
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{path}: expected a 'timestamp' column.")
        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
        if df.empty:
            raise ValueError(f"{path}: no rows for symbol={symbol}")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        # Resample by trading-day in ET so the last bar of each session
        # becomes the day's closing tick.
        df["date_et"] = df["timestamp"].dt.tz_convert("US/Eastern").dt.date
        daily = (
            df.groupby("date_et")["close"]
            .last()
            .astype(float)
        )
        daily.index = pd.Index([d for d in daily.index], name="date")
        return RegimeTagger.from_daily_close(
            daily, ema_span=ema_span, vol_window=vol_window
        )

    @staticmethod
    def from_daily_close(
        daily_close: pd.Series,
        ema_span: int = 20,
        vol_window: int = 20,
    ) -> "RegimeTagger":
        if len(daily_close) < max(ema_span, vol_window) + 5:
            raise ValueError(
                f"Need at least {max(ema_span, vol_window) + 5} daily closes; "
                f"got {len(daily_close)}."
            )
        ema = daily_close.ewm(span=ema_span, adjust=False).mean()
        log_ret = np.log(daily_close / daily_close.shift(1))
        realized_vol = log_ret.rolling(vol_window).std()
        # Terciles computed across the full sample (in-sample-only would be
        # leaky for live regime tagging; in this offline analyzer the labels
        # are uniform across the dataset by construction).
        valid_vol = realized_vol.dropna()
        if len(valid_vol) < 6:
            raise ValueError("Insufficient samples to compute vol terciles.")
        t_lo, t_hi = np.quantile(valid_vol.values, [1 / 3, 2 / 3])

        trend_by_date: Dict[date, str] = {}
        vol_by_date: Dict[date, str] = {}
        label_by_date: Dict[date, str] = {}
        for d in daily_close.index:
            try:
                px = float(daily_close.loc[d])
                ma = float(ema.loc[d])
                v = float(realized_vol.loc[d]) if not pd.isna(realized_vol.loc[d]) else None
            except (KeyError, ValueError):
                continue
            trend = "up" if px > ma else "down"
            if v is None:
                bucket = "mid"
            elif v <= t_lo:
                bucket = "low"
            elif v <= t_hi:
                bucket = "mid"
            else:
                bucket = "high"
            d_key = d if isinstance(d, date) else pd.Timestamp(d).date()
            trend_by_date[d_key] = trend
            vol_by_date[d_key] = bucket
            label_by_date[d_key] = f"{trend}_{bucket}"

        return RegimeTagger(
            label_by_date=label_by_date,
            trend_by_date=trend_by_date,
            vol_bucket_by_date=vol_by_date,
            daily_close=daily_close,
            ema=ema,
            realized_vol=realized_vol,
            vol_terciles=(float(t_lo), float(t_hi)),
            ema_span=ema_span,
            vol_window=vol_window,
        )

    def label_for_date(self, d: date) -> Optional[str]:
        return self.label_by_date.get(d)

    def dominant_label_for_period(self, start: date, end: date) -> Optional[str]:
        """Most-common regime label across the inclusive [start, end] window.
        Returns None if no day in the window has a known label."""
        labels: List[str] = []
        cur = start
        while cur <= end:
            lab = self.label_by_date.get(cur)
            if lab is not None:
                labels.append(lab)
            cur = cur.fromordinal(cur.toordinal() + 1)
        if not labels:
            return None
        return Counter(labels).most_common(1)[0][0]

    def regimes(self) -> Iterable[str]:
        """All distinct regime labels observed in the dataset."""
        return sorted(set(self.label_by_date.values()))
