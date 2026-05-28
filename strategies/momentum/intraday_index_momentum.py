"""
strategies/momentum/intraday_index_momentum.py

F1 — Intraday Index-ETF Momentum.

Backtest-harness strategy ONLY. NOT wired into main_pairs.py; does not touch
kalman_spread.py. Emits stage="SIZED" so the strategy owns its own directional
sizing and skips CandidateRanker / DynamicRiskSizer.

Anti-lookahead contract (enforced, not assumed):
  - sigma_intraday uses a TRAILING daily window that EXCLUDES the current
    session AND excludes extended-hours bars (regular 09:30–16:00 ET only).
  - A signal computed on bar t's CLOSE may only fill at bar t+1's OPEN (§2.4).
  - Per-tick management uses the tick's specific price (open/high/low/close),
    so trailing stops catch intrabar stop-outs at the bar's low (for longs)
    and high (for shorts) — not the bar's close.

Nothing here selects parameters or picks the "best" config. The harness
(walkforward_basket.py + analyze_walkforward.py) is the sole judge.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional
from zoneinfo import ZoneInfo

from core.engine.event_bus import Event, EventType

ET = ZoneInfo("America/New_York")

SESSION_OPEN = dt.time(9, 30)
SESSION_CLOSE = dt.time(16, 0)
EOD_FLAT = dt.time(15, 55)  # safety guard; EODLiquidationManager is the mechanism


# --------------------------------------------------------------------------- #
# Frozen configuration (every field corresponds to a §3 grid axis or a §2 rule)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class F1Config:
    """One instance == one grid cell from §3."""
    symbol: str

    # --- §3 grid parameters ---
    k: float                      # band-width multiplier
    trail_mult: float             # trailing stop in units of sigma_intraday
    n_vol: int                    # trailing daily lookback for sigma_intraday (days)
    no_entry_after: dt.time = dt.time(14, 0)

    # --- §2.3 frozen (NOT grid params) ---
    nominal_stop_pct: float = 0.02   # disaster backstop; repo default; not tuned

    # --- §1.2 leveraged flag: drives null choice + cost assumptions ---
    is_leveraged: bool = False

    # --- sizing (strategy owns it; SIZED stage) ---
    target_dollar_notional: float = 10_000.0

    def __post_init__(self) -> None:
        # Single validation point — runs on every construction, no duplication.
        assert self.k in (1.0, 1.5, 2.0), f"k={self.k} outside frozen grid"
        assert self.trail_mult in (1.5, 2.5), f"trail_mult={self.trail_mult} outside frozen grid"
        assert self.n_vol in (14, 20), f"n_vol={self.n_vol} outside frozen grid"
        assert self.no_entry_after == dt.time(14, 0), "no_entry_after is frozen at 14:00 ET"
        assert self.target_dollar_notional > 0, "target_dollar_notional must be positive"


class Direction(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


# --------------------------------------------------------------------------- #
# Per-symbol session state
# --------------------------------------------------------------------------- #
@dataclass
class _SessionState:
    session_date: Optional[dt.date] = None
    open_px: Optional[float] = None
    sigma_price: Optional[float] = None  # frozen at session open in PRICE terms

    direction: Direction = Direction.FLAT
    entry_px: Optional[float] = None
    shares: int = 0
    hwm: Optional[float] = None
    lwm: Optional[float] = None
    stopped_out_this_session: bool = False

    pending_entry: Optional[Direction] = None

    def reset_for_new_session(self, session_date: dt.date) -> None:
        self.session_date = session_date
        self.open_px = None
        self.sigma_price = None
        self.direction = Direction.FLAT
        self.entry_px = None
        self.shares = 0
        self.hwm = None
        self.lwm = None
        self.stopped_out_this_session = False
        self.pending_entry = None


# --------------------------------------------------------------------------- #
# Trailing daily volatility (regular session only; excludes current session)
# --------------------------------------------------------------------------- #
class _TrailingDailyVol:
    """Maintains (high-low)/open over completed prior REGULAR sessions only.

    Anti-lookahead points enforced here:
      - Only regular-hours bars (09:30 <= ts_et.time() < 16:00) are accumulated.
        Pre-market and after-hours bars are ignored, so sigma matches the
        spec's intent of a session-anchored intraday range.
      - The current session is never committed to history; sigma queried at
        session N's open uses sessions 1..N-1 only.
    """

    def __init__(self, n_vol: int) -> None:
        self._n = n_vol
        self._ratios: Deque[float] = deque(maxlen=n_vol)
        self._cur_date: Optional[dt.date] = None
        self._cur_open: Optional[float] = None
        self._cur_high: Optional[float] = None
        self._cur_low: Optional[float] = None

    def on_bar(self, ts: dt.datetime, o: float, h: float, l: float) -> None:
        ts_et = ts.astimezone(ET)
        # Reject extended-hours bars entirely. Regular session is [09:30, 16:00).
        if not (SESSION_OPEN <= ts_et.time() < SESSION_CLOSE):
            return
        d = ts_et.date()
        if self._cur_date is None:
            self._cur_date, self._cur_open, self._cur_high, self._cur_low = d, o, h, l
            return
        if d != self._cur_date:
            # session rolled: commit the completed prior REGULAR-session day
            if self._cur_open is not None and self._cur_open > 0:
                self._ratios.append(
                    (self._cur_high - self._cur_low) / self._cur_open
                )
            self._cur_date, self._cur_open, self._cur_high, self._cur_low = d, o, h, l
            return
        self._cur_high = max(self._cur_high, h)
        self._cur_low = min(self._cur_low, l)

    def ready(self) -> bool:
        return len(self._ratios) >= self._n

    def sigma_price(self, open_px: float) -> Optional[float]:
        """Return sigma in PRICE terms. None until warm."""
        if not self.ready():
            return None
        return (sum(self._ratios) / len(self._ratios)) * open_px


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
class IntradayIndexMomentum:
    """One instance per symbol. Consumes BAR_TICK; emits ORDER_CREATE (SIZED).

    Trigger (§2):
      band(t) = open_px ± k * sigma_price * sqrt(elapsed_fraction(t))
      long  if bar close > UB(t);  short if bar close < LB(t)
      one position/session, no pyramiding, no re-entry after stop-out
      no NEW entries after no_entry_after
      exit: trailing stop at trail_mult * sigma_price from HWM/LWM;
            hard EOD flat 15:55 (mechanism: EODLiquidationManager)
    """

    STRATEGY_NAME = "intraday_index_momentum"

    # Required keys on every BAR_TICK payload. Missing keys raise loudly
    # instead of silently no-op'ing — see _read_payload.
    _REQUIRED_KEYS = (
        "symbol", "timestamp", "open", "high", "low", "close",
        "is_bar_open", "is_bar_close",
    )

    def __init__(self, config: F1Config, event_bus=None) -> None:
        self.cfg = config
        self.bus = event_bus
        self.state = _SessionState()
        self.vol = _TrailingDailyVol(config.n_vol)

    # ---- harness entry point -------------------------------------------- #
    def on_bar_tick(self, event: Event) -> None:
        """Consume one BAR_TICK.

        The harness replays each 1-min bar as 4 ticks (open / low / high /
        close). The payload carries the full OHLC of the bar plus
        `is_bar_open` / `is_bar_close` flags identifying the tick within
        the bar, plus an `is_bar_low` / `is_bar_high` flag we use to drive
        intrabar stop-out detection. Missing keys raise KeyError — silent
        no-op on a contract violation is worse than failing loudly.
        """
        p = event.payload
        sym, ts, o, h, l, c, is_open_tick, is_close_tick = self._read_payload(p)
        if sym != self.cfg.symbol:
            return
        is_low_tick = bool(p.get("is_bar_low", False))
        is_high_tick = bool(p.get("is_bar_high", False))

        ts_et = ts.astimezone(ET)

        # feed the trailing-vol tracker (it ignores extended-hours bars itself)
        self.vol.on_bar(ts, o, h, l)

        # session rollover handling (safety reset)
        if self.state.session_date != ts_et.date():
            self.state.reset_for_new_session(ts_et.date())

        # capture session open + freeze sigma_price ONCE on the first
        # regular-hours bar (so an extended-hours tick can't seed it)
        if (
            self.state.open_px is None
            and is_open_tick
            and SESSION_OPEN <= ts_et.time() < SESSION_CLOSE
        ):
            self.state.open_px = o
            self.state.sigma_price = self.vol.sigma_price(o)

        # ---- fill pending entry at THIS bar's open (§2.4) ---- #
        if is_open_tick and self.state.pending_entry is not None:
            self._open_position(self.state.pending_entry, fill_px=o, ts=ts)
            self.state.pending_entry = None

        # nothing further until sigma is warm and we have an open
        if self.state.open_px is None or self.state.sigma_price is None:
            return

        # ---- manage open position using the TICK's specific price ---- #
        # Long stops trigger at the bar's LOW; short stops at the bar's HIGH.
        # Using `c` on every tick would miss intrabar stop-outs.
        if self.state.direction is Direction.LONG and is_low_tick:
            self._update_trailing_and_maybe_exit(price=l, ts=ts)
        elif self.state.direction is Direction.SHORT and is_high_tick:
            self._update_trailing_and_maybe_exit(price=h, ts=ts)
        elif is_close_tick and self.state.direction is not Direction.FLAT:
            # close-tick ratchet so HWM/LWM track even if H/L ticks didn't
            # pierce the stop this bar
            self._update_trailing_and_maybe_exit(price=c, ts=ts)

        # ---- EOD safety guard (mechanism = EODLiquidationManager) ---- #
        if (
            is_close_tick
            and ts_et.time() >= EOD_FLAT
            and self.state.direction is not Direction.FLAT
        ):
            self._close_position(fill_px=c, ts=ts, reason="eod_flat")
            # Also drop any stale pending entry so it can't fire next session.
            self.state.pending_entry = None
            return

        # ---- signal evaluation only on bar CLOSE (no intrabar lookahead) ---- #
        if not is_close_tick:
            return
        self._evaluate_entry_signal(close_px=c, ts_et=ts_et)

    # ---- payload contract ---------------------------------------------- #
    def _read_payload(self, p: dict):
        missing = [k for k in self._REQUIRED_KEYS if k not in p]
        if missing:
            raise KeyError(
                f"{self.STRATEGY_NAME}: BAR_TICK payload missing required keys "
                f"{missing}. Harness contract violation; not silently no-op'ing."
            )
        return (
            p["symbol"], p["timestamp"], p["open"], p["high"], p["low"], p["close"],
            bool(p["is_bar_open"]), bool(p["is_bar_close"]),
        )

    # ---- frozen band + entry logic (§2.1, §2.2) ------------------------- #
    def _elapsed_fraction(self, ts_et: dt.datetime) -> float:
        """Fraction of the 09:30–16:00 session elapsed, clamped to (0,1]."""
        session_start = dt.datetime.combine(ts_et.date(), SESSION_OPEN, tzinfo=ET)
        session_end = dt.datetime.combine(ts_et.date(), SESSION_CLOSE, tzinfo=ET)
        total = (session_end - session_start).total_seconds()
        elapsed = (ts_et - session_start).total_seconds()
        if total <= 0:
            return 1.0
        return min(max(elapsed / total, 1e-6), 1.0)

    def _band(self, ts_et: dt.datetime) -> tuple[float, float]:
        frac = self._elapsed_fraction(ts_et)
        width = self.cfg.k * self.state.sigma_price * math.sqrt(frac)
        ub = self.state.open_px + width
        lb = self.state.open_px - width
        return ub, lb

    def _evaluate_entry_signal(self, close_px: float, ts_et: dt.datetime) -> None:
        if self.state.direction is not Direction.FLAT:
            return
        if self.state.stopped_out_this_session:
            return
        if ts_et.time() >= self.cfg.no_entry_after:
            return
        ub, lb = self._band(ts_et)
        if close_px > ub:
            self.state.pending_entry = Direction.LONG     # fill next-bar open
        elif close_px < lb:
            self.state.pending_entry = Direction.SHORT

    # ---- position lifecycle --------------------------------------------- #
    def _open_position(self, direction: Direction, fill_px: float, ts: dt.datetime) -> None:
        shares = self._size(fill_px)
        if shares <= 0:
            return
        self.state.direction = direction
        self.state.entry_px = fill_px
        self.state.shares = shares
        self.state.hwm = fill_px
        self.state.lwm = fill_px
        # Repo convention: BUY opens a long, SELL_SHORT opens a short.
        action = "BUY" if direction is Direction.LONG else "SELL_SHORT"
        self._emit_order(action=action, shares=shares, ts=ts, reason="entry")

    def _update_trailing_and_maybe_exit(self, price: float, ts: dt.datetime) -> None:
        stop_dist = self.cfg.trail_mult * self.state.sigma_price
        if self.state.direction is Direction.LONG:
            self.state.hwm = max(self.state.hwm, price)
            trail_stop = self.state.hwm - stop_dist
            disaster = self.state.entry_px * (1.0 - self.cfg.nominal_stop_pct)
            if price <= trail_stop or price <= disaster:
                self._close_position(fill_px=price, ts=ts, reason="trail_stop")
        elif self.state.direction is Direction.SHORT:
            self.state.lwm = min(self.state.lwm, price)
            trail_stop = self.state.lwm + stop_dist
            disaster = self.state.entry_px * (1.0 + self.cfg.nominal_stop_pct)
            if price >= trail_stop or price >= disaster:
                self._close_position(fill_px=price, ts=ts, reason="trail_stop")

    def _close_position(self, fill_px: float, ts: dt.datetime, reason: str) -> None:
        if self.state.direction is Direction.FLAT:
            return
        # Repo convention: SELL closes a long, BUY_TO_COVER closes a short.
        action = "SELL" if self.state.direction is Direction.LONG else "BUY_TO_COVER"
        self._emit_order(action=action, shares=self.state.shares, ts=ts, reason=reason)
        if reason == "trail_stop":
            self.state.stopped_out_this_session = True
        self.state.direction = Direction.FLAT
        self.state.entry_px = None
        self.state.shares = 0
        self.state.hwm = None
        self.state.lwm = None

    # ---- sizing + emit -------------------------------------------------- #
    def _size(self, px: float) -> int:
        if px <= 0:
            return 0
        return int(self.cfg.target_dollar_notional // px)

    def _emit_order(self, action: str, shares: int, ts: dt.datetime, reason: str) -> None:
        """Publish a SIZED ORDER_CREATE on the real EventBus."""
        if self.bus is None:
            return
        payload = self._build_order(action=action, shares=shares, ts=ts, reason=reason)
        self.bus.publish(Event(type=EventType.ORDER_CREATE, payload=payload))

    def _build_order(self, action: str, shares: int, ts: dt.datetime, reason: str) -> dict:
        return {
            "stage": "SIZED",                  # skips ranker + sizer (frozen choice)
            "strategy": self.STRATEGY_NAME,
            "symbol": self.cfg.symbol,
            "action": action,                  # BUY / SELL / SELL_SHORT / BUY_TO_COVER
            "shares": shares,
            "timestamp": ts,
            "decision_id": f"{self.STRATEGY_NAME}:{self.cfg.symbol}:{ts.isoformat()}:{reason}",
            "meta": {
                "reason": reason,
                "k": self.cfg.k,
                "trail_mult": self.cfg.trail_mult,
                "n_vol": self.cfg.n_vol,
                "is_leveraged": self.cfg.is_leveraged,
            },
        }


# --------------------------------------------------------------------------- #
# Grid expansion helper (§3) — generates exactly the frozen family, nothing more
# --------------------------------------------------------------------------- #
def frozen_grid_for_symbol(symbol: str, is_leveraged: bool = False) -> list[F1Config]:
    """Return the 12 frozen configs (§3) for one symbol."""
    configs: list[F1Config] = [
        F1Config(symbol=symbol, k=k, trail_mult=tm, n_vol=nv, is_leveraged=is_leveraged)
        for k in (1.0, 1.5, 2.0)
        for tm in (1.5, 2.5)
        for nv in (14, 20)
    ]
    assert len(configs) == 12, "grid must be exactly 12 configs/symbol per §3"
    return configs


UNLEVERAGED_UNIVERSE = ("SPY", "QQQ", "IWM")
LEVERAGED_UNIVERSE = ("TQQQ", "SPXL")
