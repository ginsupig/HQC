"""
Production-grade Liquidity & Relative Strength Engine.

Institutional-style intraday microstructure analysis with:
- Sector-adjusted residual RS
- Volatility-normalized z-scores
- Rolling relative volume
- VWAP tracking
- Liquidity quality scoring
- Hard veto logic for thin/unstable candidates
- Comprehensive error handling & fallbacks
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import pstdev
from typing import Deque, Dict, Optional, List, Set

logger = logging.getLogger("LiquidityRSEngine")


@dataclass
class TickPoint:
    """Represents a single tick/quote."""
    price: float
    volume: float
    ts_ms: int


class DataIntegrityError(Exception):
    """Raised when required market data is missing or invalid."""
    pass


class LiquidityRelativeStrengthEngine:
    """
    Production-grade intraday microstructure + benchmark/sector-relative filter.

    Provides:
    - Raw relative strength vs benchmark
    - Sector-adjusted residual RS (volatility-normalized)
    - Rolling relative volume
    - VWAP
    - Spread proxy in bps
    - Liquidity quality score
    - Hard veto logic for thin/unstable candidates

    Key features:
    - Graceful fallbacks when data is incomplete
    - Comprehensive validation and logging
    - Safe edge-case handling (zero volumes, missing data, etc.)
    - Optional data integrity checks
    """

    # Default sector to ETF mappings (large-cap US equities)
    DEFAULT_SECTOR_MAP: Dict[str, str] = {
        # Technology / Semiconductors
        "AAPL": "XLK",
        "MSFT": "XLK",
        "NVDA": "SMH",
        "AVGO": "SMH",
        "AMD": "SMH",
        "QCOM": "SMH",
        "MU": "SMH",
        "TSM": "SMH",
        "AMAT": "SMH",
        "LRCX": "SMH",
        "KLAC": "SMH",
        "INTC": "SMH",
        # Communications
        "META": "XLC",
        "GOOGL": "XLC",
        # Consumer Discretionary
        "AMZN": "XLY",
        "TSLA": "XLY",
        "HD": "XLY",
        "MCD": "XLY",
        # Financials
        "JPM": "XLF",
        "GS": "XLF",
        "MS": "XLF",
        "BAC": "XLF",
        "C": "XLF",
        "WFC": "XLF",
        "BLK": "XLF",
        # Energy
        "XOM": "XLE",
        "CVX": "XLE",
        "EOG": "XLE",
        "SLB": "XLE",
        "COP": "XLE",
        # Industrials
        "CAT": "XLI",
        "DE": "XLI",
        "ETN": "XLI",
        "GE": "XLI",
        "BA": "XLI",
        # Healthcare
        "UNH": "XLV",
        "LLY": "XLV",
        "MRK": "XLV",
        "ABBV": "XLV",
        "JNJ": "XLV",
        # Staples
        "PG": "XLP",
        "KO": "XLP",
        "PEP": "XLP",
        "COST": "XLP",
        "WMT": "XLP",
        # Materials
        "NEM": "XLB",
        "FCX": "XLB",
        # Real Estate
        "XLRE": "XLRE",
        # Utilities
        "NEE": "XLU",
        "DUK": "XLU",
    }

    REQUIRED_SYMBOLS: Set[str] = {"SPY"}  # Minimum data requirement

    def __init__(
        self,
        benchmark: str = "SPY",
        tick_window: int = 300,
        min_ticks_for_quality: int = 8,
        max_spread_bps: float = 18.0,
        min_rvol: float = 0.90,
        min_liquidity_score: float = 0.35,
        long_rs_warn_threshold: float = -0.0025,
        short_rs_warn_threshold: float = 0.0025,
        sector_map: Optional[Dict[str, str]] = None,
        vol_window: int = 30,
        use_sector_adjusted_rs: bool = True,
        spy_weight: float = 0.60,
        sector_weight: float = 0.40,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize the liquidity/RS engine.

        Args:
            benchmark: Benchmark symbol (default: "SPY")
            tick_window: Max ticks to store per symbol (default: 300)
            min_ticks_for_quality: Minimum ticks required for liquidity score (default: 8)
            max_spread_bps: Maximum acceptable spread in basis points (default: 18.0)
            min_rvol: Minimum acceptable relative volume (default: 0.90)
            min_liquidity_score: Minimum acceptable liquidity score 0-1 (default: 0.35)
            long_rs_warn_threshold: RS threshold below which longs get warning (default: -0.0025)
            short_rs_warn_threshold: RS threshold above which shorts get warning (default: 0.0025)
            sector_map: Custom sector mappings (uses DEFAULT_SECTOR_MAP if None)
            vol_window: Window size for intraday volatility calculation (default: 30)
            use_sector_adjusted_rs: Enable sector-adjusted RS (default: True)
            spy_weight: Weight for SPY in sector adjustment (default: 0.60)
            sector_weight: Weight for sector ETF in adjustment (default: 0.40)
            strict_mode: If True, raise DataIntegrityError when required data missing (default: False)

        Raises:
            ValueError: If weights don't sum to ~1.0 or invalid parameters
        """
        # Validate inputs
        self._validate_constructor_params(
            spy_weight, sector_weight, tick_window, min_ticks_for_quality
        )

        self.benchmark = str(benchmark).upper()
        self.tick_window = int(tick_window)
        self.min_ticks_for_quality = int(min_ticks_for_quality)
        self.max_spread_bps = float(max_spread_bps)
        self.min_rvol = float(min_rvol)
        self.min_liquidity_score = float(min_liquidity_score)
        self.long_rs_warn_threshold = float(long_rs_warn_threshold)
        self.short_rs_warn_threshold = float(short_rs_warn_threshold)

        self.vol_window = int(vol_window)
        self.use_sector_adjusted_rs = bool(use_sector_adjusted_rs)
        self.spy_weight = float(spy_weight)
        self.sector_weight = float(sector_weight)
        self.strict_mode = bool(strict_mode)

        # Normalize sector map
        self.sector_map = {
            k.upper(): v.upper()
            for k, v in (sector_map or self.DEFAULT_SECTOR_MAP).items()
        }

        # Initialize internal data structures
        self.ticks: Dict[str, Deque[TickPoint]] = defaultdict(
            lambda: deque(maxlen=self.tick_window)
        )
        self.cum_pv: Dict[str, float] = defaultdict(float)  # Cumulative price*volume
        self.cum_vol: Dict[str, float] = defaultdict(float)  # Cumulative volume
        self.open_price: Dict[str, float] = {}
        self.last_price: Dict[str, float] = {}

        logger.info(
            "LiquidityRSEngine initialized: benchmark=%s, "
            "sector_adjusted=%s, strict_mode=%s, weights=[SPY:%.1f%%, SECTOR:%.1f%%]",
            self.benchmark,
            self.use_sector_adjusted_rs,
            self.strict_mode,
            self.spy_weight * 100,
            self.sector_weight * 100,
        )

    def reset_daily(self) -> None:
        """Resets intraday accumulators at the start of a new trading day."""
        self.ticks.clear()
        self.cum_pv.clear()
        self.cum_vol.clear()
        self.open_price.clear()
        self.last_price.clear()

    @staticmethod
    def _validate_constructor_params(
        spy_weight: float,
        sector_weight: float,
        tick_window: int,
        min_ticks_for_quality: int,
    ) -> None:
        """Validate constructor parameters."""
        total_weight = spy_weight + sector_weight
        if not (0.95 <= total_weight <= 1.05):
            raise ValueError(
                f"Weights must sum to ~1.0, got {total_weight:.2f} "
                f"(spy={spy_weight}, sector={sector_weight})"
            )
        if tick_window < 10:
            raise ValueError(f"tick_window must be >= 10, got {tick_window}")
        if min_ticks_for_quality < 2:
            raise ValueError(f"min_ticks_for_quality must be >= 2, got {min_ticks_for_quality}")

    def update_tick(self, symbol: str, price: float, volume: float, ts_ms: int) -> None:
        """
        Update engine with a new tick.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            price: Tick price (must be > 0)
            volume: Volume (must be >= 0)
            ts_ms: Timestamp in milliseconds

        Raises:
            DataIntegrityError: If strict_mode=True and data is invalid
        """
        symbol = str(symbol).upper()
        price = float(price)
        volume = float(volume)
        ts_ms = int(ts_ms)

        # Validate inputs
        if not symbol:
            logger.warning("Ignoring tick with empty symbol")
            return
        if price <= 0:
            logger.warning("Ignoring tick for %s with invalid price: %.2f", symbol, price)
            return
        if volume < 0:
            logger.warning("Ignoring tick for %s with negative volume: %.2f", symbol, volume)
            return

        # Store tick
        self.ticks[symbol].append(TickPoint(price=price, volume=volume, ts_ms=ts_ms))
        self.last_price[symbol] = price

        # Initialize open price on first tick
        if symbol not in self.open_price:
            self.open_price[symbol] = price

        # Update cumulative PV and volume
        self.cum_pv[symbol] += price * volume
        self.cum_vol[symbol] += volume

        logger.debug(
            "Tick update: %s price=%.2f vol=%.0f ticks=%d",
            symbol,
            price,
            volume,
            len(self.ticks[symbol]),
        )

    def get_vwap(self, symbol: str) -> Optional[float]:
        """
        Get volume-weighted average price.

        Args:
            symbol: Stock symbol

        Returns:
            VWAP or None if insufficient data
        """
        symbol = str(symbol).upper()
        vol = self.cum_vol.get(symbol, 0.0)

        if vol <= 0:
            return None

        return self.cum_pv[symbol] / vol

    def get_relative_volume(self, symbol: str) -> float:
        """
        Get relative volume (latest tick volume / baseline average).

        Args:
            symbol: Stock symbol

        Returns:
            Relative volume (1.0 = baseline, >1.0 = above average)
        """
        symbol = str(symbol).upper()
        ticks = self.ticks.get(symbol)

        if not ticks:
            return 1.0

        vols = [t.volume for t in ticks]
        if len(vols) < 2:
            return 1.0

        latest = vols[-1]
        baseline = sum(vols[:-1]) / max(1, len(vols) - 1)

        if baseline <= 0:
            return 1.0

        return latest / baseline

    def get_spread_proxy_bps(self, symbol: str) -> float:
        """
        Get spread proxy using recent high-low range.

        Args:
            symbol: Stock symbol

        Returns:
            Spread in basis points (0 if insufficient data)
        """
        symbol = str(symbol).upper()
        ticks = self.ticks.get(symbol)

        if not ticks or len(ticks) < 5:
            return 0.0

        recent = list(ticks)[-5:]
        prices = [t.price for t in recent]
        mid = prices[-1]

        if mid <= 0:
            return 0.0

        hi = max(prices)
        lo = min(prices)

        return ((hi - lo) / mid) * 10000.0

    def _get_return_from_open(self, symbol: str) -> Optional[float]:
        """
        Calculate return from open for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Return (1.0 = +100%) or None if data missing
        """
        symbol = str(symbol).upper()
        open_px = self.open_price.get(symbol)
        last_px = self.last_price.get(symbol)

        if any(v is None for v in [open_px, last_px]):
            return None
        if open_px <= 0 or last_px <= 0:
            return None

        return (last_px / open_px) - 1.0

    def _get_sector_etf(self, symbol: str) -> Optional[str]:
        """
        Get sector ETF for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Sector ETF symbol or None if not mapped
        """
        return self.sector_map.get(str(symbol).upper())

    def get_relative_strength_raw(self, symbol: str) -> float:
        """
        Get raw relative strength vs benchmark.

        Args:
            symbol: Stock symbol

        Returns:
            Raw RS (symbol return - benchmark return)
        """
        symbol = str(symbol).upper()

        if symbol == self.benchmark:
            return 0.0

        sym_ret = self._get_return_from_open(symbol)
        bench_ret = self._get_return_from_open(self.benchmark)

        if sym_ret is None or bench_ret is None:
            if self.strict_mode:
                raise DataIntegrityError(
                    f"Missing return data: {symbol}={sym_ret}, {self.benchmark}={bench_ret}"
                )
            logger.debug(
                "Missing return data for RS calculation: %s=%s, %s=%s",
                symbol,
                sym_ret,
                self.benchmark,
                bench_ret,
            )
            return 0.0

        return sym_ret - bench_ret

    def get_intraday_volatility(self, symbol: str) -> float:
        """
        Calculate intraday volatility from recent returns.

        Args:
            symbol: Stock symbol

        Returns:
            Population standard deviation of returns (minimum 0.001)
        """
        symbol = str(symbol).upper()
        ticks = self.ticks.get(symbol)

        if not ticks or len(ticks) < 3:
            return 0.001

        # Use recent ticks up to vol_window — iterate only the tail, no full copy
        import itertools as _it
        prices = [t.price for t in _it.islice(ticks, max(0, len(ticks) - self.vol_window), None) if t.price > 0]

        if len(prices) < 3:
            return 0.001

        # Calculate returns
        rets: List[float] = []
        for i in range(1, len(prices)):
            prev_px = prices[i - 1]
            curr_px = prices[i]

            if prev_px > 0 and curr_px > 0:
                rets.append((curr_px / prev_px) - 1.0)

        if len(rets) < 2:
            return 0.001

        try:
            vol = pstdev(rets)
            return max(vol, 0.001)
        except (ValueError, ZeroDivisionError) as e:
            logger.warning("Error calculating volatility for %s: %s", symbol, e)
            return 0.001

    def get_relative_strength(self, symbol: str) -> float:
        """
        Get primary RS signal used by ranker.

        If sector_adjusted_rs enabled, returns residual RS normalized by volatility.
        Otherwise, returns raw benchmark-relative RS.

        Args:
            symbol: Stock symbol

        Returns:
            RS score (typically -0.02 to 0.02 when sector-adjusted)
        """
        symbol = str(symbol).upper()

        if symbol == self.benchmark:
            return 0.0

        raw_rs = self.get_relative_strength_raw(symbol)

        if not self.use_sector_adjusted_rs:
            return raw_rs

        sym_ret = self._get_return_from_open(symbol)
        spy_ret = self._get_return_from_open(self.benchmark)

        # Graceful fallback to raw RS if data missing
        if sym_ret is None or spy_ret is None:
            logger.debug(
                "Sector adjustment skipped for %s (missing return data), using raw RS",
                symbol,
            )
            return raw_rs

        sector_etf = self._get_sector_etf(symbol)
        if not sector_etf:
            logger.debug("No sector mapping for %s, using raw RS", symbol)
            return raw_rs

        sector_ret = self._get_return_from_open(sector_etf)
        if sector_ret is None:
            logger.debug(
                "Missing sector ETF data for %s (%s), using raw RS", symbol, sector_etf
            )
            return raw_rs

        # Calculate sector-adjusted residual RS
        expected_ret = (self.spy_weight * spy_ret) + (self.sector_weight * sector_ret)
        residual_rs = sym_ret - expected_ret

        # Volatility normalize (z-score)
        vol = self.get_intraday_volatility(symbol)
        rs_z = residual_rs / vol

        # Clip to prevent single-tick explosions
        clipped = max(-0.02, min(0.02, rs_z * 0.0025))

        logger.debug(
            "Sector-adjusted RS for %s: raw=%.6f, sector_etf=%s, sector_ret=%.6f, "
            "residual=%.6f, vol=%.6f, z=%.6f, clipped=%.6f",
            symbol,
            raw_rs,
            sector_etf,
            sector_ret,
            residual_rs,
            vol,
            rs_z,
            clipped,
        )

        return clipped

    def get_dist_vwap_pct(
        self, symbol: str, reference_price: Optional[float] = None
    ) -> float:
        """
        Get distance from VWAP as percentage.

        Args:
            symbol: Stock symbol
            reference_price: Price to compare against VWAP (uses last price if None)

        Returns:
            Distance as decimal (0.01 = 1% above VWAP)
        """
        symbol = str(symbol).upper()
        vwap = self.get_vwap(symbol)

        px: Optional[float]
        if reference_price is not None:
            px = float(reference_price)
        else:
            px = self.last_price.get(symbol)

        if vwap is None or px is None or vwap <= 0 or px <= 0:
            return 0.0

        return (px - vwap) / vwap

    def get_liquidity_score(self, symbol: str) -> float:
        """
        Get liquidity quality score (0-1, higher is better).

        Composite of spread quality and volume consistency.

        Args:
            symbol: Stock symbol

        Returns:
            Score 0-1
        """
        symbol = str(symbol).upper()
        ticks = self.ticks.get(symbol)

        if not ticks:
            return 0.0

        if len(ticks) < self.min_ticks_for_quality:
            return 0.50

        spread_bps = self.get_spread_proxy_bps(symbol)
        rvol = self.get_relative_volume(symbol)

        # Spread component (0-1): tight spreads score high
        spread_component = max(0.0, 1.0 - (spread_bps / max(1.0, self.max_spread_bps)))

        # Volume component (0-1): consistent/above-average volume scores high
        rvol_component = min(1.0, max(0.0, rvol / max(1.0, self.min_rvol)))

        # Weighted composite
        score = 0.65 * spread_component + 0.35 * rvol_component

        return max(0.0, min(1.0, score))

    def evaluate_candidate(self, symbol: str, action: str, reference_price: float) -> dict:
        """
        Evaluate a candidate for trading.

        Comprehensive microstructure + macro analysis with hard veto logic.

        Args:
            symbol: Stock symbol
            action: Trade action (BUY, SELL, BUY_TO_OPEN, SELL_SHORT, etc.)
            reference_price: Entry price for evaluation

        Returns:
            Dictionary with detailed metrics and flags
        """
        symbol = str(symbol).upper()
        action = str(action).upper()
        reference_price = float(reference_price)

        # Collect metrics
        rs = self.get_relative_strength(symbol)
        rs_raw = self.get_relative_strength_raw(symbol)
        rvol = self.get_relative_volume(symbol)
        spread_bps = self.get_spread_proxy_bps(symbol)
        dist_vwap_pct = self.get_dist_vwap_pct(symbol, reference_price=reference_price)
        liquidity_score = self.get_liquidity_score(symbol)

        reasons: List[str] = []
        hard_veto = False

        # Hard veto checks. The microstructure metrics below are only
        # meaningful once we have at least min_ticks_for_quality ticks for
        # the symbol — at the very first tick of an RTH session the rolling
        # tick book has just been reset, so ``liquidity_score`` returns the
        # neutral 0.50 default and ``spread_bps`` can be computed off a
        # single bar's OHLC reconstruction. Hard-vetoing on those values
        # silently kills any strategy that emits at session open
        # (notably OvernightGapFade), so we degrade them to soft warnings
        # when the book is too thin to trust.
        ticks = self.ticks.get(symbol)
        tick_count = len(ticks) if ticks else 0
        microstructure_reliable = tick_count >= self.min_ticks_for_quality

        if not ticks:
            reasons.append("no_tick_history")
            # Soft only — downstream sizer/broker will still validate price.

        if reference_price <= 0:
            reasons.append("invalid_reference_price")
            hard_veto = True

        if spread_bps > self.max_spread_bps:
            reasons.append(f"spread_bps>{self.max_spread_bps:.1f}")
            if microstructure_reliable:
                hard_veto = True

        # Soft warnings (don't veto)
        if rvol < self.min_rvol:
            reasons.append(f"rvol<{self.min_rvol:.2f}")

        # Hard veto: insufficient liquidity. Only enforced once we have
        # enough ticks for the score to mean anything.
        if liquidity_score < self.min_liquidity_score:
            reasons.append(f"liq_score<{self.min_liquidity_score:.2f}")
            if microstructure_reliable:
                hard_veto = True

        # RS alignment checks
        rs_alignment = 0.0
        if action in {"BUY", "BUY_TO_OPEN", "BUY_TO_COVER"}:
            rs_alignment = rs
            if rs < self.long_rs_warn_threshold:
                reasons.append("negative_rs_for_long")
        elif action in {"SELL", "SELL_SHORT", "SELL_TO_OPEN"}:
            rs_alignment = -rs
            if rs > self.short_rs_warn_threshold:
                reasons.append("positive_rs_for_short")
        else:
            reasons.append("unknown_action")
            hard_veto = True

        # Composite scoring (weighted components)
        rs_score = max(0.0, min(2.5, 1.25 + (rs_alignment * 250.0)))
        rvol_score = max(0.0, min(2.0, rvol))
        spread_score = max(
            0.0, min(2.0, 2.0 - (spread_bps / max(1.0, self.max_spread_bps)) * 2.0)
        )
        vwap_score = max(0.0, min(1.5, 1.5 - min(abs(dist_vwap_pct), 0.02) * 75.0))
        liquidity_score_scaled = liquidity_score * 2.0

        total_score = (
            rs_score + rvol_score + spread_score + vwap_score + liquidity_score_scaled
        )

        # Build comprehensive output
        payload = {
            "symbol": symbol,
            "action": action,
            "reference_price": round(reference_price, 2),
            "metrics": {
                "rs": round(rs, 6),
                "rs_raw": round(rs_raw, 6),
                "rvol": round(rvol, 4),
                "spread_bps": round(spread_bps, 4),
                "dist_vwap_pct": round(dist_vwap_pct, 6),
                "liquidity_score": round(liquidity_score, 4),
                "intraday_vol": round(self.get_intraday_volatility(symbol), 6),
                "vwap": round(self.get_vwap(symbol) or 0.0, 2),
            },
            "scoring": {
                "rs_score": round(rs_score, 4),
                "rvol_score": round(rvol_score, 4),
                "spread_score": round(spread_score, 4),
                "vwap_score": round(vwap_score, 4),
                "liquidity_score_scaled": round(liquidity_score_scaled, 4),
                "total_score": round(total_score, 4),
            },
            "veto": {
                "hard_veto": hard_veto,
                "reasons": reasons,
            },
            "context": {
                "sector_etf": self._get_sector_etf(symbol),
                "tick_count": len(self.ticks.get(symbol, [])),
                "benchmark": self.benchmark,
                "sector_adjusted": self.use_sector_adjusted_rs,
            },
        }

        logger.info(
            "Candidate eval %s %s @ %.2f: score=%.2f, hard_veto=%s, reasons=%s",
            symbol,
            action,
            reference_price,
            total_score,
            hard_veto,
            ",".join(reasons) if reasons else "none",
        )

        return payload

    def check_data_integrity(self) -> Dict[str, any]:
        """
        Check data integrity and return status report.

        Useful for monitoring the engine's data quality.

        Returns:
            Dictionary with integrity status and warnings
        """
        missing_required = self.REQUIRED_SYMBOLS - set(
            k for k in self.last_price.keys()
        )
        missing_sector_etfs = {
            etf
            for etf in set(self.sector_map.values())
            if self.last_price.get(etf) is None
        }

        status = {
            "timestamp": "now",
            "symbols_tracked": len(self.ticks),
            "required_symbols_missing": list(missing_required),
            "sector_etfs_missing": list(missing_sector_etfs),
            "is_healthy": len(missing_required) == 0,
            "benchmark_data_available": self.benchmark in self.last_price,
        }

        if status["is_healthy"]:
            logger.info("Data integrity check: OK")
        else:
            logger.warning(
                "Data integrity issues: missing_required=%s, missing_etfs=%s",
                status["required_symbols_missing"],
                status["sector_etfs_missing"],
            )

        return status