"""
Comprehensive test suite for LiquidityRelativeStrengthEngine.

Run with: pytest test_liquidity_rs_engine.py -v
"""

import pytest
from liquidity_rs_engine import (
    LiquidityRelativeStrengthEngine,
    TickPoint,
    DataIntegrityError,
)


class TestConstructor:
    """Test engine initialization and parameter validation."""

    def test_default_initialization(self):
        """Test engine initializes with default parameters."""
        engine = LiquidityRelativeStrengthEngine()
        assert engine.benchmark == "SPY"
        assert engine.tick_window == 300
        assert engine.use_sector_adjusted_rs is True
        assert engine.strict_mode is False

    def test_custom_initialization(self):
        """Test engine initializes with custom parameters."""
        engine = LiquidityRelativeStrengthEngine(
            benchmark="QQQ",
            tick_window=500,
            use_sector_adjusted_rs=False,
        )
        assert engine.benchmark == "QQQ"
        assert engine.tick_window == 500
        assert engine.use_sector_adjusted_rs is False

    def test_invalid_weights_sum(self):
        """Test that invalid weight sums are rejected."""
        with pytest.raises(ValueError, match="Weights must sum"):
            LiquidityRelativeStrengthEngine(spy_weight=0.7, sector_weight=0.7)

    def test_invalid_tick_window(self):
        """Test that tick_window < 10 is rejected."""
        with pytest.raises(ValueError, match="tick_window"):
            LiquidityRelativeStrengthEngine(tick_window=5)

    def test_custom_sector_map(self):
        """Test custom sector map is applied (overrides defaults)."""
        custom_map = {"TEST": "XLK", "AAPL": "XLK"}
        engine = LiquidityRelativeStrengthEngine(sector_map=custom_map)
        assert engine.sector_map["TEST"] == "XLK"
        # Custom map replaces defaults completely
        assert engine.sector_map["AAPL"] == "XLK"
        assert len(engine.sector_map) == 2


class TestTickUpdates:
    """Test tick data ingestion and validation."""

    def test_valid_tick_update(self):
        """Test valid tick is stored correctly."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)

        assert "AAPL" in engine.last_price
        assert engine.last_price["AAPL"] == 150.0
        assert len(engine.ticks["AAPL"]) == 1

    def test_open_price_initialization(self):
        """Test open price is set on first tick."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 151.0, 1000.0, 1000000001)

        assert engine.open_price["AAPL"] == 150.0
        assert engine.last_price["AAPL"] == 151.0

    def test_invalid_price_rejected(self):
        """Test zero/negative prices are rejected."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 0.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", -150.0, 1000.0, 1000000000)

        assert "AAPL" not in engine.last_price

    def test_invalid_volume_rejected(self):
        """Test negative volumes are rejected."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 150.0, -1000.0, 1000000000)

        assert "AAPL" not in engine.last_price

    def test_empty_symbol_rejected(self):
        """Test empty symbols are rejected."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("", 150.0, 1000.0, 1000000000)

        assert "" not in engine.ticks

    def test_cumulative_pv_volume(self):
        """Test cumulative PV and volume are tracked correctly."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 101.0, 500.0, 1000000001)

        assert engine.cum_vol["AAPL"] == 1500.0
        assert engine.cum_pv["AAPL"] == (100 * 1000) + (101 * 500)

    def test_tick_window_enforcement(self):
        """Test tick window max length is enforced."""
        engine = LiquidityRelativeStrengthEngine(tick_window=10)
        for i in range(20):
            engine.update_tick("AAPL", 150.0 + i, 1000.0, 1000000000 + i)

        assert len(engine.ticks["AAPL"]) == 10


class TestVWAP:
    """Test VWAP calculation."""

    def test_vwap_calculation(self):
        """Test VWAP is calculated correctly."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 101.0, 1000.0, 1000000001)

        vwap = engine.get_vwap("AAPL")
        expected = (100 * 1000 + 101 * 1000) / 2000
        assert abs(vwap - expected) < 0.001

    def test_vwap_no_volume(self):
        """Test VWAP returns None with zero volume."""
        engine = LiquidityRelativeStrengthEngine()
        vwap = engine.get_vwap("UNKNOWN")
        assert vwap is None

    def test_vwap_single_tick(self):
        """Test VWAP with single tick equals that price."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 150.5, 1000.0, 1000000000)

        vwap = engine.get_vwap("AAPL")
        assert abs(vwap - 150.5) < 0.001


class TestRelativeVolume:
    """Test relative volume calculation."""

    def test_baseline_relative_volume(self):
        """Test relative volume with consistent volumes."""
        engine = LiquidityRelativeStrengthEngine()
        for i in range(5):
            engine.update_tick("AAPL", 150.0, 1000.0, 1000000000 + i)

        rvol = engine.get_relative_volume("AAPL")
        assert abs(rvol - 1.0) < 0.01  # Latest = baseline

    def test_elevated_relative_volume(self):
        """Test relative volume with spike."""
        engine = LiquidityRelativeStrengthEngine()
        for i in range(4):
            engine.update_tick("AAPL", 150.0, 1000.0, 1000000000 + i)
        engine.update_tick("AAPL", 150.0, 3000.0, 1000000004)  # 3x volume spike

        rvol = engine.get_relative_volume("AAPL")
        assert rvol > 2.0

    def test_no_ticks_returns_baseline(self):
        """Test relative volume returns 1.0 with no ticks."""
        engine = LiquidityRelativeStrengthEngine()
        rvol = engine.get_relative_volume("UNKNOWN")
        assert rvol == 1.0


class TestSpreadProxy:
    """Test spread proxy calculation."""

    def test_tight_spread(self):
        """Test spread proxy with tight bid-ask (0.04% range)."""
        engine = LiquidityRelativeStrengthEngine()
        for i in range(5):
            # Price range: 150.00 to 150.04 (0.04% of 150 = 4 bps)
            engine.update_tick("AAPL", 150.0 + (i * 0.01), 1000.0, 1000000000 + i)

        spread = engine.get_spread_proxy_bps("AAPL")
        # (0.04 / 150) * 10000 = 2.67 bps
        assert spread < 5.0  # Less than 5 bps is tight

    def test_wide_spread(self):
        """Test spread proxy with wide bid-ask."""
        engine = LiquidityRelativeStrengthEngine()
        prices = [150.0, 150.5, 149.8, 150.2, 149.5]
        for i, price in enumerate(prices):
            engine.update_tick("AAPL", price, 1000.0, 1000000000 + i)

        spread = engine.get_spread_proxy_bps("AAPL")
        assert spread > 5.0  # Wide spread

    def test_insufficient_ticks(self):
        """Test spread proxy returns 0 with <5 ticks."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)

        spread = engine.get_spread_proxy_bps("AAPL")
        assert spread == 0.0


class TestRelativeStrength:
    """Test relative strength calculations."""

    def test_raw_rs_vs_benchmark(self):
        """Test raw RS calculation."""
        engine = LiquidityRelativeStrengthEngine()
        # Stock up 2%, benchmark up 1%
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 102.0, 1000.0, 1000000001)
        engine.update_tick("SPY", 100.0, 1000.0, 1000000000)
        engine.update_tick("SPY", 101.0, 1000.0, 1000000001)

        rs = engine.get_relative_strength_raw("AAPL")
        assert abs(rs - 0.01) < 0.001  # 2% - 1% = 1%

    def test_rs_benchmark_symbol(self):
        """Test RS returns 0 for benchmark symbol."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("SPY", 100.0, 1000.0, 1000000000)
        engine.update_tick("SPY", 102.0, 1000.0, 1000000001)

        rs = engine.get_relative_strength("SPY")
        assert rs == 0.0

    def test_rs_missing_data_non_strict(self):
        """Test RS gracefully handles missing data in non-strict mode."""
        engine = LiquidityRelativeStrengthEngine(strict_mode=False)
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)

        # No SPY data yet
        rs = engine.get_relative_strength_raw("AAPL")
        assert rs == 0.0

    def test_rs_missing_data_strict(self):
        """Test RS raises in strict mode with missing data."""
        engine = LiquidityRelativeStrengthEngine(strict_mode=True)
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)

        with pytest.raises(DataIntegrityError):
            engine.get_relative_strength_raw("AAPL")

    def test_sector_adjusted_rs(self):
        """Test sector-adjusted RS calculation."""
        engine = LiquidityRelativeStrengthEngine(use_sector_adjusted_rs=True)

        # Setup: AAPL up 3%, XLK up 2%, SPY up 1.5%
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 103.0, 1000.0, 1000000001)

        engine.update_tick("XLK", 100.0, 1000.0, 1000000000)
        engine.update_tick("XLK", 102.0, 1000.0, 1000000001)

        engine.update_tick("SPY", 100.0, 1000.0, 1000000000)
        engine.update_tick("SPY", 101.5, 1000.0, 1000000001)

        rs = engine.get_relative_strength("AAPL")
        # Should be positive (outperforming sector/benchmark)
        assert rs > 0.0

    def test_sector_adjusted_rs_missing_sector_fallback(self):
        """Test sector-adjusted RS falls back to raw RS if sector data missing."""
        engine = LiquidityRelativeStrengthEngine(
            use_sector_adjusted_rs=True, strict_mode=False
        )

        # AAPL tracked but no XLK sector data
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 102.0, 1000.0, 1000000001)
        engine.update_tick("SPY", 100.0, 1000.0, 1000000000)
        engine.update_tick("SPY", 101.0, 1000.0, 1000000001)

        rs = engine.get_relative_strength("AAPL")
        rs_raw = engine.get_relative_strength_raw("AAPL")
        assert rs == rs_raw  # Falls back to raw


class TestVolatility:
    """Test intraday volatility calculation."""

    def test_volatile_stock(self):
        """Test volatility with volatile returns."""
        engine = LiquidityRelativeStrengthEngine()
        prices = [100.0, 102.0, 98.0, 105.0, 99.0]
        for i, price in enumerate(prices):
            engine.update_tick("AAPL", price, 1000.0, 1000000000 + i)

        vol = engine.get_intraday_volatility("AAPL")
        assert vol > 0.01  # Should be significant

    def test_stable_stock(self):
        """Test volatility with stable prices (returns floor)."""
        engine = LiquidityRelativeStrengthEngine()
        for i in range(10):
            engine.update_tick("AAPL", 150.0, 1000.0, 1000000000 + i)

        vol = engine.get_intraday_volatility("AAPL")
        assert vol == 0.001  # Should return floor when zero volatility

    def test_insufficient_ticks_returns_floor(self):
        """Test volatility returns floor with <3 ticks."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)

        vol = engine.get_intraday_volatility("AAPL")
        assert vol == 0.001


class TestLiquidityScore:
    """Test liquidity quality scoring."""

    def test_high_liquidity_score(self):
        """Test high liquidity (tight spread, good volume)."""
        engine = LiquidityRelativeStrengthEngine()
        # Tight prices, consistent volume
        for i in range(10):
            engine.update_tick("AAPL", 150.0 + (i * 0.001), 1000.0, 1000000000 + i)
            engine.update_tick("AAPL", 150.0, 1000.0, 1000000000 + i + 0.5)

        score = engine.get_liquidity_score("AAPL")
        assert score > 0.7

    def test_low_liquidity_score(self):
        """Test low liquidity (wide spread, inconsistent volume)."""
        engine = LiquidityRelativeStrengthEngine()
        # Wide prices, volume spike
        prices = [100.0, 102.0, 99.0, 103.0, 98.0]
        volumes = [100.0, 100.0, 100.0, 100.0, 1000.0]
        for i, (price, vol) in enumerate(zip(prices, volumes)):
            engine.update_tick("AAPL", price, vol, 1000000000 + i)

        score = engine.get_liquidity_score("AAPL")
        assert score < 0.6

    def test_insufficient_ticks_score(self):
        """Test liquidity score returns 0.5 with insufficient ticks."""
        engine = LiquidityRelativeStrengthEngine(min_ticks_for_quality=10)
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)

        score = engine.get_liquidity_score("AAPL")
        assert score == 0.5


class TestDistVWAP:
    """Test distance from VWAP calculation."""

    def test_price_above_vwap(self):
        """Test price above VWAP."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 101.0, 1000.0, 1000000001)

        dist = engine.get_dist_vwap_pct("AAPL", reference_price=101.0)
        assert dist > 0.0

    def test_price_below_vwap(self):
        """Test price below VWAP."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 101.0, 1000.0, 1000000001)

        dist = engine.get_dist_vwap_pct("AAPL", reference_price=100.0)
        assert dist < 0.0

    def test_price_at_vwap(self):
        """Test price at VWAP."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)

        dist = engine.get_dist_vwap_pct("AAPL", reference_price=100.0)
        assert abs(dist) < 0.001


class TestEvaluateCandidate:
    """Test the main evaluate_candidate function."""

    def test_healthy_candidate(self):
        """Test evaluation of healthy candidate."""
        engine = LiquidityRelativeStrengthEngine()

        # Setup healthy data
        for i in range(20):
            engine.update_tick("AAPL", 150.0 + (i * 0.01), 1000.0, 1000000000 + i)
            engine.update_tick("SPY", 400.0 + (i * 0.01), 10000.0, 1000000000 + i)

        result = engine.evaluate_candidate("AAPL", "BUY", 150.5)

        assert result["symbol"] == "AAPL"
        assert result["action"] == "BUY"
        assert "metrics" in result
        assert "scoring" in result
        assert "veto" in result
        assert isinstance(result["veto"]["hard_veto"], bool)
        assert result["veto"]["hard_veto"] is False

    def test_candidate_with_no_ticks(self):
        """Test evaluation rejects candidate with no ticks."""
        engine = LiquidityRelativeStrengthEngine()

        result = engine.evaluate_candidate("UNKNOWN", "BUY", 150.0)

        assert result["veto"]["hard_veto"] is True
        assert "no_tick_history" in result["veto"]["reasons"]

    def test_candidate_with_invalid_price(self):
        """Test evaluation rejects invalid reference price."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)

        result = engine.evaluate_candidate("AAPL", "BUY", -150.0)

        assert result["veto"]["hard_veto"] is True
        assert "invalid_reference_price" in result["veto"]["reasons"]

    def test_candidate_with_wide_spread(self):
        """Test evaluation rejects wide spread."""
        engine = LiquidityRelativeStrengthEngine(max_spread_bps=10.0)

        # Create wide spread
        for i in range(5):
            engine.update_tick("AAPL", 100.0 + (i * 0.5), 1000.0, 1000000000 + i)

        result = engine.evaluate_candidate("AAPL", "BUY", 102.0)

        assert result["veto"]["hard_veto"] is True
        assert any("spread" in r for r in result["veto"]["reasons"])

    def test_candidate_rs_alignment_long(self):
        """Test RS alignment for long position."""
        engine = LiquidityRelativeStrengthEngine()

        # Stock up 2%, SPY up 1% (positive RS)
        engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
        engine.update_tick("AAPL", 102.0, 1000.0, 1000000001)
        engine.update_tick("SPY", 100.0, 10000.0, 1000000000)
        engine.update_tick("SPY", 101.0, 10000.0, 1000000001)

        for _ in range(20):
            engine.update_tick("AAPL", 102.0, 1000.0, 1000000002 + _)
            engine.update_tick("SPY", 101.0, 10000.0, 1000000002 + _)

        result = engine.evaluate_candidate("AAPL", "BUY", 102.0)

        # Positive RS for long should not add warning
        assert "negative_rs_for_long" not in result["veto"]["reasons"]

    def test_candidate_unknown_action(self):
        """Test evaluation rejects unknown action."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("AAPL", 150.0, 1000.0, 1000000000)

        result = engine.evaluate_candidate("AAPL", "HODL", 150.0)

        assert result["veto"]["hard_veto"] is True
        assert "unknown_action" in result["veto"]["reasons"]

    def test_candidate_output_structure(self):
        """Test output has correct nested structure."""
        engine = LiquidityRelativeStrengthEngine()
        for i in range(20):
            engine.update_tick("AAPL", 150.0, 1000.0, 1000000000 + i)
            engine.update_tick("SPY", 400.0, 10000.0, 1000000000 + i)

        result = engine.evaluate_candidate("AAPL", "BUY", 150.0)

        assert "symbol" in result
        assert "action" in result
        assert "reference_price" in result
        assert "metrics" in result
        assert "scoring" in result
        assert "veto" in result
        assert "context" in result

        assert isinstance(result["metrics"], dict)
        assert isinstance(result["scoring"], dict)
        assert isinstance(result["veto"], dict)
        assert isinstance(result["context"], dict)

        assert "rs" in result["metrics"]
        assert "total_score" in result["scoring"]
        assert "hard_veto" in result["veto"]
        assert "benchmark" in result["context"]


class TestDataIntegrity:
    """Test data integrity checking."""

    def test_data_integrity_healthy(self):
        """Test data integrity check when healthy."""
        engine = LiquidityRelativeStrengthEngine()
        engine.update_tick("SPY", 400.0, 10000.0, 1000000000)

        status = engine.check_data_integrity()

        assert status["is_healthy"] is True
        assert len(status["required_symbols_missing"]) == 0
        assert status["benchmark_data_available"] is True

    def test_data_integrity_missing_benchmark(self):
        """Test data integrity check with missing benchmark."""
        engine = LiquidityRelativeStrengthEngine()

        status = engine.check_data_integrity()

        assert status["is_healthy"] is False
        assert "SPY" in status["required_symbols_missing"]


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_market_day_simulation(self):
        """Simulate a full market day with multiple symbols."""
        engine = LiquidityRelativeStrengthEngine()

        # Simulate market open + 100 ticks per symbol
        symbols = ["AAPL", "MSFT", "SPY", "XLK"]
        base_prices = {"AAPL": 150.0, "MSFT": 380.0, "SPY": 450.0, "XLK": 145.0}

        for tick_num in range(100):
            for symbol in symbols:
                # Random walk
                price_move = (tick_num % 3) - 1  # -1, 0, or 1
                price = base_prices[symbol] + price_move * 0.1
                volume = 1000 + (tick_num % 500)
                engine.update_tick(symbol, price, volume, 1000000000 + tick_num)

        # Evaluate all candidates
        for symbol in ["AAPL", "MSFT"]:
            result = engine.evaluate_candidate(symbol, "BUY", base_prices[symbol])
            assert "metrics" in result
            assert result["symbol"] == symbol

        # Check integrity
        status = engine.check_data_integrity()
        assert status["is_healthy"] is True

    def test_sector_weighted_decision(self):
        """Test sector-weighted RS impacts ranking."""
        # Create two engines: one with sector adjustment, one without
        engine_adjusted = LiquidityRelativeStrengthEngine(
            use_sector_adjusted_rs=True
        )
        engine_raw = LiquidityRelativeStrengthEngine(use_sector_adjusted_rs=False)

        # Stock down 1%, sector ETF up 2%, SPY up 0.5%
        for engine in [engine_adjusted, engine_raw]:
            engine.update_tick("AAPL", 100.0, 1000.0, 1000000000)
            engine.update_tick("AAPL", 99.0, 1000.0, 1000000001)

            engine.update_tick("XLK", 100.0, 1000.0, 1000000000)
            engine.update_tick("XLK", 102.0, 1000.0, 1000000001)

            engine.update_tick("SPY", 100.0, 10000.0, 1000000000)
            engine.update_tick("SPY", 100.5, 10000.0, 1000000001)

            for _ in range(20):
                engine.update_tick("AAPL", 99.0, 1000.0, 1000000002 + _)
                engine.update_tick("XLK", 102.0, 1000.0, 1000000002 + _)
                engine.update_tick("SPY", 100.5, 10000.0, 1000000002 + _)

        result_adjusted = engine_adjusted.evaluate_candidate("AAPL", "BUY", 99.0)
        result_raw = engine_raw.evaluate_candidate("AAPL", "BUY", 99.0)

        # Sector-adjusted should account for sector outperformance
        # Raw RS is: -1% - 0.5% = -1.5%
        # Adjusted RS accounts for sector strength
        assert "rs" in result_adjusted["metrics"]
        assert "rs" in result_raw["metrics"]