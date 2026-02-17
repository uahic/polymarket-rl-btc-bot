"""
Tests for data source classes (HistoricalSource and LiveSource).

Tests data loading, episode management, and data validation.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sources import HistoricalSource, LiveSource, DataSource
from features.computer import RawMarketData


# ============================================================================
# HistoricalSource Tests
# ============================================================================

def test_historical_source_reset():
    """Test HistoricalSource reset() loads an episode and returns valid RawMarketData."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC"],
        episode_length=1800,
    )

    # Reset to load episode (will use dummy data since no files exist)
    initial_data = source.reset(asset="BTC")

    # Assert returns RawMarketData
    assert isinstance(initial_data, RawMarketData), \
        f"Expected RawMarketData, got {type(initial_data)}"

    # Assert basic properties
    assert initial_data.asset == "BTC", "Asset should be BTC"
    assert 0.0 <= initial_data.prob_up <= 1.0, "prob_up should be in [0, 1]"
    assert 0.0 <= initial_data.time_remaining <= 1.0, "time_remaining should be in [0, 1]"
    assert initial_data.timestamp > 0, "timestamp should be positive"

    # Assert episode loaded
    assert source.current_episode is not None, "Episode should be loaded"
    assert len(source.current_episode) > 0, "Episode should have data"


def test_historical_source_episode_length():
    """Test that historical episodes have correct length."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC"],
        episode_length=1800,
    )

    source.reset(asset="BTC")

    # Assert episode has correct length
    assert len(source.current_episode) == 1800, \
        f"Expected 1800 ticks, got {len(source.current_episode)}"


def test_historical_source_advance():
    """Test HistoricalSource advance() increments index correctly."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC"],
        episode_length=100,  # Short episode for testing
    )

    source.reset(asset="BTC")
    initial_idx = source.current_idx

    # Advance
    has_more = source.advance()

    # Assert index incremented
    assert source.current_idx == initial_idx + 1, "Index should increment"

    # Assert returns True when data available
    assert has_more is True, "Should return True when data available"


def test_historical_source_advance_exhaustion():
    """Test advance() returns False when data exhausted."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC"],
        episode_length=5,  # Very short
    )

    source.reset(asset="BTC")

    # Advance until exhausted
    for _ in range(4):
        has_more = source.advance()
        assert has_more is True, "Should have data"

    # One more advance should exhaust
    has_more = source.advance()
    assert has_more is False, "Should be exhausted"


def test_historical_source_is_done():
    """Test is_done() matches advance() behavior."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC"],
        episode_length=5,
    )

    source.reset(asset="BTC")

    # Initially not done
    assert source.is_done() is False, "Should not be done initially"

    # Advance to end
    for _ in range(5):
        source.advance()

    # Now should be done
    assert source.is_done() is True, "Should be done after exhausting data"


def test_historical_source_get_current():
    """Test get_current() returns RawMarketData at current index."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC"],
        episode_length=10,
    )

    source.reset(asset="BTC")

    # Get current
    data1 = source.get_current()
    timestamp1 = data1.timestamp

    # Advance
    source.advance()

    # Get current again
    data2 = source.get_current()
    timestamp2 = data2.timestamp

    # Timestamps should increase
    assert timestamp2 > timestamp1, "Timestamps should increase monotonically"


def test_historical_episode_consistency():
    """Test that all ticks have valid, finite data."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC"],
        episode_length=50,
    )

    source.reset(asset="BTC")

    # Check all ticks
    for i in range(50):
        data = source.get_current()

        # Assert all numeric fields are finite
        assert np.isfinite(data.prob_up), f"Tick {i}: prob_up is not finite"
        assert np.isfinite(data.time_remaining), f"Tick {i}: time_remaining is not finite"
        assert np.isfinite(data.futures.price), f"Tick {i}: futures.price is not finite"
        assert np.isfinite(data.futures.returns_1m), f"Tick {i}: returns_1m is not finite"
        assert np.isfinite(data.futures.cvd), f"Tick {i}: cvd is not finite"
        assert np.isfinite(data.orderbook.best_bid), f"Tick {i}: best_bid is not finite"
        assert np.isfinite(data.orderbook.best_ask), f"Tick {i}: best_ask is not finite"

        # Assert valid ranges
        assert 0.0 <= data.prob_up <= 1.0, f"Tick {i}: prob_up out of range"
        assert 0.0 <= data.time_remaining <= 1.0, f"Tick {i}: time_remaining out of range"
        assert data.orderbook.spread >= 0, f"Tick {i}: spread should be non-negative"

        if i < 49:
            source.advance()


def test_historical_time_remaining_decreases():
    """Test that time_remaining decreases from 1.0 to 0.0."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC"],
        episode_length=100,
    )

    source.reset(asset="BTC")

    # First tick should have time_remaining near 1.0
    first_data = source.get_current()
    assert first_data.time_remaining >= 0.99, \
        f"First tick should have time_remaining ~1.0, got {first_data.time_remaining}"

    # Advance to near end
    for _ in range(95):
        source.advance()

    # Near end should have time_remaining near 0.0
    late_data = source.get_current()
    assert late_data.time_remaining <= 0.1, \
        f"Late tick should have time_remaining ~0.0, got {late_data.time_remaining}"


def test_historical_dummy_fallback():
    """Test behavior when no parquet files exist (dummy data generation)."""
    source = HistoricalSource(
        data_dir="nonexistent_directory",
        assets=["BTC"],
        episode_length=50,
    )

    # Should fall back to dummy data without crashing
    data = source.reset(asset="BTC")

    # Assert valid data returned
    assert isinstance(data, RawMarketData), "Should return valid RawMarketData"
    assert len(source.current_episode) == 50, "Dummy episode should have correct length"


def test_historical_multi_asset_sampling():
    """Test random asset selection from multiple assets."""
    source = HistoricalSource(
        data_dir="dataset/historical",
        assets=["BTC", "ETH", "SOL"],
        episode_length=10,
    )

    # Reset multiple times and collect assets
    assets_seen = set()
    for _ in range(5):
        data = source.reset()  # No specific asset -> random
        assets_seen.add(data.asset)

    # Should select from available assets
    assert len(assets_seen) > 0, "Should sample at least one asset"
    assert all(asset in ["BTC", "ETH", "SOL"] for asset in assets_seen), \
        "All assets should be from the provided list"


# ============================================================================
# LiveSource Tests
# ============================================================================

def test_live_source_reset():
    """Test LiveSource reset() initializes episode correctly."""
    # Create mock streamers
    mock_orderbook = Mock()
    mock_orderbook.get_latest.return_value = {
        "best_bid": 0.48,
        "best_ask": 0.52,
        "spread": 0.04,
        "bids_l5": [(0.48, 100)],
        "asks_l5": [(0.52, 100)],
    }

    mock_binance = Mock()
    mock_binance.get_latest.return_value = {
        "price": 50000.0,
    }

    mock_futures = Mock()
    mock_futures.get_latest.return_value = {
        "price": 50000.0,
        "returns_1m": 0.001,
        "returns_5m": 0.005,
        "returns_10m": 0.01,
        "cvd": 1000.0,
        "cvd_history": [900, 950, 1000],
        "trade_flow_imbalance": 0.1,
        "trade_intensity": 2.0,
        "large_trade_flag": 0.0,
        "realized_vol_5m": 0.02,
        "avg_vol": 0.02,
        "vol_regime": 0.0,
        "trend_regime": 1.0,
    }

    source = LiveSource(
        orderbook_streamer=mock_orderbook,
        binance_streamer=mock_binance,
        futures_streamer=mock_futures,
        tick_interval=0.01,  # Fast for testing
    )

    # Reset
    initial_data = source.reset(asset="BTC", market_id="test_market")

    # Assert returns valid RawMarketData
    assert isinstance(initial_data, RawMarketData), "Should return RawMarketData"
    assert source.current_asset == "BTC", "Asset should be set"
    assert source.current_market == "test_market", "Market should be set"
    assert source.episode_start_time is not None, "Start time should be set"


def test_live_source_get_current():
    """Test get_current() aggregates data from all streams."""
    # Create mock streamers with specific return values
    mock_orderbook = Mock()
    mock_orderbook.get_latest.return_value = {
        "best_bid": 0.48,
        "best_ask": 0.52,
        "spread": 0.04,
        "bids_l5": [(0.48, 100)],
        "asks_l5": [(0.52, 100)],
    }

    mock_binance = Mock()
    mock_binance.get_latest.return_value = {
        "price": 51000.0,  # Specific price
    }

    mock_futures = Mock()
    mock_futures.get_latest.return_value = {
        "price": 51000.0,
        "returns_1m": 0.002,  # Specific return
        "returns_5m": 0.010,
        "returns_10m": 0.020,
        "cvd": 2000.0,
        "cvd_history": [1900, 1950, 2000],
        "trade_flow_imbalance": 0.2,
        "trade_intensity": 3.0,
        "large_trade_flag": 1.0,
        "realized_vol_5m": 0.03,
        "avg_vol": 0.03,
        "vol_regime": 1.0,
        "trend_regime": 0.0,
    }

    source = LiveSource(
        orderbook_streamer=mock_orderbook,
        binance_streamer=mock_binance,
        futures_streamer=mock_futures,
        tick_interval=0.01,
    )

    source.reset(asset="BTC", market_id="test_market")

    # Get current data
    data = source.get_current()

    # Assert data aggregated from all streams
    assert data.orderbook.best_bid == 0.48, "Should get orderbook data"
    assert data.orderbook.best_ask == 0.52, "Should get orderbook data"
    assert data.futures.price == 51000.0, "Should get futures price"
    assert data.futures.returns_1m == 0.002, "Should get futures returns"
    assert data.spot.price == 51000.0, "Should get spot price"


def test_live_source_advance():
    """Test advance() sleeps for tick_interval and returns True."""
    mock_orderbook = Mock()
    mock_orderbook.get_latest.return_value = {"best_bid": 0.48, "best_ask": 0.52, "spread": 0.04, "bids_l5": [], "asks_l5": []}

    mock_binance = Mock()
    mock_binance.get_latest.return_value = {"price": 50000.0}

    mock_futures = Mock()
    mock_futures.get_latest.return_value = {
        "price": 50000.0, "returns_1m": 0.0, "returns_5m": 0.0, "returns_10m": 0.0,
        "cvd": 0.0, "cvd_history": [], "trade_flow_imbalance": 0.0,
        "trade_intensity": 0.0, "large_trade_flag": 0.0,
        "realized_vol_5m": 0.0, "avg_vol": 0.0, "vol_regime": 0.0, "trend_regime": 0.0,
    }

    source = LiveSource(
        orderbook_streamer=mock_orderbook,
        binance_streamer=mock_binance,
        futures_streamer=mock_futures,
        tick_interval=0.01,  # 10ms
    )

    source.reset(asset="BTC", market_id="test_market")

    # Advance should sleep and return True (not done)
    start_time = time.time()
    has_more = source.advance()
    elapsed = time.time() - start_time

    # Should sleep for ~tick_interval
    assert elapsed >= 0.01, f"Should sleep for at least tick_interval, got {elapsed}s"

    # Should return True (episode not done yet)
    assert has_more is True, "Should return True when episode not done"


def test_live_source_time_remaining():
    """Test that time_remaining starts at 1.0 and decreases."""
    mock_orderbook = Mock()
    mock_orderbook.get_latest.return_value = {"best_bid": 0.48, "best_ask": 0.52, "spread": 0.04, "bids_l5": [], "asks_l5": []}

    mock_binance = Mock()
    mock_binance.get_latest.return_value = {"price": 50000.0}

    mock_futures = Mock()
    mock_futures.get_latest.return_value = {
        "price": 50000.0, "returns_1m": 0.0, "returns_5m": 0.0, "returns_10m": 0.0,
        "cvd": 0.0, "cvd_history": [], "trade_flow_imbalance": 0.0,
        "trade_intensity": 0.0, "large_trade_flag": 0.0,
        "realized_vol_5m": 0.0, "avg_vol": 0.0, "vol_regime": 0.0, "trend_regime": 0.0,
    }

    source = LiveSource(
        orderbook_streamer=mock_orderbook,
        binance_streamer=mock_binance,
        futures_streamer=mock_futures,
        tick_interval=0.01,
    )

    # Set short episode duration for testing
    source.episode_duration = 1.0  # 1 second
    source.reset(asset="BTC", market_id="test_market")

    # Initial time_remaining should be ~1.0
    initial_data = source.get_current()
    assert initial_data.time_remaining >= 0.99, \
        f"Initial time_remaining should be ~1.0, got {initial_data.time_remaining}"

    # Sleep a bit
    time.sleep(0.5)

    # time_remaining should have decreased
    later_data = source.get_current()
    assert later_data.time_remaining < initial_data.time_remaining, \
        "time_remaining should decrease over time"


def test_live_source_is_done():
    """Test is_done() triggers after episode_duration elapsed."""
    mock_orderbook = Mock()
    mock_orderbook.get_latest.return_value = {"best_bid": 0.48, "best_ask": 0.52, "spread": 0.04, "bids_l5": [], "asks_l5": []}

    mock_binance = Mock()
    mock_binance.get_latest.return_value = {"price": 50000.0}

    mock_futures = Mock()
    mock_futures.get_latest.return_value = {
        "price": 50000.0, "returns_1m": 0.0, "returns_5m": 0.0, "returns_10m": 0.0,
        "cvd": 0.0, "cvd_history": [], "trade_flow_imbalance": 0.0,
        "trade_intensity": 0.0, "large_trade_flag": 0.0,
        "realized_vol_5m": 0.0, "avg_vol": 0.0, "vol_regime": 0.0, "trend_regime": 0.0,
    }

    source = LiveSource(
        orderbook_streamer=mock_orderbook,
        binance_streamer=mock_binance,
        futures_streamer=mock_futures,
        tick_interval=0.01,
    )

    # Set very short episode duration
    source.episode_duration = 0.1  # 100ms
    source.reset(asset="BTC", market_id="test_market")

    # Initially not done
    assert source.is_done() is False, "Should not be done initially"

    # Wait for episode to complete
    time.sleep(0.15)

    # Now should be done
    assert source.is_done() is True, "Should be done after episode_duration"


def test_live_source_prob_history():
    """Test that prob_history updates each tick and is bounded."""
    mock_orderbook = Mock()

    # Return different mid prices over time
    call_count = [0]

    def mock_get_latest(market_id):
        call_count[0] += 1
        prob = 0.5 + (call_count[0] * 0.01)  # Increasing probability
        return {
            "best_bid": prob - 0.02,
            "best_ask": prob + 0.02,
            "spread": 0.04,
            "bids_l5": [],
            "asks_l5": [],
        }

    mock_orderbook.get_latest = mock_get_latest

    mock_binance = Mock()
    mock_binance.get_latest.return_value = {"price": 50000.0}

    mock_futures = Mock()
    mock_futures.get_latest.return_value = {
        "price": 50000.0, "returns_1m": 0.0, "returns_5m": 0.0, "returns_10m": 0.0,
        "cvd": 0.0, "cvd_history": [], "trade_flow_imbalance": 0.0,
        "trade_intensity": 0.0, "large_trade_flag": 0.0,
        "realized_vol_5m": 0.0, "avg_vol": 0.0, "vol_regime": 0.0, "trend_regime": 0.0,
    }

    source = LiveSource(
        orderbook_streamer=mock_orderbook,
        binance_streamer=mock_binance,
        futures_streamer=mock_futures,
        tick_interval=0.01,
    )

    source.reset(asset="BTC", market_id="test_market")

    # Get data multiple times
    for _ in range(60):  # More than 50 to test bounding
        data = source.get_current()

    # prob_history should be bounded to 50 entries
    assert len(source.prob_history) == 50, \
        f"prob_history should be bounded to 50, got {len(source.prob_history)}"

    # All entries should be valid probabilities
    for prob in source.prob_history:
        assert 0.0 <= prob <= 1.0, f"Invalid probability in history: {prob}"
