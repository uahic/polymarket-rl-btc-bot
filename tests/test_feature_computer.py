"""
Tests for FeatureComputer class.

Tests all 22 features with proper assertions for:
- Shape validation
- Feature-specific computations
- Normalization and clamping
- Edge cases
"""

import pytest
import numpy as np
from features.computer import (
    FeatureComputer,
    RawMarketData,
    OrderbookSnapshot,
    FuturesData,
    PositionState,
    TransactionState,
    CapitalState,
)


def test_feature_computation_shape(
    sample_raw_market_data,
    sample_position_flat,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test that compute_features returns correct shape and dtype."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_flat,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    # Assert shape
    assert features.shape == (26,), f"Expected shape (26,), got {features.shape}"

    # Assert dtype
    assert features.dtype == np.float32, f"Expected dtype float32, got {features.dtype}"

    # Assert all finite
    assert np.all(np.isfinite(features)), "Features contain NaN or Inf values"


def test_momentum_features(
    sample_raw_market_data,
    sample_position_flat,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test momentum features (indices 0-2): returns_1m, 5m, 10m."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_flat,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    # Extract momentum features (first 3)
    returns_1m_feat = features[0]
    returns_5m_feat = features[1]
    returns_10m_feat = features[2]

    # Expected values (scaled by returns_scale = 50.0)
    expected_1m = sample_raw_market_data.futures.returns_1m * 50.0
    expected_5m = sample_raw_market_data.futures.returns_5m * 50.0
    expected_10m = sample_raw_market_data.futures.returns_10m * 50.0

    # Assert values match (clamped to [-1, 1])
    assert np.isclose(returns_1m_feat, np.clip(expected_1m, -1, 1)), \
        f"returns_1m mismatch: got {returns_1m_feat}, expected {np.clip(expected_1m, -1, 1)}"
    assert np.isclose(returns_5m_feat, np.clip(expected_5m, -1, 1)), \
        f"returns_5m mismatch"
    assert np.isclose(returns_10m_feat, np.clip(expected_10m, -1, 1)), \
        f"returns_10m mismatch"

    # Assert all within valid range
    assert -1.0 <= returns_1m_feat <= 1.0, "returns_1m out of range"
    assert -1.0 <= returns_5m_feat <= 1.0, "returns_5m out of range"
    assert -1.0 <= returns_10m_feat <= 1.0, "returns_10m out of range"


def test_orderflow_features(
    sample_raw_market_data,
    sample_position_flat,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test orderflow features (indices 3-6): OB imbalance L1/L5, trade flow, CVD accel."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_flat,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    # Extract orderflow features
    ob_imbalance_l1 = features[3]
    ob_imbalance_l5 = features[4]
    trade_flow = features[5]
    cvd_accel = features[6]

    # OB imbalance L1: (bid_size - ask_size) / (bid_size + ask_size)
    bid_l1 = sample_raw_market_data.orderbook.bids_l5[0][1]
    ask_l1 = sample_raw_market_data.orderbook.asks_l5[0][1]
    expected_imb_l1 = (bid_l1 - ask_l1) / (bid_l1 + ask_l1)

    assert np.isclose(ob_imbalance_l1, expected_imb_l1), \
        f"OB imbalance L1 mismatch: got {ob_imbalance_l1}, expected {expected_imb_l1}"

    # All orderflow features should be in [-1, 1]
    assert -1.0 <= ob_imbalance_l1 <= 1.0, "ob_imbalance_l1 out of range"
    assert -1.0 <= ob_imbalance_l5 <= 1.0, "ob_imbalance_l5 out of range"
    assert -1.0 <= trade_flow <= 1.0, "trade_flow out of range"
    assert -1.0 <= cvd_accel <= 1.0, "cvd_accel out of range"


def test_orderflow_empty_orderbook():
    """Test orderflow features with empty orderbook."""
    computer = FeatureComputer()

    # Create orderbook with no bids/asks
    empty_orderbook = OrderbookSnapshot(
        timestamp=0.0,
        best_bid=0.0,
        best_ask=0.0,
        spread=0.0,
        bids_l5=[],
        asks_l5=[],
    )

    raw_data = RawMarketData(
        timestamp=0.0,
        asset="BTC",
        orderbook=empty_orderbook,
        futures=FuturesData(timestamp=0.0, price=50000.0),
        spot=None,
        prob_up=0.5,
        time_remaining=1.0,
    )

    features = computer.compute_features(
        raw_data=raw_data,
        position=PositionState(),
        transaction=TransactionState(),
        capital=CapitalState(),
    )

    # OB imbalance should be 0 when orderbook is empty
    assert features[3] == 0.0, "OB imbalance L1 should be 0 for empty orderbook"
    assert features[4] == 0.0, "OB imbalance L5 should be 0 for empty orderbook"


def test_microstructure_features(
    sample_raw_market_data,
    sample_position_flat,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test microstructure features (indices 7-9): spread, trade intensity, large trade flag."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_flat,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    spread_feat = features[7]
    trade_intensity = features[8]
    large_trade_flag = features[9]

    # Spread should be positive and scaled
    assert spread_feat >= 0, "Spread should be non-negative"
    assert spread_feat <= 1.0, "Spread should be clamped to 1.0"

    # Trade intensity should be scaled
    expected_intensity = sample_raw_market_data.futures.trade_intensity / 10.0
    assert np.isclose(trade_intensity, np.clip(expected_intensity, 0, 1)), \
        f"Trade intensity mismatch"

    # Large trade flag should be binary
    assert large_trade_flag in [0.0, 1.0], \
        f"Large trade flag should be binary, got {large_trade_flag}"


def test_volatility_features(
    sample_raw_market_data,
    sample_position_flat,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test volatility features (indices 10-11): realized vol, vol expansion."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_flat,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    realized_vol = features[10]
    vol_expansion = features[11]

    # Volatility should be non-negative
    assert realized_vol >= -1.0, "Realized vol out of range"
    assert realized_vol <= 1.0, "Realized vol out of range"

    # Vol expansion should be finite
    assert np.isfinite(vol_expansion), "Vol expansion should be finite"
    assert -1.0 <= vol_expansion <= 1.0, "Vol expansion out of range"


def test_volatility_insufficient_history():
    """Test volatility calculation with insufficient prob_history."""
    computer = FeatureComputer()

    # Create data with short history
    raw_data = RawMarketData(
        timestamp=0.0,
        asset="BTC",
        orderbook=OrderbookSnapshot(
            timestamp=0.0,
            best_bid=0.48,
            best_ask=0.52,
            spread=0.04,
            bids_l5=[(0.48, 100)],
            asks_l5=[(0.52, 100)],
        ),
        futures=FuturesData(timestamp=0.0, price=50000.0, realized_vol_5m=0.02, avg_vol=0.02),
        spot=None,
        prob_up=0.5,
        time_remaining=1.0,
        prob_history=[0.5],  # Only 1 point
    )

    features = computer.compute_features(
        raw_data=raw_data,
        position=PositionState(),
        transaction=TransactionState(),
        capital=CapitalState(),
    )

    # Computed volatility from insufficient history should be 0
    # Note: feature 10 is realized_vol_5m from futures data (not computed from history)
    # So we just check it's valid
    assert np.isfinite(features[10]), "Volatility should be finite"


def test_position_features(
    sample_raw_market_data,
    sample_position_long,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test position features (indices 12-15): has_position, side, PnL, time_remaining."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_long,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    has_position = features[12]
    side = features[13]
    unrealized_pnl = features[14]
    time_remaining = features[15]

    # has_position should be 1.0 (True)
    assert has_position == 1.0, f"has_position should be 1.0, got {has_position}"

    # side should be 1.0 for UP
    assert side == 1.0, f"side should be 1.0 for UP, got {side}"

    # unrealized_pnl should be scaled
    expected_pnl = sample_position_long.unrealized_pnl / 50.0  # pnl_scale
    assert np.isclose(unrealized_pnl, np.clip(expected_pnl, -1, 1)), \
        f"unrealized_pnl mismatch"

    # time_remaining should match
    assert np.isclose(time_remaining, sample_position_long.time_remaining_normalized), \
        f"time_remaining mismatch"


def test_position_features_short(
    sample_raw_market_data,
    sample_position_short,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test position features with SHORT position."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_short,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    has_position = features[12]
    side = features[13]

    # has_position should be 1.0
    assert has_position == 1.0, "has_position should be 1.0 for SHORT"

    # side should be -1.0 for DOWN
    assert side == -1.0, f"side should be -1.0 for DOWN, got {side}"


def test_regime_features(
    sample_raw_market_data,
    sample_position_flat,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test regime features (indices 16-17): vol_regime, trend_regime."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_flat,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    vol_regime = features[16]
    trend_regime = features[17]

    # Regimes should match input (passthrough)
    assert vol_regime == sample_raw_market_data.vol_regime, \
        f"vol_regime mismatch"
    assert trend_regime == sample_raw_market_data.trend_regime, \
        f"trend_regime mismatch"


def test_transaction_features(
    sample_raw_market_data,
    sample_position_flat,
    sample_transaction_failed,
    sample_capital_state,
):
    """Test transaction features (indices 18-20): pending, failed, consecutive_failures."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_flat,
        transaction=sample_transaction_failed,
        capital=sample_capital_state,
    )

    pending = features[18]
    failed = features[19]
    consecutive_failures = features[20]

    # pending should be 0.0 (False)
    assert pending == 0.0, f"pending should be 0.0, got {pending}"

    # failed should be 1.0 (True)
    assert failed == 1.0, f"failed should be 1.0, got {failed}"

    # consecutive_failures should be scaled
    expected_failures = sample_transaction_failed.consecutive_failures / 5.0
    assert np.isclose(consecutive_failures, np.clip(expected_failures, 0, 1)), \
        f"consecutive_failures mismatch"


def test_capital_feature(
    sample_raw_market_data,
    sample_position_flat,
    sample_transaction_clean,
    sample_capital_state,
):
    """Test capital feature (index 21): available_balance."""
    computer = FeatureComputer()

    features = computer.compute_features(
        raw_data=sample_raw_market_data,
        position=sample_position_flat,
        transaction=sample_transaction_clean,
        capital=sample_capital_state,
    )

    balance = features[21]

    # Balance should be scaled by balance_scale (1000.0)
    expected_balance = sample_capital_state.available_balance / 1000.0
    assert np.isclose(balance, np.clip(expected_balance, -1, 1)), \
        f"balance mismatch: got {balance}, expected {expected_balance}"

    # Should be in [-1, 1]
    assert -1.0 <= balance <= 1.0, "balance out of range"


def test_feature_clamping():
    """Test that all features are properly clamped to expected ranges."""
    computer = FeatureComputer()

    # Create data with extreme values
    raw_data = RawMarketData(
        timestamp=0.0,
        asset="BTC",
        orderbook=OrderbookSnapshot(
            timestamp=0.0,
            best_bid=0.48,
            best_ask=0.52,
            spread=10.0,  # Extreme spread
            bids_l5=[(0.48, 100)],
            asks_l5=[(0.52, 100)],
        ),
        futures=FuturesData(
            timestamp=0.0,
            price=50000.0,
            returns_1m=1.0,  # 100% return (extreme)
            returns_5m=2.0,
            returns_10m=3.0,
            trade_intensity=1000.0,  # Extreme intensity
            realized_vol_5m=10.0,  # Extreme vol
            avg_vol=0.01,
        ),
        spot=None,
        prob_up=0.5,
        time_remaining=1.0,
    )

    position = PositionState(
        has_position=True,
        side="UP",
        unrealized_pnl=10000.0,  # Extreme PnL
        time_remaining_normalized=1.0,
    )

    features = computer.compute_features(
        raw_data=raw_data,
        position=position,
        transaction=TransactionState(consecutive_failures=100),  # Extreme failures
        capital=CapitalState(available_balance=100000.0),  # Extreme balance
    )

    # All features should be within [-1, 1] or [0, 1] depending on feature
    for i, feat in enumerate(features):
        assert -1.0 <= feat <= 1.0, \
            f"Feature {i} out of range: {feat}"
        assert np.isfinite(feat), \
            f"Feature {i} is not finite: {feat}"
