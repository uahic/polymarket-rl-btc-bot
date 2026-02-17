"""
Tests for TradingGym environment.

Tests gym interface, action execution, reward computation, and episode management.
"""

import pytest
import numpy as np
from environments.trading_gym import TradingGym, ExecutionResult, TradingAction
from features.computer import (
    FeatureComputer,
    RawMarketData,
    FuturesData,
    SpotData,
    TransactionState,
)
from structures.action import Action


def test_gym_reset(mock_data_source, mock_executor):
    """Test that reset() returns valid (obs, info) tuple."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        max_episode_steps=100,
    )

    obs, info = env.reset()

    # Assert obs shape
    assert obs.shape == (26,), f"Expected observation shape (26,), got {obs.shape}"

    # Assert obs dtype
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"

    # Assert obs in valid range
    assert np.all(np.abs(obs) <= 10.0), "Observation values out of range [-10, 10]"

    # Assert all finite
    assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"

    # Assert info dict
    assert "asset" in info, "Info should contain asset"
    assert "timestamp" in info, "Info should contain timestamp"

    # Assert episode counters zeroed
    assert env.step_count == 0, "step_count should be 0 after reset"
    assert env.episode_pnl == 0.0, "episode_pnl should be 0 after reset"
    assert env.episode_trades == 0, "episode_trades should be 0 after reset"


def test_gym_observation_space(mock_data_source, mock_executor):
    """Test observation space is correctly defined."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
    )

    # Check observation space
    assert env.observation_space.shape == (26,), "Observation space should be (26,)"
    assert env.observation_space.dtype == np.float32, "Observation space dtype should be float32"

    # Check bounds
    assert np.all(env.observation_space.low == -10.0), "Low bound should be -10.0"
    assert np.all(env.observation_space.high == 10.0), "High bound should be 10.0"


def test_gym_action_space(mock_data_source, mock_executor):
    """Test action space is Discrete(3)."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
    )

    # Check action space
    assert env.action_space.n == 3, "Action space should be Discrete(3)"


def test_gym_step_buy_action(mock_data_source, mock_executor):
    """Test executing BUY action."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
    )

    env.reset()

    # Execute BUY action
    obs, reward, terminated, truncated, info = env.step(Action.BUY)

    # Assert step completed
    assert obs.shape == (26,), "Should return valid observation"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(terminated, bool), "terminated should be bool"
    assert isinstance(truncated, bool), "truncated should be bool"
    assert isinstance(info, dict), "info should be dict"

    # Assert info contains expected keys
    assert "step" in info, "Info should contain step"
    assert "action" in info, "Info should contain action"
    assert "filled" in info, "Info should contain filled"
    assert "balance" in info, "Info should contain balance"

    # For mock executor, BUY should succeed and open position
    if info["filled"]:
        assert info["has_position"] is True, "Should have position after BUY"
        assert info["balance"] < 1000.0, "Balance should decrease after BUY"


def test_gym_step_sell_action(mock_data_source, mock_executor):
    """Test executing SELL action (close position)."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
    )

    env.reset()

    # First BUY to open position
    env.step(Action.BUY)

    # Then SELL to close
    obs, reward, terminated, truncated, info = env.step(Action.SELL)

    # Assert step completed
    assert obs.shape == (26,), "Should return valid observation"

    # For mock executor, SELL should close position
    if info["filled"]:
        assert "pnl" in info, "Info should contain pnl"
        # PnL could be positive or negative


def test_gym_step_hold_action(mock_data_source, mock_executor):
    """Test executing HOLD action."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
    )

    env.reset()
    initial_balance = mock_executor.balance

    # Execute HOLD action
    obs, reward, terminated, truncated, info = env.step(Action.HOLD)

    # Assert step completed
    assert obs.shape == (26,), "Should return valid observation"

    # HOLD should not change balance (no trade)
    assert mock_executor.balance == initial_balance, "Balance should not change on HOLD"

    # filled should be False
    assert info["filled"] is False, "HOLD should not fill"


def test_gym_reward_terminal(mock_data_source, mock_executor):
    """Test terminal reward equals realized PnL."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        shaping_reward_coef=0.0,  # Disable shaping to test only terminal reward
        normalize_rewards=False,  # Disable normalization for easier testing
    )

    env.reset()

    # BUY to open position
    env.step(Action.BUY)

    # SELL to close and realize PnL
    obs, reward, terminated, truncated, info = env.step(Action.SELL)

    # If trade filled, reward should equal realized PnL
    if info["filled"] and info["pnl"] != 0.0:
        assert np.isclose(reward, info["pnl"]), \
            f"Reward should equal realized PnL: {reward} vs {info['pnl']}"


def test_gym_reward_shaping(mock_data_source, mock_executor):
    """Test shaping reward includes unrealized PnL."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        shaping_reward_coef=0.01,  # Enable shaping
        normalize_rewards=False,
    )

    env.reset()

    # BUY to open position
    env.step(Action.BUY)

    # HOLD (should get shaping reward from unrealized PnL)
    obs, reward, terminated, truncated, info = env.step(Action.HOLD)

    # Reward should include shaping component
    # (Will be non-zero if position has unrealized PnL)
    assert isinstance(reward, (int, float)), "Reward should be numeric"


def test_gym_termination(mock_data_source, mock_executor):
    """Test episode termination conditions."""
    feature_computer = FeatureComputer()

    # Use short episode
    mock_data_source.episode_length = 5

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        max_episode_steps=10,
    )

    env.reset()

    # Run until data source exhausted
    terminated = False
    truncated = False
    steps = 0

    while not terminated and not truncated and steps < 10:
        obs, reward, terminated, truncated, info = env.step(Action.HOLD)
        steps += 1

    # Should terminate when data source is done
    assert terminated or truncated, "Episode should terminate"


def test_gym_truncation():
    """Test episode truncates at max_steps."""
    from conftest import MockDataSource, MockOrderExecutor

    # Create data source with many steps
    mock_data_source = MockDataSource(episode_length=1000)
    mock_executor = MockOrderExecutor()
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        max_episode_steps=5,  # Very low max_steps
    )

    env.reset()

    # Run 5 steps
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(Action.HOLD)

    # Should be truncated after max_steps
    assert truncated is True, "Episode should be truncated after max_steps"


def test_gym_episode_metrics(mock_data_source, mock_executor):
    """Test that episode metrics accumulate correctly."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        normalize_rewards=False,
    )

    env.reset()

    # Execute several trades
    env.step(Action.BUY)  # Open position
    env.step(Action.SELL)  # Close position
    env.step(Action.BUY)  # Open again

    # Check episode metrics
    assert env.step_count == 3, f"step_count should be 3, got {env.step_count}"

    # episode_trades should count filled trades
    assert env.episode_trades >= 0, "episode_trades should be non-negative"

    # episode_pnl should accumulate
    assert isinstance(env.episode_pnl, (int, float)), "episode_pnl should be numeric"


def test_gym_info_dict(mock_data_source, mock_executor):
    """Test that info dict contains all expected keys."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
    )

    env.reset()
    obs, reward, terminated, truncated, info = env.step(Action.HOLD)

    # Check all expected keys present
    expected_keys = [
        "step", "timestamp", "action", "filled", "pnl",
        "balance", "unrealized_pnl", "has_position", "position_side",
        "episode_pnl", "episode_trades", "episode_spent",
        "amount_spent", "rejection_reason"
    ]

    for key in expected_keys:
        assert key in info, f"Info dict missing key: {key}"


# ---------------------------------------------------------------------------
# High Priority: Error handling & edge cases
# ---------------------------------------------------------------------------

def test_step_before_reset_raises(mock_data_source, mock_executor):
    """step() before reset() must raise RuntimeError."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
    )

    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.HOLD)


def test_reward_normalization_zero_variance(mock_data_source, mock_executor):
    """_normalize_reward() with constant zero rewards must not divide by zero."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        normalize_rewards=True,
    )
    env.reset()

    # All HOLD steps produce 0.0 reward — reward_std stays 0
    rewards = []
    for _ in range(5):
        _, reward, terminated, truncated, _ = env.step(Action.HOLD)
        rewards.append(reward)
        if terminated or truncated:
            break

    assert all(np.isfinite(r) for r in rewards), "Rewards must be finite even with zero variance"


def test_data_source_exhausted_immediately(mock_executor):
    """Episode with a single tick terminates immediately after first step."""
    from conftest import MockDataSource

    source = MockDataSource(episode_length=1)
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=mock_executor,
        feature_computer=feature_computer,
        max_episode_steps=100,
    )
    env.reset()

    _, _, terminated, truncated, _ = env.step(Action.HOLD)
    assert terminated or truncated, "Single-tick episode should end after one step"


def test_max_episode_steps_one(mock_executor):
    """max_episode_steps=1 truncates on the very first step."""
    from conftest import MockDataSource

    source = MockDataSource(episode_length=1000)
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=mock_executor,
        feature_computer=feature_computer,
        max_episode_steps=1,
    )
    env.reset()

    _, _, terminated, truncated, _ = env.step(Action.HOLD)
    assert truncated is True, "Should truncate when max_episode_steps=1"


# ---------------------------------------------------------------------------
# High Priority: Action enum properties
# ---------------------------------------------------------------------------

def test_action_is_buy():
    """is_buy is True only for BUY."""
    assert Action.BUY.is_buy is True
    assert Action.HOLD.is_buy is False
    assert Action.SELL.is_buy is False


def test_action_is_sell():
    """is_sell is True only for SELL."""
    assert Action.SELL.is_sell is True
    assert Action.BUY.is_sell is False
    assert Action.HOLD.is_sell is False


def test_action_size_multiplier():
    """size_multiplier is 0.5 for BUY/SELL and 0.0 for HOLD."""
    assert Action.BUY.size_multiplier == 0.5
    assert Action.SELL.size_multiplier == 0.5
    assert Action.HOLD.size_multiplier == 0.0


def test_action_get_confidence_size_neutral():
    """At prob=0.5 (neutral), confidence size is at minimum (0.25)."""
    assert Action.BUY.get_confidence_size(0.5) == pytest.approx(0.25)
    assert Action.SELL.get_confidence_size(0.5) == pytest.approx(0.25)


def test_action_get_confidence_size_extreme():
    """At extreme probabilities, confidence size approaches 1.0."""
    assert Action.BUY.get_confidence_size(0.0) == pytest.approx(1.0)
    assert Action.BUY.get_confidence_size(1.0) == pytest.approx(1.0)


def test_action_get_confidence_size_hold():
    """HOLD always returns 0.0 regardless of probability."""
    assert Action.HOLD.get_confidence_size(0.0) == 0.0
    assert Action.HOLD.get_confidence_size(0.5) == 0.0
    assert Action.HOLD.get_confidence_size(1.0) == 0.0


def test_action_get_confidence_size_midpoint():
    """At prob=0.75 (midpoint between neutral and extreme), size is 0.625."""
    expected = 0.25 + 0.75 * (abs(0.75 - 0.5) * 2)
    assert Action.BUY.get_confidence_size(0.75) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Medium Priority: Episode metrics reset between episodes
# ---------------------------------------------------------------------------

def test_episode_metrics_reset_on_reset(mock_data_source, mock_executor):
    """All episode metrics must be zeroed when reset() is called."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        normalize_rewards=False,
    )

    # First episode: accumulate metrics
    env.reset()
    env.step(Action.BUY)
    env.step(Action.SELL)

    # Second reset must zero everything
    env.reset()
    assert env.step_count == 0, "step_count must be 0 after reset"
    assert env.episode_pnl == 0.0, "episode_pnl must be 0 after reset"
    assert env.episode_trades == 0, "episode_trades must be 0 after reset"
    assert env.episode_fees == 0.0, "episode_fees must be 0 after reset"
    assert env.episode_spent == 0.0, "episode_spent must be 0 after reset"


def test_episode_metrics_in_info_match_env(mock_data_source, mock_executor):
    """Info dict episode metrics must match env attributes at every step."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        normalize_rewards=False,
    )
    env.reset()

    for action in [Action.BUY, Action.SELL, Action.HOLD]:
        _, _, terminated, truncated, info = env.step(action)
        assert info["episode_pnl"] == env.episode_pnl
        assert info["episode_trades"] == env.episode_trades
        assert info["episode_spent"] == env.episode_spent
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Medium Priority: Reward shaping coefficient edge cases
# ---------------------------------------------------------------------------

def test_reward_shaping_coef_zero(mock_data_source, mock_executor):
    """With shaping_reward_coef=0.0, reward equals only realized PnL."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        shaping_reward_coef=0.0,
        normalize_rewards=False,
    )
    env.reset()

    # BUY to open a position
    env.step(Action.BUY)

    # HOLD — no realized PnL, no shaping → reward must be 0
    _, reward, _, _, info = env.step(Action.HOLD)
    assert reward == pytest.approx(0.0), (
        f"With shaping_coef=0 and no realized PnL, reward should be 0, got {reward}"
    )


def test_reward_equals_pnl_on_close_no_shaping(mock_data_source, mock_executor):
    """Reward on closing a position (no shaping) equals the realized PnL."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        shaping_reward_coef=0.0,
        normalize_rewards=False,
    )
    env.reset()

    env.step(Action.BUY)
    _, reward, _, _, info = env.step(Action.SELL)

    if info["filled"] and info["pnl"] != 0.0:
        assert reward == pytest.approx(info["pnl"]), (
            f"Reward should equal pnl {info['pnl']}, got {reward}"
        )


# ---------------------------------------------------------------------------
# Medium Priority: rejection_reason populated on rejected trades
# ---------------------------------------------------------------------------

def test_rejection_reason_on_hold(mock_data_source, mock_executor):
    """HOLD never fills; rejection_reason should be None (not an error)."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
    )
    env.reset()

    _, _, _, _, info = env.step(Action.HOLD)
    assert info["filled"] is False
    assert info["rejection_reason"] is None


def test_rejection_reason_type(mock_data_source, mock_executor):
    """rejection_reason in info dict is always None or a string."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
    )
    env.reset()

    for action in [Action.BUY, Action.HOLD, Action.SELL]:
        _, _, terminated, truncated, info = env.step(action)
        assert info["rejection_reason"] is None or isinstance(info["rejection_reason"], str), (
            f"rejection_reason must be None or str, got {type(info['rejection_reason'])}"
        )
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Medium Priority: FeatureComputer edge cases
# ---------------------------------------------------------------------------

def test_feature_computer_zero_prob_spread(sample_orderbook, sample_futures_data,
                                           sample_spot_data, sample_position_flat,
                                           sample_transaction_clean, sample_capital_state):
    """prob_up=0.0 must not produce NaN or Inf in spread feature."""
    fc = FeatureComputer()

    raw = RawMarketData(
        timestamp=1.0,
        asset="BTC",
        orderbook=sample_orderbook,
        futures=sample_futures_data,
        spot=sample_spot_data,
        prob_up=0.0,  # Edge: zero probability
        time_remaining=0.5,
        prob_history=[0.5] * 10,
    )
    features = fc.compute_features(raw, sample_position_flat, sample_transaction_clean,
                                   sample_capital_state)

    assert np.all(np.isfinite(features)), "Features must be finite with prob_up=0"


def test_feature_computer_empty_prob_history(sample_orderbook, sample_futures_data,
                                             sample_spot_data, sample_position_flat,
                                             sample_transaction_clean, sample_capital_state):
    """Empty prob_history must produce finite features (velocity=0, vol=0)."""
    fc = FeatureComputer()

    raw = RawMarketData(
        timestamp=1.0,
        asset="BTC",
        orderbook=sample_orderbook,
        futures=sample_futures_data,
        spot=sample_spot_data,
        prob_up=0.5,
        time_remaining=0.5,
        prob_history=[],  # Edge: no history
    )
    features = fc.compute_features(raw, sample_position_flat, sample_transaction_clean,
                                   sample_capital_state)

    assert np.all(np.isfinite(features)), "Features must be finite with empty prob_history"
    # vol and velocity features should be zero
    assert features[10] == pytest.approx(0.0), "realized_vol should be 0 with no history"


def test_feature_computer_single_element_prob_history(sample_orderbook, sample_futures_data,
                                                       sample_spot_data, sample_position_flat,
                                                       sample_transaction_clean,
                                                       sample_capital_state):
    """prob_history with a single element: velocity and vol should be 0."""
    fc = FeatureComputer()

    raw = RawMarketData(
        timestamp=1.0,
        asset="BTC",
        orderbook=sample_orderbook,
        futures=sample_futures_data,
        spot=sample_spot_data,
        prob_up=0.5,
        time_remaining=0.5,
        prob_history=[0.5],
    )
    features = fc.compute_features(raw, sample_position_flat, sample_transaction_clean,
                                   sample_capital_state)
    assert np.all(np.isfinite(features)), "Features must be finite with single-element history"


def test_feature_computer_all_features_clamped(sample_orderbook, sample_position_flat,
                                                sample_transaction_clean, sample_capital_state):
    """Every feature in the output must be within [-1, 1] (or [0, 1] for binary)."""
    fc = FeatureComputer()

    # Extreme input values
    extreme_futures = FuturesData(
        timestamp=1.0,
        price=1e9,
        returns_1m=10.0,   # way above scale
        returns_5m=-10.0,
        returns_10m=5.0,
        cvd=1e8,
        cvd_history=[0.0, 1e8],
        trade_flow_imbalance=5.0,
        trade_intensity=1e6,
        large_trade_flag=1.0,
        realized_vol_5m=1.0,
        avg_vol=1.0,
    )
    extreme_spot = SpotData(timestamp=1.0, price=1e9, change_pct=5.0)
    raw = RawMarketData(
        timestamp=1.0,
        asset="BTC",
        orderbook=sample_orderbook,
        futures=extreme_futures,
        spot=extreme_spot,
        prob_up=0.5,
        time_remaining=0.5,
        prob_history=[0.5] * 30,
    )

    features = fc.compute_features(raw, sample_position_flat, sample_transaction_clean,
                                   sample_capital_state)

    assert np.all(np.isfinite(features)), "All features must be finite"
    assert np.all(features >= -1.0), f"Feature below -1: {features[features < -1.0]}"
    assert np.all(features <= 1.0), f"Feature above 1: {features[features > 1.0]}"


def test_feature_computer_cvd_constant_history(sample_orderbook, sample_spot_data,
                                                sample_position_flat, sample_transaction_clean,
                                                sample_capital_state):
    """Constant CVD history must yield zero CVD acceleration."""
    fc = FeatureComputer()

    futures = FuturesData(
        timestamp=1.0,
        price=50000.0,
        cvd=100.0,
        cvd_history=[100.0, 100.0, 100.0],  # Constant — no acceleration
        trade_flow_imbalance=0.0,
        trade_intensity=0.0,
        large_trade_flag=0.0,
        realized_vol_5m=0.01,
        avg_vol=0.01,
    )
    raw = RawMarketData(
        timestamp=1.0,
        asset="BTC",
        orderbook=sample_orderbook,
        futures=futures,
        spot=sample_spot_data,
        prob_up=0.5,
        time_remaining=0.5,
        prob_history=[0.5] * 10,
    )
    features = fc.compute_features(raw, sample_position_flat, sample_transaction_clean,
                                   sample_capital_state)

    # Feature index 6 is CVD acceleration
    assert features[6] == pytest.approx(0.0), "Constant CVD must give zero acceleration"


# ---------------------------------------------------------------------------
# Lower Priority: consecutive_failures feature increments
# ---------------------------------------------------------------------------

def test_consecutive_failures_feature_reflects_state(sample_orderbook, sample_futures_data,
                                                      sample_spot_data, sample_position_flat,
                                                      sample_capital_state):
    """Feature index 20 must scale with consecutive_failures count."""
    fc = FeatureComputer()

    raw = RawMarketData(
        timestamp=1.0,
        asset="BTC",
        orderbook=sample_orderbook,
        futures=sample_futures_data,
        spot=sample_spot_data,
        prob_up=0.5,
        time_remaining=0.5,
        prob_history=[0.5] * 10,
    )

    txn_zero = TransactionState(consecutive_failures=0)
    txn_three = TransactionState(consecutive_failures=3)

    feats_zero = fc.compute_features(raw, sample_position_flat, txn_zero, sample_capital_state)
    feats_three = fc.compute_features(raw, sample_position_flat, txn_three, sample_capital_state)

    # Feature index 20 = consecutive_failures / failures_scale
    assert feats_zero[20] == pytest.approx(0.0)
    assert feats_three[20] == pytest.approx(3.0 / fc.failures_scale)


# ---------------------------------------------------------------------------
# Lower Priority: Info dict type consistency
# ---------------------------------------------------------------------------

def test_info_dict_types_consistent_across_steps(mock_data_source, mock_executor):
    """All info dict values must have consistent types across multiple steps."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        normalize_rewards=False,
    )
    env.reset()

    float_keys = {"pnl", "balance", "unrealized_pnl", "episode_pnl", "episode_spent",
                  "amount_spent"}
    int_keys = {"step", "episode_trades"}
    bool_keys = {"filled", "has_position"}

    for action in [Action.BUY, Action.HOLD, Action.SELL, Action.HOLD]:
        _, _, terminated, truncated, info = env.step(action)

        for key in float_keys:
            assert isinstance(info[key], (int, float)), (
                f"info['{key}'] should be numeric, got {type(info[key])}"
            )
        for key in int_keys:
            assert isinstance(info[key], (int, np.integer)), (
                f"info['{key}'] should be int, got {type(info[key])}"
            )
        for key in bool_keys:
            assert isinstance(info[key], bool), (
                f"info['{key}'] should be bool, got {type(info[key])}"
            )

        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Lower Priority: Terminal observation is all-zeros
# ---------------------------------------------------------------------------

def test_terminal_observation_is_zeros():
    """On termination, the returned observation must be all zeros."""
    from conftest import MockDataSource, MockOrderExecutor

    source = MockDataSource(episode_length=2)
    executor = MockOrderExecutor()
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=executor,
        feature_computer=feature_computer,
        max_episode_steps=1000,
    )
    env.reset()

    # Step until terminated
    obs = None
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(Action.HOLD)
        if terminated or truncated:
            break

    assert obs is not None
    assert np.all(obs == 0.0), f"Terminal observation should be all zeros, got {obs}"


# ---------------------------------------------------------------------------
# Lower Priority: Multiple consecutive resets
# ---------------------------------------------------------------------------

def test_multiple_consecutive_resets(mock_data_source, mock_executor):
    """Calling reset() multiple times in a row must not raise errors."""
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=mock_data_source,
        executor=mock_executor,
        feature_computer=feature_computer,
    )

    for _ in range(5):
        obs, info = env.reset()
        assert obs.shape == (26,)
        assert np.all(np.isfinite(obs))


def test_reset_after_episode_end():
    """reset() after a completed episode must produce a valid observation."""
    from conftest import MockDataSource, MockOrderExecutor

    source = MockDataSource(episode_length=2)
    executor = MockOrderExecutor()
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=executor,
        feature_computer=feature_computer,
        max_episode_steps=1000,
    )
    env.reset()

    # Run episode to completion
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(Action.HOLD)
        if terminated or truncated:
            break

    # Now reset for a new episode
    obs, info = env.reset()
    assert obs.shape == (26,)
    assert np.all(np.isfinite(obs))
    assert env.step_count == 0


# ---------------------------------------------------------------------------
# Lower Priority: Observation is valid throughout full episode
# ---------------------------------------------------------------------------

def test_observation_valid_throughout_episode():
    """Every observation during an episode must be finite and in [-10, 10]."""
    from conftest import MockDataSource, MockOrderExecutor

    source = MockDataSource(episode_length=20)
    executor = MockOrderExecutor()
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=executor,
        feature_computer=feature_computer,
        max_episode_steps=50,
        normalize_rewards=False,
    )

    obs, _ = env.reset()
    assert np.all(np.isfinite(obs)), "Reset obs must be finite"

    for action in [Action.BUY, Action.HOLD, Action.SELL, Action.HOLD] * 10:
        obs, _, terminated, truncated, _ = env.step(action)
        if not (terminated or truncated):
            assert np.all(np.isfinite(obs)), f"Obs must be finite mid-episode: {obs}"
            assert np.all(np.abs(obs) <= 10.0), f"Obs out of [-10,10] range: {obs}"
        if terminated or truncated:
            break
