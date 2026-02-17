"""
Integration tests for complete system.

Tests end-to-end workflows combining multiple components.
"""

import pytest
import numpy as np
import tempfile
import os
from data.sources import HistoricalSource
from environments.trading_gym import TradingGym
from features.computer import FeatureComputer
from strategies.ppo_paper_v2 import PPOStrategyV2
from structures.action import Action
from conftest import MockOrderExecutor


def test_full_historical_episode():
    """Test complete episode with HistoricalSource → TradingGym → PPOStrategyV2."""
    # Create components
    source = HistoricalSource(
        data_dir="data/historical",
        assets=["BTC"],
        episode_length=100,  # Short episode
    )

    executor = MockOrderExecutor(initial_balance=1000.0)
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        max_episode_steps=100,
    )

    strategy = PPOStrategyV2(input_dim=25)
    strategy.training = True

    # Run episode
    obs, info = env.reset()
    done = False
    truncated = False
    step_count = 0

    while not done and not truncated and step_count < 100:
        # Get action from strategy
        action = strategy.act(obs)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)

        # Store experience
        strategy.store(obs, action, reward, next_obs, done or truncated)

        obs = next_obs
        step_count += 1

    # Assert episode completed
    assert step_count > 0, "Episode should have at least 1 step"
    assert done or truncated, "Episode should terminate"

    # Assert no errors
    assert info["balance"] > 0, "Balance should be positive"


def test_training_loop():
    """Test multiple episodes with PPO updates."""
    # Create components
    source = HistoricalSource(
        data_dir="data/historical",
        assets=["BTC"],
        episode_length=50,
    )

    executor = MockOrderExecutor(initial_balance=1000.0)
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        max_episode_steps=50,
    )

    strategy = PPOStrategyV2(input_dim=25, buffer_size=64)
    strategy.training = True

    # Run multiple episodes
    num_episodes = 3
    episode_returns = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        strategy.reset()  # Reset strategy state

        done = False
        truncated = False
        episode_reward = 0.0

        while not done and not truncated:
            action = strategy.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            strategy.store(obs, action, reward, next_obs, done or truncated)

            episode_reward += reward
            obs = next_obs

            # Update if buffer full
            if strategy.should_update():
                metrics = strategy.update()
                assert metrics is not None, "Update should return metrics"

        episode_returns.append(episode_reward)

    # Assert all episodes completed
    assert len(episode_returns) == num_episodes, \
        f"Should complete {num_episodes} episodes"

    # All returns should be finite
    assert all(np.isfinite(ret) for ret in episode_returns), \
        "All episode returns should be finite"


def test_checkpoint_resume():
    """Test training checkpoint save/load and resume."""
    # Create components
    source = HistoricalSource(
        data_dir="data/historical",
        assets=["BTC"],
        episode_length=30,
    )

    executor = MockOrderExecutor(initial_balance=1000.0)
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        max_episode_steps=30,
    )

    strategy1 = PPOStrategyV2(input_dim=25, buffer_size=32)
    strategy1.training = True

    # Train for one episode
    obs, info = env.reset()
    strategy1.reset()

    for _ in range(32):  # Fill buffer
        action = strategy1.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        strategy1.store(obs, action, reward, next_obs, done or truncated)
        obs = next_obs
        if done or truncated:
            break

    # Update
    if strategy1.should_update():
        metrics1 = strategy1.update()

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        checkpoint_path = tmp.name

    try:
        strategy1.save(checkpoint_path)

        # Create new strategy and load
        strategy2 = PPOStrategyV2(input_dim=25)
        strategy2.load(checkpoint_path)

        # Both should produce same action for same state
        strategy1.training = False
        strategy2.training = False

        strategy1.reset()
        strategy2.reset()

        test_obs = np.random.randn(22).astype(np.float32)

        action1 = strategy1.act(test_obs)
        action2 = strategy2.act(test_obs)

        assert action1.value == action2.value, \
            "Loaded strategy should produce same action"

    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def test_gym_compatibility():
    """Test TradingGym works as standard gym environment."""
    from conftest import MockDataSource

    source = MockDataSource(episode_length=50)
    executor = MockOrderExecutor()
    feature_computer = FeatureComputer()

    env = TradingGym(
        data_source=source,
        executor=executor,
        feature_computer=feature_computer,
    )

    # Test reset
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), "obs should be numpy array"
    assert isinstance(info, dict), "info should be dict"

    # Test step
    obs, reward, terminated, truncated, info = env.step(Action.HOLD)
    assert isinstance(obs, np.ndarray), "obs should be numpy array"
    assert isinstance(reward, (int, float)), "reward should be numeric"
    assert isinstance(terminated, bool), "terminated should be bool"
    assert isinstance(truncated, bool), "truncated should be bool"
    assert isinstance(info, dict), "info should be dict"

    # Test action space
    assert env.action_space.n == 3, "Action space should be Discrete(3)"

    # Test observation space
    assert env.observation_space.shape == (26,), "Observation space should be (26,)"


def test_feature_consistency():
    """Test feature computation is consistent between training and inference."""
    from features.computer import RawMarketData, OrderbookSnapshot, FuturesData, SpotData
    from features.computer import PositionState, TransactionState, CapitalState

    computer = FeatureComputer()

    # Create sample data
    raw_data = RawMarketData(
        timestamp=0.0,
        asset="BTC",
        orderbook=OrderbookSnapshot(
            timestamp=0.0,
            best_bid=0.48,
            best_ask=0.52,
            spread=0.04,
            bids_l5=[(0.48, 100), (0.47, 200), (0.46, 150)],
            asks_l5=[(0.52, 100), (0.53, 200), (0.54, 150)],
        ),
        futures=FuturesData(
            timestamp=0.0,
            price=50000.0,
            returns_1m=0.001,
            returns_5m=0.005,
            returns_10m=0.01,
            cvd=1000.0,
            cvd_history=[900, 950, 1000],
            trade_flow_imbalance=0.1,
            trade_intensity=2.0,
            large_trade_flag=0.0,
            realized_vol_5m=0.02,
            avg_vol=0.02,
        ),
        spot=SpotData(timestamp=0.0, price=50000.0),
        prob_up=0.5,
        time_remaining=0.75,
        prob_history=[0.48, 0.49, 0.50],
    )

    position = PositionState()
    transaction = TransactionState()
    capital = CapitalState(available_balance=1000.0)

    # Compute features multiple times
    features1 = computer.compute_features(raw_data, position, transaction, capital)
    features2 = computer.compute_features(raw_data, position, transaction, capital)
    features3 = computer.compute_features(raw_data, position, transaction, capital)

    # All should be identical (deterministic)
    assert np.array_equal(features1, features2), \
        "Features should be identical for same input (1 vs 2)"
    assert np.array_equal(features2, features3), \
        "Features should be identical for same input (2 vs 3)"

    # All should be finite
    assert np.all(np.isfinite(features1)), "Features should be finite"
    assert np.all(np.isfinite(features2)), "Features should be finite"
    assert np.all(np.isfinite(features3)), "Features should be finite"
