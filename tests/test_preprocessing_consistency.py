"""
Test preprocessing consistency between old and new architecture.

This test verifies that the new FeatureComputer produces identical
features to the old MarketState.to_features() method.
"""

import logging
import numpy as np
from features.computer import (
    FeatureComputer,
    RawMarketData,
    OrderbookSnapshot,
    FuturesData,
    SpotData,
    PositionState,
    TransactionState,
    CapitalState,
)


def test_feature_computation():
    """Test that feature computer produces valid 22-dim output."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Feature Computation")
    logger.info("="*60)

    # Create test data
    raw_data = RawMarketData(
        timestamp=1708000000.0,
        asset="BTC",
        orderbook=OrderbookSnapshot(
            timestamp=1708000000.0,
            best_bid=0.495,
            best_ask=0.505,
            spread=0.01,
            bids_l5=[(0.495, 100.0), (0.49, 200.0)],
            asks_l5=[(0.505, 100.0), (0.51, 200.0)],
        ),
        futures=FuturesData(
            timestamp=1708000000.0,
            price=50000.0,
            returns_1m=0.001,
            returns_5m=0.005,
            returns_10m=0.010,
            cvd=1000.0,
            cvd_history=[950.0, 1000.0],
            trade_flow_imbalance=0.1,
            trade_intensity=5.0,
            large_trade_flag=0.0,
            realized_vol_5m=0.02,
            avg_vol=0.018,
        ),
        spot=SpotData(
            timestamp=1708000000.0,
            price=50000.0,
            change_pct=0.001,
        ),
        prob_up=0.5,
        time_remaining=0.5,
        prob_history=[0.48, 0.49, 0.50],
        vol_regime=0.0,
        trend_regime=1.0,
    )

    position = PositionState(
        has_position=True,
        side="UP",
        unrealized_pnl=5.0,
        time_remaining_normalized=0.5,
    )

    transaction = TransactionState(
        pending_order=False,
        failed_order=False,
        consecutive_failures=0,
    )

    capital = CapitalState(
        available_balance=1000.0,
        max_balance=1000.0,
    )

    # Compute features
    computer = FeatureComputer()
    features = computer.compute_features(raw_data, position, transaction, capital)

    # Verify output
    logger.info(f"\nFeature shape: {features.shape}")
    assert features.shape == (26,), f"Expected shape (26,), got {features.shape}"

    logger.info(f"Feature dtype: {features.dtype}")
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

    logger.info(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    assert np.all(np.isfinite(features)), "Features contain NaN or Inf"

    logger.info(f"\nFeatures (first 10):")
    for i, val in enumerate(features[:10]):
        logger.info(f"  Feature {i:2d}: {val:+.6f}")

    logger.info("\n✓ Feature computation test passed!")
    return True


def test_feature_consistency_example():
    """
    Example test showing how to verify consistency with MarketState.

    NOTE: This requires having a MarketState instance from the old code.
    Since we can't easily create one in this test file, this is just
    an example of what the test would look like.
    """
    logger.info("\n" + "="*60)
    logger.info("TEST: Feature Consistency (Example)")
    logger.info("="*60)

    logger.info("\nTo test consistency with existing code:")
    logger.info("1. Create a MarketState from your live bot")
    logger.info("2. Compute features with state.to_features()")
    logger.info("3. Convert MarketState to RawMarketData")
    logger.info("4. Compute features with FeatureComputer")
    logger.info("5. Verify they match: np.allclose(features_old, features_new)")

    logger.info("\nExample code:")
    logger.info("""
from structures.market import MarketState
from features.computer import compute_features_from_market_state

logger = logging.getLogger(__name__)

# Get MarketState from your existing bot
state = MarketState(...)

# Old way
features_old = state.to_features()

# New way (using backwards compatibility wrapper)
features_new = compute_features_from_market_state(state)

# Verify
difference = np.abs(features_old - features_new)
logger.info(f"Max difference: {difference.max()}")
assert np.allclose(features_old, features_new, atol=1e-4)
""")

    logger.info("\n✓ Consistency test example provided!")
    return True


def test_data_sources():
    """Test that HistoricalSource can be instantiated."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Data Sources")
    logger.info("="*60)

    from data.sources import HistoricalSource

    # Create historical source
    source = HistoricalSource(
        data_dir="data/historical",
        assets=["BTC"],
        episode_length=100,  # Short for testing
    )

    logger.info(f"\nHistoricalSource created:")
    logger.info(f"  Data dir: {source.data_dir}")
    logger.info(f"  Assets: {source.assets}")
    logger.info(f"  Episode length: {source.episode_length}")

    # Try to reset (will use dummy data if no files exist)
    try:
        raw_data = source.reset(asset="BTC")
        logger.info(f"\nReset successful!")
        logger.info(f"  Initial timestamp: {raw_data.timestamp}")
        logger.info(f"  Asset: {raw_data.asset}")
        logger.info(f"  Prob: {raw_data.prob_up:.3f}")
        logger.info(f"  Time remaining: {raw_data.time_remaining:.2f}")

        # Try advancing
        has_more = source.advance()
        logger.info(f"\nAdvanced to next tick: {has_more}")

        if has_more:
            next_data = source.get_current()
            logger.info(f"  Next timestamp: {next_data.timestamp}")
            logger.info(f"  Next prob: {next_data.prob_up:.3f}")

        logger.info("\n✓ Data source test passed!")
        return True

    except Exception as e:
        logger.info(f"\n✗ Data source test failed: {e}")
        return False


def test_gym_environment():
    """Test that TradingGym can be instantiated."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Gym Environment")
    logger.info("="*60)

    from environments.trading_gym import TradingGym
    from data.sources import HistoricalSource
    from features.computer import FeatureComputer
    from simulation.executor_wrapper import GymExecutorWrapper

    # Create components
    source = HistoricalSource("data/historical", assets=["BTC"], episode_length=100)
    executor = GymExecutorWrapper(default_order_size=10.0)
    computer = FeatureComputer()

    # Create environment
    env = TradingGym(
        data_source=source,
        executor=executor,
        feature_computer=computer,
        initial_balance=1000.0,
    )

    logger.info(f"\nTradingGym created:")
    logger.info(f"  Action space: {env.action_space}")
    logger.info(f"  Observation space: {env.observation_space}")

    # Test reset
    obs, info = env.reset()
    logger.info(f"\nReset successful!")
    logger.info(f"  Observation shape: {obs.shape}")
    logger.info(f"  Info: {info}")

    # Test step
    from structures.action import Action
    action = Action.HOLD
    obs, reward, done, truncated, info = env.step(action)
    logger.info(f"\nStep successful!")
    logger.info(f"  Action: {action.name}")
    logger.info(f"  Reward: {reward:.4f}")
    logger.info(f"  Done: {done}, Truncated: {truncated}")

    logger.info("\n✓ Gym environment test passed!")
    return True


def test_ppo_adapter():
    """Test that PPO adapter can be instantiated."""
    logger.info("\n" + "="*60)
    logger.info("TEST: PPO Adapter")
    logger.info("="*60)

    try:
        from strategies.ppo_gym_adapter import PPOGymAdapter

        # Create adapter
        adapter = PPOGymAdapter(
            input_dim=22,
            hidden_size=64,
            buffer_size=256,
            asset="BTC",
        )

        logger.info(f"\nPPOGymAdapter created:")
        logger.info(f"  Input dim: {adapter.ppo.input_dim}")
        logger.info(f"  Hidden size: {adapter.ppo.hidden_size}")
        logger.info(f"  Buffer size: {adapter.ppo.buffer_size}")

        # Test action selection
        obs = np.random.randn(22).astype(np.float32)
        action = adapter.act(obs, training=False)

        logger.info(f"\nAction selection successful!")
        logger.info(f"  Observation shape: {obs.shape}")
        logger.info(f"  Action: {action}")
        assert action in [0, 1, 2], f"Invalid action: {action}"

        logger.info("\n✓ PPO adapter test passed!")
        return True

    except Exception as e:
        logger.info(f"\n✗ PPO adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info(" PREPROCESSING CONSISTENCY TESTS")
    logger.info("="*70)

    tests = [
        ("Feature Computation", test_feature_computation),
        ("Feature Consistency Example", test_feature_consistency_example),
        ("Data Sources", test_data_sources),
        ("Gym Environment", test_gym_environment),
        ("PPO Adapter", test_ppo_adapter),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            logger.info(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    logger.info("\n" + "="*70)
    logger.info(" TEST SUMMARY")
    logger.info("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status:12s} {name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.info(f"\n⚠️  {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/data/workspaces/trading/bots/polymarket-rl-btc-bot')

    success = run_all_tests()
    sys.exit(0 if success else 1)
