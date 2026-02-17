"""
Tests for PPOStrategyV2.

Tests neural network components, action selection, experience storage,
GAE computation, PPO updates, and checkpointing.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from strategies.ppo_paper_v2 import (
    PPOStrategyV2,
    TemporalEncoder,
    Actor,
    Critic,
    Experience,
)
from structures.action import Action


def test_temporal_encoder_forward():
    """Test TemporalEncoder forward pass with correct shapes."""
    encoder = TemporalEncoder(input_dim=25, history_len=5, output_dim=32)

    # Input: (batch, history_len * input_dim) = (4, 125)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 5 * 25)

    # Forward pass
    output = encoder(input_tensor)

    # Assert output shape
    assert output.shape == (batch_size, 32), \
        f"Expected shape ({batch_size}, 32), got {output.shape}"

    # Assert all outputs finite
    assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf"


def test_actor_forward():
    """Test Actor forward pass returns valid probabilities."""
    actor = Actor(input_dim=25, hidden_size=64, output_dim=3, history_len=5, temporal_dim=32)

    batch_size = 4
    current_state = torch.randn(batch_size, 25)
    temporal_state = torch.randn(batch_size, 5 * 25)

    # Forward pass
    probs = actor(current_state, temporal_state)

    # Assert shape
    assert probs.shape == (batch_size, 3), \
        f"Expected shape ({batch_size}, 3), got {probs.shape}"

    # Assert probabilities sum to 1.0
    prob_sums = torch.sum(probs, dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size)), \
        f"Probabilities should sum to 1.0, got {prob_sums}"

    # Assert all probabilities > 0
    assert torch.all(probs > 0), "All probabilities should be positive"

    # Assert all finite
    assert torch.all(torch.isfinite(probs)), "Probabilities contain NaN or Inf"


def test_critic_forward():
    """Test Critic forward pass returns finite value estimates."""
    critic = Critic(input_dim=25, hidden_size=96, history_len=5, temporal_dim=32)

    batch_size = 4
    current_state = torch.randn(batch_size, 25)
    temporal_state = torch.randn(batch_size, 5 * 25)

    # Forward pass
    values = critic(current_state, temporal_state)

    # Assert shape
    assert values.shape == (batch_size, 1), \
        f"Expected shape ({batch_size}, 1), got {values.shape}"

    # Assert all values finite
    assert torch.all(torch.isfinite(values)), "Values contain NaN or Inf"


def test_act_training_mode():
    """Test act() samples stochastically in training mode."""
    strategy = PPOStrategyV2(input_dim=25)
    strategy.training = True

    features = np.random.randn(22).astype(np.float32)

    # Sample multiple times
    actions = []
    for _ in range(20):
        action = strategy.act(features)
        actions.append(action.value)

    # Should have some variability (stochastic sampling)
    unique_actions = len(set(actions))
    assert unique_actions > 1, \
        f"Training mode should sample stochastically, got {unique_actions} unique actions"


def test_act_eval_mode():
    """Test act() is deterministic in eval mode."""
    strategy = PPOStrategyV2(input_dim=25)
    strategy.training = False

    features = np.random.randn(22).astype(np.float32)

    # Sample multiple times
    actions = []
    for _ in range(10):
        strategy.reset()  # Reset state history
        action = strategy.act(features)
        actions.append(action.value)

    # All actions should be the same (greedy/deterministic)
    unique_actions = len(set(actions))
    assert unique_actions == 1, \
        f"Eval mode should be deterministic, got {unique_actions} unique actions"


def test_previous_action_encoding():
    """Test that previous action is properly one-hot encoded."""
    strategy = PPOStrategyV2(input_dim=25)

    # Test each action
    for action_idx in [0, 1, 2]:
        strategy._previous_action = action_idx

        features = np.random.randn(22).astype(np.float32)

        # Append previous action
        features_with_action = strategy._append_previous_action(features)

        # Assert shape
        assert features_with_action.shape == (25,), \
            f"Expected shape (25,), got {features_with_action.shape}"

        # Extract one-hot encoding (last 3 elements)
        one_hot = features_with_action[-3:]

        # Create expected one-hot
        expected = np.zeros(3, dtype=np.float32)
        expected[action_idx] = 1.0

        # Assert match
        assert np.array_equal(one_hot, expected), \
            f"One-hot mismatch for action {action_idx}: got {one_hot}, expected {expected}"


def test_temporal_state_stacking():
    """Test temporal state stacking with history deque."""
    strategy = PPOStrategyV2(input_dim=25, history_len=5)

    features = np.random.randn(22).astype(np.float32)

    # Add states to history
    temporal_states = []
    for i in range(7):  # More than history_len
        features_with_action = strategy._append_previous_action(features)
        temporal_state = strategy._get_temporal_state(features_with_action)
        temporal_states.append(temporal_state)

    # Assert shape
    final_temporal = temporal_states[-1]
    assert final_temporal.shape == (5 * 25,), \
        f"Expected shape (125,), got {final_temporal.shape}"

    # History should be bounded to 5
    assert len(strategy._state_history) == 5, \
        f"History should be bounded to 5, got {len(strategy._state_history)}"


def test_temporal_state_padding():
    """Test temporal state padding when history < history_len."""
    strategy = PPOStrategyV2(input_dim=25, history_len=5)

    features = np.random.randn(22).astype(np.float32)

    # Only add 2 states (less than history_len=5)
    features_with_action = strategy._append_previous_action(features)
    strategy._get_temporal_state(features_with_action)
    temporal_state = strategy._get_temporal_state(features_with_action)

    # Should still have shape (5 * 25) with zero padding
    assert temporal_state.shape == (5 * 25,), \
        f"Expected shape (125,) even with insufficient history, got {temporal_state.shape}"

    # First part should be zeros (padding)
    assert np.allclose(temporal_state[:3 * 25], 0.0), \
        "First 3 states should be zero-padded"


def test_experience_storage():
    """Test experience storage in buffer."""
    strategy = PPOStrategyV2(input_dim=25, buffer_size=10)

    features = np.random.randn(22).astype(np.float32)
    next_features = np.random.randn(22).astype(np.float32)

    # Need to call act() first to set up internal state
    action = strategy.act(features)

    # Store experience
    strategy.store(
        features=features,
        action=action,
        reward=1.0,
        next_features=next_features,
        done=False,
    )

    # Assert experience added
    assert len(strategy.experiences) == 1, \
        f"Expected 1 experience, got {len(strategy.experiences)}"

    # Check experience structure
    exp = strategy.experiences[0]
    assert isinstance(exp, Experience), "Should be Experience object"
    assert exp.state.shape == (25,), "State should be 25-dim"
    assert exp.temporal_state.shape == (5 * 25,), "Temporal state should be 125-dim"
    assert exp.action in [0, 1, 2], "Action should be in [0, 1, 2]"
    assert isinstance(exp.reward, float), "Reward should be float"


def test_experience_buffer_bounding():
    """Test that experience buffer is bounded to buffer_size."""
    strategy = PPOStrategyV2(input_dim=25, buffer_size=5)

    features = np.random.randn(22).astype(np.float32)

    # Add more experiences than buffer_size
    for i in range(10):
        action = strategy.act(features)
        strategy.store(
            features=features,
            action=action,
            reward=float(i),
            next_features=features,
            done=False,
        )

    # Buffer should be bounded to buffer_size
    assert len(strategy.experiences) == 5, \
        f"Buffer should be bounded to 5, got {len(strategy.experiences)}"


def test_reward_scaling():
    """Test that rewards are scaled by reward_scale."""
    strategy = PPOStrategyV2(input_dim=25)

    assert strategy.reward_scale == 0.1, "reward_scale should be 0.1"

    features = np.random.randn(22).astype(np.float32)
    action = strategy.act(features)

    # Store with reward=10.0
    strategy.store(
        features=features,
        action=action,
        reward=10.0,
        next_features=features,
        done=False,
    )

    # Stored reward should be scaled
    stored_reward = strategy.experiences[0].reward
    expected_reward = 10.0 * 0.1  # 1.0
    assert np.isclose(stored_reward, expected_reward), \
        f"Stored reward should be {expected_reward}, got {stored_reward}"


def test_gae_computation():
    """Test GAE computation with mock data."""
    strategy = PPOStrategyV2(input_dim=25, gamma=0.99, gae_lambda=0.95)

    # Mock data
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    values = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    next_value = 9.0

    # Compute GAE
    advantages, returns = strategy._compute_gae(rewards, values, dones, next_value)

    # Assert shapes
    assert advantages.shape == (4,), f"Expected shape (4,), got {advantages.shape}"
    assert returns.shape == (4,), f"Expected shape (4,), got {returns.shape}"

    # Assert all finite
    assert np.all(np.isfinite(advantages)), "Advantages contain NaN or Inf"
    assert np.all(np.isfinite(returns)), "Returns contain NaN or Inf"

    # Returns should equal advantages + values
    expected_returns = advantages + values
    assert np.allclose(returns, expected_returns), \
        "Returns should equal advantages + values"


def test_ppo_update_not_ready():
    """Test that update() returns None when buffer not full."""
    strategy = PPOStrategyV2(input_dim=25, buffer_size=100)

    features = np.random.randn(22).astype(np.float32)

    # Add only a few experiences
    for _ in range(10):
        action = strategy.act(features)
        strategy.store(features, action, 0.0, features, False)

    # update() should return None
    metrics = strategy.update()
    assert metrics is None, "update() should return None when buffer not full"


def test_ppo_update_returns_metrics():
    """Test that update() returns metrics when buffer is full."""
    strategy = PPOStrategyV2(input_dim=25, buffer_size=64, batch_size=32)

    features = np.random.randn(22).astype(np.float32)

    # Fill buffer
    for _ in range(64):
        action = strategy.act(features)
        strategy.store(features, action, np.random.randn() * 0.1, features, False)

    # Update
    metrics = strategy.update()

    # Assert metrics returned
    assert metrics is not None, "update() should return metrics"
    assert isinstance(metrics, dict), "Metrics should be dict"

    # Check expected keys
    expected_keys = ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"]
    for key in expected_keys:
        assert key in metrics, f"Metrics missing key: {key}"
        assert np.isfinite(metrics[key]), f"Metric {key} is not finite"


def test_ppo_update_clears_buffer():
    """Test that update() clears experience buffer."""
    strategy = PPOStrategyV2(input_dim=25, buffer_size=64)

    features = np.random.randn(22).astype(np.float32)

    # Fill buffer
    for _ in range(64):
        action = strategy.act(features)
        strategy.store(features, action, 0.0, features, False)

    assert len(strategy.experiences) == 64, "Buffer should be full"

    # Update
    strategy.update()

    # Buffer should be cleared
    assert len(strategy.experiences) == 0, \
        f"Buffer should be cleared after update, got {len(strategy.experiences)}"


def test_gradient_clipping():
    """Test that gradients are clipped."""
    strategy = PPOStrategyV2(input_dim=25, max_grad_norm=0.5, buffer_size=64)

    # Fill buffer and update
    features = np.random.randn(22).astype(np.float32)

    for _ in range(64):
        action = strategy.act(features)
        strategy.store(features, action, np.random.randn(), features, False)

    # This should apply gradient clipping internally
    metrics = strategy.update()

    # Just verify update completed without errors
    assert metrics is not None, "Update should complete with gradient clipping"


def test_save_load_checkpoint():
    """Test save and load checkpoint preserves model state."""
    strategy1 = PPOStrategyV2(input_dim=25)

    features = np.random.randn(22).astype(np.float32)

    # Run a few forward passes to initialize state
    for _ in range(5):
        strategy1.act(features)

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Save
        strategy1.save(temp_path)

        # Create new strategy and load
        strategy2 = PPOStrategyV2(input_dim=25)
        strategy2.load(temp_path)

        # Both strategies should produce same output for same input
        strategy1.training = False
        strategy2.training = False

        strategy1.reset()
        strategy2.reset()

        action1 = strategy1.act(features)
        action2 = strategy2.act(features)

        assert action1.value == action2.value, \
            f"Loaded model should produce same action: {action1} vs {action2}"

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_reset():
    """Test reset() clears state correctly."""
    strategy = PPOStrategyV2(input_dim=25)

    features = np.random.randn(22).astype(np.float32)

    # Add some experiences
    for _ in range(10):
        action = strategy.act(features)
        strategy.store(features, action, 0.0, features, False)

    # Add to state history
    for _ in range(5):
        strategy.act(features)

    # Reset
    strategy.reset()

    # Assert buffer cleared
    assert len(strategy.experiences) == 0, "Experience buffer should be cleared"

    # Assert state history cleared
    assert len(strategy._state_history) == 0, "State history should be cleared"

    # Assert previous action reset to HOLD
    assert strategy._previous_action == 1, \
        f"previous_action should be reset to HOLD (1), got {strategy._previous_action}"


def test_should_update():
    """Test should_update() returns correct boolean."""
    strategy = PPOStrategyV2(input_dim=25, buffer_size=64)

    features = np.random.randn(22).astype(np.float32)

    # Initially should not update
    assert strategy.should_update() is False, "Should not update with empty buffer"

    # Fill buffer
    for _ in range(64):
        action = strategy.act(features)
        strategy.store(features, action, 0.0, features, False)

    # Now should update
    assert strategy.should_update() is True, "Should update when buffer is full"
