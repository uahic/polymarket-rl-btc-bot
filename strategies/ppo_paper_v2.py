import torch
import numpy as np
import torch.nn as nn
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from .base_strategy import BaseStrategy, MarketState, Action
from .ml_base_strategy import MLStrategy

import logging
logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple with temporal context."""

    state: np.ndarray  # Current state features (29,) = 26 base features + 3 prev action one-hot
    temporal_state: np.ndarray  # Stacked temporal features (history_len * 29,)
    action: int
    reward: float
    next_state: np.ndarray
    next_temporal_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class TemporalEncoder(nn.Module):
    """Encodes temporal sequence of states into momentum/trend features using a GRU.

    A GRU respects the sequential ordering of states (oldest → newest), which
    a flat MLP over stacked states cannot. This lets the network learn velocity
    and acceleration patterns directly from the sequence structure.

    Input:  (batch, history_len * input_dim)  — flattened for API compatibility
    Internally reshaped to (batch, history_len, input_dim) for the GRU.
    Output: (batch, output_dim)  — last hidden state of the GRU
    """

    def __init__(self, input_dim: int = 29, history_len: int = 5, output_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.history_len = history_len
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x is (batch, history_len * input_dim)."""
        batch_size = x.size(0)
        # Reshape: (batch, history_len * input_dim) → (batch, history_len, input_dim)
        x = x.view(batch_size, self.history_len, self.input_dim)
        # GRU: output shape (batch, history_len, output_dim), h_n shape (1, batch, output_dim)
        _, h_n = self.gru(x)
        # Return last hidden state: (batch, output_dim)
        return h_n.squeeze(0)


class Actor(nn.Module):
    """Policy network with GRU-based temporal awareness.

    Architecture:
        Current state (29) + GRU temporal features (32) = 61
        → 64 → LayerNorm → tanh → 64 → LayerNorm → tanh → 3 (softmax)

    Temporal encoder is a GRU that processes the ordered state history,
    capturing velocity and acceleration patterns directly from sequence structure.
    Smaller network (64) to prevent overfitting on enhanced features.
    """

    def __init__(
        self,
        input_dim: int = 29,
        hidden_size: int = 64,
        output_dim: int = 3,
        history_len: int = 5,
        temporal_dim: int = 32,
    ):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, history_len, temporal_dim)

        # Combined input: current state + temporal features
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(
        self, current_state: torch.Tensor, temporal_state: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass. Returns action probabilities.

        Args:
            current_state: (batch, 29) current features with previous action
            temporal_state: (batch, history_len * 29) stacked history
        """
        # Encode temporal context via GRU (last hidden state)
        temporal_features = self.temporal_encoder(temporal_state)

        # Combine current + temporal
        combined = torch.cat([current_state, temporal_features], dim=-1)

        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        logits = self.fc3(h)
        probs = torch.softmax(logits, dim=-1)
        return probs


class Critic(nn.Module):
    """Value network with GRU-based temporal awareness - ASYMMETRIC (larger than actor).

    Architecture:
        Current state (29) + GRU temporal features (32) = 61
        → 96 → LayerNorm → tanh → 96 → LayerNorm → tanh → 1

    Larger network (96 vs 64) because:
    - Value estimation is harder than policy
    - Critic doesn't overfit as easily (regresses to scalar)
    - Better value estimates improve advantage computation
    """

    def __init__(
        self,
        input_dim: int = 29,
        hidden_size: int = 96,
        history_len: int = 5,
        temporal_dim: int = 32,
    ):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, history_len, temporal_dim)

        # Combined input: current state + temporal features
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(
        self, current_state: torch.Tensor, temporal_state: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass. Returns value estimate.

        Args:
            current_state: (batch, 29) current features with previous action
            temporal_state: (batch, history_len * 29) stacked history
        """
        # Encode temporal context via GRU (last hidden state)
        temporal_features = self.temporal_encoder(temporal_state)

        # Combine current + temporal
        combined = torch.cat([current_state, temporal_features], dim=-1)

        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        value = self.fc3(h)
        return value


class PPOStrategyV2(MLStrategy):
    """PPO-based strategy with GRU temporal-aware actor-critic architecture using PyTorch.

    Key features:
    - GRU temporal encoder: processes ordered state history, capturing velocity/acceleration
    - Time-of-day encoding: 4 cyclical features (hour_sin/cos, dow_sin/cos) added to base features
    - Asymmetric architecture: larger critic (96) for better value estimation
    - Low gamma (0.9): focuses on near-term rewards for 15-min horizon
    - Larger buffer (2048): diverse experiences for stable gradient updates

    Feature dimensions:
      Base features:        26  (22 market + 4 time-of-day)
      + previous action:     3  (one-hot)
      = input_dim:          29
      Temporal (GRU out):   32
      Combined:             61
    """

    def __init__(
        self,
        input_dim: int = 29,  # 26 base features (22 market + 4 time-of-day) + 3 prev action one-hot
        hidden_size: int = 64,  # Actor hidden size
        critic_hidden_size: int = 96,  # Larger critic for better value estimation
        history_len: int = 5,  # Number of past states for temporal processing
        temporal_dim: int = 32,  # GRU hidden size / temporal encoder output size
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.9,  # Low gamma for 15-min horizon - focus on near-term rewards
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.03,  # Lower entropy to allow sparse policy (mostly HOLD)
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 2048,  # Larger buffer for diverse experiences and stable gradients
        batch_size: int = 64,
        n_epochs: int = 10,
        target_kl: float = 0.02,
    ):
        super().__init__("rl")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.history_len = history_len
        self.temporal_dim = temporal_dim
        self.output_dim = 3  # BUY, HOLD, SELL (simplified)

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl

        # Networks with temporal processing
        self.actor = Actor(
            input_dim, hidden_size, self.output_dim, history_len, temporal_dim
        )
        self.critic = Critic(input_dim, critic_hidden_size, history_len, temporal_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Experience buffer
        self.experiences: List[Experience] = []

        # Temporal state history (single buffer for this strategy instance)
        self._state_history: deque = deque(maxlen=self.history_len)

        # Fixed reward scaling (not adaptive normalization)
        # Scale PnL to roughly [-1, 1] range for stability
        self.reward_scale = 0.1  # Divide PnL by 10 (e.g., $10 -> 1.0)

        # For storing last action's log prob, value, and state
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_temporal_state: Optional[np.ndarray] = None
        self._last_features_with_action: Optional[np.ndarray] = None

        # Track previous action (initialize to HOLD=1)
        self._previous_action: int = 1

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)


    def _append_previous_action(self, features: np.ndarray) -> np.ndarray:
        """Append one-hot encoded previous action to features.

        Args:
            features: Base features (26-dim: 22 market + 4 time-of-day)

        Returns:
            Extended features (29-dim) = features + one-hot previous action
        """
        # Create one-hot encoding of previous action
        prev_action_onehot = np.zeros(3, dtype=np.float32)
        prev_action_onehot[self._previous_action] = 1.0

        # Concatenate: [26 features] + [3 action dims] = 29
        return np.concatenate([features, prev_action_onehot])

    def _get_temporal_state(self, current_features_with_action: np.ndarray) -> np.ndarray:
        """Get stacked temporal state.

        Maintains a history of the last N states (with previous action appended).
        Returns flattened array of shape (history_len * input_dim,) which the
        GRU TemporalEncoder reshapes internally to (batch, history_len, input_dim).

        Args:
            current_features_with_action: 29-dim features (26 base + 3 prev action)
        """
        # Add current state to history
        self._state_history.append(current_features_with_action.copy())

        # Pad with zeros if not enough history
        if len(self._state_history) < self.history_len:
            padding = [np.zeros(self.input_dim, dtype=np.float32)] * (
                self.history_len - len(self._state_history)
            )
            stacked = np.concatenate(padding + list(self._state_history))
        else:
            stacked = np.concatenate(list(self._state_history))

        return stacked.astype(np.float32)

    def act(self, features: np.ndarray) -> Action:
        """Select action using current policy with temporal context.

        Args:
            features: 26-dimensional base feature vector (22 market + 4 time-of-day)

        Returns:
            Action index (0=BUY_UP, 1=HOLD, 2=SELL_DOWN)
        """
        # Append previous action to features (26 -> 29)
        features_with_action = self._append_previous_action(features)

        # Get temporal state (stacked history of 29-dim states)
        temporal_state = self._get_temporal_state(features_with_action)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features_with_action.reshape(1, -1), dtype=torch.float32, device=self.device)
        temporal_tensor = torch.tensor(temporal_state.reshape(1, -1), dtype=torch.float32, device=self.device)

        # Get action probabilities and value with temporal context
        with torch.no_grad():
            probs = self.actor(features_tensor, temporal_tensor)
            value = self.critic(features_tensor, temporal_tensor)

        probs_np = probs[0].cpu().numpy()
        value_np = float(value[0, 0].cpu().numpy())

        if self.training:
            # Sample from distribution
            action_idx = np.random.choice(self.output_dim, p=probs_np)
        else:
            # Greedy
            action_idx = int(np.argmax(probs_np))

        # Store for experience collection
        self._last_log_prob = float(np.log(probs_np[action_idx] + 1e-8))
        self._last_value = value_np
        self._last_temporal_state = temporal_state
        self._last_features_with_action = features_with_action

        # Update previous action for next step
        self._previous_action = action_idx

        return Action(action_idx)

    def store(
        self,
        features: np.ndarray,
        action: Action,
        reward: float,
        next_features: np.ndarray,
        done: bool,
    ):
        """Store experience for training with temporal context.

        Args:
            features: Current state features (26-dim: 22 market + 4 time-of-day)
            action: Action taken (Action enum)
            reward: Reward received
            next_features: Next state features (26-dim: 22 market + 4 time-of-day)
            done: Whether episode ended
        """
        # Apply fixed scaling to reward (not adaptive normalization)
        # This keeps rewards in a stable range without moving targets
        scaled_reward = reward * self.reward_scale

        # Use the state that act() actually saw — cached before _previous_action was updated.
        # Recomputing here would use the post-update _previous_action and produce the wrong state.
        features_with_action = (
            self._last_features_with_action
            if self._last_features_with_action is not None
            else self._append_previous_action(features)
        )

        # For next_features, append the action just taken (becomes the "previous action" at t+1)
        action_idx = action.value
        next_action_onehot = np.zeros(3, dtype=np.float32)
        next_action_onehot[action_idx] = 1.0
        next_features_with_action = np.concatenate([next_features, next_action_onehot])

        # Compute next temporal state by peeking at history without mutating it.
        # _get_temporal_state() appends to _state_history; calling it here would corrupt the
        # sequence that act() relies on (it would interleave act-states with store-states).
        history_list = list(self._state_history)  # current history after act() appended to it
        history_list.append(next_features_with_action)  # peek: what history looks like at t+1
        if len(history_list) < self.history_len:
            padding = [np.zeros(self.input_dim, dtype=np.float32)] * (
                self.history_len - len(history_list)
            )
            next_temporal_state = np.concatenate(padding + history_list).astype(np.float32)
        else:
            next_temporal_state = np.concatenate(history_list[-self.history_len :]).astype(np.float32)

        exp = Experience(
            state=features_with_action,
            temporal_state=(
                self._last_temporal_state
                if self._last_temporal_state is not None
                else np.zeros(self.history_len * self.input_dim, dtype=np.float32)
            ),
            action=action_idx,
            reward=scaled_reward,
            next_state=next_features_with_action,
            next_temporal_state=next_temporal_state,
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )
        self.experiences.append(exp)

        # Limit buffer size
        if len(self.experiences) > self.buffer_size:
            self.experiences = self.experiences[-self.buffer_size :]

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> tuple:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _clip_grad_norm(self, model: nn.Module, max_norm: float):
        """Clip gradients by global norm using PyTorch."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def should_update(self) -> bool:
        """Check if buffer is full and ready for update."""
        return len(self.experiences) >= self.buffer_size

    def update(self) -> Optional[Dict[str, float]]:
        """Update policy using PPO with PyTorch autograd and temporal context."""
        if len(self.experiences) < self.buffer_size:
            return None

        # Print training indicator
        # print(f"\n{'='*60}")
        # print(f"  PPO TRAINING UPDATE")
        # print(f"  Buffer: {len(self.experiences)}/{self.buffer_size} experiences")
        # print(f"  Epochs: {self.n_epochs} | Batch Size: {self.batch_size}")
        # print(f"{'='*60}\n")

        # Convert experiences to arrays (including temporal states)
        states = np.array([e.state for e in self.experiences], dtype=np.float32)
        temporal_states = np.array(
            [e.temporal_state for e in self.experiences], dtype=np.float32
        )
        actions = np.array([e.action for e in self.experiences], dtype=np.int32)
        rewards = np.array([e.reward for e in self.experiences], dtype=np.float32)
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)
        old_log_probs = np.array(
            [e.log_prob for e in self.experiences], dtype=np.float32
        )
        old_values = np.array([e.value for e in self.experiences], dtype=np.float32)

        # Compute next value for GAE (with temporal context)
        next_state_tensor = torch.tensor(
            self.experiences[-1].next_state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        next_temporal_tensor = torch.tensor(
            self.experiences[-1].next_temporal_state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            next_value = float(self.critic(next_state_tensor, next_temporal_tensor)[0, 0].cpu().numpy())

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, old_values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to PyTorch tensors (including temporal states)
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        temporal_states_tensor = torch.tensor(temporal_states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages.astype(np.float32), dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns.astype(np.float32), dtype=torch.float32, device=self.device)

        n_samples = len(self.experiences)
        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)

            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                # Get batch using PyTorch indexing
                batch_states = states_tensor[batch_idx]
                batch_temporal = temporal_states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]

                # Actor update
                self.actor_optimizer.zero_grad()
                probs = self.actor(batch_states, batch_temporal)

                # Get log probs for taken actions
                batch_size_local = batch_actions.shape[0]
                action_indices = torch.arange(batch_size_local, device=self.device)
                selected_probs = probs[action_indices, batch_actions]
                log_probs = torch.log(selected_probs + 1e-8)

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Entropy bonus (encourages exploration)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                entropy_mean = torch.mean(entropy)
                actor_loss = policy_loss - self.entropy_coef * entropy_mean

                # Metrics
                approx_kl = torch.mean(batch_old_log_probs - log_probs)
                clip_frac = torch.mean(
                    (
                        (ratio < 1 - self.clip_epsilon)
                        | (ratio > 1 + self.clip_epsilon)
                    ).float()
                )

                # Backward pass and update actor
                actor_loss.backward()
                self._clip_grad_norm(self.actor, self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                self.critic_optimizer.zero_grad()
                values = self.critic(batch_states, batch_temporal).squeeze(-1)

                # Simple MSE loss (no clipping - allows critic to adapt quickly)
                critic_loss = self.value_coef * torch.mean((batch_returns - values) ** 2)

                # Backward pass and update critic
                critic_loss.backward()
                self._clip_grad_norm(self.critic, self.max_grad_norm)
                self.critic_optimizer.step()

                # Record metrics
                all_metrics["policy_loss"].append(float(actor_loss.detach().cpu().numpy()))
                all_metrics["value_loss"].append(float(critic_loss.detach().cpu().numpy()))
                all_metrics["entropy"].append(float(entropy_mean.detach().cpu().numpy()))
                all_metrics["approx_kl"].append(float(approx_kl.detach().cpu().numpy()))
                all_metrics["clip_fraction"].append(float(clip_frac.detach().cpu().numpy()))

                epoch_kl += float(approx_kl.detach().cpu().numpy())
                n_batches += 1

            # Early stopping on KL divergence
            avg_kl = epoch_kl / max(1, n_batches)
            if avg_kl > self.target_kl:
                logger.info(f"[RL] Early stop epoch {epoch}, KL={avg_kl:.4f}")
                break

        # Clear buffer after update
        self.experiences.clear()

        # Compute explained variance
        y_pred = old_values
        y_true = returns
        var_y = np.var(y_true)
        explained_var = (
            1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0
        )


        # Prepare return metrics
        metrics = {
            "policy_loss": np.mean(all_metrics["policy_loss"]),
            "value_loss": np.mean(all_metrics["value_loss"]),
            "entropy": np.mean(all_metrics["entropy"]),
            "approx_kl": np.mean(all_metrics["approx_kl"]),
            "clip_fraction": np.mean(all_metrics["clip_fraction"]),
            "explained_variance": explained_var,
        }

        # Print training summary
        # print(f"{'='*60}")
        # print(f"  TRAINING COMPLETE")
        # print(f"  Policy Loss:     {metrics['policy_loss']:>8.4f}")
        # print(f"  Value Loss:      {metrics['value_loss']:>8.4f}")
        # print(f"  Entropy:         {metrics['entropy']:>8.4f}")
        # print(f"  KL Divergence:   {metrics['approx_kl']:>8.4f}")
        # print(f"  Clip Fraction:   {metrics['clip_fraction']:>8.4f}")
        # print(f"  Explained Var:   {metrics['explained_variance']:>8.4f}")
        # print(f"{'='*60}\n")

        return metrics

    def reset(self):
        """Clear experience buffer and state history for new episode."""
        self.experiences.clear()
        self._state_history.clear()
        self._last_temporal_state = None
        self._last_features_with_action = None
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._previous_action = 1  # Reset to HOLD

    def save(self, path: str):
        """Save model and training state."""
        weights_path = str(Path(path).with_suffix(".pth"))

        # Save complete checkpoint with PyTorch
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'reward_scale': self.reward_scale,
            # Architecture params for reconstruction
            'input_dim': self.input_dim,
            'hidden_size': self.hidden_size,
            'critic_hidden_size': self.critic_hidden_size,
            'history_len': self.history_len,
            'temporal_dim': self.temporal_dim,
            'gamma': self.gamma,
            'buffer_size': self.buffer_size,
        }
        torch.save(checkpoint, weights_path)

    def load(self, path: str):
        """Load model and training state."""
        weights_path = str(Path(path).with_suffix(".pth"))

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Load model state
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Load optimizer state
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Load reward scaling
        self.reward_scale = float(checkpoint.get('reward_scale', 0.1))

        # Move to device and restore train mode (load_state_dict preserves the saved
        # training flag, which may differ from our intended mode)
        self.actor.to(self.device)
        self.critic.to(self.device)
        if self.training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()
