import torch
import numpy as np
import torch.nn as nn
from collections import deque
from typing import List, Dict, Optional
from dataclasses import dataclass
from .base_strategy import BaseStrategy, MarketState, Action
from .ml_base_strategy import MLStrategy


@dataclass
class Experience:
    """Single experience tuple with temporal context."""

    state: np.ndarray  # Current state features (21,)
    temporal_state: np.ndarray  # Stacked temporal features (history_len * 21,)
    action: int
    reward: float
    next_state: np.ndarray
    next_temporal_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class TemporalEncoder(nn.Module):
    """Encodes temporal sequence of states into momentum/trend features.

    Takes last N states and compresses them into a fixed-size representation
    that captures velocity, acceleration, and trend direction.

    Architecture: (history_len * 21) → 64 → LayerNorm → tanh → 32
    Output is concatenated with current state features.
    """

    def __init__(self, input_dim: int = 21, history_len: int = 5, output_dim: int = 32):
        super().__init__()
        self.history_len = history_len
        self.temporal_input = input_dim * history_len
        self.fc1 = nn.Linear(self.temporal_input, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x is (batch, history_len * input_dim)."""
        h = torch.tanh(self.ln1(self.fc1(x)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        return h


class Actor(nn.Module):
    """Policy network with temporal awareness.

    Architecture:
        Current state (21) + Temporal features (32) = 50
        → 64 → LayerNorm → tanh → 64 → LayerNorm → tanh → 3 (softmax)

    Temporal encoder captures momentum/trends from state history.
    Smaller network (64) to prevent overfitting on enhanced features.
    """

    def __init__(
        self,
        input_dim: int = 21,
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
            current_state: (batch, 21) current features
            temporal_state: (batch, history_len * 21) stacked history
        """
        # Encode temporal context
        temporal_features = self.temporal_encoder(temporal_state)

        # Combine current + temporal
        combined = torch.cat([current_state, temporal_features], dim=-1)

        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        logits = self.fc3(h)
        probs = torch.softmax(logits, dim=-1)
        return probs


class Critic(nn.Module):
    """Value network with temporal awareness - ASYMMETRIC (larger than actor).

    Architecture:
        Current state (21) + Temporal features (32) = 50
        → 96 → LayerNorm → tanh → 96 → LayerNorm → tanh → 1

    Larger network (96 vs 64) because:
    - Value estimation is harder than policy
    - Critic doesn't overfit as easily (regresses to scalar)
    - Better value estimates improve advantage computation
    """

    def __init__(
        self,
        input_dim: int = 21,
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
            current_state: (batch, 21) current features
            temporal_state: (batch, history_len * 21) stacked history
        """
        # Encode temporal context
        temporal_features = self.temporal_encoder(temporal_state)

        # Combine current + temporal
        combined = torch.cat([current_state, temporal_features], dim=-1)

        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        value = self.fc3(h)
        return value


class PPOStrategy(MLStrategy):
    """PPO-based strategy with temporal-aware actor-critic architecture using PyTorch.

    Key features:
    - Temporal processing: maintains history of last N states to capture momentum
    - Asymmetric architecture: larger critic (96) for better value estimation
    - Lower gamma (0.95): appropriate for 15-min trading horizon
    - Smaller buffer (256): faster adaptation to regime changes
    """

    def __init__(
        self,
        input_dim: int = 21,
        hidden_size: int = 64,  # Actor hidden size
        critic_hidden_size: int = 96,  # Larger critic for better value estimation
        history_len: int = 5,  # Number of past states for temporal processing
        temporal_dim: int = 32,  # Temporal encoder output size
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.95,  # Lower gamma for 15-min horizon (was 0.99)
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.03,  # Lower entropy to allow sparse policy (mostly HOLD)
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 256,  # Smaller buffer for faster adaptation (was 512)
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

        # Temporal state history (per-market, keyed by asset)
        self._state_history: Dict[str, deque] = {}

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # For storing last action's log prob and value
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_temporal_state: Optional[np.ndarray] = None

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)

    def _get_temporal_state(
        self, asset: str, current_features: np.ndarray
    ) -> np.ndarray:
        """Get stacked temporal state for an asset.

        Maintains a history of the last N states per asset.
        Returns flattened array of shape (history_len * input_dim,).
        """
        if asset not in self._state_history:
            self._state_history[asset] = deque(maxlen=self.history_len)

        history = self._state_history[asset]

        # Add current state to history
        history.append(current_features.copy())

        # Pad with zeros if not enough history
        if len(history) < self.history_len:
            padding = [np.zeros(self.input_dim, dtype=np.float32)] * (
                self.history_len - len(history)
            )
            stacked = np.concatenate(padding + list(history))
        else:
            stacked = np.concatenate(list(history))

        return stacked.astype(np.float32)

    def act(self, state: MarketState) -> Action:
        """Select action using current policy with temporal context."""
        features = state.to_features()

        # Get temporal state (stacked history)
        temporal_state = self._get_temporal_state(state.asset, features)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32, device=self.device)
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

        return Action(action_idx)

    def store(
        self,
        state: MarketState,
        action: Action,
        reward: float,
        next_state: MarketState,
        done: bool,
    ):
        """Store experience for training with temporal context."""
        # Update running reward stats for normalization
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            (
                (self.reward_count - 1) * self.reward_std**2
                + delta * (reward - self.reward_mean)
            )
            / max(1, self.reward_count)
        )

        # Normalize reward
        norm_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)

        # Get next temporal state (updates history with next_state)
        next_features = next_state.to_features()
        next_temporal_state = self._get_temporal_state(next_state.asset, next_features)

        exp = Experience(
            state=state.to_features(),
            temporal_state=(
                self._last_temporal_state
                if self._last_temporal_state is not None
                else np.zeros(self.history_len * self.input_dim, dtype=np.float32)
            ),
            action=action.value,
            reward=norm_reward,
            next_state=next_features,
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

    def update(self) -> Optional[Dict[str, float]]:
        """Update policy using PPO with PyTorch autograd and temporal context."""
        if len(self.experiences) < self.buffer_size:
            return None

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
        old_values_tensor = torch.tensor(old_values, dtype=torch.float32, device=self.device)

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
                batch_old_values = old_values_tensor[batch_idx]

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
                values = self.critic(batch_states, batch_temporal).squeeze()

                # Value loss with clipping (PPO2 style)
                values_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss1 = (batch_returns - values) ** 2
                value_loss2 = (batch_returns - values_clipped) ** 2
                critic_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))

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
                print(f"  [RL] Early stop epoch {epoch}, KL={avg_kl:.4f}")
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

        return {
            "policy_loss": np.mean(all_metrics["policy_loss"]),
            "value_loss": np.mean(all_metrics["value_loss"]),
            "entropy": np.mean(all_metrics["entropy"]),
            "approx_kl": np.mean(all_metrics["approx_kl"]),
            "clip_fraction": np.mean(all_metrics["clip_fraction"]),
            "explained_variance": explained_var,
        }

    def reset(self):
        """Clear experience buffer and state history."""
        self.experiences.clear()
        self._state_history.clear()
        self._last_temporal_state = None

    def save(self, path: str):
        """Save model and training state."""
        # Convert path to .pth if needed
        weights_path = path.replace(".npz", "") + ".pth"

        # Save complete checkpoint with PyTorch
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count,
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
        # Convert path to .pth if needed
        weights_path = path.replace(".npz", "") + ".pth"

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Load model state
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Load optimizer state
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Load training stats
        self.reward_mean = float(checkpoint['reward_mean'])
        self.reward_std = float(checkpoint['reward_std'])
        self.reward_count = int(checkpoint['reward_count'])

        # Set models to appropriate mode
        self.actor.to(self.device)
        self.critic.to(self.device)
