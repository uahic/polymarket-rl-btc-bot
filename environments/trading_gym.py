"""
Unified Gym environment for trading bot.

This environment provides a standard Gymnasium interface that works
seamlessly with both historical and live data sources.

The environment is compatible with:
- Stable-Baselines3
- RLlib
- CleanRL
- Custom training loops

Usage:
    # Offline training
    from data.sources import HistoricalSource
    source = HistoricalSource("data/historical")
    env = TradingGym(source, executor, feature_computer)

    # Live trading
    from data.sources import LiveSource
    source = LiveSource(orderbook_stream, binance_stream, futures_stream)
    env = TradingGym(source, executor, feature_computer)

The environment handles:
- Feature computation (via FeatureComputer)
- Order execution (via OrderExecutor)
- Reward computation
- Episode management
"""

import logging
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass


from data.sources import DataSource
from features.computer import (
    FeatureComputer,
    RawMarketData,
    PositionState,
    TransactionState,
    CapitalState,
)
from structures.action import Action

logger = logging.getLogger(__name__)


@dataclass
class TradingAction:
    """Trading action."""

    action: Action
    size: float = 1.0  # USD size


@dataclass
class ExecutionResult:
    """Result of order execution."""

    success: bool
    filled: bool
    balance: float
    position: Optional[Any]  # Current position (None if flat)
    pnl: float  # Realized PnL from this action
    fee: float  # Fees paid
    slippage: float  # Slippage incurred
    amount_spent: float = 0.0  # USD/USDC spent on this trade
    rejection_reason: Optional[str] = None


class OrderExecutor:
    """
    Abstract order executor interface.

    Subclasses handle:
    - SimulatedOrderExecutor: Paper trading with realistic simulation
    - LiveOrderExecutor: Real trading via CLOB API
    """

    def reset(self, balance: float):
        """Reset executor state."""
        raise NotImplementedError

    def execute(
        self, action: TradingAction, market_data: RawMarketData
    ) -> ExecutionResult:
        """Execute trading action."""
        raise NotImplementedError

    def get_position_state(self) -> PositionState:
        """Get current position state."""
        raise NotImplementedError

    def get_transaction_state(self) -> TransactionState:
        """Get current transaction state."""
        raise NotImplementedError

    def get_capital_state(self) -> CapitalState:
        """Get current capital state."""
        raise NotImplementedError

    def compute_position_state(self, current_price: float, time_remaining: float) -> PositionState:
        """Compute position state with current market price and time remaining."""
        raise NotImplementedError


class TradingGym(gym.Env):
    """
    Unified Gym environment for trading.

    Action Space:
        Discrete(3): [BUY_UP, HOLD, SELL_DOWN]

    Observation Space:
        Box(N,): Normalized features in [-10, 10] (dynamic based on FeatureConfig)
        Shape is determined at runtime from FeatureComputer output

    Rewards:
        - Terminal reward: Realized P&L when position closes
        - Shaping reward: Unrealized P&L * 0.01 (optional)

    Episode Termination:
        - Market expiration (15 minutes for Polymarket)
        - Max steps reached
        - Data source exhausted (historical mode)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_source: DataSource,
        executor: OrderExecutor,
        feature_computer: FeatureComputer,
        initial_balance: float = 1000.0,
        max_episode_steps: int = 1800,  # 15 min @ 500ms ticks
        shaping_reward_coef: float = 0.01,
        normalize_rewards: bool = True,
    ):
        """
        Initialize trading environment.

        Args:
            data_source: Data source (historical or live)
            executor: Order executor (simulated or live)
            feature_computer: Feature computation
            initial_balance: Starting capital
            max_episode_steps: Max ticks per episode
            shaping_reward_coef: Weight for unrealized PnL shaping
            normalize_rewards: Whether to z-score normalize rewards
        """
        super().__init__()

        self.data_source = data_source
        self.executor = executor
        self.feature_computer = feature_computer
        self.initial_balance = initial_balance
        self.max_episode_steps = max_episode_steps
        self.shaping_reward_coef = shaping_reward_coef
        self.normalize_rewards = normalize_rewards

        # Gym spaces
        self.action_space = gym.spaces.Discrete(3)  # BUY_UP, HOLD, SELL_DOWN
        # Observation space will be set dynamically on first reset
        self.observation_space = None

        # Episode state
        self.step_count = 0
        self.current_market_data: Optional[RawMarketData] = None
        self.pending_rewards: Dict[str, float] = {}  # For terminal rewards

        # Episode metrics
        self.episode_pnl = 0.0
        self.episode_fees = 0.0
        self.episode_trades = 0
        self.episode_spent = 0.0  # Total USD/USDC spent on trades

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed
            options: Episode options (asset, start_time, etc.)

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # Reset episode state
        self.step_count = 0
        self.pending_rewards = {}
        self.episode_pnl = 0.0
        self.episode_fees = 0.0
        self.episode_trades = 0
        self.episode_spent = 0.0

        # Reset data source
        reset_kwargs = options or {}
        self.current_market_data = self.data_source.reset(**reset_kwargs)

        # Reset executor
        self.executor.reset(balance=self.initial_balance)

        # Compute initial observation
        obs = self._get_observation()

        # Initialize observation space if not yet set
        if self.observation_space is None:
            self.observation_space = gym.spaces.Box(
                low=-10.0,
                high=10.0,
                shape=obs.shape,
                dtype=np.float32,
            )
            logger.info(f"Observation space initialized: {obs.shape}")

        info = {
            "asset": self.current_market_data.asset,
            "timestamp": self.current_market_data.timestamp,
        }

        return obs, info

    async def step(
        self,
        action: Action,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Args:
            action: Action enum (HOLD, BUY, SELL)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.current_market_data is None:
            raise RuntimeError("Must call reset() before step()")

        # Advance time FIRST to get fresh market data
        has_more_data = await self.data_source.advance()
        self.step_count += 1

        # Check termination
        terminated = self.data_source.is_done() or not has_more_data
        truncated = self.step_count >= self.max_episode_steps

        # Update market data BEFORE executing action (prevents stale data)
        if not terminated and not truncated:
            self.current_market_data = self.data_source.get_current()
        else:
            # Terminal state: refresh market data if available so PnL is accurate
            latest = self.data_source.get_current()
            if latest is not None:
                self.current_market_data = latest

        # Execute action with FRESH market data
        trading_action = TradingAction(action=action, size=1.0)
        result = self.executor.execute(trading_action, self.current_market_data)

        # Track metrics
        if result.filled:
            self.episode_trades += 1
            self.episode_pnl += result.pnl
            self.episode_fees += result.fee
            self.episode_spent += result.amount_spent

        # Compute reward
        reward = self._compute_reward(result)

        # Get observation for next step
        if not terminated and not truncated:
            obs = self._get_observation()
        else:
            obs = np.zeros(26, dtype=np.float32)

        # Get current unrealized PnL
        if self.current_market_data:
            position = self.executor.compute_position_state(
                current_price=self.current_market_data.prob_up,
                time_remaining=self.current_market_data.time_remaining,
            )
            unrealized_pnl = position.unrealized_pnl
            has_position = position.has_position
            position_side = position.side
        else:
            unrealized_pnl = 0.0
            has_position = False
            position_side = None

        # Info dict
        info = {
            "step": self.step_count,
            "timestamp": (
                self.current_market_data.timestamp if self.current_market_data else 0.0
            ),
            "action": action,
            "filled": result.filled,
            "pnl": result.pnl,
            "balance": result.balance,
            "unrealized_pnl": unrealized_pnl,
            "has_position": has_position,
            "position_side": position_side,
            "episode_pnl": self.episode_pnl,
            "episode_trades": self.episode_trades,
            "episode_spent": self.episode_spent,
            "amount_spent": result.amount_spent,
            "rejection_reason": result.rejection_reason,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Compute current observation.

        Aggregates market data and agent state into feature vector.

        Returns:
            26-dimensional feature vector (22 market + 4 time-of-day)
        """
        # Get agent-dependent state with current market price
        position = self.executor.compute_position_state(
            current_price=self.current_market_data.prob_up,
            time_remaining=self.current_market_data.time_remaining,
        )

        transaction = self.executor.get_transaction_state()
        capital = self.executor.get_capital_state()

        # Compute features
        features = self.feature_computer.compute_features(
            raw_data=self.current_market_data,
            position=position,
            transaction=transaction,
            capital=capital,
        )

        return features

    def _compute_reward(self, result: ExecutionResult) -> float:
        """
        Compute reward for this step.

        Uses two-component reward:
        1. Terminal reward: Realized P&L when position closes (primary signal)
        2. Shaping reward: Unrealized P&L * coefficient (optional, helps with credit assignment)

        Redundant-action penalties bypass normalization so they always register
        as a strong fixed negative, preventing the z-score mean from absorbing them.
        All other rewards are clipped to [-3, 3] instead of z-scored, which keeps
        the scale stable without shifting the penalty signal.

        Args:
            result: Execution result

        Returns:
            Scalar reward
        """
        # Redundant action: return the raw penalty directly, skip normalization.
        # This guarantees the gradient signal is always a hard negative regardless
        # of running reward statistics.
        is_redundant = (
            result.rejection_reason is not None
            and "redundant" in result.rejection_reason
        )
        if is_redundant:
            return result.pnl

        reward = 0.0

        # Terminal reward (realized P&L)
        if result.pnl != 0.0:
            reward += result.pnl

        # Shaping reward (unrealized P&L)
        if self.shaping_reward_coef > 0 and self.current_market_data:
            # Compute position state with current market data to get actual unrealized PnL
            position = self.executor.compute_position_state(
                current_price=self.current_market_data.prob_up,
                time_remaining=self.current_market_data.time_remaining,
            )
            # Add positive or negative reward, weighted by a tiny coefficient to combat sparse rewards
            if position.has_position:
                reward += position.unrealized_pnl * self.shaping_reward_coef

        # Clip reward to stable range instead of z-scoring.
        # Z-scoring shifts the penalty signal as the running mean drifts;
        # clipping keeps the scale fixed while bounding outliers.
        if self.normalize_rewards:
            reward = float(np.clip(reward, -3.0, 3.0))

        return reward

    def render(self):
        """Render environment (optional)."""
        if self.current_market_data is None:
            return

        logger.info(f"=== Step {self.step_count} ===")
        logger.info(f"Asset: {self.current_market_data.asset}")
        logger.info(f"Prob UP: {self.current_market_data.prob_up:.3f}")
        logger.info(f"Time remaining: {self.current_market_data.time_remaining:.2%}")
        logger.info(f"Balance: ${self.executor.get_capital_state().available_balance:.2f}")
        logger.info(f"Episode PnL: ${self.episode_pnl:.2f}")

    def close(self):
        """Clean up resources."""
        pass
