"""
Gym-based trading runner for live and paper trading.

This runner replaces engine.py with a cleaner architecture based on TradingGym.
It supports both live CLOB trading and paper trading with simulated execution.

Key benefits over engine.py:
- Uses standard OpenAI Gym interface
- Shares code with offline training (train_offline.py)
- Cleaner separation: streams → environment → agent
- Compatible with standard RL libraries

Usage:
    python run_gym.py btc_ppo --live
    python run_gym.py btc_ppo --paper --train
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Any, List

# Environment imports
from environments.trading_gym import TradingGym
from data.sources import LiveSource
from features.computer import FeatureComputer
from executors.live_executor import LiveOrderExecutor
from executors.executor_wrapper import GymExecutorWrapper

# Strategy imports
from structures.market import MarketInfo
from structures.action import Action
from strategies.base_strategy import BaseStrategy
from strategies.ml_base_strategy import MLStrategy

# Stream imports
from streams.orderbook import OrderbookStreamer
from streams.binance import BinanceStreamer
from streams.binance_futures import FuturesStreamer
from streams.polymarket_api import get_15m_markets

# Logging
from logger.colors import Colors

# Config
from config_loader import Config, load_runner_config

_runner_cfg = load_runner_config()
_rcfg_runner = _runner_cfg.get("runner", {})
_rcfg_env = _runner_cfg.get("environment", {})
_rcfg_streams = _runner_cfg.get("streams", {})
_rcfg_loop = _runner_cfg.get("loop", {})
_rcfg_logging = _runner_cfg.get("logging", {})

logger = logging.getLogger(__name__)

action_color_dict = {
    Action.BUY: f"{Colors.BOLD}{Colors.GREEN}BUY{Colors.RESET}",
    Action.HOLD: f"{Colors.BOLD}{Colors.BLUE}HOLD{Colors.RESET}",
    Action.SELL: f"{Colors.BOLD}{Colors.RED}SELL{Colors.RESET}",
}

position_color_dict = {
    "UP": f"{Colors.BOLD}{Colors.GREEN}UP{Colors.RESET}",
    "DOWN": f"{Colors.BOLD}{Colors.RED}DOWN{Colors.RESET}",
    "None": f"{Colors.BOLD}{Colors.GRAY}None{Colors.RESET}",
}


def log_run_episode(
    step_count, action: Action, prev_position_pnl, reward, info: Dict[str, Any]
) -> None:
    episode_pnl = info.get("unrealized_pnl", 0)  # Episode-wide PnL
    position_pnl = info.get("position_unrealized_pnl", 0)  # Current position PnL
    delta_position_pnl = position_pnl - prev_position_pnl
    amount_spent = info.get("amount_spent", 0)
    realized_pnl = info.get("pnl", 0)
    has_position = info.get("has_position", False)
    position_side = info.get("position_side", None)
    balance = info.get('balance', 0)

    # Build log message with fixed-width columns
    # Pad action manually since colored strings don't respect format width
    action_str = action_color_dict[action]
    # BUY=3, SELL=4, HOLD=4 chars - pad to 4 chars visually
    action_padding = " " if action == Action.BUY else ""

    log_msg = (
        f"Step {step_count:4d} | "
        f"Action: {action_str}{action_padding} | "
        f"Reward: {reward:+7.4f} | "
        f"Balance: ${balance:8.2f}"
    )

    # Add position info - show both episode and position PnL with fixed widths
    if has_position:
        # Color-code position and pad manually (UP=2 chars, DOWN=4 chars -> pad to 4)
        pos_colored = position_color_dict.get(position_side, position_side)
        pos_padding = "  " if position_side == "UP" else ""  # UP needs 2 spaces to match DOWN
        log_msg += (
            f" | Pos: {pos_colored}{pos_padding} | "
            f"Pos UPnL: ${position_pnl:+8.2f} (Δ: ${delta_position_pnl:+7.2f}) | "
            f"Ep UPnL: ${episode_pnl:+8.2f}"
        )
    else:
        # None is 4 chars, no padding needed
        log_msg += f" | Pos: {position_color_dict['None']} | Ep UPnL: ${episode_pnl:+8.2f}"

    # Add trade info (distinguish between open, close, and hold)
    rejection_reason = info.get("rejection_reason", "") or ""
    if action.name == "HOLD":
        # HOLD action - no trade expected
        pass
    elif "redundant" in rejection_reason:
        # Redundant same-direction order
        log_msg += f" | 🚫 REDUNDANT {action.name} | Penalty: ${realized_pnl:+.2f}"
    elif realized_pnl != 0 and amount_spent > 0:
        # Closed old position AND opened new one (switched positions)
        log_msg += f" | 🔄 SWITCHED | Realized: ${realized_pnl:+8.2f} | Opened: ${amount_spent:8.2f}"
    elif realized_pnl != 0 and not info.get("filled", False):
        # Closed position but failed to open new one (ended up flat)
        log_msg += f" | ⚠️  CLOSED ONLY | Realized: ${realized_pnl:+8.2f} | Reopen failed: {rejection_reason}"
    elif realized_pnl != 0:
        # Just closed position
        log_msg += f" | ✓ CLOSED | Realized: ${realized_pnl:+8.2f}"
    elif amount_spent > 0 and info.get("filled", False):
        # Just opened position
        log_msg += f" | ✓ OPENED | Size: ${amount_spent:8.2f}"
    elif not info.get("filled", False):
        # Trade was rejected
        log_msg += f" | ❌ REJECTED: {rejection_reason}"

    logger.info(log_msg)


class GymTradingRunner:
    """
    Unified runner using TradingGym for both training and live trading.

    Architecture:
        Streams → LiveSource → TradingGym → Strategy

    One strategy instance per asset for clean state separation.
    """

    def __init__(
        self,
        strategy_factory: Callable,
        config: Config,
        mode: str = _rcfg_runner.get("mode", "paper"),
        trade_size: float = _rcfg_runner.get("default_trade_size", 10.0),
        assets: List[str] = None,
        max_episode_steps: int = _rcfg_runner.get("max_episode_steps", 1800),
        enable_dashboard: bool = _rcfg_runner.get("enable_dashboard", False),
        feature_config=None,
    ):
        """
        Initialize gym trading runner.

        Args:
            strategy_factory: Function that creates strategy instances
            config: Config object with API keys and settings
            mode: "live" for real CLOB trading, "paper" for simulated
            trade_size: Default trade size in USD
            assets: List of assets to track (default: ["BTC"])
            max_episode_steps: Maximum steps per episode (default: 1800 = 15min @ 500ms)
            enable_dashboard: Enable professional dashboard integration
            feature_config: FeatureConfig for feature computation (default: baseline)
        """
        self.strategy_factory = strategy_factory
        self.config = config
        self.mode = mode
        self.trade_size = trade_size
        self.assets = assets or _rcfg_runner.get("assets", ["BTC"])
        self.max_episode_steps = max_episode_steps
        self.enable_dashboard = enable_dashboard
        self.feature_config = feature_config

        # State
        self.strategies: Dict[str, any] = {}  # asset → strategy instance
        self.active_markets: Dict[str, MarketInfo] = {}  # asset → market info

        # Components (initialized in run())
        self.orderbook_streamer = None
        self.binance_streamer = None
        self.futures_streamer = None
        self.transaction_client = None

        # Dashboard tracking
        self.cumulative_pnl = 0.0
        self.cumulative_trades = 0
        self.cumulative_wins = 0
        self.episode_count = 0

    async def run(self):
        """Start the trading runner."""
        logger.info(
            f"GYM TRADING RUNNER | Mode: {self.mode.upper()} | Assets: {', '.join(self.assets)} | Trade size: ${self.trade_size:.2f}"
        )

        # Initialize streams
        logger.info("Initializing data streams...")
        self.orderbook_streamer = OrderbookStreamer()
        self.binance_streamer = BinanceStreamer(assets=self.assets)
        self.futures_streamer = FuturesStreamer(assets=self.assets)

        # Initialize CLOB client for live trading
        if self.mode == "live":
            from transactions.async_client import AsyncClobClient

            self.transaction_client = AsyncClobClient(
                host=self.config.clob_url,
                key=self.config.private_key,
                chain_id=self.config.chain_id,
            )
            logger.info("CLOB client initialized (LIVE MODE)")
        else:
            logger.info("Paper trading mode (simulated execution)")

        # Start streams and trading loop concurrently
        await asyncio.gather(
            self.orderbook_streamer.stream(),
            self.binance_streamer.stream(),
            self.futures_streamer.stream(),
            self._trading_loop(),
        )

    async def _trading_loop(self):
        """Main trading loop using gym environment."""
        # Wait for streams to initialize
        logger.info("Waiting for streams to initialize...")
        await asyncio.sleep(_rcfg_streams.get("init_wait_seconds", 3))

        # Create data source
        live_source = LiveSource(
            self.orderbook_streamer,
            self.binance_streamer,
            self.futures_streamer,
            tick_interval=_rcfg_streams.get("tick_interval", 0.5),
        )

        # Create feature computer with the same config as the strategy
        feature_computer = FeatureComputer(feature_config=self.feature_config)

        logger.info("All systems ready")

        # Main loop: discover markets and trade
        while True:
            # Discover active markets
            markets = get_15m_markets(assets=self.assets)
            logger.info(f"Discovered {len(markets)} active market(s) for {self.assets}")

            # Subscribe to orderbook streams for discovered markets first,
            # then clean up stale subscriptions. This order ensures the current
            # market's condition_id is already in the active set before stale
            # cleanup runs, preventing a spurious reconnect at episode boundaries.
            for market in markets:
                self.orderbook_streamer.subscribe(
                    market.condition_id, market.token_up, market.token_down
                )

            # Clean up stale orderbook subscriptions for expired markets
            active_condition_ids = {market.condition_id for market in markets}
            self.orderbook_streamer.clear_stale(active_condition_ids)

            if not markets:
                logger.warning(f"No active markets found for {self.assets}. Retrying in {_rcfg_loop.get('market_discovery_interval_seconds', 10)}s...")
                await asyncio.sleep(_rcfg_loop.get("market_discovery_interval_seconds", 10))
                continue

            for market in markets:
                asset = market.asset

                # Get or create executor for this market
                if self.mode == "live":
                    executor = LiveOrderExecutor(
                        transaction_client=self.transaction_client,
                        config=self.config,
                        default_order_size=self.trade_size,
                    )
                else:
                    executor = GymExecutorWrapper(default_order_size=self.trade_size)

                # Set market context for executor
                if hasattr(executor, "set_market_context"):
                    executor.set_market_context(
                        market.condition_id,
                        market.token_up,
                        market.token_down,
                    )

                # Create gym environment
                env = TradingGym(
                    data_source=live_source,
                    executor=executor,
                    feature_computer=feature_computer,
                    initial_balance=(
                        _rcfg_env.get("paper_initial_balance", 1000.0)
                        if self.mode == "paper"
                        else _rcfg_env.get("live_initial_balance", 100.0)
                    ),
                    max_episode_steps=self.max_episode_steps,
                    shaping_reward_coef=_rcfg_env.get("shaping_reward_coef", 0.01),
                    normalize_rewards=_rcfg_env.get("normalize_rewards", True),
                )

                # Get or create strategy for this asset
                # _get_strategy calls the strategy_factory passed to this class instance
                strategy = self._get_strategy(asset)

                # Run MULTIPLE episodes for this market
                # Note: is_done() check happens AFTER first episode reset
                # Detect if this is a new market by comparing condition_id
                is_new_market = (
                    not hasattr(self, '_last_market_id') or
                    self._last_market_id != market.condition_id
                )
                is_first_episode = is_new_market
                market_expired = False

                # Track this market for next iteration
                self._last_market_id = market.condition_id

                # Run at least one episode, then check if market expired
                while True:
                    # Run one episode
                    market_expired = await self._run_episode(
                        env, strategy, market, is_first_episode
                    )

                    is_first_episode = False

                    if market_expired or live_source.is_done():
                        # Market expired during or after episode
                        logger.info(f"Market {market.condition_id} expired")
                        break

            # Wait before next market discovery
            await asyncio.sleep(_rcfg_loop.get("market_discovery_interval_seconds", 10))

    async def _run_episode(
        self, env: TradingGym, strategy: BaseStrategy, market: MarketInfo, is_first_episode: bool = True
    ):
        """
        Run one RL episode.

        Args:
            env: TradingGym environment
            strategy: Strategy instance
            market: Market information
            is_first_episode: If True, this is the first episode for this market (cold reset)

        Returns:
            True if market expired during episode, False otherwise
        """
        # Pretty multiline episode header
        logger.info(
            f"\n{'='*80}\n"
            f"  Episode {self.episode_count + 1} | {market.asset} | {'NEW MARKET' if is_first_episode else 'WARM RESET'}\n"
            f"  Market: {market.description[:60]}...\n"
            f"  ID: {market.condition_id}\n"
            f"{'='*80}"
        )

        # Reset environment
        obs, info = env.reset(
            options={
                "asset": market.asset,
                "market_id": market.condition_id,
                "is_first_episode": is_first_episode
            }
        )

        # Only reset strategy on first episode of a new market
        if is_first_episode:
            strategy.reset()

        done = False  # Episode done?
        truncated = False  # Episode truncated?
        step_count = 0
        episode_reward = 0.0
        # prev_obs = obs
        prev_position_pnl = 0.0  # Track previous position PnL for delta calculation

        # Track trades for dashboard
        episode_trades = 0
        episode_wins = 0

        while not (done or truncated):
            # Get action from strategy
            action = strategy.act(obs)

            # Step environment
            next_obs, reward, done, truncated, info = await env.step(action)

            # RL training (if enabled)
            if isinstance(strategy, MLStrategy) and strategy.training:
                strategy.store(obs, action, reward, next_obs, done or truncated)

                # Train for 1 step
                if strategy.should_update():
                    metrics = strategy.update()
                    if metrics:
                        self._log_training_metrics(metrics)

                        # Update dashboard with training metrics
                        if self.enable_dashboard:
                            self._update_dashboard_training(metrics, strategy)

            episode_reward += reward
            step_count += 1

            # Track completed trades for dashboard
            # Log any trade execution (open or close)
            if info.get("filled", False) or info.get("pnl", 0) != 0:
                # Count only closing trades for stats
                if info.get("pnl", 0) != 0:
                    episode_trades += 1
                    if info.get("pnl", 0) > 0:
                        episode_wins += 1

                # Log to dashboard
                if self.enable_dashboard:
                    self._update_dashboard_trade(info, market.asset)

            # Update dashboard with current PnL
            if (
                self.enable_dashboard
                and step_count % _rcfg_logging.get("dashboard_pnl_update_steps", 10)
                == 0
            ):
                self._update_dashboard_pnl(info)

            # Logging
            if step_count % _rcfg_logging.get("log_frequency_steps", 1) == 0:
                log_run_episode(step_count, action, prev_position_pnl, reward, info)
                prev_position_pnl = info.get("position_unrealized_pnl", 0)

            obs = next_obs

            if done or truncated:
                break

        # Episode complete
        final_pnl = info.get("episode_pnl", 0.0)
        final_balance = info.get("balance", 0.0)
        initial_balance = (
            _rcfg_env.get("paper_initial_balance", 1000.0)
            if self.mode == "paper"
            else _rcfg_env.get("live_initial_balance", 100.0)
        )
        episode_profit = final_balance - initial_balance

        # Update cumulative stats
        self.cumulative_pnl += final_pnl
        self.cumulative_trades += episode_trades
        self.cumulative_wins += episode_wins
        self.episode_count += 1

        # Update dashboard with episode metrics
        if self.enable_dashboard:
            self._update_dashboard_episode(episode_reward, step_count)

        logger.info(
            f"Episode complete | Steps: {step_count} | "
            f"Reason: {'MARKET EXPIRED' if done else 'TRUNCATED'} | "
            f"Reward: {episode_reward:.4f} | "
            f"Episode Profit: ${episode_profit:+.2f} (Balance: ${final_balance:.2f}) | "
            f"Cumulative PnL: ${self.cumulative_pnl:+.2f}"
        )

        # Return True if market expired (done=True), False if just truncated
        return done


    def _get_strategy(self, asset: str):
        """Get or create strategy instance for asset."""
        if asset not in self.strategies:
            self.strategies[asset] = self.strategy_factory()
            if hasattr(self.strategies[asset], "reset"):
                self.strategies[asset].reset()
            logger.info(f"Created strategy instance for {asset}")

        return self.strategies[asset]

    def _log_training_metrics(self, metrics: dict):
        """Log training metrics."""
        logger.info(
            f"[Training] Policy Loss: {metrics.get('policy_loss', 0):.4f} | Value Loss: {metrics.get('value_loss', 0):.4f} | Entropy: {metrics.get('entropy', 0):.4f}"
        )

    def _update_dashboard_training(self, metrics: dict, strategy):
        """Update dashboard with training metrics."""
        try:
            from dashboard.professional_dashboard import (
                update_training_metrics,
                update_buffer_size,
            )

            update_training_metrics(
                policy_loss=metrics.get("policy_loss"),
                value_loss=metrics.get("value_loss"),
                entropy=metrics.get("entropy"),
                kl_divergence=metrics.get("approx_kl"),
                clip_fraction=metrics.get("clip_fraction"),
                explained_variance=metrics.get("explained_variance"),
            )

            # Update buffer size
            if hasattr(strategy, "experiences"):
                buffer_size = len(strategy.experiences)
                max_buffer = (
                    strategy.buffer_size if hasattr(strategy, "buffer_size") else 256
                )
                update_buffer_size(buffer_size, max_buffer)

        except Exception as e:
            # Silently fail if dashboard is not available
            pass

    def _update_dashboard_pnl(self, info: dict):
        """Update dashboard with current PnL."""
        try:
            from dashboard.professional_dashboard import update_pnl

            unrealized_pnl = info.get("unrealized_pnl", 0.0)
            realized_pnl = self.cumulative_pnl

            update_pnl(
                total_pnl=realized_pnl + unrealized_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
            )
        except Exception as e:
            pass

    def _update_dashboard_trade(self, info: dict, asset: str):
        """Log a completed trade to the dashboard."""
        try:
            from dashboard.professional_dashboard import log_trade
            from datetime import datetime

            # Extract trade information
            pnl = info.get("pnl", 0.0)
            amount_spent = info.get("amount_spent", 0.0)
            position_side = info.get("position_side", "UNKNOWN")
            filled = info.get("filled", False)

            # Get prices if available
            entry_price = info.get("entry_price", 0.5)
            exit_price = info.get("exit_price", entry_price)

            # Only log if there's actual trade activity (open or close)
            if filled or pnl != 0:
                # For opening trades, pnl will be 0
                # For closing trades, pnl will be the realized profit/loss
                log_trade(
                    asset=asset,
                    side=position_side if position_side != "UNKNOWN" else "LONG",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=(
                        amount_spent
                        if amount_spent > 0
                        else abs(pnl) if pnl != 0 else 10.0
                    ),
                    pnl=pnl,
                    duration_sec=info.get("trade_duration", 0),
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                )
        except Exception as e:
            # Silent fail - dashboard is optional
            pass

    def _update_dashboard_episode(self, episode_reward: float, step_count: int):
        """Update dashboard with episode completion metrics."""
        try:
            from dashboard.professional_dashboard import (
                update_episode_metrics,
                update_pnl,
            )

            # Calculate average episode reward (last 10 episodes would be better)
            avg_reward = episode_reward / step_count if step_count > 0 else 0

            update_episode_metrics(
                episode_count=self.episode_count,
                avg_reward=avg_reward,
                avg_length=step_count,
            )

            # Update final PnL
            update_pnl(
                total_pnl=self.cumulative_pnl,
                realized_pnl=self.cumulative_pnl,
                unrealized_pnl=0.0,
            )
        except Exception as e:
            pass


async def main(args):
    """Entry point for gym-based runner."""
    from strategies import registry, MLStrategy

    # Create strategy factory
    def strategy_factory():
        strategy = registry.create(args.strategy)

        # Setup ML-based strategy
        if isinstance(strategy, MLStrategy):
            if args.load:
                strategy.load(args.load)
                logger.info(f"Loaded model from {args.load}")
            if args.train:
                strategy.train()
                logger.info("Training mode active")
            else:
                strategy.eval()

        return strategy

    # Create runner
    mode = "live" if args.live else "paper"
    runner = GymTradingRunner(
        strategy_factory=strategy_factory,
        config=args.config,
        mode=mode,
        trade_size=args.size,
        assets=_rcfg_runner.get("assets", ["BTC"]),
    )

    # Run
    await runner.run()


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    from config_loader import load_config
    from strategies import ALL_STRATEGIES

    parser = argparse.ArgumentParser(description="Gym-based Polymarket Trading")
    parser.add_argument("strategy", choices=ALL_STRATEGIES, help="Strategy to run")
    parser.add_argument(
        "--train", action="store_true", help="Enable training mode for RL"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=_rcfg_runner.get("default_trade_size", 10.0),
        help="Trade size in $",
    )
    parser.add_argument("--load", type=str, help="Load RL model from file")
    parser.add_argument("--live", action="store_true", help="Enable live trading mode")

    args = parser.parse_args()

    # Load config
    load_dotenv()
    args.config = load_config()

    # Run
    asyncio.run(main(args))
