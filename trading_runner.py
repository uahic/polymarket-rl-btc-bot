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
from typing import Callable, Dict, Optional, List
from dataclasses import dataclass

# Environment imports
from environments.trading_gym import TradingGym
from data.sources import LiveSource
from features.computer import FeatureComputer
from executors.live_executor import LiveOrderExecutor
from executors.executor_wrapper import GymExecutorWrapper

# Strategy imports
from structures.action import Action
from strategies.base_strategy import BaseStrategy

# Stream imports
from streams.orderbook import OrderbookStreamer
from streams.binance import BinanceStreamer
from streams.binance_futures import FuturesStreamer

# Logging
from logger.colors import Colors

# Config
from config_loader import Config

logger = logging.getLogger(__name__)

action_color_dict ={
    Action.BUY: f"{Colors.BOLD}{Colors.GREEN}BUY{Colors.RESET}",
    Action.HOLD: f"{Colors.BOLD}{Colors.BLUE}HOLD{Colors.RESET}",
    Action.SELL: f"{Colors.BOLD}{Colors.RED}RED{Colors.RESET}"
}

@dataclass
class MarketInfo:
    """Information about an active Polymarket market."""
    condition_id: str
    asset: str
    token_up: str
    token_down: str
    expiry: float
    description: str


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
        mode: str = "paper",  # "live" or "paper"
        trade_size: float = 10.0,
        assets: List[str] = None,
        max_episode_steps: int = 1800,
        enable_dashboard: bool = False,
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
        """
        self.strategy_factory = strategy_factory
        self.config = config
        self.mode = mode
        self.trade_size = trade_size
        self.assets = assets or ["BTC"]
        self.max_episode_steps = max_episode_steps
        self.enable_dashboard = enable_dashboard

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
        logger.info(f"GYM TRADING RUNNER | Mode: {self.mode.upper()} | Assets: {', '.join(self.assets)} | Trade size: ${self.trade_size:.2f}")

        # Initialize streams
        logger.info("Initializing data streams...")
        self.orderbook_streamer = OrderbookStreamer()
        self.binance_streamer = BinanceStreamer(assets=self.assets)
        self.futures_streamer = FuturesStreamer(assets=self.assets)

        # Initialize CLOB client for live trading
        if self.mode == "live":
            from py_clob_client.client import ClobClient

            self.transaction_client = ClobClient(
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
        await asyncio.sleep(3)

        # Create data source
        live_source = LiveSource(
            self.orderbook_streamer,
            self.binance_streamer,
            self.futures_streamer,
            tick_interval=0.5,
        )

        # Create feature computer
        feature_computer = FeatureComputer()

        logger.info("All systems ready")

        # Main loop: discover markets and trade
        while True:
            # Discover active markets
            markets = await self._discover_markets()

            # Subscribe to orderbook streams for discovered markets first,
            # then clean up stale subscriptions. This order ensures the current
            # market's condition_id is already in the active set before stale
            # cleanup runs, preventing a spurious reconnect at episode boundaries.
            for market in markets:
                self.orderbook_streamer.subscribe(
                    market.condition_id,
                    market.token_up,
                    market.token_down
                )

            # Clean up stale orderbook subscriptions for expired markets
            active_condition_ids = {market.condition_id for market in markets}
            self.orderbook_streamer.clear_stale(active_condition_ids)

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
                if hasattr(executor, 'set_market_context'):
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
                    initial_balance=1000.0 if self.mode == "paper" else 100.0,
                    max_episode_steps=self.max_episode_steps,
                    shaping_reward_coef=0.01,
                    normalize_rewards=True,
                )

                # Get or create strategy for this asset
                strategy = self._get_strategy(asset)

                # Run episode for this market
                await self._run_episode(env, strategy, market)

            # Wait before next market discovery
            await asyncio.sleep(10)

    async def _run_episode(self, env: TradingGym, strategy: BaseStrategy, market: MarketInfo):
        """
        Run one episode (one market) using gym interface.

        Args:
            env: TradingGym environment
            strategy: Strategy instance
            market: Market information
        """
        logger.info(f"Starting episode: {market.asset} | Market: {market.description[:60]}... | ID: {market.condition_id} | Expiry: {time.strftime('%H:%M:%S', time.localtime(market.expiry))}")

        # Reset environment and strategy
        obs, info = env.reset(options={"asset": market.asset, "market_id": market.condition_id})
        strategy.reset()

        done = False
        truncated = False
        step_count = 0
        episode_reward = 0.0
        # prev_obs = obs
        prev_pnl = 0.0  # Track previous PnL for delta calculation

        # Track trades for dashboard
        episode_trades = 0
        episode_wins = 0

        while not (done or truncated):
            # Get action from strategy
            action = strategy.act(obs)

            # Step environment
            next_obs, reward, done, truncated, info = await env.step(action)

            # RL training (if enabled)
            if hasattr(strategy, 'store') and hasattr(strategy, 'training') and strategy.training:
                strategy.store(obs, action, reward, next_obs, done or truncated)

                # Update (train) if buffer is full
                if hasattr(strategy, 'should_update') and strategy.should_update():
                    metrics = strategy.update() if hasattr(strategy, 'update') else None
                    if metrics:
                        self._log_training_metrics(metrics)

                        # Update dashboard with training metrics
                        if self.enable_dashboard:
                            self._update_dashboard_training(metrics, strategy)

            episode_reward += reward
            step_count += 1

            # Track completed trades for dashboard
            # Log any trade execution (open or close)
            if info.get('filled', False) or info.get('pnl', 0) != 0:
                # Count only closing trades for stats
                if info.get('pnl', 0) != 0:
                    episode_trades += 1
                    if info.get('pnl', 0) > 0:
                        episode_wins += 1

                # Log to dashboard
                if self.enable_dashboard:
                    self._update_dashboard_trade(info, market.asset)

            # Update dashboard with current PnL
            if self.enable_dashboard and step_count % 10 == 0:
                self._update_dashboard_pnl(info)

            # Logging
            if step_count % 1 == 0:
                current_pnl = info.get('unrealized_pnl', 0)
                delta_pnl = current_pnl - prev_pnl
                amount_spent = info.get('amount_spent', 0)
                realized_pnl = info.get('pnl', 0)
                has_position = info.get('has_position', False)
                position_side = info.get('position_side', None)


                # Build log message
                log_msg = (
                    f"Step {step_count:4d} | "
                    f"Action: {action_color_dict[action]:4s} | "
                    f"Reward: {reward:+.4f} | "
                    f"Balance: ${info.get('balance', 0):.2f}"
                )

                # Add position info
                if has_position:
                    log_msg += f" | Pos: {position_side} | UPnL: ${current_pnl:+.2f} | Δ: ${delta_pnl:+.2f}"
                else:
                    log_msg += f" | Pos: FLAT"

                # Add trade info (distinguish between open, close, and hold)
                rejection_reason = info.get('rejection_reason', '') or ''
                if action.name == 'HOLD':
                    # HOLD action - no trade expected
                    pass
                elif 'redundant' in rejection_reason:
                    # Redundant same-direction order
                    log_msg += f" | 🚫 REDUNDANT {action.name} | Penalty: ${realized_pnl:+.2f}"
                elif realized_pnl != 0 and amount_spent > 0:
                    # Closed old position AND opened new one (switched positions)
                    log_msg += f" | 🔄 SWITCHED | Realized: ${realized_pnl:+.2f} | Spent: ${amount_spent:.2f}"
                elif realized_pnl != 0:
                    # Just closed position
                    log_msg += f" | ✓ CLOSED | Realized: ${realized_pnl:+.2f}"
                elif amount_spent > 0 and info.get('filled', False):
                    # Just opened position
                    log_msg += f" | ✓ OPENED | Spent: ${amount_spent:.2f}"
                elif not info.get('filled', False):
                    # Trade was rejected
                    log_msg += f" | ❌ REJECTED: {rejection_reason}"

                logger.info(log_msg)
                prev_pnl = current_pnl

            obs = next_obs

            if done or truncated:
                break

        # Episode complete
        final_pnl = info.get("episode_pnl", 0.0)
        final_balance = info.get("balance", 0.0)
        total_spent = info.get("episode_spent", 0.0)

        # Update cumulative stats
        self.cumulative_pnl += final_pnl
        self.cumulative_trades += episode_trades
        self.cumulative_wins += episode_wins
        self.episode_count += 1

        # Update dashboard with episode metrics
        if self.enable_dashboard:
            self._update_dashboard_episode(episode_reward, step_count)

        logger.info(f"Episode complete: {market.asset} | Steps: {step_count} | Reward: {episode_reward:.4f} | Balance: ${final_balance:.2f} | PnL: ${final_pnl:+.2f} | Spent: ${total_spent:.2f} | Cumulative PnL: ${self.cumulative_pnl:+.2f}")

    async def _discover_markets(self) -> List[MarketInfo]:
        """
        Discover active Polymarket markets for tracked assets.

        Returns:
            List of active markets ready to trade
        """
        # TODO: Implement market discovery via CLOB API
        # For now, return mock data for testing

        markets = []

        # Calculate next 15-minute slot expiry
        # This ensures the same expiry is used for the same 15-min period
        current_time = time.time()
        minutes_into_hour = int(time.localtime(current_time).tm_min)
        next_slot = ((minutes_into_hour // 15) + 1) * 15

        # Get current hour as timestamp
        current_hour = time.mktime(time.localtime(current_time)[:4] + (0, 0, 0, 0, 0))

        # Calculate expiry at the next 15-minute boundary
        expiry = current_hour + (next_slot * 60)
        if expiry <= current_time:
            expiry += 900  # Add 15 minutes if we're exactly on a boundary

        for asset in self.assets:
            # Use expiry timestamp in condition_id to make it unique per slot
            slot_id = int(expiry)
            # Mock market for testing
            markets.append(
                MarketInfo(
                    condition_id=f"{asset.lower()}_15m_{slot_id}",
                    asset=asset,
                    token_up=f"mock_token_up_{asset.lower()}",
                    token_down=f"mock_token_down_{asset.lower()}",
                    expiry=expiry,
                    description=f"Will {asset} price go up in next 15 minutes?",
                )
            )

        return markets

    def _get_strategy(self, asset: str):
        """Get or create strategy instance for asset."""
        if asset not in self.strategies:
            self.strategies[asset] = self.strategy_factory()
            if hasattr(self.strategies[asset], 'reset'):
                self.strategies[asset].reset()
            logger.info(f"Created strategy instance for {asset}")

        return self.strategies[asset]

    def _log_training_metrics(self, metrics: dict):
        """Log training metrics."""
        logger.info(f"[Training] Policy Loss: {metrics.get('policy_loss', 0):.4f} | Value Loss: {metrics.get('value_loss', 0):.4f} | Entropy: {metrics.get('entropy', 0):.4f}")

    def _update_dashboard_training(self, metrics: dict, strategy):
        """Update dashboard with training metrics."""
        try:
            from dashboard.professional_dashboard import (
                update_training_metrics,
                update_buffer_size,
            )

            update_training_metrics(
                policy_loss=metrics.get('policy_loss'),
                value_loss=metrics.get('value_loss'),
                entropy=metrics.get('entropy'),
                kl_divergence=metrics.get('approx_kl'),
                clip_fraction=metrics.get('clip_fraction'),
                explained_variance=metrics.get('explained_variance'),
            )

            # Update buffer size
            if hasattr(strategy, 'experiences'):
                buffer_size = len(strategy.experiences)
                max_buffer = strategy.buffer_size if hasattr(strategy, 'buffer_size') else 256
                update_buffer_size(buffer_size, max_buffer)

        except Exception as e:
            # Silently fail if dashboard is not available
            pass

    def _update_dashboard_pnl(self, info: dict):
        """Update dashboard with current PnL."""
        try:
            from dashboard.professional_dashboard import update_pnl

            unrealized_pnl = info.get('unrealized_pnl', 0.0)
            realized_pnl = self.cumulative_pnl

            update_pnl(
                total_pnl=realized_pnl + unrealized_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl
            )
        except Exception as e:
            pass

    def _update_dashboard_trade(self, info: dict, asset: str):
        """Log a completed trade to the dashboard."""
        try:
            from dashboard.professional_dashboard import log_trade
            from datetime import datetime

            # Extract trade information
            pnl = info.get('pnl', 0.0)
            amount_spent = info.get('amount_spent', 0.0)
            position_side = info.get('position_side', 'UNKNOWN')
            filled = info.get('filled', False)

            # Get prices if available
            entry_price = info.get('entry_price', 0.5)
            exit_price = info.get('exit_price', entry_price)

            # Only log if there's actual trade activity (open or close)
            if filled or pnl != 0:
                # For opening trades, pnl will be 0
                # For closing trades, pnl will be the realized profit/loss
                log_trade(
                    asset=asset,
                    side=position_side if position_side != 'UNKNOWN' else 'LONG',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=amount_spent if amount_spent > 0 else abs(pnl) if pnl != 0 else 10.0,
                    pnl=pnl,
                    duration_sec=info.get('trade_duration', 0),
                    timestamp=datetime.now().strftime('%H:%M:%S')
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
                avg_length=step_count
            )

            # Update final PnL
            update_pnl(
                total_pnl=self.cumulative_pnl,
                realized_pnl=self.cumulative_pnl,
                unrealized_pnl=0.0
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
        assets=["BTC"],  # TODO: Make configurable
    )

    # Run
    await runner.run()


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    from config_loader import load_config
    from strategies import ALL_STRATEGIES

    parser = argparse.ArgumentParser(description="Gym-based Polymarket Trading")
    parser.add_argument(
        "strategy", choices=ALL_STRATEGIES, help="Strategy to run"
    )
    parser.add_argument(
        "--train", action="store_true", help="Enable training mode for RL"
    )
    parser.add_argument("--size", type=float, default=10.0, help="Trade size in $")
    parser.add_argument("--load", type=str, help="Load RL model from file")
    parser.add_argument("--live", action="store_true", help="Enable live trading mode")

    args = parser.parse_args()

    # Load config
    load_dotenv()
    args.config = load_config()

    # Run
    asyncio.run(main(args))
