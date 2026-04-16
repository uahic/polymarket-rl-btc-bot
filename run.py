import asyncio
import argparse
import logging
import sys
import signal

from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent))
from strategies import ALL_STRATEGIES, registry, MLStrategy
from config_loader import load_config
from trading_runner import GymTradingRunner
from features.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)



def show_strategy_list_and_usage():
    logger.info(f"Available Strategies:\n\n- {'\n- '.join(registry.list_all())}\n\n")
    logger.info("Usage: python run.py <strategy>")
    logger.info("       python run.py <strategy> --train")
    logger.info("       python run.py <strategy> --train --dashboard")
    logger.info("       python run.py <strategy> --episode-length 3600  # 30min episodes")


async def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading")
    parser.add_argument(
        "strategy", nargs="?", choices=ALL_STRATEGIES, help="Strategy to run"
    )
    parser.add_argument(
        "--train", action="store_true", help="Enable training mode for RL"
    )
    parser.add_argument("--size", type=float, default=1.0, help="Trade size in $")
    parser.add_argument("--load", nargs="?", const="__auto__", default=None,
                        help="Load RL model from file; omit path to auto-load the most recent .pth")
    parser.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Dashboard port")
    parser.add_argument("--live", action="store_true", help="Enable live trading mode")
    parser.add_argument("--episode-length", type=int, default=1800, help="Maximum steps per episode (default: 1800 = 15min @ 500ms)")
    parser.add_argument("--feature-config", type=str, default=None, help="Path to feature config YAML (default: baseline 22 features)")

    args = parser.parse_args()

    if not args.strategy:
        show_strategy_list_and_usage()
        return

    # Auto-load .env file
    load_dotenv()

    # Load config file from ./config.yaml
    config = load_config()

    # Load feature config
    if args.feature_config:
        feature_config = FeatureRegistry.from_yaml(args.feature_config)
        logger.info(f"Loaded feature config from {args.feature_config}")
    else:
        feature_config = FeatureRegistry.get_baseline_config()
        logger.info("Using baseline feature config (22 features, no time-of-day)")

    # Start dashboard if requested
    dashboard_thread = None
    if args.dashboard:
        import threading
        import time
        from dashboard.professional_dashboard import run_dashboard

        logger.info(f"STARTING  DASHBOARD - URL: http://localhost:{args.port}")

        dashboard_thread = threading.Thread(
            target=lambda: run_dashboard(host='0.0.0.0', port=args.port),
            daemon=True
        )
        dashboard_thread.start()
        time.sleep(2)  # Give dashboard time to initialize

    # Use gym-based runner
    mode = "live" if args.live else "paper"
    runner = GymTradingRunner(
        strategy_factory=lambda: create_strategy(args, feature_config),
        config=config,
        mode=mode,
        trade_size=args.size,
        assets=["BTC"],  # TODO: Make configurable
        max_episode_steps=args.episode_length,
        enable_dashboard=args.dashboard,
        feature_config=feature_config,
    )

    # Setup signal handler for graceful shutdown with auto-save
    def signal_handler(sig, frame):
        logger.info("Interrupt received, saving models and shutting down gracefully...")

        # Save all strategy instances if training mode was enabled
        if args.train:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for asset, strategy in runner.strategies.items():
                if isinstance(strategy, MLStrategy) and hasattr(strategy, 'save'):
                    save_path = f"models/{args.strategy}_{asset}_{timestamp}.pth"
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        strategy.save(save_path)
                        logger.info(f"Saved {asset} model to {save_path}")
                    except Exception as e:
                        logger.error(f"Failed to save {asset} model: {e}")

        logger.info("Shutdown complete.")
        # Raise KeyboardInterrupt to let asyncio handle cancellation properly
        raise KeyboardInterrupt

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await runner.run()
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except asyncio.CancelledError:
        pass


def find_latest_model(strategy_name: str) -> str | None:
    """Return the path of the most recently modified .pth in models/, or None."""
    models_dir = Path("models")
    if not models_dir.is_dir():
        return None
    candidates = sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def create_strategy(args, feature_config):
    """Create strategy instance with proper configuration.

    Args:
        args: Command-line arguments
        feature_config: FeatureConfig instance to use for the strategy
    """
    # Try creating with feature_config first, fall back to no params if not needed
    try:
        strategy = registry.create(args.strategy, feature_config=feature_config)
    except TypeError:
        # Strategy doesn't need feature_config parameter
        strategy = registry.create(args.strategy)

    # Setup ML-based strategy
    if isinstance(strategy, MLStrategy):
        load_path = args.load
        if load_path == "__auto__":
            load_path = find_latest_model(args.strategy)
            if load_path:
                logger.info(f"Auto-detected most recent model: {load_path}")
            else:
                logger.info("No .pth models found in models/ — starting fresh.")
        if load_path:
            strategy.load(load_path)
            logger.info(f"Loaded model from {load_path}")
        if args.train:
            strategy.train()
            logger.info("Training mode active")
        else:
            strategy.eval()

    return strategy


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Graceful shutdown already handled in main()
        pass
