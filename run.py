import asyncio
import argparse
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strategies import ALL_STRATEGIES, registry, MLStrategy
from engine import TradingEngine


def show_strategy_list_and_usage():
    print(f"Available Strategies:\n\n- {'\n- '.join(registry.list_all())}\n\n")
    print("Usage: python run.py <strategy>")
    print("       python run.py rl --train")
    print("       python run.py rl --train --dashboard")


async def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading")
    parser.add_argument(
        "strategy", nargs="?", choices=ALL_STRATEGIES, help="Strategy to run"
    )
    parser.add_argument(
        "--train", action="store_true", help="Enable training mode for RL"
    )
    parser.add_argument("--size", type=float, default=10.0, help="Trade size in $")
    parser.add_argument("--load", type=str, help="Load RL model from file")
    parser.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Dashboard port")

    args = parser.parse_args()

    if not args.strategy:
        show_strategy_list_and_usage()
        return

    strategy = registry.create(args.strategy)

    # Setup ML-based strategy
    if isinstance(strategy, MLStrategy):
        if args.load:
            strategy.load(args.load)
            print(f"Loaded model from {args.load}")
        if args.train:
            strategy.train()
            print("Training mode active")
        else:
            strategy.eval()

    engine = TradingEngine(strategy, trade_size=args.size)
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
