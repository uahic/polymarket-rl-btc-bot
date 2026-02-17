"""Test script to verify strategy auto-registration works."""

import logging
from strategies import registry, BaseStrategy, STRATEGIES

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Strategy Auto-Registration Test")
    logger.info("=" * 60)
    logger.info()

    # Show registry info
    logger.info(f"Registry: {registry}")
    logger.info()

    # List all discovered strategies
    logger.info(f"Discovered {len(registry)} strategies:")
    for name in registry.list_names():
        logger.info(f"  - {name}")
    logger.info()

    # Show STRATEGIES dictionary
    logger.info("STRATEGIES dictionary:")
    for name, cls in STRATEGIES.items():
        logger.info(f"  {name}: {cls}")
    logger.info()

    # Test getting a strategy
    if 'ExampleStrategy' in registry:
        logger.info("Getting ExampleStrategy class:")
        strategy_class = registry.get('ExampleStrategy')
        logger.info(f"  Class: {strategy_class}")
        logger.info()

        # Test creating an instance
        logger.info("Creating ExampleStrategy instance:")
        strategy = registry.create('ExampleStrategy', config={'test': 'value'})
        logger.info(f"  Instance: {strategy}")
        logger.info()

        # Test executing the strategy
        logger.info("Executing strategy:")
        result = strategy.execute()
        logger.info(f"  Result: {result}")
        logger.info()

    # Test all strategies
    logger.info("Testing all strategies:")
    for name in registry.list_names():
        strategy = registry.create(name, config={'test_param': 123})
        result = strategy.execute()
        logger.info(f"  {name}: {result}")
    logger.info()

    logger.info("=" * 60)
    logger.info("All tests passed! ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
