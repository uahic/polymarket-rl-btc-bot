"""Test script to verify strategy auto-registration works."""

from strategies import registry, BaseStrategy, STRATEGIES


def main():
    print("=" * 60)
    print("Strategy Auto-Registration Test")
    print("=" * 60)
    print()

    # Show registry info
    print(f"Registry: {registry}")
    print()

    # List all discovered strategies
    print(f"Discovered {len(registry)} strategies:")
    for name in registry.list_names():
        print(f"  - {name}")
    print()

    # Show STRATEGIES dictionary
    print("STRATEGIES dictionary:")
    for name, cls in STRATEGIES.items():
        print(f"  {name}: {cls}")
    print()

    # Test getting a strategy
    if 'ExampleStrategy' in registry:
        print("Getting ExampleStrategy class:")
        strategy_class = registry.get('ExampleStrategy')
        print(f"  Class: {strategy_class}")
        print()

        # Test creating an instance
        print("Creating ExampleStrategy instance:")
        strategy = registry.create('ExampleStrategy', config={'test': 'value'})
        print(f"  Instance: {strategy}")
        print()

        # Test executing the strategy
        print("Executing strategy:")
        result = strategy.execute()
        print(f"  Result: {result}")
        print()

    # Test all strategies
    print("Testing all strategies:")
    for name in registry.list_names():
        strategy = registry.create(name, config={'test_param': 123})
        result = strategy.execute()
        print(f"  {name}: {result}")
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
