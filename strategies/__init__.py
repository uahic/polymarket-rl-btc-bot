"""
Trading strategies package with auto-discovery and registration.

This package automatically discovers and registers all strategy classes
that inherit from BaseStrategy. To create a new strategy:

1. Create a new Python file in this directory (e.g., my_strategy.py)
2. Import BaseStrategy: from .base import BaseStrategy
3. Create a class that inherits from BaseStrategy
4. Implement the required execute() method

The strategy will be automatically discovered and available through the registry.

Example:
    from strategies import registry, BaseStrategy

    # List all available strategies
    print(registry.list_names())

    # Get a specific strategy class
    MyStrategy = registry.get('MyStrategy')

    # Create an instance
    strategy = registry.create('MyStrategy', config={'key': 'value'})

    # Or instantiate directly
    strategy = MyStrategy(config={'key': 'value'})
"""

from .base_strategy import BaseStrategy
from .ml_base_strategy import MLStrategy
from .registry import StrategyRegistry

# Create global registry instance
registry = StrategyRegistry()

# Auto-discover all strategies in this package
# This will find all classes that inherit from BaseStrategy
ALL_STRATEGIES = registry.discover(BaseStrategy, __name__)

# Export public API
__all__ = [
    'BaseStrategy',
    'registry',
    'ALL_STRATEGIES',
]
