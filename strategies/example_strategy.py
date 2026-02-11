"""
Example strategy to demonstrate auto-registration.

This file can be deleted once you create your own strategies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_strategy import BaseStrategy
from structures.market import MarketState
from structures.action import Action


class ExampleStrategy(BaseStrategy):
    """
    Example strategy implementation.

    This class demonstrates how to create a strategy that will be
    automatically discovered and registered by the system.
    """

    def act(self, state: MarketState) -> Action:
        """Select action given current state."""
        # Example: Always buy with small size
        return Action(side='buy', size=0.1)

    def execute(self, market_data=None):
        """
        Execute the example strategy.

        Args:
            market_data: Optional market data to process

        Returns:
            A simple message indicating the strategy executed
        """
        return f"{self.name} executed with config: {self.config}"


class AnotherExampleStrategy(BaseStrategy):
    """Another example strategy to show multiple auto-registration."""

    def act(self, state: MarketState) -> Action:
        """Select action given current state."""
        # Example: Always sell with small size
        return Action(side='sell', size=0.1)

    def execute(self, *args, **kwargs):
        """Execute the strategy."""
        return f"{self.name} is running!"
