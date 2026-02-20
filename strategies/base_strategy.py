"""Base class for all trading strategies."""

import sys
import re
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from structures.market import MarketState
from structures.action import Action


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategy implementations should inherit from this class
    and will be automatically discovered and registered.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the strategy.

        Args:
            config: Optional configuration dictionary for the strategy
        """
        self.config = config or {}
        self.name = self.__class__.__name__

    @property
    def MODEL_NAME(self) -> str:
        """
        Get the model name for config loading and checkpoint validation.

        Automatically derived from class name by converting CamelCase to snake_case.
        Can be overridden by defining MODEL_NAME as a class variable.

        Examples:
            PPOStrategyV2 -> ppo_strategy_v2
            DebugFeatures -> debug_features
            MyCustomModel -> my_custom_model

        Returns:
            Model name in snake_case
        """
        # Check if class has explicitly defined MODEL_NAME as a class variable
        cls = self.__class__
        if 'MODEL_NAME' in cls.__dict__ and not callable(cls.__dict__['MODEL_NAME']):
            return cls.__dict__['MODEL_NAME']

        # Auto-derive from class name
        class_name = cls.__name__

        # Convert CamelCase to snake_case
        # Insert underscore before uppercase letters (except at start)
        snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', class_name)
        # Handle sequences of capitals (like "PPO" -> "ppo", "V2" -> "v2")
        snake_case = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', snake_case)

        return snake_case.lower()

    @abstractmethod
    def act(self, state: MarketState) -> Action:
        """Select action given current state."""
        pass

    def reset(self) -> None:
        """Reset any internal state (called between episodes/markets)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
