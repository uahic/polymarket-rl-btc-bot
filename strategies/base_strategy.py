"""Base class for all trading strategies."""

import sys
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

    @abstractmethod
    def act(self, state: MarketState) -> Action:
        """Select action given current state."""
        pass

    def reset(self):
        """Reset any internal state (called between episodes/markets)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
