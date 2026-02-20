from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

# import sys
import numpy as np

# sys.path.insert(0, str(Path(__file__).parent))
from .base_strategy import BaseStrategy, Action


class MLStrategy(BaseStrategy, ABC):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.training = False

    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        """Train the model"""
        pass

    @abstractmethod
    def should_update(self) -> bool:
        """Signals teh Trading Runner if an update is necessary
        Is called when self.training = True"""
        pass

    @abstractmethod
    def store(
        self,
        features: np.ndarray,
        action: Action,
        reward: float,
        next_features: np.ndarray,
        done: bool,
    ) -> None:
        """Store recent experiences"""
        pass

    def train(self):
        """Set to training mode."""
        self.training = True

    def eval(self):
        """Set to evaluation mode."""
        self.training = False
