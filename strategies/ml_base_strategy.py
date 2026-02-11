from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .base_strategy import BaseStrategy


class MLStrategy(BaseStrategy, ABC):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.training = False

    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        pass

    def train(self):
        """Set to training mode."""
        self.training = True

    def eval(self):
        """Set to evaluation mode."""
        self.training = False
