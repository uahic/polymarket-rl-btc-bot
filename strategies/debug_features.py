import torch
import numpy as np
import torch.nn as nn
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from .base_strategy import BaseStrategy, MarketState, Action
from .ml_base_strategy import MLStrategy

import logging

logger = logging.getLogger(__name__)


FEATURE_LABELS = [
    # Ultra-short momentum
    "returns_1m",
    "returns_5m",
    "returns_10m",
    # Order flow
    "ob_imbalance_l1",
    "ob_imbalance_l5",
    "trade_flow_imbalance",
    "cvd_acceleration",
    # Microstructure
    "spread_pct",
    "trade_intensity",
    "large_trade_flag",
    # Volatility
    "realized_vol_5m",
    "vol_expansion",
    # Position state
    "has_position",
    "position_side",
    "unrealized_pnl",
    "time_remaining_normalized",
    # Regime
    "vol_regime",
    "trend_regime",
    # Transaction status
    "pending_order",
    "failed_order",
    "consecutive_failures",
    # Capital
    "available_balance",
    # Time-of-day encoding
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Previous action one-hot (PPO extension)
    "prev_action_BUY_UP",
    "prev_action_HOLD",
    "prev_action_SELL_DOWN",
]


def _format_features(arr: np.ndarray) -> str:
    lines = []
    for i, val in enumerate(arr):
        label = FEATURE_LABELS[i] if i < len(FEATURE_LABELS) else f"feature_{i}"
        lines.append(f"  {label:<30} {val:+.6f}")
    return "\n".join(lines)


@dataclass
class Experience:
    features: np.ndarray
    action: Action
    reward: float
    next_features: np.ndarray
    done: bool

    def __str__(self) -> str:
        sep = "-" * 50
        return (
            f"{sep}\n"
            f"Experience\n"
            f"{sep}\n"
            f"action : {self.action.name}\n"
            f"reward : {self.reward:+.6f}\n"
            f"done   : {self.done}\n"
            f"\nfeatures ({len(self.features)}):\n"
            f"{_format_features(self.features)}\n"
            f"\nnext_features ({len(self.next_features)}):\n"
            f"{_format_features(self.next_features)}\n"
            f"{sep}"
        )


def _make_file_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_logger = logging.getLogger("debug_features.experience")
    file_logger.setLevel(logging.DEBUG)
    if not file_logger.handlers:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s\n%(message)s\n"))
        file_logger.addHandler(handler)
    file_logger.propagate = False
    return file_logger


_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "debug_features.log"


class DebugFeatures(MLStrategy):
    def __init__(self):
        super().__init__("debug_features")
        self.buffer_size: int = 2048
        self.experiences: deque = deque(maxlen=self.buffer_size)
        self._file_logger = _make_file_logger(_LOG_PATH)

    def act(self, features: np.ndarray) -> Action:
        logger.debug("Calling act()")
        return Action.HOLD

    def store(
        self,
        features: np.ndarray,
        action: Action,
        reward: float,
        next_features: np.ndarray,
        done: bool,
    ) -> None:

        exp = Experience(features, action, reward, next_features, done)
        self.experiences.append(exp)

        self._file_logger.debug(str(exp))

    def should_update(self) -> bool:
        logger.debug("Calling should_update() = True")
        return True

    def update(self) -> Optional[Dict[str, float]]:
        logger.debug("Calling update()")

    def reset(self):
        logger.debug("Calling reset()")

    def save(self, path: str):
        logger.debug("Calling save()")

    def load(self, path: str):
        logger.debug("Calling load()")
