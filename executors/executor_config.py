"""Shared config loader for executor modules."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "paper_executor_config.yaml"


def load_executor_config() -> Dict[str, Any]:
    """Load paper executor config from yaml, falling back to hardcoded defaults."""
    try:
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(
            f"paper_executor_config.yaml not found at {_CONFIG_PATH}, using defaults"
        )
        return {}
