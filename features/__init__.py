"""
Feature computation for trading bot.
Provides unified preprocessing for both historical and live data.
"""

from .computer import FeatureComputer, RawMarketData

__all__ = ["FeatureComputer", "RawMarketData"]
