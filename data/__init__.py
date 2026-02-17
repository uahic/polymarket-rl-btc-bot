"""
Data layer for trading bot.

Provides unified interface for both historical and live data sources.
"""

from .sources import DataSource, HistoricalSource, LiveSource

__all__ = ["DataSource", "HistoricalSource", "LiveSource"]
