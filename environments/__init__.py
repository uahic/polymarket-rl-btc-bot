"""
Gym environments for trading bot.

Provides unified Gymnasium environment that works with both:
- Historical data (offline training)
- Live data (real-time trading)
"""

from .trading_gym import TradingGym

__all__ = ["TradingGym"]
