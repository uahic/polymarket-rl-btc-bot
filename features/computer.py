"""
Unified feature computation for trading bot.

This module provides the SINGLE SOURCE OF TRUTH for all feature engineering.
The same FeatureComputer is used for both:
- Historical data (offline training)
- Live data (real-time trading)

This guarantees that preprocessing is identical across training and deployment.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from features.orderbook_utils import compute_orderbook_imbalance_l1, compute_orderbook_imbalance_l5
from features.feature_registry import FeatureConfig, FeatureRegistry


@dataclass
class OrderbookSnapshot:
    """Orderbook state at a point in time."""
    timestamp: float
    best_bid: float
    best_ask: float
    spread: float
    bids_l5: List[tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks_l5: List[tuple[float, float]] = field(default_factory=list)

    @property
    def mid_price(self) -> float:
        """Midpoint price."""
        return (self.best_bid + self.best_ask) / 2 if self.best_bid and self.best_ask else 0.0


@dataclass
class FuturesData:
    """Binance futures market data."""
    timestamp: float
    price: float
    returns_1m: float = 0.0
    returns_5m: float = 0.0
    returns_10m: float = 0.0
    cvd: float = 0.0  # Cumulative volume delta
    cvd_history: List[float] = field(default_factory=list)  # For acceleration
    trade_flow_imbalance: float = 0.0  # Recent buy vs sell pressure
    trade_intensity: float = 0.0  # Trades per second
    large_trade_flag: float = 0.0  # Binary: large trade detected
    realized_vol_5m: float = 0.0
    avg_vol: float = 0.0  # For vol expansion calculation


@dataclass
class SpotData:
    """Binance spot price data."""
    timestamp: float
    price: float
    change_pct: float = 0.0  # % change from reference point


@dataclass
class PositionState:
    """Current position state (depends on agent's actions)."""
    has_position: bool = False
    side: Optional[str] = None  # "UP" or "DOWN"
    unrealized_pnl: float = 0.0
    time_remaining_normalized: float = 0.0  # [0, 1]


@dataclass
class TransactionState:
    """Transaction/order state (depends on agent's actions)."""
    pending_order: bool = False
    failed_order: bool = False
    consecutive_failures: int = 0
    pending_order_age: float = 0.0


@dataclass
class CapitalState:
    """Capital management state."""
    available_balance: float = 1000.0
    max_balance: float = 1000.0


@dataclass
class RawMarketData:
    """
    Raw market data input to feature computer.

    This is the interface between data sources and feature computation.
    Both HistoricalSource and LiveSource must provide this structure.
    """
    timestamp: float
    asset: str
    orderbook: OrderbookSnapshot
    futures: FuturesData
    spot: SpotData

    # Market-specific
    prob_up: float  # Polymarket probability for UP outcome
    time_remaining: float  # Fraction of episode remaining [0, 1]

    # Historical context (for velocity/volatility features)
    prob_history: List[float] = field(default_factory=list)

    # Optional regime features
    vol_regime: float = 0.0  # 0 or 1
    trend_regime: float = 0.0  # 0 or 1


class FeatureComputer:
    """
    Stateless feature computation.

    This class is the SINGLE SOURCE OF TRUTH for feature engineering.
    It computes all 26 features from raw market data in a deterministic,
    reproducible way.

    Features are identical whether computed from:
    - Historical data during offline training
    - Live data during real-time trading

    The feature computation logic is extracted from structures/market.py
    and refactored to be stateless and testable.
    """

    def __init__(self, feature_config: FeatureConfig):
        """Initialize feature computer with configuration."""
        self.feature_config = feature_config

        # Feature normalization constants (tuned from historical data)
        self.returns_scale = 50.0  # Typical returns: -0.02 to 0.02
        self.cvd_accel_scale = 10.0  # CVD acceleration is small
        self.spread_scale = 20.0  # Spread ~0-5%
        self.trade_intensity_scale = 10.0  # Typical max intensity
        self.vol_scale = 20.0  # Volatility ~0-5%
        self.pnl_scale = 50.0  # Typical PnL range
        self.failures_scale = 5.0  # Max expected failures
        self.balance_scale = 1000.0  # Typical bankroll

        # Preallocated output buffer — avoids heap allocation on every compute_features() call
        self._feature_buf = np.zeros(26, dtype=np.float32)

    def compute_features(
        self,
        raw_data: RawMarketData,
        position: PositionState,
        transaction: TransactionState,
        capital: CapitalState,
    ) -> np.ndarray:
        """
        Compute all 26 features from raw market data.

        Args:
            raw_data: Raw market observations
            position: Current position state (agent-dependent)
            transaction: Transaction status (agent-dependent)
            capital: Capital management state (agent-dependent)

        Returns:
            26-dimensional feature vector, normalized to [-1, 1]

        Features (in order):
            1-3:   Ultra-short momentum (returns_1m, 5m, 10m)
            4-7:   Order flow (OB imbalance L1, L5, trade flow, CVD accel)
            8-10:  Microstructure (spread, trade intensity, large trade flag)
            11-12: Volatility (realized 5m, vol expansion)
            13-16: Position (has_position, side, PnL, time remaining)
            17-18: Regime (vol regime, trend regime)
            19-21: Transaction status (pending, failed, consecutive failures)
            22:    Capital (available balance)
            23-26: Time-of-day encoding (hour_sin, hour_cos, dow_sin, dow_cos)
        """

        # Compute derived features
        ob_imbalance_l1 = compute_orderbook_imbalance_l1(raw_data.orderbook.bids_l5, raw_data.orderbook.asks_l5) # l1 be filtered out
        ob_imbalance_l5 = compute_orderbook_imbalance_l5(raw_data.orderbook.bids_l5, raw_data.orderbook.asks_l5)
        spread_pct = self._compute_spread_pct(raw_data.orderbook, raw_data.prob_up)
        velocity = self._compute_velocity(raw_data.prob_history, raw_data.prob_up, window=3)
        vol_5m = self._compute_volatility(raw_data.prob_history, window=30)

        # Helper to clamp values to [-1, 1]
        def clamp(x: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
            return max(min_val, min(max_val, x))

        # Time-of-day cyclical encoding from Unix timestamp
        t = raw_data.timestamp
        hour = (t % 86400) / 3600       # hour of day in [0, 24)
        dow = (t // 86400) % 7          # day of week in [0, 7)
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin  = math.sin(2 * math.pi * dow / 7)
        dow_cos  = math.cos(2 * math.pi * dow / 7)

        f = self._feature_buf
        # 1-3: Ultra-short momentum
        f[0] = clamp(raw_data.futures.returns_1m * self.returns_scale)
        f[1] = clamp(raw_data.futures.returns_5m * self.returns_scale)
        f[2] = clamp(raw_data.futures.returns_10m * self.returns_scale)
        # 4-7: Order flow
        f[3] = clamp(ob_imbalance_l1)
        f[4] = clamp(ob_imbalance_l5)
        f[5] = clamp(raw_data.futures.trade_flow_imbalance)
        f[6] = clamp(self._compute_cvd_acceleration(raw_data.futures.cvd_history) * self.cvd_accel_scale)
        # 8-10: Microstructure
        f[7] = clamp(spread_pct * self.spread_scale)
        f[8] = clamp(raw_data.futures.trade_intensity / self.trade_intensity_scale)
        f[9] = raw_data.futures.large_trade_flag
        # 11-12: Volatility
        f[10] = clamp(vol_5m * self.vol_scale)
        f[11] = clamp(raw_data.futures.avg_vol / max(0.001, raw_data.futures.realized_vol_5m))
        # 13-16: Position
        f[12] = float(position.has_position)
        f[13] = 1.0 if position.side == "UP" else (-1.0 if position.side == "DOWN" else 0.0)
        f[14] = clamp(position.unrealized_pnl / self.pnl_scale)
        f[15] = position.time_remaining_normalized
        # 17-18: Regime
        f[16] = raw_data.vol_regime
        f[17] = raw_data.trend_regime
        # 19-21: Transaction status
        f[18] = float(transaction.pending_order)
        f[19] = float(transaction.failed_order)
        f[20] = clamp(transaction.consecutive_failures / self.failures_scale)
        # 22: Capital management
        f[21] = clamp(capital.available_balance / self.balance_scale)
        # 23-26: Time-of-day encoding
        f[22] = hour_sin
        f[23] = hour_cos
        f[24] = dow_sin
        f[25] = dow_cos

        # Apply feature selection based on config
        if self.feature_config.input_mode == "auto_adjust":
            # Return only enabled features
            enabled_indices = self.feature_config.get_enabled_indices()
            return f[enabled_indices].copy()
        else:  # zero_fill
            # Return all 26, but zero out disabled
            result = f.copy()
            for feat in FeatureRegistry.FEATURES:
                if not self.feature_config.enabled_features[feat.name]:
                    result[feat.index] = 0.0
            return result

    def _compute_spread_pct(self, orderbook: OrderbookSnapshot, prob: float) -> float:
        """
        Compute spread as percentage of price.

        Spread % = spread / mid_price
        """
        if prob <= 0:
            return 0.0

        return orderbook.spread / max(0.01, prob)

    def _compute_cvd_acceleration(self, cvd_history: List[float]) -> float:
        """
        Compute CVD acceleration (second derivative).

        Acceleration = CVD change rate over recent window
        """
        if len(cvd_history) < 2:
            return 0.0

        # Simple finite difference
        recent_change = cvd_history[-1] - cvd_history[-2] if len(cvd_history) >= 2 else 0.0
        return recent_change

    def _compute_velocity(self, prob_history: List[float], current_prob: float, window: int = 5) -> float:
        """
        Compute probability velocity (first derivative).

        Velocity = prob change over last N ticks
        """
        if len(prob_history) < window:
            return 0.0

        return current_prob - prob_history[-window]

    def _compute_volatility(self, prob_history: List[float], window: int = 10) -> float:
        """
        Compute rolling volatility of probability.

        Volatility = std(prob) over window
        """
        if len(prob_history) < window:
            return 0.0

        recent = prob_history[-window:]
        return float(np.std(recent))

