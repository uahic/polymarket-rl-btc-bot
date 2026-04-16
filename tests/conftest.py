"""
Shared pytest fixtures for test suite.

Provides common objects and mocks used across tests.
"""

import pytest
import numpy as np
import time
from typing import List, Optional
from dataclasses import dataclass

# Import core classes
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.computer import (
    RawMarketData,
    OrderbookSnapshot,
    FuturesData,
    SpotData,
    PositionState,
    TransactionState,
    CapitalState,
)
from environments.trading_gym import ExecutionResult, TradingAction
from structures.action import Action


@pytest.fixture
def sample_orderbook():
    """Create a sample orderbook snapshot."""
    return OrderbookSnapshot(
        timestamp=time.time(),
        best_bid=0.48,
        best_ask=0.52,
        spread=0.04,
        bids_l5=[
            (0.48, 100.0),
            (0.47, 200.0),
            (0.46, 150.0),
            (0.45, 100.0),
            (0.44, 80.0),
        ],
        asks_l5=[
            (0.52, 100.0),
            (0.53, 200.0),
            (0.54, 150.0),
            (0.55, 100.0),
            (0.56, 80.0),
        ],
    )


@pytest.fixture
def sample_futures_data():
    """Create sample futures data."""
    return FuturesData(
        timestamp=time.time(),
        price=50000.0,
        returns_1m=0.001,
        returns_5m=0.005,
        returns_10m=0.01,
        cvd=1000.0,
        cvd_history=[900.0, 950.0, 1000.0],
        trade_flow_imbalance=0.1,
        trade_intensity=2.0,
        large_trade_flag=0.0,
        realized_vol_5m=0.02,
        avg_vol=0.02,
    )


@pytest.fixture
def sample_spot_data():
    """Create sample spot data."""
    return SpotData(
        timestamp=time.time(),
        price=50000.0,
        change_pct=0.005,
    )


@pytest.fixture
def sample_raw_market_data(sample_orderbook, sample_futures_data, sample_spot_data):
    """Create sample raw market data."""
    return RawMarketData(
        timestamp=time.time(),
        asset="BTC",
        orderbook=sample_orderbook,
        futures=sample_futures_data,
        spot=sample_spot_data,
        prob_up=0.50,
        time_remaining=0.75,
        prob_history=[0.48, 0.49, 0.50],
        vol_regime=0.0,
        trend_regime=1.0,
    )


@pytest.fixture
def sample_position_flat():
    """Create a flat position (no position)."""
    return PositionState(
        has_position=False,
        side=None,
        unrealized_pnl=0.0,
        time_remaining_normalized=0.75,
    )


@pytest.fixture
def sample_position_long():
    """Create a long position (UP)."""
    return PositionState(
        has_position=True,
        side="UP",
        unrealized_pnl=5.0,
        time_remaining_normalized=0.75,
    )


@pytest.fixture
def sample_position_short():
    """Create a short position (DOWN)."""
    return PositionState(
        has_position=True,
        side="DOWN",
        unrealized_pnl=-3.0,
        time_remaining_normalized=0.75,
    )


@pytest.fixture
def sample_transaction_clean():
    """Create clean transaction state (no pending/failed orders)."""
    return TransactionState(
        pending_order=False,
        failed_order=False,
        consecutive_failures=0,
        pending_order_age=0.0,
    )


@pytest.fixture
def sample_transaction_failed():
    """Create transaction state with failed orders."""
    return TransactionState(
        pending_order=False,
        failed_order=True,
        consecutive_failures=3,
        pending_order_age=0.0,
    )


@pytest.fixture
def sample_capital_state():
    """Create sample capital state."""
    return CapitalState(
        available_balance=1000.0,
        max_balance=1000.0,
    )


class MockOrderExecutor:
    """Mock order executor for testing."""

    def __init__(self, initial_balance: float = 1000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = None
        self.position_entry_price = None
        self.transaction_state = TransactionState()

    def reset(self, balance: float):
        """Reset executor."""
        self.balance = balance
        self.initial_balance = balance
        self.position = None
        self.position_entry_price = None
        self.transaction_state = TransactionState()

    def execute(self, action: TradingAction, market_data: RawMarketData) -> ExecutionResult:
        """Execute trading action (mock)."""
        # Simple mock: always succeed
        pnl = 0.0
        filled = False

        if action.action == 0:  # BUY
            if not self.position:
                self.position = "UP"
                self.position_entry_price = market_data.prob_up
                self.balance -= action.size
                filled = True
        elif action.action == 2:  # SELL
            if self.position == "UP":
                pnl = (market_data.prob_up - self.position_entry_price) * 10
                self.balance += pnl
                self.position = None
                self.position_entry_price = None
                filled = True

        return ExecutionResult(
            success=True,
            filled=filled,
            balance=self.balance,
            position=self.position,
            pnl=pnl,
            fee=0.0,
            slippage=0.0,
            amount_spent=action.size if filled else 0.0,
            rejection_reason=None,
        )

    def compute_position_state(self, market_data: RawMarketData, time_remaining: float) -> PositionState:
        """Compute current position state."""
        if self.position:
            # Use prob_up for mock (tests don't have realistic orderbook)
            unrealized_pnl = (market_data.prob_up - self.position_entry_price) * 10
            return PositionState(
                has_position=True,
                side=self.position,
                unrealized_pnl=unrealized_pnl,
                time_remaining_normalized=time_remaining,
            )
        else:
            return PositionState(
                has_position=False,
                side=None,
                unrealized_pnl=0.0,
                time_remaining_normalized=time_remaining,
            )

    def get_position_state(self) -> PositionState:
        """Get position state."""
        # Create a simple mock market data for the get_position_state call
        from features.computer import RawMarketData, OrderbookData, FuturesData
        mock_market_data = RawMarketData(
            asset="BTC",
            timestamp=0.0,
            prob_up=0.5,
            orderbook=OrderbookData(best_bid=None, best_ask=None, spread=None),
            futures=FuturesData(trade_flow_imbalance=0.0),
            time_remaining=0.5,
        )
        return self.compute_position_state(mock_market_data, 0.5)

    def get_transaction_state(self) -> TransactionState:
        """Get transaction state."""
        return self.transaction_state

    def get_capital_state(self) -> CapitalState:
        """Get capital state."""
        return CapitalState(
            available_balance=self.balance,
            max_balance=self.initial_balance,
        )


@pytest.fixture
def mock_executor():
    """Create a mock order executor."""
    return MockOrderExecutor()


class MockDataSource:
    """Mock data source for testing."""

    def __init__(self, episode_length: int = 100):
        self.episode_length = episode_length
        self.current_idx = 0
        self.episode = None

    def reset(self, **kwargs) -> RawMarketData:
        """Reset to new episode."""
        self.current_idx = 0
        self.episode = self._generate_episode()
        return self.episode[0]

    def get_current(self) -> RawMarketData:
        """Get current observation."""
        return self.episode[self.current_idx]

    def advance(self) -> bool:
        """Advance to next tick."""
        self.current_idx += 1
        return self.current_idx < len(self.episode)

    def is_done(self) -> bool:
        """Check if episode ended."""
        return self.current_idx >= len(self.episode)

    def _generate_episode(self) -> List[RawMarketData]:
        """Generate mock episode."""
        episode = []
        for i in range(self.episode_length):
            prob = 0.5 + 0.1 * np.sin(i / 10)
            raw_data = RawMarketData(
                timestamp=time.time() + i * 0.5,
                asset="BTC",
                orderbook=OrderbookSnapshot(
                    timestamp=time.time() + i * 0.5,
                    best_bid=prob - 0.02,
                    best_ask=prob + 0.02,
                    spread=0.04,
                    bids_l5=[(prob - 0.02, 100.0)],
                    asks_l5=[(prob + 0.02, 100.0)],
                ),
                futures=FuturesData(
                    timestamp=time.time() + i * 0.5,
                    price=50000.0,
                    returns_1m=0.001,
                    returns_5m=0.005,
                    returns_10m=0.01,
                    cvd=float(i * 10),
                    cvd_history=[float(max(0, i - 1) * 10), float(i * 10)],
                    trade_flow_imbalance=0.0,
                    trade_intensity=2.0,
                    large_trade_flag=0.0,
                    realized_vol_5m=0.02,
                    avg_vol=0.02,
                ),
                spot=SpotData(
                    timestamp=time.time() + i * 0.5,
                    price=50000.0,
                    change_pct=0.0,
                ),
                prob_up=prob,
                time_remaining=1.0 - (i / self.episode_length),
                prob_history=[prob] * min(i + 1, 50),
                vol_regime=0.0,
                trend_regime=0.0,
            )
            episode.append(raw_data)
        return episode


@pytest.fixture
def mock_data_source():
    """Create a mock data source."""
    return MockDataSource()
