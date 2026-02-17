"""
Wrapper for SimulatedOrderExecutor to work with Gym environment.

This module bridges the existing SimulatedOrderExecutor with the new
Gym environment interface, allowing seamless integration without
changing the existing simulator code.
"""

from typing import Optional
from executors.executor import SimulatedOrderExecutor as OldExecutor
from environments.trading_gym import OrderExecutor, TradingAction, ExecutionResult
from structures.action import Action
from features.computer import (
    PositionState,
    TransactionState,
    CapitalState,
    RawMarketData,
)


class Position:
    """Tracks current open position."""

    def __init__(self, side: str, entry_price: float, shares: float, asset: str):
        self.side = side  # "UP" or "DOWN"
        self.entry_price = entry_price
        self.shares = shares
        self.asset = asset
        self.entry_value = entry_price * shares

    def compute_pnl(self, current_price: float) -> float:
        """
        Compute unrealized P&L.

        For Polymarket:
        - UP token: profit when price increases
        - DOWN token: profit when price decreases

        Args:
            current_price: Current UP token probability (prob_up)
        """
        if self.side == "UP":
            # UP token value increases with probability
            current_value = current_price * self.shares
            return current_value - self.entry_value
        else:  # DOWN
            # DOWN token value is inverse
            # When we bought DOWN, we paid (1 - up_prob) per share
            # Current value is (1 - current_up_prob) per share
            current_down_price = 1.0 - current_price
            current_value = current_down_price * self.shares
            return current_value - self.entry_value


class GymExecutorWrapper(OrderExecutor):
    """
    Wraps SimulatedOrderExecutor for Gym environment interface.

    This wrapper:
    1. Translates Gym actions to executor API calls
    2. Tracks position state for feature computation
    3. Manages transaction status (pending, failed, etc.)
    4. Provides state getters for FeatureComputer
    """

    def __init__(self, default_order_size: float = 10.0):
        """
        Initialize wrapper.

        Args:
            default_order_size: Default trade size in USD
        """
        self.executor = OldExecutor()
        self.default_order_size = default_order_size

        # Position tracking
        self.current_position: Optional[Position] = None

        # Transaction state
        self.pending_order = False
        self.failed_order = False
        self.consecutive_failures = 0
        self.last_fill_result = None

        # For time remaining calculation
        self.position_entry_time = 0.0
        self.episode_duration = 900.0  # 15 min

    def reset(self, balance: float):
        """Reset executor state for new episode."""
        self.executor.reset(new_balance=balance)
        self.current_position = None
        self.pending_order = False
        self.failed_order = False
        self.consecutive_failures = 0
        self.last_fill_result = None
        self.position_entry_time = 0.0

    def execute(self, action: TradingAction, market_data: RawMarketData) -> ExecutionResult:
        """
        Execute trading action.

        Args:
            action: Trading action (0=BUY_UP, 1=HOLD, 2=SELL_DOWN)
            market_data: Current market state

        Returns:
            ExecutionResult with fill information
        """
        # Reset transaction flags
        self.pending_order = False
        self.failed_order = False

        # Map action to trading decision
        action_idx = action.action

        if action_idx == Action.HOLD:  # HOLD
            # No action
            return self._create_hold_result()

        # Determine desired position side
        desired_side = "UP" if action_idx == Action.BUY else "DOWN"

        # Check if we already have a position in the SAME direction
        if self.current_position is not None:
            if self.current_position.side == desired_side:
                # Same direction order - ignore and penalize
                return ExecutionResult(
                    success=False,
                    filled=False,
                    balance=self.executor.get_balance(),
                    position=self.current_position,
                    pnl=-0.5,  # Penalty for redundant action
                    fee=0.0,
                    slippage=0.0,
                    amount_spent=0.0,
                    rejection_reason=f"redundant_{desired_side.lower()}_order",
                )
            else:
                # Different direction - close existing position first
                close_result = self._close_position(market_data)

                # If close failed, don't open new position
                if not close_result.success:
                    self.failed_order = True
                    self.consecutive_failures += 1
                    return close_result

        # Open new position
        if action_idx == Action.BUY:  # BUY_UP
            return self._open_position("BUY", market_data, action.size)
        elif action_idx == Action.SELL:  # SELL_DOWN (buy DOWN token)
            return self._open_position("SELL", market_data, action.size)
        else:
            raise ValueError(f"Invalid action: {action_idx}")

    def _open_position(self, side: str, market_data: RawMarketData, size: float) -> ExecutionResult:
        """
        Open new position.

        Args:
            side: "BUY" (UP token) or "SELL" (DOWN token)
            market_data: Current market state
            size: Trade size in USD

        Returns:
            ExecutionResult
        """
        # Simulate order fill using existing executor
        result = self.executor.simulate_order_fill(
            side=side,
            asset=market_data.asset,
            size=size,
            current_prob=market_data.prob_up,
            current_bid=market_data.orderbook.best_bid,
            current_ask=market_data.orderbook.best_ask,
            spread=market_data.orderbook.spread,
            order_book_imbalance=market_data.futures.trade_flow_imbalance,
            order_type="GTC",
        )

        if result["filled"]:
            # Create position
            shares = size / result["fill_price"]
            self.current_position = Position(
                side="UP" if side == "BUY" else "DOWN",
                entry_price=result["fill_price"],
                shares=shares,
                asset=market_data.asset,
            )
            self.position_entry_time = market_data.timestamp
            self.consecutive_failures = 0

            return ExecutionResult(
                success=True,
                filled=True,
                balance=result["balance_remaining"],
                position=self.current_position,
                pnl=0.0,  # No realized PnL yet
                fee=size * self.executor.calculate_fee_per_share(result["fill_price"]),
                slippage=result["slippage"],
                amount_spent=size,
            )
        else:
            # Order rejected
            self.failed_order = True
            self.consecutive_failures += 1

            return ExecutionResult(
                success=False,
                filled=False,
                balance=result["balance_remaining"],
                position=self.current_position,
                pnl=0.0,
                fee=0.0,
                slippage=0.0,
                amount_spent=0.0,
                rejection_reason=result["reason"],
            )

    def _close_position(self, market_data: RawMarketData) -> ExecutionResult:
        """
        Close current position by selling shares.

        Args:
            market_data: Current market state

        Returns:
            ExecutionResult with realized P&L
        """
        if self.current_position is None:
            return self._create_hold_result()

        # Calculate exit price based on market conditions
        # When selling, we get the bid price (if UP) or 1-ask (if DOWN)
        if self.current_position.side == "UP":
            # Selling UP tokens - get bid price
            exit_price = market_data.orderbook.best_bid if market_data.orderbook.best_bid is not None else market_data.prob_up
            # Apply slippage (worse price when selling)
            slippage = -0.0001 * market_data.orderbook.spread if market_data.orderbook.spread else -0.0001
            exit_price = max(0.01, exit_price + slippage)
        else:  # DOWN
            # Selling DOWN tokens - get bid price for DOWN (which is 1 - ask_up)
            exit_price = (1 - market_data.orderbook.best_ask) if market_data.orderbook.best_ask is not None else (1 - market_data.prob_up)
            slippage = -0.0001 * market_data.orderbook.spread if market_data.orderbook.spread else -0.0001
            exit_price = max(0.01, exit_price + slippage)

        # Calculate proceeds from selling shares
        exit_value = exit_price * self.current_position.shares
        entry_value = self.current_position.entry_value

        # Calculate fee for selling
        fee = self.executor.calculate_fee_per_share(exit_price) * self.current_position.shares

        # Net proceeds after fees
        net_proceeds = exit_value - fee

        # Realized P&L = net proceeds - what we originally paid
        pnl = net_proceeds - entry_value

        # Add proceeds to balance
        self.executor.balance += net_proceeds
        self.executor.total_fees_paid += fee

        # Clear position
        self.current_position = None
        self.consecutive_failures = 0

        return ExecutionResult(
            success=True,
            filled=True,
            balance=self.executor.get_balance(),
            position=None,
            pnl=pnl,
            fee=fee,
            slippage=slippage,
            amount_spent=0.0,  # Closing position - we receive money
        )

    def _create_hold_result(self) -> ExecutionResult:
        """Create result for HOLD action."""
        return ExecutionResult(
            success=True,
            filled=False,
            balance=self.executor.get_balance(),
            position=self.current_position,
            pnl=0.0,
            fee=0.0,
            slippage=0.0,
            amount_spent=0.0,
        )

    def get_position_state(self) -> PositionState:
        """Get current position state for feature computation."""
        if self.current_position is None:
            return PositionState(
                has_position=False,
                side=None,
                unrealized_pnl=0.0,
                time_remaining_normalized=0.0,
            )

        # Note: Need current price to compute unrealized PnL
        # This will be provided by FeatureComputer with market data
        # For now, return zero (will be computed in _get_observation)
        return PositionState(
            has_position=True,
            side=self.current_position.side,
            unrealized_pnl=0.0,  # Computed by environment with current price
            time_remaining_normalized=0.0,  # Computed by environment
        )

    def compute_position_state(self, current_price: float, time_remaining: float) -> PositionState:
        """
        Compute position state with current market data.

        Args:
            current_price: Current market price (prob_up)
            time_remaining: Time remaining in episode [0, 1]

        Returns:
            PositionState with computed unrealized PnL
        """
        if self.current_position is None:
            return PositionState(
                has_position=False,
                side=None,
                unrealized_pnl=0.0,
                time_remaining_normalized=time_remaining,
            )

        # Compute unrealized P&L
        unrealized_pnl = self.current_position.compute_pnl(current_price)

        return PositionState(
            has_position=True,
            side=self.current_position.side,
            unrealized_pnl=unrealized_pnl,
            time_remaining_normalized=time_remaining,
        )

    def get_transaction_state(self) -> TransactionState:
        """Get transaction state for feature computation."""
        return TransactionState(
            pending_order=self.pending_order,
            failed_order=self.failed_order,
            consecutive_failures=self.consecutive_failures,
        )

    def get_capital_state(self) -> CapitalState:
        """Get capital state for feature computation."""
        return CapitalState(
            available_balance=self.executor.get_balance(),
            max_balance=self.executor.initial_balance,
        )

    def get_statistics(self):
        """Get execution statistics."""
        return self.executor.get_statistics()
