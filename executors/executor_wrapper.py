"""
Wrapper for SimulatedOrderExecutor to work with Gym environment.

This module bridges the existing SimulatedOrderExecutor with the new
Gym environment interface, allowing seamless integration without
changing the existing simulator code.
"""

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Optional
import numpy as np
from executors.paper_executor import SimulatedOrderExecutor
from executors.executor_config import load_executor_config as _load_executor_config
from environments.trading_gym import OrderExecutor, TradingAction, ExecutionResult
from structures.action import Action
from structures.position import Position
from features.computer import (
    PositionState,
    TransactionState,
    CapitalState,
    RawMarketData,
)

_cfg = _load_executor_config()
_sp = _cfg.get("spread", {})
_MAX_EXIT_SLIPPAGE_BPS: float = _sp.get("max_exit_slippage_bps", 10.0)
_EXIT_SPREAD_SLIPPAGE_WEIGHT: float = _sp.get("exit_spread_slippage_weight", 0.5)
_EXIT_SIZE_SLIPPAGE_WEIGHT: float = _sp.get("exit_size_slippage_weight", 0.3)
_STANDARD_SPREAD: float = _sp.get("standard_spread", 0.02)


class GymExecutorWrapper(OrderExecutor):
    """
    Wraps SimulatedOrderExecutor for Gym environment interface.

    This wrapper:
    1. Translates Gym actions to executor API calls
    2. Tracks position state for feature computation
    3. Manages transaction status (pending, failed, etc.)
    4. Provides state getters for FeatureComputer
    """

    def __init__(self, default_order_size: float = 1.0):
        """
        Initialize wrapper.

        Args:
            default_order_size: Default trade size in USD
        """
        self.executor = SimulatedOrderExecutor()
        self.default_order_size = default_order_size

        # Position tracking
        self.current_position: Optional[Position] = None

        # Transaction state
        self.pending_order = False
        self.failed_order = False
        self.consecutive_failures = 0

    def reset(self, balance: float):
        """Reset executor state for new episode."""
        self.executor.reset(new_balance=balance)
        self.current_position = None
        self.pending_order = False
        self.failed_order = False
        self.consecutive_failures = 0

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
                    pnl=-2.0,  # Penalty for redundant action
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

                # Close succeeded - try to open new position
                if action_idx == Action.BUY:  # BUY_UP
                    open_result = self._open_position("BUY", market_data, action.size)
                elif action_idx == Action.SELL:  # SELL_DOWN (buy DOWN token)
                    open_result = self._open_position("SELL", market_data, action.size)
                else:
                    raise ValueError(f"Invalid action: {action_idx}")

                # If open failed, preserve close PnL in the result
                if not open_result.filled:
                    return ExecutionResult(
                        success=False,
                        filled=False,
                        balance=open_result.balance,
                        position=None,  # Position was closed
                        pnl=close_result.pnl,  # Preserve realized PnL from close
                        fee=close_result.fee,
                        slippage=close_result.slippage,
                        amount_spent=0.0,
                        rejection_reason=open_result.rejection_reason,
                    )

                # Both close and open succeeded - combine PnLs
                return ExecutionResult(
                    success=True,
                    filled=True,
                    balance=open_result.balance,
                    position=open_result.position,
                    pnl=close_result.pnl,  # Return close PnL (open PnL is 0 anyway)
                    fee=close_result.fee + open_result.fee,
                    slippage=close_result.slippage + open_result.slippage,
                    amount_spent=open_result.amount_spent,
                )

        # Open new position (no existing position to close)
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

            # TODO Maybe get the number of shares directly from the executor
            shares = size / result["fill_price"]
            self.current_position = Position(
                side="UP" if side == "BUY" else "DOWN",
                entry_price=result["fill_price"],
                shares=shares,
                asset=market_data.asset,
            )
            self.consecutive_failures = 0

            fee = self.executor.calculate_fee_per_share(result["fill_price"]) * shares
            return ExecutionResult(
                success=True,
                filled=True,
                balance=result["balance_remaining"],
                position=self.current_position,
                pnl=0.0,  # No realized PnL yet
                fee=fee,
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
        else:  # DOWN
            # Selling DOWN tokens - get bid price for DOWN (which is 1 - ask_up)
            exit_price = (1 - market_data.orderbook.best_ask) if market_data.orderbook.best_ask is not None else (1 - market_data.prob_up)

        # Market-impact slippage: mirrors entry model in paper_executor.
        # spread_factor amplifies impact in illiquid markets; size_factor
        # amplifies impact for larger positions.
        effective_spread = market_data.orderbook.spread or _STANDARD_SPREAD
        spread_factor = effective_spread / self.executor.standard_spread
        exit_size = self.current_position.entry_value
        size_factor = exit_size / self.executor.gtc_typical_order_size
        slippage_bps = (
            np.random.uniform(0, _MAX_EXIT_SLIPPAGE_BPS)
            * (1 + _EXIT_SPREAD_SLIPPAGE_WEIGHT * spread_factor)
            * (1 + _EXIT_SIZE_SLIPPAGE_WEIGHT * size_factor)
        )
        slippage = -(exit_price * slippage_bps / 10000)

        # exit_price cant be zero
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

        # Add proceeds to balance via executor's public interface
        self.executor.realize_pnl(net_proceeds, fee=fee)

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

    def compute_position_state(self, market_data: RawMarketData, time_remaining: float) -> PositionState:
        """
        Compute position state with current market data.

        Args:
            market_data: Current market data with orderbook prices
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

        # Compute unrealized P&L using actual exit price (what we'd get if we closed now)
        # This properly accounts for bid-ask spread
        if self.current_position.side == "UP":
            # For UP position, we'd sell at the bid price
            exit_price = market_data.orderbook.best_bid if market_data.orderbook.best_bid is not None else market_data.prob_up
        else:  # DOWN
            # For DOWN position, we'd sell at (1 - ask)
            exit_price = (1 - market_data.orderbook.best_ask) if market_data.orderbook.best_ask is not None else (1 - market_data.prob_up)

        unrealized_pnl = self.current_position.compute_pnl(exit_price)

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
