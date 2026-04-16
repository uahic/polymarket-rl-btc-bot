"""
Live order executor for real CLOB trading via Gym interface.

This executor wraps AsyncClobClient to provide gym-compatible order execution
with asynchronous fire-and-forget order submission and status tracking.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from environments.trading_gym import OrderExecutor, TradingAction, ExecutionResult
from structures.action import Action
from structures.position import ExtendedPosition
from features.computer import (
    PositionState,
    TransactionState,
    CapitalState,
    RawMarketData,
)
from py_clob_client.clob_types import OrderArgs, CreateOrderOptions
from py_clob_client.order_builder.constants import BUY as ORDER_BUY, SELL as ORDER_SELL


@dataclass
class PendingOrder:
    """Tracks pending order state."""
    task: asyncio.Task
    action: TradingAction
    timestamp: float
    order_id: Optional[str] = None
    side: Optional[str] = None


class LiveOrderExecutor(OrderExecutor):
    """
    Async order executor for live CLOB trading.

    Orders are submitted asynchronously (fire-and-forget) and status is
    checked on subsequent calls. This matches the gym interface while
    allowing non-blocking order submission.
    """

    def __init__(
        self,
        transaction_client,
        config,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        default_order_size: float = 10.0,
        order_timeout: float = 5.0,
    ):
        """
        Initialize live order executor.

        Args:
            transaction_client: AsyncClobClient instance
            config: Config object with market/trading settings
            loop: Event loop for async operations
            default_order_size: Default trade size in USD
            order_timeout: Cancel orders older than this (seconds)
        """
        self.transaction_client = transaction_client
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.default_order_size = default_order_size
        self.order_timeout = order_timeout

        # State
        self.balance = 0.0
        self.current_position: Optional[ExtendedPosition] = None
        self.pending_orders: Dict[asyncio.Task, PendingOrder] = {}

        # For tracking fills
        self.last_fill_pnl = 0.0
        self.last_fill_fee = 0.0
        self.last_fill_slippage = 0.0
        self.last_amount_spent = 0.0

        # Transaction state
        self.consecutive_failures = 0
        self.last_action_status = "none"  # "pending", "success", "failed"

        # Order options
        self.order_type = getattr(config, "order_type", "GTC")
        self.order_options = CreateOrderOptions(tick_size="0.01", neg_risk=False)

        # Current market context
        self.current_market_id: Optional[str] = None
        self.current_token_up: Optional[str] = None
        self.current_token_down: Optional[str] = None

    def reset(self, balance: float):
        """Reset for new episode/market."""
        self.balance = balance
        self.current_position = None
        self.pending_orders.clear()
        self.last_fill_pnl = 0.0
        self.last_fill_fee = 0.0
        self.last_fill_slippage = 0.0
        self.last_amount_spent = 0.0
        self.consecutive_failures = 0
        self.last_action_status = "none"

    def set_market_context(self, condition_id: str, token_up: str, token_down: str):
        """Set current market context for order execution."""
        self.current_market_id = condition_id
        self.current_token_up = token_up
        self.current_token_down = token_down

    def execute(
        self, action: TradingAction, market_data: RawMarketData
    ) -> ExecutionResult:
        """
        Execute order asynchronously (fire-and-forget).

        Returns immediately with success=True if order submitted.
        Actual fill status checked on next call via _check_pending_fills().

        Args:
            action: Trading action (0=BUY_UP, 1=HOLD, 2=SELL_DOWN)
            market_data: Current market data

        Returns:
            ExecutionResult with immediate status
        """
        # Check and process any completed orders from previous actions
        self._check_pending_fills()

        # Cancel stale orders
        self._cancel_stale_orders()

        # HOLD action - no order needed
        if action.action == Action.HOLD:
            return ExecutionResult(
                success=True,
                filled=False,
                balance=self.balance,
                position=self.current_position,
                pnl=0.0,
                fee=0.0,
                slippage=0.0,
            )

        # Submit new order asynchronously
        try:
            order_task = self.loop.create_task(
                self._submit_order_async(action, market_data)
            )

            # Track pending order (fire-and-forget)
            self.pending_orders[order_task] = PendingOrder(
                task=order_task,
                action=action,
                timestamp=time.time(),
            )

            self.last_action_status = "pending"

            return ExecutionResult(
                success=True,
                filled=False,  # Not yet, will check on next call
                balance=self.balance,
                position=self.current_position,
                pnl=self.last_fill_pnl,
                fee=self.last_fill_fee,
                slippage=self.last_fill_slippage,
                amount_spent=self.last_amount_spent,
            )

        except Exception as e:
            self.consecutive_failures += 1
            self.last_action_status = "failed"

            return ExecutionResult(
                success=False,
                filled=False,
                balance=self.balance,
                position=self.current_position,
                pnl=0.0,
                fee=0.0,
                slippage=0.0,
                amount_spent=0.0,
                rejection_reason=f"Order submission failed: {str(e)}",
            )

    async def _submit_order_async(self, action: TradingAction, market_data: RawMarketData):
        """
        Actually submit order to CLOB (async).

        Returns dict with fill result when complete.
        """
        result = {
            "filled": False,
            "pnl": 0.0,
            "fee": 0.0,
            "slippage": 0.0,
            "amount_spent": 0.0,
            "error": None,
        }

        try:
            # Determine desired position side
            desired_side = "UP" if action.action == Action.BUY else "DOWN"

            # 1. Check if we already have a position in the SAME direction
            if self.current_position:
                if self.current_position.side == desired_side:
                    # Same direction order - reject with penalty
                    result["filled"] = False
                    result["pnl"] = -0.5  # Penalty for redundant action
                    result["error"] = f"redundant_{desired_side.lower()}_order"
                    return result
                else:
                    # Different direction - close existing position first
                    close_result = await self._close_position_async(market_data)
                    result["pnl"] += close_result.get("pnl", 0.0)
                    result["fee"] += close_result.get("fee", 0.0)

            # 2. Open new position
            if action.action in [Action.BUY, Action.SELL]:  # BUY_UP or SELL_DOWN
                open_result = await self._open_position_async(action, market_data)
                result["filled"] = open_result.get("filled", False)
                result["fee"] += open_result.get("fee", 0.0)
                result["slippage"] = open_result.get("slippage", 0.0)
                result["amount_spent"] = open_result.get("amount_spent", 0.0)

        except Exception as e:
            result["error"] = str(e)
            result["filled"] = False

        return result

    async def _close_position_async(self, market_data: RawMarketData) -> Dict:
        """Close current position."""
        if not self.current_position:
            return {"pnl": 0.0, "fee": 0.0}

        pos = self.current_position

        # Determine sell side (opposite of current position)
        sell_side = ORDER_SELL if pos.side == "UP" else ORDER_BUY
        token_id = pos.token_id

        # Use current price for close
        current_price = market_data.prob_up if pos.side == "UP" else (1.0 - market_data.prob_up)

        # Create order
        order_args = OrderArgs(
            token_id=token_id,
            price=current_price,
            size=pos.shares,
            side=sell_side,
            fee_rate_bps=0,  # Fetched from API
            nonce=0,
        )

        # Submit order
        response = await self.transaction_client.create_and_post_order(
            order_args, self.order_options, self.order_type
        )

        # Compute P&L
        pnl = pos.compute_pnl(current_price)

        # Estimate fee (will be corrected when order fills)
        fee = current_price * (1 - current_price) * pos.shares * 0.0001  # 1 bps

        # Update balance
        self.balance += (current_price * pos.shares) - fee

        # Clear position
        self.current_position = None

        return {
            "pnl": pnl,
            "fee": fee,
            "order_id": response.get("orderID"),
        }

    async def _open_position_async(
        self, action: TradingAction, market_data: RawMarketData
    ) -> Dict:
        """Open new position."""
        # Determine side and token
        if action.action == Action.BUY:  # BUY_UP
            side = ORDER_BUY
            token_id = self.current_token_up
            position_side = "UP"
            price = market_data.prob_up
        else:  # SELL_DOWN (action == 2)
            side = ORDER_SELL
            token_id = self.current_token_down
            position_side = "DOWN"
            price = 1.0 - market_data.prob_up

        # Calculate shares from USD amount
        trade_amount = action.size
        shares = trade_amount / price if price > 0 else 0

        if shares <= 0:
            return {"filled": False, "error": "Invalid shares calculation"}

        # Check balance
        if trade_amount > self.balance:
            return {"filled": False, "error": "Insufficient balance"}

        # Create order
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side=side,
            fee_rate_bps=0,
            nonce=0,
        )

        # Submit order
        response = await self.transaction_client.create_and_post_order(
            order_args, self.order_options, self.order_type
        )

        # Create position
        self.current_position = ExtendedPosition(
            side=position_side,
            entry_price=price,
            shares=shares,
            asset=market_data.asset,
            token_id=token_id,
            condition_id=self.current_market_id,
            entry_time=time.time(),
        )

        # Estimate fee
        fee = price * (1 - price) * shares * 0.0001

        # Update balance
        self.balance -= (trade_amount + fee)

        return {
            "filled": True,
            "fee": fee,
            "slippage": 0.0,  # Unknown until actual fill
            "amount_spent": trade_amount,
            "order_id": response.get("orderID"),
        }

    def _check_pending_fills(self):
        """Check status of pending orders and update state."""
        for task, pending in list(self.pending_orders.items()):
            if task.done():
                try:
                    result = task.result()

                    if result.get("filled"):
                        self.last_fill_pnl = result.get("pnl", 0.0)
                        self.last_fill_fee = result.get("fee", 0.0)
                        self.last_fill_slippage = result.get("slippage", 0.0)
                        self.last_amount_spent = result.get("amount_spent", 0.0)
                        self.last_action_status = "success"
                        self.consecutive_failures = 0
                    else:
                        self.last_action_status = "failed"
                        self.consecutive_failures += 1

                except Exception as e:
                    self.last_action_status = "failed"
                    self.consecutive_failures += 1

                # Remove completed order
                del self.pending_orders[task]

    def _cancel_stale_orders(self):
        """Cancel orders older than timeout."""
        current_time = time.time()

        for task, pending in list(self.pending_orders.items()):
            if current_time - pending.timestamp > self.order_timeout:
                if not task.done():
                    task.cancel()
                del self.pending_orders[task]

    def get_position_state(self) -> PositionState:
        """Return current position for features."""
        if self.current_position is None:
            return PositionState(
                has_position=False,
                side=None,
                unrealized_pnl=0.0,
                time_remaining_normalized=1.0,
            )

        # Compute unrealized PnL (need current price from market data)
        # This is approximation - actual PnL computed on close
        return PositionState(
            has_position=True,
            side=self.current_position.side,
            unrealized_pnl=0.0,  # Updated externally with current price
            time_remaining_normalized=1.0,
        )

    def compute_position_state(
        self, market_data: RawMarketData, time_remaining: float
    ) -> PositionState:
        """Compute position state with current market data for PnL calculation."""
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
        """Return transaction status for features."""
        return TransactionState(
            pending_order=self.last_action_status == "pending" or len(self.pending_orders) > 0,
            failed_order=self.last_action_status == "failed",
            consecutive_failures=self.consecutive_failures,
        )

    def get_capital_state(self) -> CapitalState:
        """Return balance for features."""
        return CapitalState(
            available_balance=self.balance,
            max_balance=self.balance,
        )
