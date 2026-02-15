"""Simulated order executor for realistic paper trading."""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class SimulatedFill:
    """Record of a simulated order fill."""
    timestamp: datetime
    side: str  # "BUY" or "SELL"
    asset: str
    size: float  # Dollar amount
    fill_price: float
    slippage: float
    reason: str  # "success", "insufficient_balance", "no_liquidity", etc.


class SimulatedOrderExecutor:
    """Simulates realistic order execution for training without spending money.

    Features:
    - Balance tracking with initial capital
    - Realistic fill probability based on spread and liquidity
    - Slippage simulation based on order book imbalance
    - Fill rejection (insufficient balance, no liquidity)
    - Trade history tracking
    """

    def __init__(self, initial_balance: float = 1000.0):
        """Initialize simulator with starting capital.

        Args:
            initial_balance: Starting USDC balance
        """
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.fills_history: List[SimulatedFill] = []
        self.total_fees_paid = 0.0

        # Statistics
        self.total_fills = 0
        self.total_rejections = 0
        self.rejection_reasons: Dict[str, int] = {}

    def simulate_order_fill(
        self,
        side: str,  # "BUY" or "SELL"
        asset: str,
        size: float,  # Dollar amount
        current_prob: float,  # Current UP probability
        current_bid: float,
        current_ask: float,
        spread: float,
        order_book_imbalance: float,  # [-1, 1]: negative = more sellers, positive = more buyers
    ) -> Dict[str, Any]:
        """Simulate whether order fills and at what price.

        Args:
            side: "BUY" (UP token) or "SELL" (DOWN token)
            asset: Asset identifier (e.g., "BTC")
            size: Dollar amount to trade
            current_prob: Current UP probability (0-1)
            current_bid: Best bid price
            current_ask: Best ask price
            spread: Bid-ask spread
            order_book_imbalance: Order book pressure [-1, 1]

        Returns:
            Dict with keys:
                - filled (bool): Whether order filled
                - fill_price (float): Actual fill price (0 if not filled)
                - slippage (float): Difference from base price
                - reason (str): Fill/rejection reason
                - balance_remaining (float): Balance after fill
        """
        # 1. Check balance (including estimated fees)
        estimated_fee = size * 0.002  # 0.2% taker fee (typical Polymarket)
        total_cost = size + estimated_fee

        if total_cost > self.balance:
            self._record_rejection("insufficient_balance", side, asset, size)
            return {
                "filled": False,
                "fill_price": 0.0,
                "slippage": 0.0,
                "reason": "insufficient_balance",
                "balance_remaining": self.balance
            }

        # 2. Simulate liquidity constraints
        # Large orders in wide spreads are less likely to fill
        base_fill_probability = 0.95

        # Spread penalty: wider spread = lower fill rate
        spread_penalty = min(spread / 0.05, 1.0)  # Normalize to [0, 1]

        # Size penalty: larger orders harder to fill
        size_penalty = min(size / 100, 0.5)  # Orders >$100 get penalized

        fill_probability = base_fill_probability * (1 - 0.3 * spread_penalty) * (1 - 0.2 * size_penalty)

        if np.random.random() > fill_probability:
            self._record_rejection("no_liquidity", side, asset, size)
            return {
                "filled": False,
                "fill_price": 0.0,
                "slippage": 0.0,
                "reason": "no_liquidity",
                "balance_remaining": self.balance
            }

        # 3. Simulate fill price with realistic slippage
        if side == "BUY":
            # Buying UP token: pay the ask + slippage
            base_price = current_ask if current_ask > 0 else current_prob

            # Slippage depends on:
            # - Order book imbalance: more buyers = worse price
            # - Spread: wider spread = more slippage
            # - Order size: larger orders = more slippage
            imbalance_factor = max(0, order_book_imbalance)  # Only penalize if buying into demand
            spread_factor = spread / 0.02  # Normalize to typical spread
            size_factor = size / 50  # Normalize to typical order size

            # Slippage in basis points (0-50 bps typical)
            slippage_bps = np.random.uniform(0, 30) * (1 + imbalance_factor) * (1 + 0.5 * spread_factor) * (1 + 0.3 * size_factor)
            fill_price = base_price * (1 + slippage_bps / 10000)
            fill_price = min(fill_price, 0.99)  # Cap at 0.99 (Polymarket limit)

        else:  # SELL (buying DOWN token)
            # DOWN token price = 1 - UP price
            # When selling (buying DOWN), pay (1 - bid) + slippage
            down_ask = 1 - current_bid if current_bid > 0 else 1 - current_prob
            base_price = down_ask

            # Slippage logic (reverse for DOWN)
            imbalance_factor = max(0, -order_book_imbalance)  # Penalize if selling into supply
            spread_factor = spread / 0.02
            size_factor = size / 50

            slippage_bps = np.random.uniform(0, 30) * (1 + imbalance_factor) * (1 + 0.5 * spread_factor) * (1 + 0.3 * size_factor)
            fill_price = base_price * (1 + slippage_bps / 10000)
            fill_price = min(fill_price, 0.99)  # DOWN token also capped at 0.99

        # 4. Calculate actual costs
        actual_fee = size * 0.002
        total_cost = size + actual_fee

        # 5. Update balance
        self.balance -= total_cost
        self.total_fees_paid += actual_fee

        # 6. Record fill
        fill = SimulatedFill(
            timestamp=datetime.now(timezone.utc),
            side=side,
            asset=asset,
            size=size,
            fill_price=fill_price,
            slippage=fill_price - base_price,
            reason="success"
        )
        self.fills_history.append(fill)
        self.total_fills += 1

        return {
            "filled": True,
            "fill_price": fill_price,
            "slippage": fill_price - base_price,
            "reason": "success",
            "balance_remaining": self.balance
        }

    def realize_pnl(self, pnl: float):
        """Add realized P&L back to balance (when closing position).

        Args:
            pnl: Profit or loss (can be negative)
        """
        self.balance += pnl
        self.balance = max(0, self.balance)  # Can't go negative

    def _record_rejection(self, reason: str, side: str, asset: str, size: float):
        """Record order rejection for statistics."""
        self.total_rejections += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

        # Still record in history for analysis
        fill = SimulatedFill(
            timestamp=datetime.now(timezone.utc),
            side=side,
            asset=asset,
            size=size,
            fill_price=0.0,
            slippage=0.0,
            reason=reason
        )
        self.fills_history.append(fill)

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics.

        Returns:
            Dict with performance metrics
        """
        total_orders = self.total_fills + self.total_rejections
        fill_rate = self.total_fills / total_orders if total_orders > 0 else 0.0

        # Calculate average slippage from successful fills
        successful_fills = [f for f in self.fills_history if f.reason == "success"]
        avg_slippage = np.mean([f.slippage for f in successful_fills]) if successful_fills else 0.0

        # Calculate P&L
        total_pnl = self.balance - self.initial_balance
        pnl_pct = (total_pnl / self.initial_balance) * 100

        return {
            "current_balance": self.balance,
            "initial_balance": self.initial_balance,
            "total_pnl": total_pnl,
            "pnl_pct": pnl_pct,
            "total_orders": total_orders,
            "total_fills": self.total_fills,
            "total_rejections": self.total_rejections,
            "fill_rate": fill_rate,
            "total_fees_paid": self.total_fees_paid,
            "avg_slippage": avg_slippage,
            "rejection_reasons": self.rejection_reasons,
        }

    def reset(self, new_balance: Optional[float] = None):
        """Reset simulator state.

        Args:
            new_balance: New starting balance (defaults to initial_balance)
        """
        self.balance = new_balance if new_balance is not None else self.initial_balance
        self.fills_history.clear()
        self.total_fees_paid = 0.0
        self.total_fills = 0
        self.total_rejections = 0
        self.rejection_reasons.clear()

    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance

    def print_statistics(self):
        """Print human-readable statistics."""
        stats = self.get_statistics()

        print("\n" + "="*50)
        print("SIMULATION STATISTICS")
        print("="*50)
        print(f"Balance:           ${stats['current_balance']:.2f} (start: ${stats['initial_balance']:.2f})")
        print(f"Total P&L:         ${stats['total_pnl']:+.2f} ({stats['pnl_pct']:+.2f}%)")
        print(f"Total Fees:        ${stats['total_fees_paid']:.2f}")
        print(f"Orders:            {stats['total_orders']} ({stats['total_fills']} fills, {stats['total_rejections']} rejections)")
        print(f"Fill Rate:         {stats['fill_rate']*100:.1f}%")
        print(f"Avg Slippage:      {stats['avg_slippage']*10000:.2f} bps")

        if stats['rejection_reasons']:
            print("\nRejection Reasons:")
            for reason, count in stats['rejection_reasons'].items():
                print(f"  - {reason}: {count}")
        print("="*50 + "\n")
