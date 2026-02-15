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
    - Probability-dependent Polymarket fee calculation
    - Fill rejection (insufficient balance, no liquidity)
    - Trade history tracking
    """

    def __init__(self, initial_balance: float = 1000.0, default_fee_rate_bps: int = 625):
        """Initialize simulator with starting capital.

        Args:
            initial_balance: Starting USDC balance
            default_fee_rate_bps: Default fee_rate_bps for 15-min markets (625 typical)
                                  Produces ~1.56% effective fee at p=0.50
        """
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.fills_history: List[SimulatedFill] = []
        self.total_fees_paid = 0.0
        self.default_fee_rate_bps = default_fee_rate_bps

        # Statistics
        self.total_fills = 0
        self.total_rejections = 0
        self.rejection_reasons: Dict[str, int] = {}

    def calculate_fee_per_share(self, probability: float, fee_rate_bps: int = None) -> float:
        """Calculate Polymarket fee per share using probability-dependent formula.

        Formula: fee(p) = p × (1 − p) × r
        Where:
            p = probability/price (0.01 to 0.99)
            r = fee_rate_bps / 10000 (convert basis points to decimal)

        Fee peaks at p=0.50 (~1.56% effective for fee_rate_bps=625)
        Fee drops toward 0% at extremes (p→0.01 or p→0.99)

        Args:
            probability: Current probability/price (0.01 to 0.99)
            fee_rate_bps: Fee rate in basis points (default: use self.default_fee_rate_bps)

        Returns:
            Fee per share in USDC
        """
        if fee_rate_bps is None:
            fee_rate_bps = self.default_fee_rate_bps

        # Clamp probability to valid range
        p = max(0.01, min(0.99, probability))
        r = fee_rate_bps / 10000.0
        return p * (1 - p) * r

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
        order_type: str = "GTC",  # "GTC" (Good-Til-Cancelled) or "FOK" (Fill-Or-Kill)
        limit_price: Optional[float] = None,  # Limit price for FOK orders
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
            order_type: "GTC" (Good-Til-Cancelled) or "FOK" (Fill-Or-Kill)
            limit_price: Limit price for FOK orders (required for FOK)

        Returns:
            Dict with keys:
                - filled (bool): Whether order filled
                - fill_price (float): Actual fill price (0 if not filled)
                - slippage (float): Difference from base price
                - reason (str): Fill/rejection reason
                - balance_remaining (float): Balance after fill
                - order_type (str): Order type used
        """
        # Route to appropriate order type handler
        if order_type == "FOK":
            return self._simulate_fok_order(
                side, asset, size, current_prob, current_bid, current_ask,
                spread, order_book_imbalance, limit_price
            )
        else:  # GTC or other market orders
            return self._simulate_gtc_order(
                side, asset, size, current_prob, current_bid, current_ask,
                spread, order_book_imbalance, order_type
            )

    def _simulate_gtc_order(
        self,
        side: str,
        asset: str,
        size: float,
        current_prob: float,
        current_bid: float,
        current_ask: float,
        spread: float,
        order_book_imbalance: float,
        order_type: str
    ) -> Dict[str, Any]:
        """Simulate GTC (Good-Til-Cancelled) market order.

        GTC orders are more flexible and have higher fill rates.
        """
        # 1. Check balance (including estimated fees)
        # Use probability-dependent fee formula: fee(p) = p × (1 − p) × r
        base_price = current_prob if side == "BUY" else (1 - current_prob)
        fee_per_share = self.calculate_fee_per_share(base_price)
        estimated_shares = size / base_price if base_price > 0 else 0
        estimated_fee = fee_per_share * estimated_shares
        total_cost = size + estimated_fee

        if total_cost > self.balance:
            self._record_rejection("insufficient_balance", side, asset, size)
            return {
                "filled": False,
                "fill_price": 0.0,
                "slippage": 0.0,
                "reason": "insufficient_balance",
                "balance_remaining": self.balance,
                "order_type": order_type
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
                "balance_remaining": self.balance,
                "order_type": order_type
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

        # 4. Calculate actual costs using probability-dependent fees
        shares = size / fill_price
        actual_fee = self.calculate_fee_per_share(fill_price) * shares
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
            "balance_remaining": self.balance,
            "order_type": order_type
        }

    def _simulate_fok_order(
        self,
        side: str,
        asset: str,
        size: float,
        current_prob: float,
        current_bid: float,
        current_ask: float,
        spread: float,
        order_book_imbalance: float,
        limit_price: Optional[float]
    ) -> Dict[str, Any]:
        """Simulate FOK (Fill-Or-Kill) order.

        FOK orders must:
        1. Fill completely and immediately
        2. Fill at limit price or better
        3. Otherwise get rejected (killed)

        Characteristics:
        - Higher rejection rate than GTC (stricter requirements)
        - No slippage beyond limit price
        - Instant execution or rejection
        """
        order_type = "FOK"

        # 1. Check balance (including estimated fees)
        # Use probability-dependent fee formula
        base_price = current_prob if side == "BUY" else (1 - current_prob)
        fee_per_share = self.calculate_fee_per_share(base_price)
        estimated_shares = size / base_price if base_price > 0 else 0
        estimated_fee = fee_per_share * estimated_shares
        total_cost = size + estimated_fee

        if total_cost > self.balance:
            self._record_rejection("insufficient_balance", side, asset, size)
            return {
                "filled": False,
                "fill_price": 0.0,
                "slippage": 0.0,
                "reason": "insufficient_balance",
                "balance_remaining": self.balance,
                "order_type": order_type
            }

        # 2. Validate limit price
        if limit_price is None:
            self._record_rejection("fok_no_limit_price", side, asset, size)
            return {
                "filled": False,
                "fill_price": 0.0,
                "slippage": 0.0,
                "reason": "fok_no_limit_price",
                "balance_remaining": self.balance,
                "order_type": order_type
            }

        # 3. Check if limit price is achievable
        if side == "BUY":
            # BUY: can we get price <= limit_price?
            best_available = current_ask if current_ask > 0 else current_prob
            if best_available > limit_price:
                self._record_rejection("fok_price_not_met", side, asset, size)
                return {
                    "filled": False,
                    "fill_price": 0.0,
                    "slippage": 0.0,
                    "reason": "fok_price_not_met",
                    "balance_remaining": self.balance,
                    "order_type": order_type
                }
            base_price = best_available
        else:  # SELL
            # SELL (DOWN token): can we get price <= limit_price?
            best_available = 1 - current_bid if current_bid > 0 else 1 - current_prob
            if best_available > limit_price:
                self._record_rejection("fok_price_not_met", side, asset, size)
                return {
                    "filled": False,
                    "fill_price": 0.0,
                    "slippage": 0.0,
                    "reason": "fok_price_not_met",
                    "balance_remaining": self.balance,
                    "order_type": order_type
                }
            base_price = best_available

        # 4. Simulate immediate fill probability
        # FOK has stricter requirements - must fill immediately
        fok_base_fill_rate = 0.80  # Lower than GTC's 0.95

        # Penalties
        spread_penalty = min(spread / 0.05, 1.0)
        size_penalty = min(size / 100, 0.6)  # Larger orders harder to fill immediately

        # Imbalance penalty: if market is moving against us
        if side == "BUY" and order_book_imbalance > 0:  # Buying into demand
            imbalance_penalty = order_book_imbalance * 0.3
        elif side == "SELL" and order_book_imbalance < 0:  # Selling into supply
            imbalance_penalty = -order_book_imbalance * 0.3
        else:
            imbalance_penalty = 0.0

        fill_probability = fok_base_fill_rate * (1 - 0.4 * spread_penalty) * (1 - 0.3 * size_penalty) * (1 - imbalance_penalty)

        if np.random.random() > fill_probability:
            self._record_rejection("fok_no_immediate_fill", side, asset, size)
            return {
                "filled": False,
                "fill_price": 0.0,
                "slippage": 0.0,
                "reason": "fok_no_immediate_fill",
                "balance_remaining": self.balance,
                "order_type": order_type
            }

        # 5. Fill at limit price or better
        # FOK fills at best available price up to limit
        # Simulate small price improvement possibility
        price_improvement_bps = np.random.uniform(0, 5)  # 0-5 bps improvement possible
        fill_price = base_price * (1 - price_improvement_bps / 10000)
        fill_price = max(fill_price, 0.01)  # Floor at 0.01

        # Ensure we don't exceed limit
        fill_price = min(fill_price, limit_price)

        # 6. Update balance using probability-dependent fees
        shares = size / fill_price
        actual_fee = self.calculate_fee_per_share(fill_price) * shares
        total_cost = size + actual_fee
        self.balance -= total_cost
        self.total_fees_paid += actual_fee

        # 7. Record fill
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
            "balance_remaining": self.balance,
            "order_type": order_type
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
