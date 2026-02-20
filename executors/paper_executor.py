"""Simulated order executor for realistic paper trading."""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
logger = logging.getLogger(__name__)

from executors.executor_config import load_executor_config as _load_executor_config

def clamp_probability(current_prob: float) -> float:
    return max(0.01, min(0.99, current_prob))

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
        cfg = _load_executor_config()
        ex  = cfg.get("executor", {})
        sp  = cfg.get("spread", {})
        gtc = cfg.get("gtc", {})
        fok = cfg.get("fok", {})

        self.balance = ex.get("initial_balance", initial_balance)
        self.initial_balance = self.balance
        self.default_fee_rate_bps = ex.get("default_fee_rate_bps", default_fee_rate_bps)

        # Spread baselines
        self.near_expiry_spread   = sp.get("near_expiry_spread", 0.05)
        self.standard_spread      = sp.get("standard_spread", 0.02)

        # GTC parameters
        self.gtc_base_fill_probability = gtc.get("base_fill_probability", 0.95)
        self.gtc_spread_fill_weight    = gtc.get("spread_fill_weight", 0.3)
        self.gtc_size_penalty_cap      = gtc.get("size_penalty_cap", 0.5)
        self.gtc_size_penalty_threshold = gtc.get("size_penalty_threshold", 100.0)
        self.gtc_size_fill_weight      = gtc.get("size_fill_weight", 0.2)
        self.gtc_typical_order_size    = gtc.get("typical_order_size", 50.0)
        self.gtc_max_slippage_bps      = gtc.get("max_slippage_bps", 30.0)
        self.gtc_spread_slippage_weight = gtc.get("spread_slippage_weight", 0.5)
        self.gtc_size_slippage_weight  = gtc.get("size_slippage_weight", 0.3)

        # FOK parameters
        self.fok_base_fill_probability = fok.get("base_fill_probability", 0.80)
        self.fok_spread_fill_weight    = fok.get("spread_fill_weight", 0.4)
        self.fok_size_penalty_cap      = fok.get("size_penalty_cap", 0.6)
        self.fok_size_penalty_threshold = fok.get("size_penalty_threshold", 100.0)
        self.fok_size_fill_weight      = fok.get("size_fill_weight", 0.3)
        self.fok_imbalance_coefficient = fok.get("imbalance_coefficient", 0.3)
        self.fok_max_price_improvement_bps = fok.get("max_price_improvement_bps", 5.0)

        self.fills_history: List[SimulatedFill] = []
        self.total_fees_paid = 0.0

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
        if probability < 0.01 or probability > 0.99:
            logger.debug(f"Clamped probability to [0,1]. Original probability was: {probability}")
        r = fee_rate_bps / 10000.0
        return p * (1 - p) * r

    def _get_effective_spread(self, spread: float = 0.02) -> float:
        """Get effective spread, handling None values.

        Args:
            spread: Bid-ask spread (can be None if orderbook is empty)

        Returns:
            Effective spread to use in calculations (defaults to 0.02 if None)
        """
        return spread 

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
        # Clamp probability to Polymarket's valid range [0.01, 0.99]
        current_prob = clamp_probability(current_prob)

        # Clamp bid/ask to valid range if provided
        if current_bid is not None:
            current_bid = clamp_probability(current_bid)
        if current_ask is not None:
            current_ask = clamp_probability(current_ask)

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
        effective_spread = self._get_effective_spread(spread)
        spread_penalty = min(effective_spread / self.near_expiry_spread, 1.0)
        size_penalty = min(size / self.gtc_size_penalty_threshold, self.gtc_size_penalty_cap)

        fill_probability = (
            self.gtc_base_fill_probability
            * (1 - self.gtc_spread_fill_weight * spread_penalty)
            * (1 - self.gtc_size_fill_weight * size_penalty)
        )

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
        # Slippage depends on:
        # - Order book imbalance: more buyers = worse price for BUY, more sellers = worse price for SELL
        # - Spread: wider spread = more slippage
        # - Order size: larger orders = more slippage
        spread_factor = effective_spread / self.standard_spread
        size_factor = size / self.gtc_typical_order_size

        if side == "BUY":
            # Buying UP token: pay the ask + slippage
            base_price = current_ask
            imbalance_factor = max(0, order_book_imbalance)  # Only penalize if buying into demand
        else:  # SELL (buying DOWN token)
            # DOWN token price = 1 - UP price
            # When selling (buying DOWN), pay (1 - bid) + slippage
            base_price = 1 - current_bid
            imbalance_factor = max(0, -order_book_imbalance)  # Penalize if selling into supply

        slippage_bps = (
            np.random.uniform(0, self.gtc_max_slippage_bps)
            * (1 + imbalance_factor)
            * (1 + self.gtc_spread_slippage_weight * spread_factor)
            * (1 + self.gtc_size_slippage_weight * size_factor)
        )

        fill_price = base_price * (1 + slippage_bps / 10000)
        fill_price = min(fill_price, 0.99)  

        # 4. Calculate actual costs using probability-dependent fees
        shares = size / fill_price
        actual_fee = self.calculate_fee_per_share(fill_price) * shares
        total_cost = size + actual_fee

        # 5. Update balance (re-check after slippage inflated the actual cost)
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

        # Clamp probability to Polymarket's valid range [0.01, 0.99]
        current_prob = clamp_probability(current_prob)

        # Clamp bid/ask to valid range if provided
        if current_bid is not None:
            current_bid = clamp_probability(current_bid)
        if current_ask is not None:
            current_ask = clamp_probability(current_ask)

        # Clamp limit_price to valid range if provided
        if limit_price is not None:
            limit_price = clamp_probability(limit_price)

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
            best_available = current_ask
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
            best_available = 1 - current_bid
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
        effective_spread = self._get_effective_spread(spread)
        spread_penalty = min(effective_spread / self.near_expiry_spread, 1.0)
        size_penalty = min(size / self.fok_size_penalty_threshold, self.fok_size_penalty_cap)

        # Imbalance penalty: if market is moving against us (capped at 1.0 to keep fill_probability >= 0)
        if side == "BUY" and order_book_imbalance > 0:  # Buying into demand
            imbalance_penalty = min(order_book_imbalance * self.fok_imbalance_coefficient, 1.0)
        elif side == "SELL" and order_book_imbalance < 0:  # Selling into supply
            imbalance_penalty = min(-order_book_imbalance * self.fok_imbalance_coefficient, 1.0)
        else:
            imbalance_penalty = 0.0

        fill_probability = (
            self.fok_base_fill_probability
            * (1 - self.fok_spread_fill_weight * spread_penalty)
            * (1 - self.fok_size_fill_weight * size_penalty)
            * (1 - imbalance_penalty)
        )

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
        # FOK fills at best available price, with small price improvement possibility
        # (price improvement only moves fill_price below base_price, so limit_price cap never fires)
        price_improvement_bps = np.random.uniform(0, self.fok_max_price_improvement_bps)
        fill_price = base_price * (1 - price_improvement_bps / 10000)
        fill_price = max(fill_price, 0.01)  # Floor at 0.01

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

    def realize_pnl(self, pnl: float, fee: float = 0.0):
        """Add realized P&L back to balance (when closing position).

        Args:
            pnl: Net proceeds after fees (can be negative)
            fee: Exit fee already deducted from pnl, tracked for statistics
        """
        self.balance += pnl
        self.total_fees_paid += fee

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

        logger.info("=" * 50)
        logger.info("SIMULATION STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Balance:           ${stats['current_balance']:.2f} (start: ${stats['initial_balance']:.2f})")
        logger.info(f"Total P&L:         ${stats['total_pnl']:+.2f} ({stats['pnl_pct']:+.2f}%)")
        logger.info(f"Total Fees:        ${stats['total_fees_paid']:.2f}")
        logger.info(f"Orders:            {stats['total_orders']} ({stats['total_fills']} fills, {stats['total_rejections']} rejections)")
        logger.info(f"Fill Rate:         {stats['fill_rate']*100:.1f}%")
        logger.info(f"Avg Slippage:      {stats['avg_slippage']*10000:.2f} bps")

        if stats['rejection_reasons']:
            logger.info("Rejection Reasons:")
            for reason, count in stats['rejection_reasons'].items():
                logger.info(f"  - {reason}: {count}")
        logger.info("=" * 50)
