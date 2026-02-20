"""
Shared orderbook utility functions.

This module provides the SINGLE SOURCE OF TRUTH for orderbook calculations.
Used by both:
- streams/orderbook.py (live streaming)
- features/computer.py (feature computation)
"""

from typing import List, Tuple


def compute_orderbook_imbalance_l1(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]]
) -> float:
    """
    Compute L1 orderbook imbalance (top of book).

    Args:
        bids: List of (price, size) tuples, sorted descending by price
        asks: List of (price, size) tuples, sorted ascending by price

    Returns:
        Imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        Range: [-1, 1]
        - Positive: more buying pressure (bid-heavy)
        - Negative: more selling pressure (ask-heavy)
        - Zero: balanced or empty book
    """
    if not bids or not asks:
        return 0.0

    bid_size = bids[0][1]  # Best bid size
    ask_size = asks[0][1]  # Best ask size

    total = bid_size + ask_size
    if total == 0:
        return 0.0

    return (bid_size - ask_size) / total


def compute_orderbook_imbalance_l5(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    mid_price: float = None,
    max_spread: float = 0.03
) -> float:
    """
    Compute L5 orderbook imbalance (depth across top 5 levels).

    Args:
        bids: List of (price, size) tuples, sorted descending by price
        asks: List of (price, size) tuples, sorted ascending by price
        mid_price: Optional mid price for filtering levels within max_spread
        max_spread: Maximum spread from mid_price to include (default: 3%)

    Returns:
        Imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        Range: [-1, 1]
        - Positive: more depth on bid side
        - Negative: more depth on ask side
        - Zero: balanced or empty book
    """
    if not bids or not asks:
        return 0.0

    # If mid_price provided, filter levels within max_spread
    if mid_price is not None:
        bid_depth = sum(size for price, size in bids[:5] if mid_price - price <= max_spread)
        ask_depth = sum(size for price, size in asks[:5] if price - mid_price <= max_spread)
    else:
        # No filtering, just sum top 5 levels
        bid_depth = sum(size for _, size in bids[:5])
        ask_depth = sum(size for _, size in asks[:5])

    total = bid_depth + ask_depth
    if total == 0:
        return 0.0

    return (bid_depth - ask_depth) / total
