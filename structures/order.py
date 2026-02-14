import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from functools import cached_property

class OrderSide(Enum):
    BUY = 0
    SELL = 1 

# USDC has 6 decimal places
USDC_DECIMALS = 6


@dataclass(frozen=True)
class Order:
    """
    Represents a Polymarket order.

    Attributes:
        token_id: The ERC-1155 token ID for the market outcome
        price: Price per share (0-1, e.g., 0.65 = 65%)
        size: Number of shares
        side: Order side ('BUY' or 'SELL')
        maker: The maker's wallet address (Safe/Proxy)
        nonce: Unique order nonce (usually timestamp)
        fee_rate_bps: Fee rate in basis points (usually 0)
        signature_type: Signature type (2 = Gnosis Safe)
    """

    token_id: str
    price: Decimal
    size: Decimal
    side: OrderSide
    maker: str
    nonce: Optional[int] = None
    fee_rate_bps: int = 0
    signature_type: int = 2

    def __post_init__(self):
        """Validate order parameters."""
        if not 0 < self.price <= 1:
            raise ValueError(f"Invalid price: {self.price}")

        if self.size <= 0:
            raise ValueError(f"Invalid size: {self.size}")

        if not 0 <= self.fee_rate_bps <= 10000:
            raise ValueError(f"Invalid fee_rate_bps: {self.fee_rate_bps}")

        # Set nonce if not provided (frozen=True requires object.__setattr__)
        if self.nonce is None:
            object.__setattr__(self, 'nonce', int(time.time()))

    @cached_property
    def maker_amount(self) -> str:
        """Amount in USDC (with decimals) that maker provides."""
        return str(int(self.size * self.price * 10**USDC_DECIMALS))

    @cached_property
    def taker_amount(self) -> str:
        """Amount in USDC (with decimals) that taker provides."""
        return str(int(self.size * 10**USDC_DECIMALS))

    @cached_property
    def side_value(self) -> int:
        """Numeric representation of side (0=BUY, 1=SELL)."""
        return self.side.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            'token_id': self.token_id,
            'price': str(self.price),
            'size': str(self.size),
            'side': self.side.name,
            'maker': self.maker,
            'nonce': self.nonce,
            'fee_rate_bps': self.fee_rate_bps,
            'signature_type': self.signature_type,
            'maker_amount': self.maker_amount,
            'taker_amount': self.taker_amount,
            'side_value': self.side_value
        }
