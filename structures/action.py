from enum import Enum


class Action(Enum):
    BUY = 0   # Buy UP token (action index 0)
    HOLD = 1  # Hold position (action index 1)
    SELL = 2  # Sell UP token / Buy DOWN token (action index 2)

    @property
    def is_buy(self) -> bool:
        return self == Action.BUY

    @property
    def is_sell(self) -> bool:
        return self == Action.SELL

    @property
    def size_multiplier(self) -> float:
        """Base 50% sizing for trades (adjusted by confidence in TradingEngine)."""
        return 0.5 if self in (Action.BUY, Action.SELL) else 0.0

    def get_confidence_size(self, prob: float) -> float:
        """
        Get position size multiplier based on probability extremeness.

        At extreme probabilities (near 0 or 1), we have higher edge due to
        asymmetric payoffs in binary markets. Scale size accordingly.

        Returns: size multiplier in [0.25, 1.0]
        """
        if self == Action.HOLD:
            return 0.0

        # Distance from 0.5 - higher = more extreme
        extremeness = abs(prob - 0.5) * 2  # [0, 1]

        # Scale from 0.25 (at 0.5) to 1.0 (at extremes)
        # More aggressive at extremes where edge is higher
        base = 0.25
        scale = 0.75  # max additional size

        return base + (scale * extremeness)

