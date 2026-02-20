from dataclasses import dataclass


@dataclass
class Position:
    """Tracks current open position (paper/sim trading)."""

    side: str  # "UP" or "DOWN"
    entry_price: float
    shares: float
    asset: str

    @property
    def entry_value(self) -> float:
        return self.entry_price * self.shares

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


@dataclass
class ExtendedPosition(Position):
    """Position with live-trading metadata (Polymarket token IDs and entry timestamp)."""

    token_id: str = ""
    condition_id: str = ""
    entry_time: float = 0.0
