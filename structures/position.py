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

    def compute_pnl(self, current_token_price: float) -> float:
        """
        Compute unrealized P&L.

        For Polymarket:
        - UP token: profit when price increases
        - DOWN token: profit when price decreases

        Args:
            current_token_price: Current price of the token we hold
                                 (For UP: this is prob_up or best_bid)
                                 (For DOWN: this is (1-prob_up) or (1-best_ask))
        """
        # Simple calculation: current value - entry value
        current_value = current_token_price * self.shares
        return current_value - self.entry_value


@dataclass
class ExtendedPosition(Position):
    """Position with live-trading metadata (Polymarket token IDs and entry timestamp)."""

    token_id: str = ""
    condition_id: str = ""
    entry_time: float = 0.0
