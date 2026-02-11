from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone

@dataclass
class Position:
    """Track position for a market."""
    asset: str
    side: Optional[str] = None
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_prob: float = 0.0
    time_remaining_at_entry: float = 0.0

