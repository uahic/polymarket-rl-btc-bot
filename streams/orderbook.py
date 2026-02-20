"""
Polymarket CLOB WebSocket helpers for orderbook streaming.
"""
import asyncio
import logging
import json
import websockets
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

CLOB_WSS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

logger = logging.getLogger(__name__)


@dataclass
class OrderbookState:
    """Orderbook state for a market."""
    condition_id: str
    token_id: str
    side: str  # "UP" or "DOWN"
    bids: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    asks: List[tuple] = field(default_factory=list)
    last_update: Optional[datetime] = None

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class OrderbookStreamer:
    """Stream orderbook data from Polymarket CLOB."""

    def __init__(self):
        self.orderbooks: Dict[str, OrderbookState] = {}
        self.running = False
        self.callbacks: List[Callable] = []
        self._subscriptions: List[tuple] = []  # [(condition_id, token_id, side), ...]
        self._pending_subs: List[str] = []  # New token IDs to subscribe
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._active_condition_ids: set = set()  # Track currently active markets

    def subscribe(self, condition_id: str, token_up: str, token_down: str):
        """Subscribe to orderbook for a market."""
        # Check if already subscribed
        existing_tokens = {t for _, t, _ in self._subscriptions}
        added = []

        if token_up not in existing_tokens:
            self._subscriptions.append((condition_id, token_up, "UP"))
            self._pending_subs.append(token_up)
            added.append("UP")

        if token_down not in existing_tokens:
            self._subscriptions.append((condition_id, token_down, "DOWN"))
            self._pending_subs.append(token_down)
            added.append("DOWN")

        # if added:
        #     logger.debug(f"  [OB] Queued {condition_id[:8]}... ({', '.join(added)}) - pending: {len(self._pending_subs)}")

        # Initialize orderbook states
        self.orderbooks[f"{condition_id}_UP"] = OrderbookState(
            condition_id=condition_id,
            token_id=token_up,
            side="UP"
        )
        self.orderbooks[f"{condition_id}_DOWN"] = OrderbookState(
            condition_id=condition_id,
            token_id=token_down,
            side="DOWN"
        )

    def clear_stale(self, active_condition_ids: set):
        """Remove orderbooks for expired markets and trigger reconnection only when markets change."""
        # Check if the active markets have actually changed
        if active_condition_ids == self._active_condition_ids:
            # No change in markets, skip cleanup
            return

        stale_keys = [k for k in self.orderbooks.keys()
                      if k.rsplit('_', 1)[0] not in active_condition_ids]

        had_stale = len(stale_keys) > 0
        for k in stale_keys:
            del self.orderbooks[k]

        # Also clean up subscriptions list
        self._subscriptions = [(cid, tid, side) for cid, tid, side in self._subscriptions
                               if cid in active_condition_ids]

        # Update the tracked active markets
        self._active_condition_ids = active_condition_ids.copy()

        if had_stale:
            logger.info(f"  [OB] Cleared {len(stale_keys)} stale orderbooks")

    def on_update(self, callback: Callable):
        """Register a callback for orderbook updates."""
        self.callbacks.append(callback)

    def get_orderbook(self, condition_id: str, side: str) -> Optional[OrderbookState]:
        """Get orderbook state for a market side."""
        return self.orderbooks.get(f"{condition_id}_{side}")

    def get_latest(self, condition_id: str) -> dict:
        """
        Get latest orderbook data for LiveSource integration.

        Returns dict with aggregated orderbook metrics for both sides.
        """
        ob_up = self.get_orderbook(condition_id, "UP")
        ob_down = self.get_orderbook(condition_id, "DOWN")

        if not ob_up or not ob_down or not ob_up.bids or not ob_up.asks:
            return {
                "best_bid": 0.5,
                "best_ask": 0.5,
                "spread": 0.0,
                "mid_price": 0.5,
                "bids_l5": [],
                "asks_l5": [],
            }

        return {
            "best_bid": ob_up.best_bid,
            "best_ask": ob_up.best_ask,
            "spread": ob_up.spread,
            "mid_price": ob_up.mid_price,
            "bids_l5": ob_up.bids[:5],
            "asks_l5": ob_up.asks[:5],
        }

    async def stream(self):
        """Start streaming orderbooks."""

        if self.running is True:
            return

        self.running = True

        while self.running:
            # Wait for subscriptions if none exist yet
            if not self._subscriptions and not self._pending_subs:
                await asyncio.sleep(0.5)
                continue

            try:
                async with websockets.connect(CLOB_WSS) as ws:
                    logger.info("Connected to Polymarket CLOB WSS")

                    # Collect all token IDs for initial subscription
                    token_ids = [token_id for _, token_id, _ in self._subscriptions]

                    # Also include any pending subs
                    if self._pending_subs:
                        token_ids.extend(self._pending_subs)
                        self._pending_subs.clear()

                    if token_ids:
                        # Send single subscription with all assets
                        sub_msg = {
                            "assets_ids": token_ids,
                            "type": "market"
                        }
                        await ws.send(json.dumps(sub_msg))
                        logger.info(f"Subscribed to {len(token_ids)} orderbooks")

                    self._ws = ws

                    # Listen for updates
                    while self.running:
                        try:
                            # Check for pending subscriptions FIRST (new markets added dynamically)
                            if self._pending_subs:
                                new_tokens = self._pending_subs.copy()
                                self._pending_subs.clear()
                                sub_msg = {
                                    "assets_ids": new_tokens,
                                    "type": "market"
                                }
                                await ws.send(json.dumps(sub_msg))
                                logger.info(f"[OB] Sent subscription for {len(new_tokens)} new tokens")

                            # Short timeout to check pending subs frequently
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                            data = json.loads(msg)

                            # Handle different message types
                            if isinstance(data, list):
                                # Initial snapshot is an array
                                for item in data:
                                    if isinstance(item, dict):
                                        self._handle_book_update(item)
                            elif isinstance(data, dict):
                                # Check for orderbook update (has bids/asks)
                                if "bids" in data or "asks" in data:
                                    self._handle_book_update(data)
                                # Check for price_changes
                                elif "price_changes" in data:
                                    self._handle_price_change(data)

                        except asyncio.TimeoutError:
                            pass
                        except json.JSONDecodeError:
                            pass

                    self._ws = None

            except Exception as e:
                logger.warning(f"CLOB WSS error: {e}, reconnecting...")
                await asyncio.sleep(1)

    def _handle_book_update(self, data: dict):
        """Handle orderbook update message."""
        asset_id = data.get("asset_id")

        ob = next((ob for ob in self.orderbooks.values() if ob.token_id == asset_id), None)
        if ob is None:
            return

        def merge_side(existing: list, updates: list, descending: bool) -> list:
            # Start from the current book so we preserve levels not mentioned in this update.
            # Polymarket sends incremental diffs: size=0 means the level was removed,
            # any other size replaces the existing quantity at that price.
            book = dict(existing)
            for entry in updates:
                price, size = float(entry["price"]), float(entry["size"])
                if size == 0:
                    # Remove price level if not present in the new data
                    book.pop(price, None)
                else:
                    # Upsert new price/size pair
                    book[price] = size
            return sorted(book.items(), key=lambda x: x[0], reverse=descending)[:10]

        if "bids" in data:
            # Merge/Update new bid data from data object with existing object for orderbook
            ob.bids = merge_side(ob.bids, data["bids"], descending=True)
        if "asks" in data:
            # Merge/Update new ask data from data object with existing object for orderbook
            ob.asks = merge_side(ob.asks, data["asks"], descending=False)

        # Bids and asks arrive in separate messages, so after updating one side the book
        # can temporarily appear crossed (best_bid >= best_ask) due to the other side
        # being stale. Strip any levels that violate the invariant using the current
        # best prices as the boundary before notifying callbacks.

        # Yes, that's exactly what a crossed book is — and it's valid in real markets. 
        # Someone can place a bid higher than an existing ask. But on Polymarket's CLOB,#
        # this shouldn't persist because the exchange matches orders immediately when they cross. 
        # A bid at 0.75 against an ask at 0.72 would be filled at the ask price before either side 
        # ever appears in the orderbook snapshot you receive. So if you're seeing crossed levels in the feed, it's one of:
        # 1. Stale data — the case we already fixed, where bid and ask updates arrive in separate messages
        # 2. Feed inconsistency — a race between the match engine and the WebSocket broadcast, resolved within milliseconds
        # 3. Bug in the exchange feed — unlikely but possible

        if ob.bids and ob.asks:
            # Sort out crossed book entries
            best_bid, best_ask = ob.bids[0][0], ob.asks[0][0]
            ob.bids = [(p, s) for p, s in ob.bids if p < best_ask]
            ob.asks = [(p, s) for p, s in ob.asks if p > best_bid]

        ob.last_update = datetime.now(timezone.utc)

        for cb in self.callbacks:
            try:
                cb(ob)
            except:
                pass

    def _handle_price_change(self, data: dict):
        """Handle price change message (simpler update)."""
        changes = data.get("price_changes", [])
        for change in changes:
            asset_id = change.get("asset_id")
            price = change.get("price")

            # Find matching orderbook and update mid estimate
            for key, ob in self.orderbooks.items():
                if ob.token_id == asset_id:
                    ob.last_update = datetime.now(timezone.utc)
                    break

    def stop(self):
        """Stop streaming."""
        self.running = False


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s")
    logging.getLogger("__main__").setLevel(logging.DEBUG)


    # Test with a real market
    from polymarket_api import get_15m_markets

    logger.info("Testing Orderbook WSS...")

    async def test():
        markets = get_15m_markets(['BTC'])

        if not markets:
            logger.warning("No active markets!")
            return

        m = markets[0]
        logger.info(f"Subscribing to: {m.description}...")

        streamer = OrderbookStreamer()
        streamer.subscribe(m.condition_id, m.token_up, m.token_down)

        def on_update(ob: OrderbookState):
            logger.info(f"  {ob.side}: bid={ob.best_bid:.3f} ask={ob.best_ask:.3f} spread={ob.spread:.3f}")

        streamer.on_update(on_update)

        # Run for 15 seconds
        task = asyncio.create_task(streamer.stream())
        await asyncio.sleep(15)
        streamer.stop()
        task.cancel()

    asyncio.run(test())
