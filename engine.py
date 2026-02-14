import asyncio
import sys
import copy
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from logger.training_logger import get_logger
from config import Config
from strategies import BaseStrategy, MLStrategy
from streams.binance import BinanceStreamer
from streams.orderbook import OrderbookStreamer
from streams.binance_futures import FuturesStreamer
from streams.polymarket_api import Market, get_15m_markets
from structures.market import MarketState
from structures.action import Action
from structures.position import Position
from structures.order import Order, OrderSide
from security.order_signer import OrderSigner
from config_loader import decrypt_private_key
from transactions.transaction_client import TransactionClient
from transactions.async_http import OrderError

# Dashboard integration (optional)
try:
    from dashboard.flask_dashboard import (
        update_dashboard_state,
        update_rl_metrics,
        emit_rl_buffer,
        run_dashboard,
        emit_trade,
    )

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

    def update_dashboard_state(**kwargs):
        pass

    def update_rl_metrics(metrics):
        pass

    def emit_rl_buffer(buffer_size, max_buffer=256, avg_reward=None):
        pass

    def emit_trade(action, asset, size=0, pnl=None):
        pass


class TradingEngine:
    """
    Paper trading engine with strategy harness.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: Config,
        trade_size: float = 10.0,
        live_trading: bool = False,
    ):
        self.strategy = strategy
        self.trade_size = trade_size
        self.config = config
        self.live_trading = live_trading

        # Streamers
        # self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.price_streamer = BinanceStreamer(["BTC"])
        self.orderbook_streamer = OrderbookStreamer()
        # self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.futures_streamer = FuturesStreamer(["BTC"])

        # Order management (initialized only if live_trading=True)
        self.order_signer: Optional[OrderSigner] = None
        self.transaction_client: Optional[TransactionClient] = None

        # Order tracking
        self.pending_orders: Dict[str, Dict[str, Any]] = (
            {}
        )  # cid -> {order_id, side, size, price}
        self.order_timestamps: Dict[str, float] = {}  # cid -> unix_timestamp
        self.order_type: str = "GTC"  # "GTC" or "FOK"

        # Initialize live trading components
        if self.live_trading:
            # Load private key and initialize signer
            private_key = decrypt_private_key()
            self.order_signer = OrderSigner(private_key)

            # Initialize async CLOB client
            # Use checksummed address from signer for consistency with API credentials
            self.transaction_client = TransactionClient(
                host=config.clob.host,
                chain_id=config.clob.chain_id,
                signature_type=config.clob.signature_type,
                funder=self.order_signer.address,  # Use checksummed address
                builder_creds=(
                    config.builder if config.builder.is_configured() else None
                ),
            )

            self.order_type = getattr(config, "order_type", "GTC")
            print(f"[LIVE TRADING] Mode enabled, using {self.order_type} orders")

        # State
        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Position] = {}
        self.states: Dict[str, MarketState] = {}
        self.prev_states: Dict[str, MarketState] = {}  # For RL transitions
        self.open_prices: Dict[str, float] = {}  # Binance price at market open
        self.running = False

        # Stats
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # Pending rewards for RL (set on position close)
        self.pending_rewards: Dict[str, float] = {}

        # Logger (for RL training)
        self.logger = get_logger() if isinstance(strategy, MLStrategy) else None

    def refresh_markets(self):
        """Find active 15-min markets."""
        print("\n" + "=" * 60)
        print(f"STRATEGY: {self.strategy.name.upper()}")
        print("=" * 60)

        # markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
        markets = get_15m_markets(assets=["BTC"])
        now = datetime.now(timezone.utc)

        # Clear stale data
        self.markets.clear()
        self.states.clear()

        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            if mins_left < 0.5:
                continue

            print(f"\n{m.asset} 15m | {mins_left:.1f}m left")
            print(f"  UP: {m.price_up:.3f} | DOWN: {m.price_down:.3f}")

            self.markets[m.condition_id] = m
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)

            # Init state
            self.states[m.condition_id] = MarketState(
                asset=m.asset,
                prob=m.price_up,
                time_remaining=mins_left / 15.0,
            )

            # Init position
            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)

            # Record open price
            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0:
                self.open_prices[m.condition_id] = current_price

        if not self.markets:
            print("\nNo active markets!")
        else:
            # Clear stale orderbook subscriptions
            active_cids = set(self.markets.keys())
            self.orderbook_streamer.clear_stale(active_cids)

    def execute_action(self, cid: str, action: Action, state: MarketState):
        """Execute action - paper or live based on mode."""
        if action == Action.HOLD:
            return

        if self.live_trading:
            # Schedule async live order execution
            asyncio.create_task(self._execute_action_live(cid, action, state))
        else:
            # Keep existing paper trading logic
            self._execute_action_paper(cid, action, state)

    def _execute_action_paper(self, cid: str, action: Action, state: MarketState):
        """Execute paper trade with flexible sizing."""
        pos = self.positions.get(cid)
        if not pos:
            return

        price = state.prob
        trade_amount = self.trade_size * action.size_multiplier

        # Close existing position if switching sides
        if pos.size > 0:
            if action.is_sell and pos.side == "UP":
                shares = pos.size / pos.entry_price
                pnl = (price - pos.entry_price) * shares
                self._record_trade(pos, price, pnl, "CLOSE UP", cid=cid)
                self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                pos.size = 0
                pos.side = None
                return

            elif action.is_buy and pos.side == "DOWN":
                exit_down_price = 1 - price  # Current DOWN token price
                shares = pos.size / pos.entry_price
                pnl = (
                    exit_down_price - pos.entry_price
                ) * shares  # DOWN token went up = profit
                self._record_trade(pos, price, pnl, "CLOSE DOWN", cid=cid)
                self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                pos.size = 0
                pos.side = None
                return

        # Open new position
        if pos.size == 0:
            size_label = {0.25: "SM", 0.5: "MD", 1.0: "LG"}.get(
                action.size_multiplier, ""
            )

            if action.is_buy:
                pos.side = "UP"
                pos.size = trade_amount
                pos.entry_price = price
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = price
                pos.time_remaining_at_entry = state.time_remaining
                print(
                    f"    OPEN {pos.asset} UP ({size_label}) ${trade_amount:.0f} @ {price:.3f}"
                )
                emit_trade(f"BUY_{size_label}", pos.asset, pos.size)

            elif action.is_sell:
                pos.side = "DOWN"
                pos.size = trade_amount
                pos.entry_price = 1 - price  # DOWN token price = 1 - UP prob
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = price  # Keep original UP prob for reference
                pos.time_remaining_at_entry = state.time_remaining
                print(
                    f"    OPEN {pos.asset} DOWN ({size_label}) ${trade_amount:.0f} @ {1 - price:.3f}"
                )
                emit_trade(f"SELL_{size_label}", pos.asset, pos.size)

    async def _execute_action_live(self, cid: str, action: Action, state: MarketState):
        """Execute live order via CLOB API."""
        try:
            pos = self.positions.get(cid)
            m = self.markets.get(cid)
            if not pos or not m:
                return

            # Cancel pending orders first if switching sides
            if cid in self.pending_orders:
                await self._cancel_pending_order(cid)

            price = state.prob
            trade_amount = self.trade_size * action.size_multiplier

            # Close position if switching sides
            if pos.size > 0:
                if (action.is_sell and pos.side == "UP") or (
                    action.is_buy and pos.side == "DOWN"
                ):
                    # Close existing position first
                    # If this fails, exception bubbles up and we skip opening new position
                    await self._close_position_live(cid, pos, state, pos.side)
                    # Position is now closed (pos.size = 0)

            # Open new position (only if we're flat)
            # This runs after successful close OR if we had no position
            if pos.size == 0:
                await self._open_position_live(cid, action, state, trade_amount, m)
            else:
                # Position still open (same side, add to existing?)
                # For now, skip - strategy shouldn't double down without closing first
                print(
                    f"    [SKIP] Position already open on {pos.side}, action wants same side"
                )

        except OrderError as e:
            print(f"  [ORDER ERROR] {m.asset}: {e}")
            self._handle_order_failure(cid, str(e))
        except Exception as e:
            print(f"  [ERROR] {m.asset}: {e}")
            self._handle_order_failure(cid, str(e))

    async def _open_position_live(
        self,
        cid: str,
        action: Action,
        state: MarketState,
        trade_amount: float,
        market: Market,
    ):
        """Open new position with live order."""

        # Determine token and side
        if action.is_buy:
            token_id = market.token_up
            side = OrderSide.BUY
            price = state.prob
            position_side = "UP"
        else:  # action.is_sell
            token_id = market.token_down
            side = OrderSide.BUY  # Buy DOWN token
            price = 1 - state.prob
            position_side = "DOWN"

        # Calculate shares
        size = trade_amount / price

        # Sign and submit order
        signed_order = self.order_signer.sign_order_dict(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            maker=self.config.safe_address,
            nonce=None,
            fee_rate_bps=0,
        )

        response = await self.transaction_client.post_order(
            signed_order=signed_order, order_type=self.order_type
        )

        # Track order if GTC
        if self.order_type == "GTC":
            order_id = response.get("orderID")
            if order_id:
                self.pending_orders[cid] = {
                    "order_id": order_id,
                    "side": position_side,
                    "size": trade_amount,
                    "price": price,
                }
                self.order_timestamps[cid] = datetime.now(timezone.utc).timestamp()

        # Update position (optimistic)
        pos = self.positions[cid]
        pos.side = position_side
        pos.size = trade_amount
        pos.entry_price = price
        pos.entry_time = datetime.now(timezone.utc)
        pos.entry_prob = state.prob
        pos.time_remaining_at_entry = state.time_remaining

        size_label = {0.25: "SM", 0.5: "MD", 1.0: "LG"}.get(action.size_multiplier, "")
        print(
            f"    [LIVE] OPEN {market.asset} {position_side} ({size_label}) ${trade_amount:.0f} @ {price:.3f}"
        )
        emit_trade(f"BUY_{size_label}", market.asset, trade_amount)

        # Update state
        state.last_action_status = "pending" if self.order_type == "GTC" else "success"
        state.consecutive_failures = 0

    async def _close_position_live(
        self, cid: str, pos: Position, state: MarketState, side: str
    ):
        """Close position by selling tokens."""
        m = self.markets[cid]

        # Determine token and price
        token_id = m.token_up if side == "UP" else m.token_down
        exit_price = state.prob if side == "UP" else (1 - state.prob)

        # Calculate shares
        shares = pos.size / pos.entry_price

        # Sign and submit SELL order (use FOK for exits)
        signed_order = self.order_signer.sign_order_dict(
            token_id=token_id,
            price=exit_price,
            size=shares,
            side=OrderSide.SELL,
            maker=self.config.safe_address,
            nonce=None,
            fee_rate_bps=0,
        )

        await self.transaction_client.post_order(signed_order, order_type="FOK")

        # Calculate PnL
        pnl = (exit_price - pos.entry_price) * shares

        # Record and clear position
        self._record_trade(pos, exit_price, pnl, f"CLOSE {side}", cid=cid)
        self.pending_rewards[cid] = pnl
        pos.size = 0
        pos.side = None

        # Update state
        state.last_action_status = "success"
        state.consecutive_failures = 0

    async def _cancel_pending_order(self, cid: str):
        """Cancel pending GTC order for a market."""
        if cid not in self.pending_orders:
            return

        order_id = self.pending_orders[cid].get("order_id")
        if not order_id:
            return

        try:
            await self.transaction_client.cancel_order(order_id)
            print(f"    [CANCEL] Order {order_id[:8]}... cancelled")
            del self.pending_orders[cid]
            if cid in self.order_timestamps:
                del self.order_timestamps[cid]
        except Exception as e:
            print(f"    [CANCEL ERROR] {order_id[:8]}...: {e}")

    def _handle_order_failure(self, cid: str, error_msg: str):
        """Handle order failure - fail fast strategy."""
        state = self.states.get(cid)
        if state:
            state.last_action_status = "failed"
            state.consecutive_failures += 1
            print(
                f"    [FAILURE] {error_msg} (consecutive: {state.consecutive_failures})"
            )

    async def _cancel_expired_orders(self, expired_cids: List[str]):
        """Cancel pending orders for expired markets."""
        for cid in expired_cids:
            if cid in self.pending_orders:
                await self._cancel_pending_order(cid)

    def _record_trade(
        self, pos: Position, price: float, pnl: float, action: str, cid: str = None
    ):
        """Record completed trade."""
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        print(f"    {action} {pos.asset} @ {price:.3f} | PnL: ${pnl:+.2f}")
        # Emit to dashboard
        emit_trade(action, pos.asset, pos.size, pnl)

        # Log to CSV
        if self.logger and pos.entry_time:
            duration = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
            binance_change = 0.0
            if cid and cid in self.open_prices:
                current = self.price_streamer.get_price(pos.asset)
                if current > 0 and self.open_prices[cid] > 0:
                    binance_change = (
                        current - self.open_prices[cid]
                    ) / self.open_prices[cid]

            self.logger.log_trade(
                asset=pos.asset,
                action="BUY" if "UP" in action else "SELL",
                side=pos.side or "UNKNOWN",
                entry_price=pos.entry_price,
                exit_price=price,
                size=pos.size,
                pnl=pnl,
                duration_sec=duration,
                time_remaining=pos.time_remaining_at_entry,
                prob_at_entry=pos.entry_prob,
                prob_at_exit=price,
                binance_change=binance_change,
                condition_id=cid,
            )

    def _compute_step_reward(
        self, cid: str, state: MarketState, action: Action, pos: Position
    ) -> float:
        """Compute reward signal for RL training - pure realized PnL."""
        # Only reward on position close - cleaner signal
        # Reward is set when trade closes in _execute_trade via self.pending_rewards
        return self.pending_rewards.pop(cid, 0.0)

    def close_all_positions(self):
        """Close all positions at current prices."""
        for cid, pos in self.positions.items():
            if pos.size > 0:
                state = self.states.get(cid)
                if state:
                    price = state.prob
                    shares = pos.size / pos.entry_price
                    if pos.side == "UP":
                        pnl = (price - pos.entry_price) * shares
                    else:
                        exit_down_price = 1 - price
                        pnl = (exit_down_price - pos.entry_price) * shares

                    self._record_trade(
                        pos, price, pnl, f"FORCE CLOSE {pos.side}", cid=cid
                    )
                    self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                    pos.size = 0
                    pos.side = None

    async def close_all_positions_async(self):
        """Close all positions asynchronously for live trading."""
        tasks = []
        for cid, pos in self.positions.items():
            if pos.size > 0:
                state = self.states.get(cid)
                if state:
                    tasks.append(self._close_position_live(cid, pos, state, pos.side))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def update_state(self, cid: str, m: Market, state: MarketState, time_now: datetime):

        # Update state from orderbook - CRITICAL for 15-min
        ob = self.orderbook_streamer.get_orderbook(cid, "UP")
        if ob and ob.mid_price:
            state.prob = ob.mid_price
            state.prob_history.append(ob.mid_price)
            if len(state.prob_history) > 100:
                state.prob_history = state.prob_history[-100:]
            state.best_bid = ob.best_bid or 0.0
            state.best_ask = ob.best_ask or 0.0
            state.spread = ob.spread or 0.0

            # Orderbook imbalance - L1 (top of book)
            if ob.bids and ob.asks:
                bid_vol_l1 = ob.bids[0][1] if ob.bids else 0
                ask_vol_l1 = ob.asks[0][1] if ob.asks else 0
                total_l1 = bid_vol_l1 + ask_vol_l1
                state.order_book_imbalance_l1 = (
                    (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0.0
                )

                # Orderbook imbalance - L5 (depth)
                bid_vol_l5 = sum(size for _, size in ob.bids[:5])
                ask_vol_l5 = sum(size for _, size in ob.asks[:5])
                total_l5 = bid_vol_l5 + ask_vol_l5
                state.order_book_imbalance_l5 = (
                    (bid_vol_l5 - ask_vol_l5) / total_l5 if total_l5 > 0 else 0.0
                )

        # Update binance price
        binance_price = self.price_streamer.get_price(m.asset)
        state.binance_price = binance_price
        open_price = self.open_prices.get(cid, binance_price)
        if open_price > 0:
            state.binance_change = (binance_price - open_price) / open_price

        # Update futures data (focused on fast-updating features)
        futures = self.futures_streamer.get_state(m.asset)
        if futures:
            # Order flow - THE EDGE
            old_cvd = state.cvd
            state.cvd = futures.cvd
            state.cvd_acceleration = (
                (futures.cvd - old_cvd) / 1e6 if old_cvd != 0 else 0.0
            )
            state.trade_flow_imbalance = futures.trade_flow_imbalance

            # Ultra-short momentum
            state.returns_1m = futures.returns_1m
            state.returns_5m = futures.returns_5m
            state.returns_10m = futures.returns_10m  # Properly computed from klines

            # Microstructure - CRITICAL for 15-min
            state.trade_intensity = futures.trade_intensity
            state.large_trade_flag = futures.large_trade_flag

            # Volatility
            state.realized_vol_5m = (
                futures.realized_vol_1h / 3.5 if futures.realized_vol_1h > 0 else 0.0
            )
            state.vol_expansion = futures.vol_ratio - 1.0

            # Regime context (slow but useful for context)
            state.vol_regime = 1.0 if futures.realized_vol_1h > 0.01 else 0.0
            state.trend_regime = 1.0 if abs(futures.returns_1h) > 0.005 else 0.0

        # Time remaining - CRITICAL
        state.time_remaining = (m.end_time - time_now).total_seconds() / 900

        # Update position info in state
        pos = self.positions.get(cid)
        if pos and pos.size > 0:
            state.has_position = True
            state.position_side = pos.side
            shares = pos.size / pos.entry_price
            if pos.side == "UP":
                state.position_pnl = (state.prob - pos.entry_price) * shares
            else:
                current_down_price = 1 - state.prob
                state.position_pnl = (current_down_price - pos.entry_price) * shares
        else:
            state.has_position = False
            state.position_side = None
            state.position_pnl = 0.0

        # Update order status tracking
        if cid in self.pending_orders and cid in self.order_timestamps:
            # Calculate pending order age (normalized to 0-1, max 15 min)
            age_seconds = (
                datetime.now(timezone.utc).timestamp() - self.order_timestamps[cid]
            )
            state.pending_order_age = min(age_seconds / 900, 1.0)

            # Auto-cancel stale orders (>5 min) for high-frequency trading
            if age_seconds > 300 and self.live_trading:
                # Schedule as fire-and-forget task (update_state is NOT async)
                asyncio.create_task(self._cancel_pending_order(cid))
        elif state.last_action_status == "pending":
            # Order filled or cancelled
            state.last_action_status = "success"
            state.pending_order_age = 0.0

        # consecutive_failures updated in _handle_order_failure

    async def decision_loop(self):
        """Main trading loop."""
        tick = 0
        tick_interval = 0.5  # 500ms ticks for faster decisions
        while self.running:
            await asyncio.sleep(tick_interval)
            tick += 1
            now = datetime.now(timezone.utc)

            # Check expired markets
            expired = [cid for cid, m in self.markets.items() if m.end_time <= now]

            # Cancel pending orders for expired markets
            if self.live_trading and expired:
                await self._cancel_expired_orders(expired)

            for cid in expired:
                print(f"\n  EXPIRED: {self.markets[cid].asset}")

                # RL: Store terminal experience with final PnL
                if isinstance(self.strategy, MLStrategy) and self.strategy.training:
                    state = self.states.get(cid)
                    prev_state = self.prev_states.get(cid)
                    pos = self.positions.get(cid)
                    if state and prev_state:
                        # Terminal reward is the realized PnL
                        terminal_reward = (
                            state.position_pnl if pos and pos.size > 0 else 0.0
                        )
                        self.strategy.store(
                            prev_state, Action.HOLD, terminal_reward, state, done=True
                        )

                    # Clean up prev_state
                    if cid in self.prev_states:
                        del self.prev_states[cid]

                del self.markets[cid]

            if not self.markets:
                print("\nAll markets expired. Refreshing...")
                if self.live_trading:
                    await self.close_all_positions_async()
                else:
                    self.close_all_positions()
                self.refresh_markets()
                if not self.markets:
                    print("No new markets. Waiting...")
                    await asyncio.sleep(30)
                continue

            # Update states and make decisions
            for cid, m in self.markets.items():
                state = self.states.get(cid)
                if not state:
                    continue

                self.update_state(cid, m, state, now)
                pos = self.positions.get(cid)

                # For non-RL strategies, force close near expiry as safety
                # For RL, let it learn to close on its own (gets penalty at expiry)
                if pos and pos.size > 0 and state.very_near_expiry:
                    if not isinstance(self.strategy, MLStrategy):
                        print(f"    ⏰ EARLY CLOSE: {pos.asset}")
                        close_action = Action.SELL if pos.side == "UP" else Action.BUY
                        self.execute_action(cid, close_action, state)
                        continue

                # Get action from strategy
                action = self.strategy.act(state)

                # RL: Store experience EVERY tick (dense learning signal)
                if isinstance(self.strategy, MLStrategy) and self.strategy.training:
                    prev_state = self.prev_states.get(cid)
                    if prev_state:
                        step_reward = self._compute_step_reward(cid, state, action, pos)
                        # Episode not done unless market expired
                        self.strategy.store(
                            prev_state, action, step_reward, state, done=False
                        )

                    # Deep copy state for next iteration
                    self.prev_states[cid] = copy.deepcopy(state)

                # Execute
                if action != Action.HOLD:
                    self.execute_action(cid, action, state)

            # Status update every 10 ticks (console), but dashboard every tick
            if tick % 10 == 0:
                self.print_status()
            else:
                # Update dashboard state every tick for responsiveness
                self._update_dashboard_only()

            # RL training: emit buffer progress every tick
            if isinstance(self.strategy, MLStrategy) and self.strategy.training:
                buffer_size = len(self.strategy.experiences)
                # Compute average reward from recent experiences
                avg_reward = None
                if buffer_size > 0:
                    recent_rewards = [
                        exp.reward for exp in self.strategy.experiences[-50:]
                    ]  # Last 50
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                emit_rl_buffer(buffer_size, self.strategy.buffer_size, avg_reward)

                # PPO update when buffer is full
                if buffer_size >= self.strategy.buffer_size:
                    # Get buffer rewards before update clears them
                    buffer_rewards = [exp.reward for exp in self.strategy.experiences]
                    metrics = self.strategy.update()
                    if metrics:
                        print(
                            f"  [RL] loss={metrics['policy_loss']:.4f} "
                            f"v_loss={metrics['value_loss']:.4f} "
                            f"ent={metrics['entropy']:.3f} "
                            f"kl={metrics['approx_kl']:.4f} "
                            f"ev={metrics['explained_variance']:.2f}"
                        )
                        # Send to dashboard
                        metrics["buffer_size"] = len(self.strategy.experiences)
                        update_rl_metrics(metrics)
                        # Log to CSV
                        if self.logger:
                            self.logger.log_update(
                                metrics=metrics,
                                buffer_rewards=buffer_rewards,
                                cumulative_pnl=self.total_pnl,
                                cumulative_trades=self.trade_count,
                                cumulative_wins=self.win_count,
                            )

    def _update_dashboard_only(self):
        """Update dashboard state without printing to console."""
        now = datetime.now(timezone.utc)
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                vel = state._velocity()
                dashboard_markets[cid] = {
                    "asset": m.asset,
                    "prob": state.prob,
                    "time_left": mins_left,
                    "velocity": vel,
                }
                if pos:
                    dashboard_positions[cid] = {
                        "side": pos.side,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                    }

        update_dashboard_state(
            strategy_name=self.strategy.name,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count,
            positions=dashboard_positions,
            markets=dashboard_markets,
        )

    def print_status(self):
        """Print current status."""
        now = datetime.now(timezone.utc)
        win_rate = self.win_count / max(1, self.trade_count) * 100

        print(f"\n[{now.strftime('%H:%M:%S')}] {self.strategy.name.upper()}")
        print(
            f"  PnL: ${self.total_pnl:+.2f} | Trades: {self.trade_count} | Win: {win_rate:.0f}%"
        )

        # Prepare dashboard data
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                pos_str = (
                    f"{pos.side} ${pos.size:.0f}" if pos and pos.size > 0 else "FLAT"
                )
                vel = state._velocity()
                print(
                    f"  {m.asset}: prob={state.prob:.3f} vel={vel:+.3f} | {pos_str} | {mins_left:.1f}m"
                )

                # Dashboard data
                dashboard_markets[cid] = {
                    "asset": m.asset,
                    "prob": state.prob,
                    "time_left": mins_left,
                    "velocity": vel,
                }
                if pos:
                    dashboard_positions[cid] = {
                        "side": pos.side,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                    }

        # Update dashboard
        update_dashboard_state(
            strategy_name=self.strategy.name,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count,
            positions=dashboard_positions,
            markets=dashboard_markets,
        )

    def print_final_stats(self):
        """Print final results."""
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Strategy: {self.strategy.name}")
        print(f"Total PnL: ${self.total_pnl:+.2f}")
        print(f"Trades: {self.trade_count}")
        print(f"Win Rate: {self.win_count / max(1, self.trade_count) * 100:.1f}%")

    async def run(self):
        """Run the trading engine."""
        self.running = True

        # Derive API credentials for authenticated endpoints
        print(f"[DEBUG] live_trading={self.live_trading}, transaction_client={self.transaction_client is not None}")
        if self.live_trading and self.transaction_client:
            print("[AUTH] Deriving L2 API credentials...")
            try:
                api_creds = await self.transaction_client.derive_l2_api_credentials(
                    self.order_signer
                )
                self.transaction_client.set_api_credentials(api_creds)
                print("[AUTH] ✓ API credentials configured")
            except Exception as e:
                print(f"[AUTH ERROR] Failed to derive credentials: {e}")
                import traceback
                traceback.print_exc()

        self.refresh_markets()

        if not self.markets:
            print("No markets to trade!")
            return

        tasks = [
            self.price_streamer.stream(),
            self.orderbook_streamer.stream(),
            self.futures_streamer.stream(),
            self.decision_loop(),
        ]

        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass  # Handle in finally
        finally:
            print("\n\nShutting down...")
            self.running = False
            self.price_streamer.stop()
            self.orderbook_streamer.stop()
            self.futures_streamer.stop()

            # Cancel all pending orders and close client
            if self.live_trading and self.transaction_client:
                try:
                    print("  Cancelling all pending orders...")
                    await self.transaction_client.cancel_all_orders()
                    await self.transaction_client.close()
                except Exception as e:
                    print(f"  Error during order cleanup: {e}")

            # Close all positions
            if self.live_trading:
                await self.close_all_positions_async()
            else:
                self.close_all_positions()

            self.print_final_stats()

            # Save RL model if training
            if isinstance(self.strategy, MLStrategy) and self.strategy.training:
                self.strategy.save("rl_model")
                print("  [RL] Model saved to rl_model.safetensors")
