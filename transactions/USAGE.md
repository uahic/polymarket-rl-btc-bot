# Async Order Client Usage Guide

## Overview

The `order_client.py` module provides async HTTP clients for Polymarket APIs using `aiohttp`. This ensures non-blocking operations in your decision loop.

## Integration with engine.py

### 1. Initialize in TradingEngine.__init__

```python
from transactions.order_client import AsyncClobClient, AsyncRelayerClient, ApiCredentials

class TradingEngine:
    def __init__(self, strategy: BaseStrategy, config: Config, trade_size: float = 10.0):
        # ... existing code ...

        # Initialize async order clients
        self.order_client = AsyncClobClient(
            host=config.clob.host,
            chain_id=config.clob.chain_id,
            signature_type=config.clob.signature_type,
            funder=config.safe_address,
            api_creds=self._api_creds,  # if you have them
            builder_creds=config.builder,
            timeout=30
        )

        self.relayer_client = AsyncRelayerClient(
            host=config.relayer.host,
            chain_id=config.clob.chain_id,
            builder_creds=config.builder,
            timeout=60
        )
```

### 2. Use in decision_loop with await

```python
async def decision_loop(self):
    """Main trading loop."""
    tick = 0
    tick_interval = 0.5

    while self.running:
        await asyncio.sleep(tick_interval)
        tick += 1
        now = datetime.now(timezone.utc)

        # Update states and make decisions
        for cid, m in self.markets.items():
            state = self.states.get(cid)
            if not state:
                continue

            self.update_state(cid, m, state, now)

            # Get action from strategy
            action = self.strategy.act(state)

            # Execute via order client (non-blocking!)
            if action != Action.HOLD:
                try:
                    # Example: Place order asynchronously
                    result = await self.order_client.post_order(
                        signed_order=your_signed_order,
                        order_type="GTC"
                    )
                    print(f"Order placed: {result.get('orderID')}")

                    # While this HTTP call happens, other tasks keep running:
                    # - price_streamer keeps updating prices
                    # - orderbook_streamer keeps updating
                    # - futures_streamer keeps processing

                except Exception as e:
                    print(f"Order failed: {e}")
                    # Continue to next market even if one fails
```

### 3. Cleanup in shutdown

```python
async def run(self):
    """Run the trading engine."""
    self.running = True
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
        pass
    finally:
        print("\n\nShutting down...")
        self.running = False

        # Close streamers
        self.price_streamer.stop()
        self.orderbook_streamer.stop()
        self.futures_streamer.stop()

        # Close order clients (IMPORTANT!)
        if self.order_client:
            await self.order_client.close()
        if self.relayer_client:
            await self.relayer_client.close()

        self.close_all_positions()
        self.print_final_stats()

        # Save RL model if training
        if isinstance(self.strategy, MLStrategy) and self.strategy.training:
            self.strategy.save("rl_model")
            print("  [RL] Model saved to rl_model.safetensors")
```

## Key Methods

### AsyncClobClient

```python
# Get open orders
orders = await client.get_open_orders()

# Get order book for a token
book = await client.get_order_book(token_id="0x123...")

# Get market price
price = await client.get_market_price(token_id="0x123...")

# Post an order
result = await client.post_order(
    signed_order={"order": {...}, "signature": "0x..."},
    order_type="GTC"  # or "FOK", "GTD"
)

# Cancel an order
await client.cancel_order(order_id="0xabc...")

# Cancel all orders
await client.cancel_all_orders()

# Cancel orders for a specific market
await client.cancel_market_orders(market=condition_id)

# Get trade history
trades = await client.get_trades(token_id="0x123...", limit=100)
```

### AsyncRelayerClient

```python
# Deploy Safe wallet (gasless)
result = await relayer.deploy_safe(safe_address="0x...")

# Approve USDC (gasless)
result = await relayer.approve_usdc(
    safe_address="0x...",
    spender="0x...",
    amount=1000000  # 1 USDC (6 decimals)
)

# Approve token (gasless)
result = await relayer.approve_token(
    safe_address="0x...",
    token_id="0x...",
    spender="0x...",
    amount=1000
)
```

## Error Handling

All methods use automatic retry with exponential backoff (3 attempts by default).

```python
try:
    result = await client.post_order(signed_order, "GTC")
except ApiError as e:
    print(f"API error: {e}")
except AuthenticationError as e:
    print(f"Auth error: {e}")
except OrderError as e:
    print(f"Order error: {e}")
```

## Performance Benefits

### Before (synchronous requests):
```
[0.0s] Decision loop iteration starts
[0.1s] Call requests.post() to place order
[0.1s - 2.5s] 🔒 ENTIRE EVENT LOOP FROZEN 🔒
              - No price updates
              - No orderbook updates
              - No futures updates
[2.5s] Order completes, continue with STALE data
```

### After (async aiohttp):
```
[0.0s] Decision loop iteration starts
[0.1s] Call await client.post_order()
[0.1s] HTTP request sent, yields control to event loop
[0.1s - 2.5s] ✅ Event loop keeps running ✅
              - price_streamer continues updating
              - orderbook_streamer continues updating
              - futures_streamer continues processing
[2.5s] Order completes, resume with FRESH data
```

## Session Management

The clients use a single `aiohttp.ClientSession` that's reused across all requests:

- **Lazy initialization**: Session created on first request
- **Connection pooling**: Reuses HTTP connections
- **Automatic cleanup**: Call `await client.close()` in finally block

This is much more efficient than creating a new session per request.
