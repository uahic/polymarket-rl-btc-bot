# FOK (Fill-Or-Kill) Order Support

## Overview

The simulator now supports **FOK (Fill-Or-Kill) orders** in addition to standard GTC (Good-Til-Cancelled) market orders.

## What is a FOK Order?

**FOK orders** have strict requirements:
- ✅ Must fill **completely and immediately**
- ✅ Must fill at **limit price or better**
- ❌ Otherwise gets **rejected (killed)**

This is different from GTC orders which are more flexible and can accept market prices with slippage.

## Key Differences: FOK vs GTC

| Feature | GTC (Market Order) | FOK (Limit Order) |
|---------|-------------------|-------------------|
| Fill Rate | ~95% base | ~80% base |
| Price Control | No (takes market price) | Yes (limit price required) |
| Slippage | 0-50 bps typical | Limited by limit price |
| Execution | Best effort | Immediate or reject |
| Use Case | Quick fills, liquid markets | Price discipline, volatile markets |

## Usage

### In Engine Configuration

```python
from engine import TradingEngine
from strategies.ppo_paper import PPOStrategy
from config import Config

# GTC orders (default)
engine = TradingEngine(
    strategy=PPOStrategy(),
    config=Config(),
    simulation_mode=True,
    # order_type defaults to "GTC"
)

# FOK orders
config = Config()
config.order_type = "FOK"  # Set order type

engine = TradingEngine(
    strategy=PPOStrategy(),
    config=config,
    simulation_mode=True,
)
```

### Direct Simulation

```python
from simulation.executor import SimulatedOrderExecutor

executor = SimulatedOrderExecutor(initial_balance=1000.0)

# GTC order
result_gtc = executor.simulate_order_fill(
    side="BUY",
    asset="BTC",
    size=10.0,
    current_prob=0.50,
    current_bid=0.495,
    current_ask=0.505,
    spread=0.01,
    order_book_imbalance=0.2,
    order_type="GTC",  # Default
)

# FOK order
result_fok = executor.simulate_order_fill(
    side="BUY",
    asset="BTC",
    size=10.0,
    current_prob=0.50,
    current_bid=0.495,
    current_ask=0.505,
    spread=0.01,
    order_book_imbalance=0.2,
    order_type="FOK",
    limit_price=0.510,  # Required for FOK
)
```

## FOK Fill Logic

### 1. **Balance Check**
Same as GTC - must have sufficient balance for order + fees.

### 2. **Limit Price Validation**
```python
if limit_price is None:
    return REJECT("fok_no_limit_price")
```

### 3. **Price Achievability Check**
For **BUY** orders:
```python
if current_ask > limit_price:
    return REJECT("fok_price_not_met")
```

For **SELL** orders (DOWN token):
```python
if (1 - current_bid) > limit_price:
    return REJECT("fok_price_not_met")
```

### 4. **Immediate Fill Probability**
FOK orders are harder to fill immediately:

```python
base_fill_rate = 0.80  # vs 0.95 for GTC

# Penalties
spread_penalty = min(spread / 0.05, 1.0)
size_penalty = min(size / 100, 0.6)  # Higher than GTC
imbalance_penalty = 0.0 to 0.3  # If trading against market flow

fill_probability = base_fill_rate * (1 - penalties)
```

### 5. **Fill Price**
If filled, FOK orders get:
- Base price = best available (ask for BUY, 1-bid for SELL)
- Small price improvement possible (0-5 bps)
- **Capped at limit price** (never worse than limit)

## Rejection Reasons

FOK orders have additional rejection reasons:

| Reason | Description |
|--------|-------------|
| `insufficient_balance` | Not enough capital (same as GTC) |
| `fok_no_limit_price` | FOK requires limit_price parameter |
| `fok_price_not_met` | Current market price > limit price |
| `fok_no_immediate_fill` | Can't fill immediately (liquidity/conditions) |

## Example Output

### GTC Order (Successful)
```
OPEN BTC UP (MD) $10 @ 0.523 [GTC] (slip: +12.3bps, bal: $842.15)
```

### FOK Order (Successful)
```
OPEN BTC UP (MD) $10 @ 0.508 [FOK] (slip: +3.2bps, bal: $842.15)
```

### FOK Order (Rejected)
```
✗ REJECTED [FOK]: fok_price_not_met (bal: $852.15)
✗ REJECTED [FOK]: fok_no_immediate_fill (bal: $852.15)
```

## When to Use FOK vs GTC

### Use **GTC** when:
- ✅ You want high fill rate (need to get in/out quickly)
- ✅ Market is liquid with tight spreads
- ✅ You accept market price + reasonable slippage
- ✅ Training agent to maximize fill rate

### Use **FOK** when:
- ✅ You want price discipline (protect against bad fills)
- ✅ Market is volatile (spreads widening)
- ✅ You can tolerate rejections
- ✅ Training agent to avoid overpaying

## Training Considerations

### FOK Orders and RL Training

**Pros:**
- Teaches agent price discipline
- Reduces catastrophic losses from wide spreads
- More realistic for professional trading

**Cons:**
- Higher rejection rate means sparser rewards
- Agent needs to learn when NOT to trade
- Slower learning initially

### Recommended Strategy

1. **Start with GTC** - Learn basic trading logic
2. **Switch to FOK** - Add price discipline once profitable
3. **Monitor statistics** - Compare fill rates and P&L

## Statistics Tracking

FOK-specific metrics in simulation statistics:

```python
stats = executor.get_statistics()

print(f"Fill Rate: {stats['fill_rate']*100:.1f}%")
print(f"Rejection Reasons:")
for reason, count in stats['rejection_reasons'].items():
    print(f"  - {reason}: {count}")
```

Example output:
```
Fill Rate: 78.3%
Rejection Reasons:
  - fok_no_immediate_fill: 15
  - fok_price_not_met: 8
  - insufficient_balance: 3
```

## Advanced: Adaptive Limit Pricing

For more sophisticated strategies, you can adjust limit prices based on market conditions:

```python
# Tight limit in calm markets
if state.spread < 0.01:
    limit_price = current_ask * 1.001  # Only +10 bps

# Looser limit in volatile markets
elif state.spread > 0.03:
    limit_price = current_ask * 1.005  # Up to +50 bps
```

This allows your agent to balance fill rate vs price discipline dynamically.

## Implementation Details

### Code Structure

1. **[executor.py](bots/polymarket-rl-btc-bot/simulation/executor.py)**:
   - `simulate_order_fill()` - Routes to appropriate handler
   - `_simulate_gtc_order()` - Handles GTC/market orders
   - `_simulate_fok_order()` - Handles FOK orders

2. **[engine.py](bots/polymarket-rl-btc-bot/engine.py)**:
   - Reads `self.order_type` from config
   - Calculates limit price for FOK (ask + buffer)
   - Passes to simulator

### Key Parameters

```python
# FOK-specific
fok_base_fill_rate = 0.80  # Lower than GTC
price_improvement_bps = 0-5  # Small improvements possible
limit_buffer = 0.002  # 20 bps above ask for auto-generated limits
```

## Testing

Test both order types:

```python
# Test script
executor = SimulatedOrderExecutor(1000.0)

# Same market conditions, different order types
conditions = {
    "side": "BUY",
    "asset": "BTC",
    "size": 10.0,
    "current_prob": 0.50,
    "current_bid": 0.495,
    "current_ask": 0.505,
    "spread": 0.01,
    "order_book_imbalance": 0.2
}

# GTC
result_gtc = executor.simulate_order_fill(**conditions, order_type="GTC")
print(f"GTC: {result_gtc}")

# FOK
result_fok = executor.simulate_order_fill(
    **conditions,
    order_type="FOK",
    limit_price=0.508  # Willing to pay 0.508
)
print(f"FOK: {result_fok}")
```

## Troubleshooting

**Q: All FOK orders are getting rejected**
- Check limit prices - they might be too tight
- Increase limit buffer: `limit_price = ask * 1.005` (50 bps)
- Check market conditions (wide spreads?)

**Q: FOK fill rate too low (<50%)**
- Normal in volatile markets with wide spreads
- Consider using GTC in these conditions
- Or increase limit price tolerance

**Q: No difference between FOK and GTC**
- Check that `order_type` is being passed correctly
- Verify limit_price is set for FOK
- Markets might be very liquid (tight spreads, both fill easily)
