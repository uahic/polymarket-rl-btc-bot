# Simulation Mode Guide

## Overview

The trading bot now includes **realistic simulation mode** for training your PPO agent without spending real money. This mode simulates:

- ✅ **Balance tracking** - Starts with initial capital, tracks spending
- ✅ **Order rejections** - Insufficient balance, no liquidity
- ✅ **Realistic fills** - Based on spread, order book imbalance, order size
- ✅ **Slippage** - Market impact from order size and liquidity
- ✅ **Fees** - 0.2% taker fee (typical Polymarket)
- ✅ **Statistics** - Fill rate, average slippage, rejection reasons

## Usage

### Basic Usage (Default $1000 starting capital)

```python
from engine import TradingEngine
from strategies.ppo_paper import PPOStrategy
from config import Config

# Create strategy
strategy = PPOStrategy()
strategy.training = True  # Enable training mode

# Create engine with simulation enabled
engine = TradingEngine(
    strategy=strategy,
    config=Config(),
    trade_size=10.0,
    live_trading=False,  # Paper trading
    simulation_mode=True,  # Enable realistic simulation
    initial_balance=1000.0  # Start with $1000
)

# Run
await engine.run()
```

### Fetch Balance from API

If you want to use your actual Polymarket balance for simulation:

```python
engine = TradingEngine(
    strategy=strategy,
    config=Config(),
    trade_size=10.0,
    live_trading=False,
    simulation_mode=True,
    initial_balance=None  # Will fetch from API if transaction_client is configured
)
```

**Note**: This requires your API credentials to be set up. If the fetch fails, it falls back to $1000.

### Disable Simulation (Old Instant Fill Mode)

```python
engine = TradingEngine(
    strategy=strategy,
    config=Config(),
    live_trading=False,
    simulation_mode=False  # Use old instant-fill paper trading
)
```

## How It Works

### 1. **Balance Tracking**

The simulator starts with `initial_balance` and tracks:
- Order costs (size + fees)
- Realized P&L (when closing positions)
- Current available balance

The balance is fed into the agent's state via `state.available_balance`, allowing it to learn:
- Don't trade when balance is low
- Adjust position sizing based on capital
- Risk management

### 2. **Order Fill Simulation**

When you place an order, the simulator:

1. **Checks balance** - Rejects if `size + fee > balance`
2. **Simulates liquidity** - Fill probability depends on:
   - Spread (wide spread = lower fill rate)
   - Order size (large orders harder to fill)
   - Base fill rate: 95%

3. **Calculates fill price** with slippage:
   - Base price: `ask` for BUY, `bid` for SELL (DOWN token)
   - Slippage factors:
     - Order book imbalance (buying into demand = worse price)
     - Spread width
     - Order size
   - Typical slippage: 0-50 basis points

4. **Deducts balance** - Subtracts `size + 0.2% fee`

### 3. **Position Closing**

When closing a position:
- Calculates P&L based on entry vs exit price
- Returns `principal + P&L` to balance
- Balance can go to $0 but not negative

### 4. **State Integration**

The agent sees balance in every state via:

```python
state.available_balance  # Current USDC balance (normalized /1000 in features)
```

This is feature #22 in the state vector, allowing the agent to condition its policy on available capital.

## Statistics

At the end of a session, you'll see:

```
==================================================
SIMULATION STATISTICS
==================================================
Balance:           $856.32 (start: $1000.00)
Total P&L:         -$143.68 (-14.37%)
Total Fees:        $12.45
Orders:            127 (115 fills, 12 rejections)
Fill Rate:         90.6%
Avg Slippage:      18.32 bps

Rejection Reasons:
  - insufficient_balance: 10
  - no_liquidity: 2
==================================================
```

## Training Workflow

1. **Start with simulation** - Train agent without risk
2. **Monitor statistics** - Check fill rate, slippage, rejections
3. **Tune hyperparameters** - Adjust based on simulation performance
4. **Validate on held-out data** - Test on different market conditions
5. **Go live (optional)** - Switch to `live_trading=True` when confident

## Key Differences from Old Paper Mode

| Feature | Old Paper Mode | Simulation Mode |
|---------|---------------|-----------------|
| Balance tracking | ❌ No | ✅ Yes |
| Order rejections | ❌ No | ✅ Yes (balance, liquidity) |
| Fill price | Current price | Realistic (ask/bid + slippage) |
| Slippage | ❌ No | ✅ Yes (0-50 bps) |
| Fees | ❌ No | ✅ Yes (0.2%) |
| Agent sees balance | ❌ No | ✅ Yes |
| Statistics | Basic P&L | Full fill stats |

## Tips

1. **Start small** - Use `trade_size=5.0` or `10.0` when learning
2. **Monitor rejections** - High rejection rate means agent isn't learning capital constraints
3. **Check slippage** - High slippage means orders too large for liquidity
4. **Balance awareness** - Agent should learn to trade less as balance decreases
5. **Fill rate** - Should be >90%; if lower, market conditions are extreme

## Advanced: Custom Initial Balance

```python
# Simulate with different bankrolls to test robustness
for balance in [100, 500, 1000, 5000]:
    engine = TradingEngine(
        strategy=PPOStrategy(),
        config=Config(),
        simulation_mode=True,
        initial_balance=balance
    )
    await engine.run()
```

## Troubleshooting

**Q: Agent keeps getting "insufficient_balance" rejections**
- A: Reduce `trade_size` or increase `initial_balance`
- The agent should learn to avoid this by seeing `available_balance` in state

**Q: Fill rate is very low (<50%)**
- A: Market has extremely wide spreads or you're placing huge orders
- Check if `trade_size` is appropriate for market liquidity

**Q: No statistics printed**
- A: Make sure `simulation_mode=True` and `live_trading=False`

**Q: Want to disable simulation**
- A: Set `simulation_mode=False` to use old instant-fill mode
