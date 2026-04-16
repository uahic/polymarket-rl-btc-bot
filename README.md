# Polymarket RL BTC Bot

A reinforcement learning trading bot for Polymarket BTC prediction markets. 

Ships with a simple PPO (Proximal Policy Optimization) implementation with a GRU-based actor-critic to trade 15-minute BTC binary outcome markets on the Polymarket CLOB.

Supports auto-discovery for custom python classes (for other RL implementations/bots) as well.

Supports offline training on historical data and live or paper-mode deployment.

---

## Features

- PPO actor-critic agent with GRU temporal encoding for sequential state modeling
- Paper trading (simulated) and live trading (on-chain order signing via Polymarket CLOB)
- Offline training on historical data, decoupled from live deployment
- Configurable feature sets: momentum, order flow, volatility, microstructure, regime, position, capital
- Automatic market discovery — finds active BTC 15-minute markets at runtime
- Real-time web dashboard for monitoring training and execution
- Model auto-save on SIGINT/SIGTERM
- Encrypted private key storage with password protection
- Per-model feature and hyperparameter configs

---

## Setup

```bash
pip install -r requirements_offline.txt
python setup.py   # configure credentials and generate config.yaml
```

Copy `.env.example` to `.env` and fill in your credentials:

```
POLY_PRIVATE_KEY=<your 64-char hex private key>
POLY_SAFE_ADDRESS=<your Polymarket Safe address>
```

For live trading with the builder program, also set:

```
POLY_BUILDER_API_KEY=...
POLY_BUILDER_API_SECRET=...
POLY_BUILDER_API_PASSPHRASE=...
```

---

## Training

Train offline on historical data:

```bash
python train_offline.py --assets BTC --episodes 5000 --output models/ppo_v2.pt
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--assets` | BTC | Asset(s) to train on |
| `--episodes` | — | Number of training episodes |
| `--output` | — | Path to save trained model |
| `--data-dir` | dataset/historical | Directory with historical data |
| `--resume-from` | auto | Resume from checkpoint |
| `--checkpoint-interval` | 1000 | Save checkpoint every N episodes |
| `--batch-size`, `--lr-actor`, `--lr-critic`, `--hidden-size` | — | Hyperparameters |

---

## Inference / Live Trading

Run with a pre-trained model:

```bash
# Paper trading (simulated, no real orders)
python run.py ppo_paper_v2 --load __auto__

# Live trading
python run.py ppo_paper_v2 --live --load models/ppo_paper_v2_BTC_20250222_143012.pth --size 10.0

# With online training and dashboard
python run.py ppo_paper_v2 --train --dashboard --load __auto__
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--live` | off | Enable live order execution |
| `--paper` | on | Explicit paper/simulation mode |
| `--load PATH` | — | Model path; `__auto__` loads latest `.pth` in `models/` |
| `--train` | off | Accumulate experience and update model during trading |
| `--size` | 1.0 | Trade size in USD |
| `--dashboard` | off | Enable web dashboard |
| `--port` | 5050 | Dashboard port |
| `--episode-length` | 1800 | Max steps per episode (500ms steps, ~15 min) |
| `--feature-config` | baseline | Path to feature YAML config |

Dashboard URL: `http://localhost:5050`

---

## Configuration

| File | Purpose |
|---|---|
| `config.yaml` | API keys, CLOB host, Safe address, RPC URL, chain ID |
| `config/trading_runner_config.yaml` | Trading loop params: assets, mode, episode length, logging |
| `config/feature_configs/ppo_paper_v2/` | Feature set configs (baseline, full, minimal, ablation) |
| `.env` | Environment variable overrides |

---

## Project Structure

```
run.py                    # entry point for trading
train_offline.py          # entry point for offline training
trading_runner.py         # main trading loop
strategies/               # PPO strategy implementations
environments/             # Gym trading environment
features/                 # feature computation
executors/                # paper and live order execution
data/                     # data sources and loaders
security/                 # credential management and order signing
models/                   # saved model checkpoints
```
