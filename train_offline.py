"""
Offline training script using historical data.

This script trains RL agents on pre-downloaded historical data.
Training happens much faster than real-time since we can replay
historical episodes at full speed.

The same preprocessing (FeatureComputer) is used for both offline
training and live deployment, ensuring consistency.

Usage:
    # Basic training with defaults
    python train_offline.py

    # Train on specific assets with custom episodes
    python train_offline.py --assets BTC ETH --episodes 5000

    # Full customization
    python train_offline.py \
        --data-dir dataset/historical \
        --assets BTC ETH SOL \
        --episodes 10000 \
        --output models/ppo_custom.pt \
        --checkpoint-interval 1000 \
        --hidden-size 64 \
        --critic-hidden-size 96 \
        --lr-actor 1e-4 \
        --lr-critic 3e-4 \
        --gamma 0.95 \
        --buffer-size 256 \
        --batch-size 64

    # Quick test run
    python train_offline.py --assets BTC --episodes 100 --output models/test.pt

    # Resume from checkpoint
    python train_offline.py --resume-from models/checkpoint_1000.pt --episodes 10000

    # Auto-resume (finds latest checkpoint automatically)
    python train_offline.py --episodes 10000  # Will resume if checkpoints exist

See --help for all available options.
"""

import argparse
import asyncio
import logging
import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Any, Dict
from datetime import datetime
import yaml

# Environment imports
from environments.trading_gym import TradingGym
from data.sources import HistoricalSource
from features.computer import FeatureComputer
from features.config_loader import load_full_config
from features.feature_registry import FeatureConfig

# Import executor wrapper
from executors.executor_wrapper import GymExecutorWrapper

logger = logging.getLogger(__name__)


def save_training_state(
    output_path: str,
    episode: int,
    episode_rewards: List[float],
    episode_lengths: List[int],
    episode_pnls: List[float],
    update_count: int,
):
    """Save training progress metadata."""
    state_path = Path(output_path).parent / "training_state.json"
    state = {
        "last_episode": episode,
        "total_episodes_completed": episode + 1,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_pnls": episode_pnls,
        "update_count": update_count,
        "timestamp": datetime.now().isoformat(),
    }
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def load_training_state(output_path: str) -> Optional[Dict]:
    """Load training progress metadata if it exists."""
    state_path = Path(output_path).parent / "training_state.json"
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return None


async def train_offline(
    strategy: Any,
    feature_config: FeatureConfig,
    data_dir: str,
    assets: List[str],
    num_episodes: int,
    output_path: str,
    checkpoint_interval: int = 1000,
    resume_from: Optional[str] = None,
):
    """
    Train RL agent on historical data.

    Args:
        strategy: RL strategy instance (must have act, store, reset methods)
        feature_config: Feature configuration for FeatureComputer
        data_dir: Directory with historical data
        assets: List of assets to train on
        num_episodes: Number of episodes to train
        output_path: Where to save final model
        checkpoint_interval: Save checkpoint every N episodes
        resume_from: Path to checkpoint to resume from (if None, checks for latest)
    """
    # Check for existing training state
    start_episode = 0
    episode_rewards = []
    episode_lengths = []
    episode_pnls = []
    update_count = 0

    if resume_from is not None or Path(output_path).parent.exists():
        # Try to load training state
        training_state = load_training_state(output_path)

        if training_state:
            start_episode = training_state["last_episode"] + 1
            episode_rewards = training_state.get("episode_rewards", [])
            episode_lengths = training_state.get("episode_lengths", [])
            episode_pnls = training_state.get("episode_pnls", [])
            update_count = training_state.get("update_count", 0)

            # Find the checkpoint file
            if resume_from is None:
                checkpoint_file = Path(output_path).parent / f"checkpoint_{training_state['last_episode'] + 1}.pt"
                if not checkpoint_file.exists():
                    # Try to find the latest checkpoint
                    checkpoint_files = sorted(Path(output_path).parent.glob("checkpoint_*.pt"))
                    if checkpoint_files:
                        checkpoint_file = checkpoint_files[-1]
                        start_episode = int(checkpoint_file.stem.split('_')[1])
                resume_from = str(checkpoint_file) if checkpoint_file.exists() else None

            if resume_from and Path(resume_from).exists():
                logger.info(f"RESUMING TRAINING | Checkpoint: {resume_from} | From episode: {start_episode} | Episodes completed: {len(episode_rewards)} | Updates: {update_count}")

                # Load model checkpoint
                if hasattr(strategy, 'load'):
                    strategy.load(resume_from)
                    logger.info("Model weights loaded")
            else:
                logger.warning("Training state found but no checkpoint file - starting fresh from episode 0")
                start_episode = 0
                episode_rewards = []
                episode_lengths = []
                episode_pnls = []
                update_count = 0

    logger.info(f"OFFLINE TRAINING | Strategy: {strategy.__class__.__name__} | Data: {data_dir} | Assets: {', '.join(assets)} | Episodes: {num_episodes} (from {start_episode}) | Output: {output_path}")

    # Initialize components
    logger.info("Initializing environment...")

    data_source = HistoricalSource(
        data_dir=data_dir,
        assets=assets,
        episode_length=1800,  # 15 min @ 500ms ticks
        random_start=True,
    )

    executor = GymExecutorWrapper(default_order_size=1.0)
    feature_computer = FeatureComputer(feature_config)

    env = TradingGym(
        data_source=data_source,
        executor=executor,
        feature_computer=feature_computer,
        initial_balance=1000.0,
        max_episode_steps=1800,
        shaping_reward_coef=0.01,
        normalize_rewards=True,
    )

    # Enable training mode
    if hasattr(strategy, 'train'):
        strategy.train()
        logger.info("Strategy set to training mode")

    # Training loop
    episodes_to_train = num_episodes - start_episode
    logger.info(f"Starting training for {episodes_to_train} episodes (from {start_episode} to {num_episodes})")

    for episode in range(start_episode, num_episodes):
        obs, info = env.reset()
        strategy.reset()

        done = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0

        while not (done or truncated):
            # Get action from strategy
            action = strategy.act(obs)

            # Convert Action enum to int for storage (PPO expects int)
            action_int = action.value if hasattr(action, 'value') else action

            # Step environment
            next_obs, reward, done, truncated, info = await env.step(action)

            # Store experience (use int for PPO)
            strategy.store(obs, action_int, reward, next_obs, done or truncated)

            # Update if buffer full
            if hasattr(strategy, 'should_update') and strategy.should_update():
                metrics = strategy.update() if hasattr(strategy, 'update') else None
                update_count += 1

                if metrics and update_count % 10 == 0:
                    logger.info(f"Update {update_count} | Policy Loss: {metrics.get('policy_loss', 0):.4f} | Value Loss: {metrics.get('value_loss', 0):.4f} | Entropy: {metrics.get('entropy', 0):.4f}")

            episode_reward += reward
            episode_length += 1
            obs = next_obs

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_pnls.append(info.get("episode_pnl", 0.0))

        # Logging
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_pnls = episode_pnls[-10:]
            buffer_size = len(strategy.experiences) if hasattr(strategy, 'experiences') else 0
            buffer_max = strategy.buffer_size if hasattr(strategy, 'buffer_size') else 0
            logger.info(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {np.mean(recent_rewards):.2f} | Avg PnL: ${np.mean(recent_pnls):.2f} | Buffer: {buffer_size}/{buffer_max}")

        # Checkpointing
        if (episode + 1) % checkpoint_interval == 0 and hasattr(strategy, 'save'):
            checkpoint_file = Path(output_path).parent / f"checkpoint_{episode + 1}.pt"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving checkpoint to {checkpoint_file}")
            strategy.save(str(checkpoint_file))

            # Save training state
            save_training_state(
                output_path=output_path,
                episode=episode,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                episode_pnls=episode_pnls,
                update_count=update_count,
            )

    # Save final model
    logger.info(f"Training complete! | Total episodes: {num_episodes} | Total updates: {update_count} | Avg reward: {np.mean(episode_rewards):.2f} | Avg PnL: ${np.mean(episode_pnls):.2f}")

    if hasattr(strategy, 'save'):
        logger.info(f"Saving final model to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        strategy.save(output_path)

        # Save final training state
        save_training_state(
            output_path=output_path,
            episode=num_episodes - 1,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            episode_pnls=episode_pnls,
            update_count=update_count,
        )

    logger.info(f"Next steps: Evaluate model, deploy to paper trading, monitor performance. To resume: python train_offline.py --resume-from {output_path}")


async def main():
    """Main entry point for offline training."""
    parser = argparse.ArgumentParser(
        description="Train RL agent on historical market data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help="Feature configuration name (e.g., 'baseline', 'full', 'minimal')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_paper_v2",
        help="Model name for config loading (e.g., 'ppo_paper_v2', 'debug_features')",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/historical",
        help="Directory containing historical data",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC"],
        help="Assets to train on (e.g., BTC ETH SOL)",
    )

    # Training arguments
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ppo_pretrained.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Actor hidden layer size",
    )
    parser.add_argument(
        "--critic-hidden-size",
        type=int,
        default=96,
        help="Critic hidden layer size",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=5,
        help="Number of past states for temporal encoding",
    )
    parser.add_argument(
        "--temporal-dim",
        type=int,
        default=32,
        help="Temporal encoder output dimension",
    )
    parser.add_argument(
        "--lr-actor",
        type=float,
        default=1e-4,
        help="Actor learning rate",
    )
    parser.add_argument(
        "--lr-critic",
        type=float,
        default=3e-4,
        help="Critic learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="PPO clipping parameter",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.03,
        help="Entropy coefficient for exploration",
    )
    parser.add_argument(
        "--value-coef",
        type=float,
        default=0.5,
        help="Value loss coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=256,
        help="Experience buffer size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for PPO updates",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs per PPO update",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="Target KL divergence for early stopping",
    )

    args = parser.parse_args()

    # Load full config from YAML
    logger.info(f"Loading config '{args.config}' for model '{args.model}'")
    config_data = load_full_config(args.config, model=args.model)

    # Parse sections
    feature_config = FeatureConfig.from_yaml_dict(config_data['features'])
    model_params = config_data['model']
    training_params = config_data['training']

    # CLI args override YAML
    if args.hidden_size is not None:
        model_params['hidden_size'] = args.hidden_size
    if args.critic_hidden_size is not None:
        model_params['critic_hidden_size'] = args.critic_hidden_size
    if args.history_len is not None:
        model_params['history_len'] = args.history_len
    if args.temporal_dim is not None:
        model_params['temporal_dim'] = args.temporal_dim
    if args.lr_actor is not None:
        training_params['lr_actor'] = args.lr_actor
    if args.lr_critic is not None:
        training_params['lr_critic'] = args.lr_critic
    if args.gamma is not None:
        training_params['gamma'] = args.gamma
    if args.gae_lambda is not None:
        training_params['gae_lambda'] = args.gae_lambda
    if args.clip_epsilon is not None:
        training_params['clip_epsilon'] = args.clip_epsilon
    if args.entropy_coef is not None:
        training_params['entropy_coef'] = args.entropy_coef
    if args.value_coef is not None:
        training_params['value_coef'] = args.value_coef
    if args.max_grad_norm is not None:
        training_params['max_grad_norm'] = args.max_grad_norm
    if args.buffer_size is not None:
        training_params['buffer_size'] = args.buffer_size
    if args.batch_size is not None:
        training_params['batch_size'] = args.batch_size
    if args.n_epochs is not None:
        training_params['n_epochs'] = args.n_epochs
    if args.target_kl is not None:
        training_params['target_kl'] = args.target_kl

    logger.info(f"Loaded config: {config_data['name']}")
    logger.info(f"  {config_data['description']}")
    logger.info(f"  Enabled features: {feature_config.get_num_enabled()}/26")
    logger.info(f"  Input mode: {feature_config.input_mode}")

    # Import strategy here to avoid issues if imports fail
    from strategies.ppo_paper_v2 import PPOStrategyV2

    # Initialize PPO strategy with config
    logger.info(f"Initializing PPO | actor_hidden={model_params['hidden_size']} critic_hidden={model_params['critic_hidden_size']} history={model_params['history_len']} temporal_dim={model_params['temporal_dim']} lr_actor={training_params['lr_actor']} lr_critic={training_params['lr_critic']} gamma={training_params['gamma']} buffer={training_params['buffer_size']} batch={training_params['batch_size']}")

    strategy = PPOStrategyV2(
        feature_config=feature_config,
        hidden_size=model_params['hidden_size'],
        critic_hidden_size=model_params['critic_hidden_size'],
        history_len=model_params['history_len'],
        temporal_dim=model_params['temporal_dim'],
        lr_actor=training_params['lr_actor'],
        lr_critic=training_params['lr_critic'],
        gamma=training_params['gamma'],
        gae_lambda=training_params['gae_lambda'],
        clip_epsilon=training_params['clip_epsilon'],
        entropy_coef=training_params['entropy_coef'],
        value_coef=training_params['value_coef'],
        max_grad_norm=training_params['max_grad_norm'],
        buffer_size=training_params['buffer_size'],
        batch_size=training_params['batch_size'],
        n_epochs=training_params['n_epochs'],
        target_kl=training_params['target_kl'],
    )

    logger.info(f"Model input_dim: {strategy.input_dim}")

    # Start training
    await train_offline(
        strategy=strategy,
        feature_config=feature_config,
        data_dir=args.data_dir,
        assets=args.assets,
        num_episodes=args.episodes,
        output_path=args.output,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    asyncio.run(main())
