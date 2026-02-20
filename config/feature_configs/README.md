# Feature Configs - Model-Specific Configuration System

This directory contains feature configurations organized by model. Each model has its own subdirectory with one or more configuration presets.

## Directory Structure

```
feature_configs/
├── ppo_paper_v2/              # PPO model configs
│   ├── baseline.yaml          # 22 features (no time-of-day)
│   ├── full.yaml              # All 26 features
│   ├── minimal.yaml           # Minimal feature set
│   └── ablation_no_orderflow.yaml  # Ablation study config
├── debug_features/            # Debug model configs
│   ├── baseline.yaml          # Minimal features for debugging
│   └── full.yaml              # All features for debugging
└── README.md                  # This file
```

## Usage

### Loading Configs in Python

```python
from features.config_loader import load_full_config, list_available_configs, list_available_models

# Load a specific config for a model
config_data = load_full_config('baseline', model='ppo_paper_v2')

# List all available models
models = list_available_models()
# ['ppo_paper_v2', 'debug_features']

# List configs for a specific model
configs = list_available_configs('ppo_paper_v2')
# ['ablation_no_orderflow', 'baseline', 'full', 'minimal']
```

### Using in Training Scripts

```bash
# Train with baseline config for ppo_paper_v2 (default)
python train_offline.py --config baseline --model ppo_paper_v2

# Train with full features
python train_offline.py --config full --model ppo_paper_v2

# Use debug_features model
python train_offline.py --config baseline --model debug_features
```

## Config File Format

### PPO Model Configs

PPO configs include three sections: `features`, `model`, and `training`.

```yaml
name: "baseline"
description: "Baseline PPO configuration with 22 market features"

# Feature selection
features:
  input_mode: "auto_adjust"  # or "zero_fill"

  momentum:
    returns_1m: true
    returns_5m: true
    # ... more features

  # ... other feature groups

# Model architecture (PPO only)
model:
  hidden_size: 64
  critic_hidden_size: 96
  history_len: 5
  temporal_dim: 32

# Training hyperparameters (PPO only)
training:
  lr_actor: 1.0e-4
  lr_critic: 3.0e-4
  gamma: 0.9
  # ... more params

# Optional validation
expected_input_dim: 25
```

### Debug Model Configs

Debug configs only include the `features` section (no model or training):

```yaml
name: "debug_baseline"
description: "Minimal feature set for debugging"

features:
  input_mode: "auto_adjust"

  momentum:
    returns_1m: true
    # ... feature toggles

expected_input_dim: 16
```

## Feature Groups

All configs support the following feature groups:

- **momentum**: Short-term price movements (returns_1m, returns_5m, returns_10m)
- **order_flow**: Order book and trade flow metrics
- **microstructure**: Spread, trade intensity, large trades
- **volatility**: Realized volatility and expansion
- **position**: Current position state
- **regime**: Market regime indicators
- **transaction**: Order status and failures
- **capital**: Available balance
- **time_of_day**: Temporal encoding (hour/day of week)

## Input Modes

- `auto_adjust`: Only include enabled features in the input vector
- `zero_fill`: Include all features, but zero out disabled ones

## Creating New Configs

### For an Existing Model

1. Navigate to the model's subdirectory (e.g., `ppo_paper_v2/`)
2. Create a new YAML file (e.g., `my_config.yaml`)
3. Follow the format of existing configs
4. Include all required sections for that model type

### For a New Model

1. Create a new subdirectory: `feature_configs/my_model/`
2. Add at least one config file (e.g., `baseline.yaml`)
3. Update your model class to:
   - Accept `FeatureConfig` in constructor
   - Save/load `model_name` in checkpoints
   - Optionally override `MODEL_NAME` if auto-derived name doesn't match directory name

#### MODEL_NAME Auto-Derivation

By default, `MODEL_NAME` is automatically derived from the class name by converting CamelCase to snake_case:

- `MyCustomModel` → `my_custom_model`
- `PPOStrategyV2` → `ppo_strategy_v2`
- `DebugFeatures` → `debug_features`

**When to Override:**
- If the auto-derived name doesn't match your desired directory name
- Example: `PPOStrategyV2` uses `MODEL_NAME = "ppo_paper_v2"` (not `ppo_strategy_v2`)

**When to Use Auto-Derivation:**
- If the class name naturally converts to the desired directory name
- Example: `DebugFeatures` → `debug_features` (perfect match, no override needed)

Example model class with auto-derivation:

```python
class MyCustomModel(MLStrategy):
    # MODEL_NAME auto-derives to "my_custom_model" - no override needed!

    def __init__(self, feature_config: FeatureConfig):
        super().__init__("my_strategy")
        self.feature_config = feature_config
        # ... initialization

    def save(self, path: str):
        checkpoint = {
            'model_name': self.MODEL_NAME,  # Uses auto-derived value
            'feature_config': {
                'enabled_features': self.feature_config.enabled_features,
                'input_mode': self.feature_config.input_mode,
            },
            # ... other fields
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        checkpoint = torch.load(path)

        # Verify model name
        if checkpoint['model_name'] != self.MODEL_NAME:
            raise ValueError("Model name mismatch")

        # Load feature config
        self.feature_config = FeatureConfig(**checkpoint['feature_config'])
```

Example with explicit override:

```python
class PPOStrategyV2(MLStrategy):
    MODEL_NAME = "ppo_paper_v2"  # Override auto-derived "ppo_strategy_v2"

    def __init__(self, feature_config: FeatureConfig):
        # ... same as above
```

## Naming Conventions

### Models (subdirectory names)
- Use snake_case
- Lowercase only
- Descriptive but concise
- Examples: `ppo_paper_v2`, `debug_features`, `dqn_baseline`

### Configs (YAML filenames)
- Use snake_case
- Lowercase only
- Descriptive of what makes this config unique
- Examples: `baseline.yaml`, `full.yaml`, `ablation_no_orderflow.yaml`

## Validation

Validate a config file:

```bash
python -c "from features.config_loader import load_full_config; load_full_config('baseline', 'ppo_paper_v2')"
```

## Checkpoints and Configs

**Important**: Checkpoints are self-contained and include:
- The full feature configuration used during training
- The model name for validation

This means:
- You can load a checkpoint without needing the original config file
- The model will validate that you're loading the correct checkpoint
- Feature configuration is preserved across training runs

When loading a checkpoint:

```python
strategy = PPOStrategyV2(feature_config=some_config, ...)
strategy.load('path/to/checkpoint.pth')
# Checkpoint's feature_config will override some_config if they differ
# Warning will be logged about the mismatch
```

## Migration from Flat Structure

Previous versions stored all configs in a flat directory:
```
feature_configs/
├── baseline.yaml
├── full.yaml
└── ...
```

This has been replaced with model-specific subdirectories. The new structure:
- Makes it clear which configs belong to which models
- Allows different models to have different config requirements
- Scales better as more models are added
- Provides better organization and discoverability

All existing configs have been moved to `ppo_paper_v2/` subdirectory.
