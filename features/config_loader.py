"""Utility functions for loading and validating feature configs."""

import yaml
from pathlib import Path
from typing import Dict, List
from .feature_registry import FeatureConfig, FeatureRegistry


def load_feature_config(path: str) -> FeatureConfig:
    """Load feature config from YAML file."""
    return FeatureRegistry.from_yaml(path)


def load_full_config(config_name: str, model: str) -> Dict:
    """
    Load full config (features + model + training) for a specific model.

    Args:
        config_name: Name of config file without .yaml extension (e.g., 'baseline', 'full')
        model: Model name (e.g., 'ppo_paper_v2', 'debug_features')

    Returns:
        Dict containing the full configuration

    Raises:
        FileNotFoundError: If config file not found

    Example:
        >>> config = load_full_config('baseline', model='ppo_paper_v2')
    """
    config_dir = Path(__file__).parent.parent / "config" / "feature_configs"
    config_path = config_dir / model / f"{config_name}.yaml"

    if not config_path.exists():
        # Provide helpful error message
        available = list_available_configs(model)
        raise FileNotFoundError(
            f"Config '{config_name}' not found for model '{model}'.\n"
            f"Path: {config_path}\n"
            f"Available configs for {model}: {', '.join(available) if available else 'none'}"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_config(config_path: str) -> bool:
    """Validate feature config file."""
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert 'name' in data, "Missing 'name' field"
        assert 'features' in data, "Missing 'features' section"
        assert 'model' in data, "Missing 'model' section"
        assert 'training' in data, "Missing 'training' section"

        # Validate feature config
        feature_config = FeatureConfig.from_yaml_dict(data['features'])

        # Check expected_input_dim if provided
        if 'expected_input_dim' in data:
            actual_dim = feature_config.get_effective_dim(include_action=True)
            expected_dim = data['expected_input_dim']
            if actual_dim != expected_dim:
                print(f"WARNING: expected_input_dim={expected_dim} but actual={actual_dim}")
                return False

        print(f"✓ Config '{data['name']}' is valid")
        print(f"  Enabled features: {feature_config.get_num_enabled()}")
        print(f"  Input mode: {feature_config.input_mode}")
        print(f"  Effective dim: {feature_config.get_effective_dim()}")
        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def list_available_configs(model: str) -> List[str]:
    """
    List all available feature config presets for a specific model.

    Args:
        model: Model name (e.g., 'ppo_paper_v2', 'debug_features')

    Returns:
        List of config names (without .yaml extension)

    Example:
        >>> configs = list_available_configs('ppo_paper_v2')
        >>> print(configs)  # ['baseline', 'full', 'minimal', 'ablation_no_orderflow']
    """
    config_dir = Path(__file__).parent.parent / "config" / "feature_configs" / model

    if not config_dir.exists():
        return []

    configs = []
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        configs.append(yaml_file.stem)

    return configs


def list_available_models() -> List[str]:
    """
    List all available models that have feature configs.

    Returns:
        List of model names

    Example:
        >>> models = list_available_models()
        >>> print(models)  # ['ppo_paper_v2', 'debug_features']
    """
    config_dir = Path(__file__).parent.parent / "config" / "feature_configs"

    if not config_dir.exists():
        return []

    models = []
    for subdir in sorted(config_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            models.append(subdir.name)

    return models
