"""Feature registry and configuration management."""

from dataclasses import dataclass
from typing import Dict, List
import yaml


@dataclass
class FeatureDefinition:
    """Metadata for a single feature."""
    name: str
    index: int  # Position in 26-feature vector
    group: str  # momentum, order_flow, etc.
    description: str
    default_enabled: bool


@dataclass
class FeatureConfig:
    """Configuration for which features are enabled."""
    enabled_features: Dict[str, bool]
    input_mode: str  # "auto_adjust" or "zero_fill"

    def get_enabled_indices(self) -> List[int]:
        """Returns indices of enabled features in order."""
        enabled = []
        for feat in FeatureRegistry.FEATURES:
            if self.enabled_features.get(feat.name, feat.default_enabled):
                enabled.append(feat.index)
        return enabled

    def get_num_enabled(self) -> int:
        """Number of enabled features."""
        return sum(self.enabled_features.values())

    def get_effective_dim(self, include_action: bool = True) -> int:
        """Input dimension for model (market features + action)."""
        if self.input_mode == "zero_fill":
            # Zero-fill mode always returns 26-dim
            market_dim = 26
        else:  # auto_adjust
            # Auto-adjust returns only enabled features
            market_dim = self.get_num_enabled()
        return market_dim + 3 if include_action else market_dim

    @classmethod
    def from_yaml_dict(cls, data: Dict) -> 'FeatureConfig':
        """Load from YAML dict (flattened feature groups)."""
        enabled = {}
        input_mode = data.get('input_mode', 'auto_adjust')

        # Flatten grouped features
        for group_name, features in data.items():
            if group_name == 'input_mode':
                continue
            if isinstance(features, dict):
                enabled.update(features)

        return cls(enabled_features=enabled, input_mode=input_mode)


class FeatureRegistry:
    """Registry of all 26 features with metadata."""

    FEATURES = [
        # Momentum (1-3)
        FeatureDefinition("returns_1m", 0, "momentum", "1-min return", True),
        FeatureDefinition("returns_5m", 1, "momentum", "5-min return", True),
        FeatureDefinition("returns_10m", 2, "momentum", "10-min return", True),

        # Order Flow (4-7)
        FeatureDefinition("ob_imbalance_l1", 3, "order_flow", "L1 OB imbalance", True),
        FeatureDefinition("ob_imbalance_l5", 4, "order_flow", "L5 OB imbalance", True),
        FeatureDefinition("trade_flow", 5, "order_flow", "Trade flow imbalance", True),
        FeatureDefinition("cvd_accel", 6, "order_flow", "CVD acceleration", True),

        # Microstructure (8-10)
        FeatureDefinition("spread_pct", 7, "microstructure", "Spread %", True),
        FeatureDefinition("trade_intensity", 8, "microstructure", "Trade intensity", True),
        FeatureDefinition("large_trade_flag", 9, "microstructure", "Large trade flag", True),

        # Volatility (11-12)
        FeatureDefinition("realized_vol_5m", 10, "volatility", "5m realized vol", True),
        FeatureDefinition("vol_expansion", 11, "volatility", "Vol expansion", True),

        # Position (13-16)
        FeatureDefinition("has_position", 12, "position", "Has position", True),
        FeatureDefinition("position_side", 13, "position", "Position side", True),
        FeatureDefinition("unrealized_pnl", 14, "position", "Unrealized PnL", True),
        FeatureDefinition("time_remaining", 15, "position", "Time remaining", True),

        # Regime (17-18)
        FeatureDefinition("vol_regime", 16, "regime", "Volatility regime", True),
        FeatureDefinition("trend_regime", 17, "regime", "Trend regime", True),

        # Transaction (19-21)
        FeatureDefinition("pending_order", 18, "transaction", "Pending order", True),
        FeatureDefinition("failed_order", 19, "transaction", "Failed order", True),
        FeatureDefinition("consecutive_failures", 20, "transaction", "Consecutive failures", True),

        # Capital (22)
        FeatureDefinition("available_balance", 21, "capital", "Available balance", True),

        # Time-of-day (23-26) - DISABLED BY DEFAULT
        FeatureDefinition("hour_sin", 22, "time_of_day", "Hour sine", False),
        FeatureDefinition("hour_cos", 23, "time_of_day", "Hour cosine", False),
        FeatureDefinition("dow_sin", 24, "time_of_day", "Day-of-week sine", False),
        FeatureDefinition("dow_cos", 25, "time_of_day", "Day-of-week cosine", False),
    ]

    @classmethod
    def get_baseline_config(cls) -> FeatureConfig:
        """Returns baseline config (22 market features, no time-of-day)."""
        enabled = {feat.name: feat.default_enabled for feat in cls.FEATURES}
        return FeatureConfig(enabled_features=enabled, input_mode="auto_adjust")

    @classmethod
    def get_full_config(cls) -> FeatureConfig:
        """Returns full config (all 26 features)."""
        enabled = {feat.name: True for feat in cls.FEATURES}
        return FeatureConfig(enabled_features=enabled, input_mode="auto_adjust")

    @classmethod
    def from_yaml(cls, path: str) -> FeatureConfig:
        """Load feature config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return FeatureConfig.from_yaml_dict(data['features'])
