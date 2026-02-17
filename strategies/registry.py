"""Registry for auto-discovering and managing strategy classes."""

import logging
import inspect
import importlib
import pkgutil
from typing import Type, Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Registry for auto-discovering and managing strategy classes."""

    def __init__(self):
        self._strategies: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None):
        """
        Decorator to manually register a strategy.

        Args:
            name: Optional custom name for the strategy. If not provided,
                  uses the class name.

        Usage:
            @registry.register()
            class MyStrategy(BaseStrategy):
                pass
        """
        def decorator(cls):
            strategy_name = name or cls.__name__
            self._strategies[strategy_name] = cls
            return cls
        return decorator

    def discover(self, base_class: Type, package_name: str) -> Dict[str, Type]:
        """
        Auto-discover all subclasses of base_class in the given package.

        Args:
            base_class: The base class to search for subclasses
            package_name: Name of the package to search (e.g., 'strategies')

        Returns:
            Dictionary mapping strategy names to strategy classes
        """

        logger.info(f"Running Strategy discovery in package {package_name}")
        try:
            # Import the package
            package = importlib.import_module(package_name)
            package_path = package.__path__
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import package {package_name}: {e}")
            return self._strategies

        # Iterate through all modules in the package
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package_path,
            prefix=f"{package_name}."
        ):
            # Skip the base and registry modules to avoid circular imports
            if modname.endswith(('.base_strategy', '.registry')):
                continue

            # Import the module
            try:
                module = importlib.import_module(modname)
            except ImportError as e:
                logger.warning(f"Failed to import {modname}: {e}")
                continue

            # Find all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a subclass of base_class (but not base_class itself)
                # and that it's defined in this module (not imported from elsewhere)
                # and that it's not an abstract class
                if (issubclass(obj, base_class) and
                    obj is not base_class and
                    obj.__module__ == modname and
                    not inspect.isabstract(obj)):
                    self._strategies[obj.__name__] = obj

        return self._strategies

    def get(self, name: str) -> Optional[Type]:
        """
        Get a strategy class by name.

        Args:
            name: Name of the strategy

        Returns:
            Strategy class if found, None otherwise
        """
        return self._strategies.get(name)

    def list_all(self) -> Dict[str, Type]:
        """
        Get all registered strategies.

        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        return self._strategies.copy()

    def list_names(self) -> List[str]:
        """
        Get list of all registered strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def create(self, name: str, **kwargs):
        """
        Create an instance of a strategy by name.

        Args:
            name: Name of the strategy
            **kwargs: Arguments to pass to the strategy constructor

        Returns:
            Instance of the strategy

        Raises:
            KeyError: If strategy not found
        """
        strategy_class = self._strategies.get(name)
        if strategy_class is None:
            raise KeyError(f"Strategy '{name}' not found. Available strategies: {self.list_names()}")
        return strategy_class(**kwargs)

    def clear(self):
        """Clear all registered strategies."""
        self._strategies.clear()

    def __len__(self) -> int:
        """Return number of registered strategies."""
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return name in self._strategies

    def __repr__(self) -> str:
        return f"StrategyRegistry({len(self)} strategies: {self.list_names()})"
