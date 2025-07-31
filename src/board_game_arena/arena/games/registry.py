"""
Game Registry

Game registry for OpenSpiel Arena.
This module provides a registry for game loaders and environments.
Games can be loaded and instantiated based on their configuration.
"""

from typing import Dict, Any, Callable, Type, Optional, List
from importlib import import_module


class GameRegistration:
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._loaders_imported = False

    def _ensure_loaders_imported(self):
        """Lazy import of all game loaders to populate the registry."""
        # This method is now a no-op since loaders are imported at module level
        self._loaders_imported = True

    def register(
        self,
        name: str,
        module_path: str,
        class_name: str,
        environment_path: str,
        display_name: str
    ) -> Callable[[Type], Type]:
        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(f"Game '{name}' is already registered.")
            self._registry[name] = {
                "module_path": module_path,
                "class_name": class_name,
                "environment_path": environment_path,
                "display_name": display_name,
                "config_class": cls
            }
            return cls
        return decorator

    def get_game_loader(self, name: str) -> Callable:
        # Ensure loaders are imported and registered
        self._ensure_loaders_imported()

        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(
                f"Game '{name}' not found. Available games: {available}")

        module_path = self._registry[name]["module_path"]
        class_name = self._registry[name]["class_name"]
        # Assumes every loader has a static load() method.
        method_name = "load"
        cls = getattr(import_module(module_path), class_name)
        return getattr(cls, method_name)

    def get_env_instance(self,
                         game_name: str,
                         game: Any,
                         player_types: List[str],
                         max_game_rounds: Optional[int] = None,
                         seed: Optional[int] = None) -> Any:
        # Ensure loaders are imported and registered
        self._ensure_loaders_imported() #TODO:check this is correct

        if game_name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(
                f"Game '{game_name}' not found. Available games: {available}")
        module_path, class_name = (self._registry[game_name]
                                   ["environment_path"].rsplit(".", 1))
        environment_class = getattr(import_module(module_path), class_name)
        return environment_class(
            game, game_name, player_types, max_game_rounds, seed)

    def make_env(self, game_name: str, config: Dict[str, Any]) -> Any:
        """
        Creates an environment instance for the given game.
        This function is analogous to gym.make().

        Args:
            game_name: The internal game name.
            config: The simulation configuration (must include keys such as
                'env_config', 'agents', and optionally 'seed').

        Returns:
            An initialized environment simulator instance.
        """
        # Get player types from agent configuration.
        agents_dict = config.get("agents")
        player_types = [
            agent["type"] for _, agent in sorted(agents_dict.items())
        ]
        max_game_rounds = config["env_config"].get("max_game_rounds")
        seed = config.get("seed", 42)

        # Call the static load() method.
        game_loader = self.get_game_loader(game_name)()

        # Retrieve the environment simulator class.
        env = self.get_env_instance(
            game_name=game_name,
            game=game_loader,
            player_types=player_types,  # See if this is still needed
            max_game_rounds=max_game_rounds,
            seed=seed
        )
        return env


# Singleton registry instance.
registry = GameRegistration()

# Import registered game loaders to populate the registry
from . import loaders  # noqa: E402,F401
