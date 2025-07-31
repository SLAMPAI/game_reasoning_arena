"""
Environment Initializer

Factory functions for creating and initializing OpenSpiel environments
based on configuration parameters.
"""

import gymnasium as gym

from typing import Dict, Any
from .open_spiel_env import OpenSpielEnv
from games.registry import registry # Initilizes an empty registry dict for the games
import logging

# Configure logger
logger = logging.getLogger(__name__)


def env_creator(game_name:str,config: Dict[str, Any]) -> OpenSpielEnv:
    """
    Creates and initializes an OpenSpiel environment based on the configuration.

    Args:
        game_name (str): The name of the game to simulate.
        Expects config to have:
      - "env_config" with at least "game_name" (and optionally "max_game_rounds").
      - "agents" dictionary (used here to determine player types).

    Returns:
        OpenSpielEnv: An instance of OpenSpielEnv configured for the specified game.

    Raises:
        KeyError: If required configuration keys are missing.
        ValueError: If the game name is not registered.
    """
    player_types = [agent["type"] for _, agent in sorted(config["agents"].items())]
    game_loader_cls = registry.get_game_loader(game_name)
    if game_loader_cls is None:
        raise ValueError(f"Game loader for '{game_name}' not found in registry.")
    game_loader = game_loader_cls()

    # Retrieve the environment simulator instance configured for the game
    env = registry.get_env_instance(
        game_name=game_name,
        game=game_loader,
        player_types=player_types,  #TODO: see if the environment really needs this!
        max_game_rounds=config["env_config"].get("max_game_rounds"),
        seed=config.get("seed", 42)
    )

    logger.info(f"Environment initialized: {game_name} with {len(player_types)} players.")

    return env


def register_env():
    """
    Registers the OpenSpiel environment with Gym-like API.
    """
    gym.register(
        id="OpenSpielEnv-v0",
        entry_point="envs.initializer:env_creator",
    )
