"""
Policy Manager

Policy assignment, initialization, and mapping.
A policy is a class (human, llm, random) that receives an observation 
(board state) and returns an action (move).
"""

import logging
from typing import Dict, Any
from .agent_registry import AGENT_REGISTRY
from ..games.registry import registry

# Configure logger
logger = logging.getLogger(__name__)

def initialize_policies(config: Dict[str, Any],
                        game_name: str,
                        seed: int
                        ) -> Dict[str, Any]:
    """
    Dynamically assigns policies to agents and initializes them in a single step.

    This function:
      - Ensures `config["agents"]` is correctly assigned.
      - Initializes policies for all assigned agents.
      - Returns a dictionary of initialized policies keyed by policy IDs.

    Args:
        config (Dict[str, Any]): Simulation configuration.
        game_name (str): The game being played.
        seed (int): Random seed.

    Returns:
        Dict[str, Any]: A dictionary mapping policy names to agent instances.
    """
    # Assign LLM models to players in the game
    num_players = registry.get_game_loader(game_name)().num_players()

    logger.info("Initializing %d agents for %s", num_players, game_name)

    # Initialize policies based on configured agents
    policies = {}
    for i in range(num_players):
        # Convert numeric index to string key (0 -> "player_0")
        player_key = f"player_{i}"
        agent_config = config["agents"].get(player_key)

        agent_type = agent_config["type"].lower()

        if agent_type not in AGENT_REGISTRY:
            raise ValueError(f"Unsupported agent type: '{agent_type}'")

        agent_class = AGENT_REGISTRY[agent_type]

        if agent_type == "llm":
            model_name = agent_config.get("model")
            policies[f"policy_{i}"] = agent_class(
                model_name=model_name,
                game_name=game_name
            )
        elif agent_type == "random":
            policies[f"policy_{i}"] = agent_class(seed=seed)
        elif agent_type == "human":
            policies[f"policy_{i}"] = agent_class()

        logger.info("Assigned: policy_%d -> %s (%s)",
                    i, agent_type.upper(),
                    agent_config.get('model'))

    return policies


def policy_mapping_fn(agent_id: str) -> str:
    """
    Maps an agent ID to a policy key.

    Args:
        agent_id (str): The agent's identifier.

    Returns:
        str: The corresponding policy key (e.g., "policy_0").
    """
    agent_id_str = str(agent_id)
    index = agent_id_str.split("_")[-1]
    policy_key = f"policy_{index}"
    return policy_key
