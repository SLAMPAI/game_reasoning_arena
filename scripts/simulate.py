#!/usr/bin/env python3
"""
simulate.py

Core simulation logic for a single game simulation.
Handles environment creation, policy initialization, and the simulation loop.
"""

import logging
from typing import Dict, Any
from board_game_arena.arena.utils.seeding import set_seed
from board_game_arena.arena.games.registry import registry  # Games registry
from board_game_arena.backends import initialize_llm_registry

from board_game_arena.arena.agents.policy_manager import (
    initialize_policies, policy_mapping_fn
)
from board_game_arena.arena.utils.loggers import SQLiteLogger
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)

def log_llm_action(agent_id: int,
                   agent_model: str,
                   observation: Dict[str, Any],
                   chosen_action: int,
                   reasoning: str,
                   flag: bool = False
                   ) -> None:
    """Logs the LLM agent's decision."""
    logger.info("Board state: \n%s", observation['state_string'])
    logger.info(f"Legal actions: {observation['legal_actions']}")
    logger.info(
        f"Agent {agent_id} ({agent_model}) chose action: {chosen_action} "
        f"with reasoning: {reasoning}"
    )
    if flag == True:
       logger.error(f"Terminated due to illegal move: {chosen_action}.")


def compute_actions(
    env, player_to_agent: Dict[int, Any], observations: Dict[int, Any]
) -> Dict[int, int]:
    """
    Computes actions for all players using the RLLib-style approach.
    Handles both turn-based and simultaneous games consistently.

    Args:
        env: The game environment.
        player_to_agent (Dict[int, Any]): Mapping from player index to agent.
        observations (Dict[int, Any]): Dictionary mapping player IDs to
            observations.

    Returns:
        Dict[int, int]: A dictionary mapping player indices to selected
            actions.
    """
    def extract_action(agent_response):
        """Extract action from agent response (handle both int and dict)."""
        if isinstance(agent_response, int):
            return agent_response
        elif isinstance(agent_response, dict):
            return agent_response.get("action", -1)
        else:
            return -1

    if env.state.is_simultaneous_node():
        # Simultaneous-move game: All players act at once
        return {player: extract_action(player_to_agent[player](observations[player]))
                for player in player_to_agent}
    else:
        # Turn-based game: Only the current player acts
        current_player = env.state.current_player()
        return {current_player: extract_action(player_to_agent[current_player](observations[current_player]))}


def simulate_game(game_name: str, config: Dict[str, Any], seed: int) -> str:
    """
    Runs a game simulation, logs agent actions and final rewards to TensorBoard.

    Args:
        game_name: The name of the game.
        config: Simulation configuration.
        seed: Random seed for reproducibility.

    Returns:
        str: Confirmation that the simulation is complete.
    """

    # Set global seed for reproducibility across all random number generators
    set_seed(seed)

    # Initialize LLM registry
    initialize_llm_registry(config)

    # Initialize loggers for all agents
    logger.info(f"Initializing environment for {game_name}.")

    # Assign players to their policy classes
    policies_dict = initialize_policies(config, game_name, seed)

    # Initialize loggers and writers for all agents
    agent_loggers_dict = {}
    for agent_id, policy_name in enumerate(policies_dict.keys()):
        # Get agent config and pass it to the logger
        player_key = f"player_{agent_id}"
        default_config = {"type": "unknown", "model": "None"}
        agent_config = config["agents"].get(player_key, default_config)

        # Sanitize model name for filename use
        model_name = agent_config.get("model", "None")
        sanitized_model_name = model_name.replace("-", "_").replace("/", "_")
        agent_loggers_dict[policy_name] = SQLiteLogger(
            agent_type=agent_config["type"],
            model_name=sanitized_model_name
        )
    writer = SummaryWriter(log_dir=f"runs/{game_name}")  # Tensorboard writer

    # Create player_to_agent mapping for RLLib-style action computation
    player_to_agent = {}
    for i, policy_name in enumerate(policies_dict.keys()):
        player_to_agent[i] = policies_dict[policy_name]

    # Loads the pyspiel game and the env simulator
    env = registry.make_env(game_name, config)

    for episode in range(config["num_episodes"]):
        episode_seed = seed + episode
        observation_dict, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        rewards_dict = {}  # Initialize rewards_dict

        logger.info(
            "Episode %d started with seed %d.", episode + 1, episode_seed
            )
        turn = 0

        while not (terminated or truncated):
            # Use RLLib-style action computation
            try:
                action_dict = compute_actions(
                    env, player_to_agent, observation_dict
                )
            except Exception as e:
                logger.error("Error computing actions: %s", e)
                truncated = True
                break

            # Process each action for logging and validation
            for agent_id, chosen_action in action_dict.items():
                policy_key = policy_mapping_fn(agent_id)
                agent_logger = agent_loggers_dict[policy_key]
                observation = observation_dict[agent_id]

                # Get agent config for logging
                agent_type = None
                agent_model = "None"
                for key, value in config["agents"].items():
                    if (key.startswith("player_") and
                            int(key.split("_")[1]) == agent_id):
                        agent_type = value["type"]
                        agent_model = value.get("model", "None")
                        break

                # Check if the chosen action is legal
                if (chosen_action is None or
                        chosen_action not in observation["legal_actions"]):
                    if agent_type == "llm":
                        log_llm_action(
                            agent_id, agent_model, observation,
                            chosen_action, "Illegal action", flag=True
                        )
                    agent_logger.log_illegal_move(
                        game_name=game_name, episode=episode + 1, turn=turn,
                        agent_id=agent_id, illegal_action=chosen_action,
                        reason="Illegal action",
                        board_state=observation["state_string"]
                    )
                    truncated = True
                    break

                # Get reasoning if available (for LLM agents)
                reasoning = "None"
                if (agent_type == "llm" and
                        hasattr(player_to_agent[agent_id], 'last_reasoning')):
                    reasoning = getattr(
                        player_to_agent[agent_id], 'last_reasoning', "None"
                    )

                # Logging
                opponents_list = []
                for a_id in config["agents"]:
                    if a_id != f"player_{agent_id}":
                        agent_type = config['agents'][a_id]['type']
                        model = config['agents'][a_id].get('model', 'None')
                        model_clean = model.replace('-', '_')
                        opponents_list.append(f"{agent_type}_{model_clean}")
                opponents = ", ".join(opponents_list)

                agent_logger.log_move(
                    game_name=game_name,
                    episode=episode + 1,
                    turn=turn,
                    action=chosen_action,
                    reasoning=reasoning,
                    opponent=opponents,
                    generation_time=0.0,  # TODO: Add timing back
                    agent_type=agent_type,
                    agent_model=agent_model,
                    seed=episode_seed
                )

                if agent_type == "llm":
                    log_llm_action(
                        agent_id, agent_model, observation,
                        chosen_action, reasoning
                    )

            # Step forward in the environment
            if not truncated:
                (observation_dict, rewards_dict,
                 terminated, truncated, _) = env.step(action_dict)
                turn += 1

        # Logging
        game_status = "truncated" if truncated else "terminated"
        logger.info(
            "Game status: %s with rewards dict: %s", game_status, rewards_dict
        )

        for agent_id, reward in rewards_dict.items():
            policy_key = policy_mapping_fn(agent_id)
            agent_logger = agent_loggers_dict[policy_key]
            agent_logger.log_game_result(
                    game_name=game_name,
                    episode=episode + 1,
                    status=game_status,
                    reward=reward
                )
            # Tensorboard logging
            agent_type = "unknown"
            agent_model = "None"

            # Find the agent config by index - handle both string and int keys
            for key, value in config["agents"].items():
                if (key.startswith("player_") and
                        int(key.split("_")[1]) == agent_id):
                    agent_type = value["type"]
                    agent_model = value.get("model", "None")
                    break
                elif str(key) == str(agent_id):
                    agent_type = value["type"]
                    agent_model = value.get("model", "None")
                    break

            tensorboard_key = f"{agent_type}_{agent_model.replace('-', '_')}"
            writer.add_scalar(
                f"Rewards/{tensorboard_key}", reward, episode + 1
            )

        logger.info(
            "Simulation for game %s, Episode %d completed.",
            game_name, episode + 1
        )
    writer.close()
    return "Simulation Completed"

# start tensorboard from the terminal:
# tensorboard --logdir=runs

# In the browser:
# http://localhost:6006/
