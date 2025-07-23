#!/usr/bin/env python3
"""
simulate.py

Core simulation logic for a single game simulation.
Handles environment creation, policy initialization, and the simulation loop.
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Tuple
from arena.utils.seeding import set_seed
from arena.games.registry import registry  # Games registry
from backends import initialize_llm_registry

from arena.agents.policy_manager import initialize_policies, policy_mapping_fn
from arena.utils.loggers import SQLiteLogger
from torch.utils.tensorboard import SummaryWriter

#TODO: count the illegal moves and log them
# from here: /Users/lucia/Desktop/LLM_research/open_spiel_arena/src/arena/envs/open_spiel_env.py

# TODO: randomize the initial player order for each game instead of always starting with player 0


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
    logger.info(f"Agent {agent_id} ({agent_model}) chose action: {chosen_action} with reasoning: {reasoning}")
    if flag == True:
       logger.error(f"Terminated due to illegal move: {chosen_action}.")


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
    writer = SummaryWriter(log_dir=f"runs/{game_name}") # Tensorboard writer

    # Loads the pyspiel game and the env simulator
    env = registry.make_env(game_name, config)

    for episode in range(config["num_episodes"]):
        episode_seed = seed + episode
        observation_dict, _ = env.reset(seed=episode_seed)
        terminated = truncated = False

        logger.info(f"Episode {episode + 1} started with seed {episode_seed}.")
        turn = 0

        while not (terminated or truncated):
            actions = {}
            for agent_id, observation in observation_dict.items():
                policy_key = policy_mapping_fn(agent_id) # Map agentID to policy key
                policy = policies_dict[policy_key]  # Policy class
                agent_logger = agent_loggers_dict[policy_key]  # Data logger
                agent_type = None
                agent_model = "None"

                # Find the agent config by index - handle both string and int keys
                for key, value in config["agents"].items():
                    if key.startswith("player_") and int(key.split("_")[1]) == agent_id:
                        agent_type = value["type"]
                        agent_model = value.get("model", "None")
                        break
                    elif str(key) == str(agent_id):
                        agent_type = value["type"]
                        agent_model = value.get("model", "None")
                        break

                start_time = time.perf_counter()
                action_metadata = policy(observation)  # Calls `__call__()` -> `_process_action()` -> `log_move()` #noq:E501
                duration = time.perf_counter() - start_time

                if isinstance(action_metadata, int): # Non-LLM agents
                    chosen_action = action_metadata
                    reasoning = "None"  # No reasoning for non-LLM agents
                else: # LLM agents
                    chosen_action = action_metadata.get("action", -1)  # Default to -1 if missing
                    reasoning = str(action_metadata.get("reasoning", "None") or "None")

                actions[agent_id] = chosen_action

                # Check if the chosen action is legal
                if chosen_action is None or chosen_action not in observation["legal_actions"]:
                    if agent_type == "llm":
                       log_llm_action(agent_id, agent_model, observation, chosen_action, reasoning, flag = True)
                    agent_logger.log_illegal_move(game_name=game_name, episode=episode + 1,turn=turn,
                                                   agent_id=agent_id, illegal_action=chosen_action,
                                                   reason=reasoning, board_state=observation["state_string"])
                    truncated = True
                    break  # exit the for-loop over agents

                # Loggins
                opponents = ", ".join(
                    f"{config['agents'][a_id]['type']}_{config['agents'][a_id].get('model', 'None').replace('-', '_')}"
                    for a_id in config["agents"] if a_id != agent_id
                )

                agent_logger.log_move(
                    game_name=game_name,
                    episode=episode + 1,
                    turn=turn,
                    action=chosen_action,
                    reasoning=reasoning,
                    opponent= opponents,  # Get all opponents
                    generation_time=duration,
                    agent_type=agent_type,
                    agent_model=agent_model,
                    seed = episode_seed
                )

                if agent_type == "llm":
                   log_llm_action(agent_id, agent_model, observation, chosen_action, reasoning)

            # Step forward in the environment #TODO: check if this works for turn-based games (track the agent playing)
            if not truncated:
                observation_dict, rewards_dict, terminated, truncated, _ = env.step(actions)
                turn += 1

        # Logging
        game_status = "truncated" if truncated else "terminated"
        logger.info(f"Game status: {game_status} with rewards dict: {rewards_dict}")

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
                if key.startswith("player_") and int(key.split("_")[1]) == agent_id:
                    agent_type = value["type"]
                    agent_model = value.get("model", "None")
                    break
                elif str(key) == str(agent_id):
                    agent_type = value["type"]
                    agent_model = value.get("model", "None")
                    break

            tensorboard_key = f"{agent_type}_{agent_model.replace('-', '_')}"
            writer.add_scalar(f"Rewards/{tensorboard_key}", reward, episode + 1)

        logger.info(f"Simulation for game {game_name}, Episode {episode + 1} completed.")
    writer.close()
    return "Simulation Completed"

# start tensorboard from the terminal:
# tensorboard --logdir=runs

# In the browser:
# http://localhost:6006/
