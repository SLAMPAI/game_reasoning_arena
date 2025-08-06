#!/usr/bin/env python3
"""
simulate.py

Core simulation logic for a single game simulation.
Handles environment creation, policy initialization, and the simulation loop.
"""

import logging
from typing import Dict, Any
from game_reasoning_arena.arena.utils.seeding import set_seed
from game_reasoning_arena.arena.games.registry import registry  # Games registry
from game_reasoning_arena.backends import initialize_llm_registry

from game_reasoning_arena.arena.agents.policy_manager import (
    initialize_policies, policy_mapping_fn
)
from game_reasoning_arena.arena.utils.loggers import SQLiteLogger
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
    logger.info("Legal actions: %s", observation['legal_actions'])
    logger.info(
        "Agent %s (%s) chose action: %s with reasoning: %s",
        agent_id, agent_model, chosen_action, reasoning
    )
    if flag:
        logger.error("Terminated due to illegal move: %s.", chosen_action)


def compute_actions(env, player_to_agent, observations):
    """
    Computes actions for all agents in the current state.

    Args:
        env: The environment (OpenSpiel env).
        player_to_agent: Dictionary mapping player IDs to agent instances.
        observations: Dictionary of observations for each player.

    Returns:
        Dictionary mapping player IDs to their chosen actions.
        Also stores reasoning in agent objects for later retrieval.
    """

    def extract_action_and_store_reasoning(player_id, agent_response):
        agent = player_to_agent[player_id]
        if isinstance(agent_response, dict) and "action" in agent_response:
            # Store reasoning in the agent object for later retrieval
            if "reasoning" in agent_response:
                agent.last_reasoning = agent_response["reasoning"]
            else:
                agent.last_reasoning = "None"
            return agent_response.get("action", -1)
        else:
            agent.last_reasoning = "None"
            return -1

    if env.state.is_simultaneous_node():
        # Simultaneous-move game: All players act at once
        actions = {}
        for player in player_to_agent:
            agent_response = player_to_agent[player](observations[player])
            actions[player] = extract_action_and_store_reasoning(
                player, agent_response)
        return actions
    else:
        # Turn-based game: Only the current player acts
        current_player = env.state.current_player()
        agent_response = player_to_agent[current_player](
            observations[current_player])
        return {current_player: extract_action_and_store_reasoning(
            current_player, agent_response)}


def simulate_game(game_name: str, config: Dict[str, Any], seed: int) -> str:
    """
    Runs a game simulation, logs agent actions and final rewards to
    TensorBoard.

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
    logger.info("Initializing environment for %s.", game_name)

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

                # Get agent config for logging - ensure we get the right
                # agent's config
                agent_type = None
                agent_model = "None"
                player_key = f"player_{agent_id}"
                if player_key in config["agents"]:
                    agent_config = config["agents"][player_key]
                    agent_type = agent_config["type"]
                    # Only set model for LLM agents
                    if agent_type == "llm":
                        agent_model = agent_config.get("model", "None")
                    else:
                        agent_model = "None"

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
                        opp_agent_type = config['agents'][a_id]['type']
                        model = config['agents'][a_id].get('model', 'None')
                        model_clean = model.replace('-', '_')
                        opponents_list.append(f"{opp_agent_type}_{model_clean}")
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
                    seed=episode_seed,
                    board_state=observation["state_string"]
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

            # Calculate opponents for this agent
            opponents_list = []
            for a_id in config["agents"]:
                if a_id != f"player_{agent_id}":
                    opp_agent_type = config['agents'][a_id]['type']
                    opp_model = config['agents'][a_id].get('model', 'None')
                    opp_model_clean = opp_model.replace('-', '_')
                    opponent_str = f"{opp_agent_type}_{opp_model_clean}"
                    opponents_list.append(opponent_str)
            opponents = ", ".join(opponents_list)

            agent_logger.log_game_result(
                    game_name=game_name,
                    episode=episode + 1,
                    status=game_status,
                    reward=reward,
                    opponent=opponents
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
