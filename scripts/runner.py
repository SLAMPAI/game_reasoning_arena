#!/usr/bin/env python3

"""
runner.py

Entry point for game simulations.
Handles Ray initialization, SLURM environment variables, and orchestration.
"""

import os
import sys

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "src")
sys.path.insert(0, os.path.abspath(src_dir))


import logging
import subprocess
import ray
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

from simulate import simulate_game
from board_game_arena.arena.utils.cleanup import full_cleanup
from board_game_arena.arena.utils.seeding import set_seed
from board_game_arena.configs.config_parser import build_cli_parser, parse_config


# Set the soft and hard core file size limits to 0 (disable core dumps)
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename="run_logs.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_ray(config=None):
    """
    Initializes Ray if not already initialized.

    Args:
        config: Optional configuration dictionary containing Ray settings
    """
    if not ray.is_initialized():
        ray_config = config.get("ray_config", {}) if config else {}

        # Extract Ray initialization parameters
        init_params = {
            "ignore_reinit_error": True,
        }

        # Add optional parameters if specified
        if ray_config.get("num_cpus"):
            init_params["num_cpus"] = ray_config["num_cpus"]
        if ray_config.get("num_gpus"):
            init_params["num_gpus"] = ray_config["num_gpus"]
        if ray_config.get("object_store_memory"):
            init_params["object_store_memory"] = (
                ray_config["object_store_memory"]
            )
        if ray_config.get("include_dashboard") is not None:
            init_params["include_dashboard"] = ray_config["include_dashboard"]
        if ray_config.get("dashboard_port"):
            init_params["dashboard_port"] = ray_config["dashboard_port"]

        ray.init(**init_params)
        logger.info("Ray initialized with config: %s", init_params)


@ray.remote
def simulate_game_ray(
    game_name: str,
    config: Dict[str, Any],
    seed: int
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ray remote wrapper for parallel game simulation.
    Calls the standard simulate_game function.
    """
    return simulate_game(game_name, config, seed)


def run_simulation(config):
    """
    Orchestrates simulation runs across multiple games and agent matchups.
    Uses the provided configuration, sets up Ray if enabled, and collects
    simulation results.
    """

    seed = config.get("seed", 42)
    set_seed(seed)

    use_ray = config.get("use_ray", False)  # Default to False for stability
    if use_ray:
        initialize_ray(config)
        logger.info("Ray enabled - using distributed execution")
    else:
        logger.info("Ray disabled - using sequential execution")

    # Handle both single game and multiple games configuration
    game_configs = []
    if "env_config" in config:
        # Single game configuration (legacy)
        game_configs = [config["env_config"]]
    elif "env_configs" in config:
        # Multiple games configuration
        game_configs = config["env_configs"]
    else:
        raise ValueError(
            "Configuration must contain either 'env_config' or 'env_configs'"
        )

    # Prepare results collection
    all_results = []

    if use_ray and len(game_configs) > 1:
        # Use Ray for parallel execution of multiple games
        ray_futures = []
        for game_config in game_configs:
            game_name = game_config["game_name"]
            output_path = game_config.get(
                "output_path",
                f"results/{game_name}_simulation_results.json"
            )
            game_specific_config = {
                **config,  # Inherit global settings
                "env_config": game_config,  # Game configuration
                "max_game_rounds": game_config.get("max_game_rounds", None),
                "num_episodes": config.get("num_episodes", 1),
                "agents": config.get("agents", {}),
                "output_path": output_path,
            }

            # Check if we should parallelize episodes too
            num_episodes = config.get("num_episodes", 1)
            parallel_episodes = (
                config.get("parallel_episodes", False) and num_episodes > 1
            )

            if parallel_episodes:
                # Parallelize episodes within each game
                episode_futures = []
                for episode in range(num_episodes):
                    episode_config = {
                        **game_specific_config,
                        "num_episodes": 1
                    }
                    episode_seed = seed + episode
                    episode_futures.append(
                        simulate_game_ray.remote(
                            game_name, episode_config, episode_seed
                        )
                    )
                ray_futures.append((game_name, episode_futures))
            else:
                # Run entire game as one Ray task
                future = simulate_game_ray.remote(
                    game_name, game_specific_config, seed
                )
                ray_futures.append((game_name, [future]))

        # Collect results
        for game_name, futures in ray_futures:
            if isinstance(futures, list):
                # Multiple episode futures
                episode_results = ray.get(futures)
                all_results.extend(episode_results)
                logger.info(
                    "Parallel simulation results for %s completed "
                    "(%d episodes)",
                    game_name, len(episode_results)
                )
            else:
                # Single game future
                result = ray.get(futures)
                all_results.append(result)
                logger.info(
                    "Parallel simulation results for %s completed",
                    game_name
                )
    else:
        # Sequential execution (Ray disabled or single game)
        for game_config in game_configs:
            game_name = game_config["game_name"]
            output_path = game_config.get(
                "output_path",
                f"results/{game_name}_simulation_results.json"
            )
            game_specific_config = {
                **config,  # Inherit global settings
                "env_config": game_config,  # Game configuration
                "max_game_rounds": game_config.get("max_game_rounds", None),
                "num_episodes": config.get("num_episodes", 1),
                "agents": config.get("agents", {}),
                "output_path": output_path,
            }

            result = simulate_game(game_name, game_specific_config, seed)
            all_results.append(result)
            logger.info(
                "Sequential simulation results for %s completed",
                game_name
            )

    logger.info(
        "All simulations completed. Total results: %d",
        len(all_results)
    )
    return all_results


def main():
    """Main entry point."""

    parser = build_cli_parser()
    args = parser.parse_args()

    # Parse config once in main
    config = parse_config(args)
    logger.info(config)

    try:
        # Run simulation with parsed config
        print("Running simulation...")
        run_simulation(config)

        print("Running post-game processing...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(
            current_dir, "..", "analysis", "post_game_processing.py"
        )
        subprocess.run(["python3", script_path], check=True)

        print("Simulation completed.")

    finally:
        # Clean up resources - backend type no longer needed
        # since it's automatic
        full_cleanup("auto")


if __name__ == "__main__":
    main()


# TODO: add HF board in the README file

# TODO: count the illegal moves and log them
# from here: /Users/lucia/Desktop/LLM_research/open_spiel_arena/src/
# arena/envs/open_spiel_env.py

# TODO: randomize the initial player order for each game instead of
# always starting with player 0
