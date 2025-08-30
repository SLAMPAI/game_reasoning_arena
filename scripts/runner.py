#!/usr/bin/env python3

"""
runner.py

Entry point for game simulations.
Handles Ray initialization, SLURM environment variables, and orchestration.
"""

import logging
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ray
from dotenv import load_dotenv

from simulate import simulate_game
from game_reasoning_arena.arena.utils.cleanup import full_cleanup
from game_reasoning_arena.arena.utils.seeding import set_seed
from game_reasoning_arena.configs.config_parser import (
    build_cli_parser,
    parse_config
)

# Ensure the src directory is in the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / ".." / "src"
sys.path.insert(0, str(src_dir.resolve()))

# Set the soft and hard core file size limits to 0 (disable core dumps)
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


def create_episode_tasks(
    game_name: str,
    game_config: Dict[str, Any],
    seed: int,
    num_episodes: int
) -> List[Any]:
    """
    Create Ray tasks for individual episodes of a game.

    Args:
        game_name: Name of the game
        game_config: Game-specific configuration
        seed: Base seed for random number generation
        num_episodes: Number of episodes to create tasks for

    Returns:
        List of Ray task futures
    """
    episode_tasks = []
    for episode in range(num_episodes):
        episode_config = {
            **game_config,
            "num_episodes": 1  # Each task handles only 1 episode
        }
        episode_seed = seed + episode
        episode_task = simulate_game_ray.remote(
            game_name, episode_config, episode_seed
        )
        episode_tasks.append(episode_task)
    return episode_tasks


def create_game_tasks(
    game_configs: List[Dict[str, Any]],
    base_config: Dict[str, Any],
    seed: int
) -> List[Tuple[str, List[Any]]]:
    """
    Create Ray tasks for all games, handling episode parallelization strategy.

    Args:
        game_configs: List of game configurations
        base_config: Base configuration dictionary
        seed: Random seed

    Returns:
        List of (game_name, task_futures) tuples
    """
    pending_game_tasks = []

    for game_config in game_configs:
        game_name = game_config["game_name"]
        game_specific_config = create_game_config(base_config, game_config)

        # Decide parallelization strategy for episodes
        num_episodes = base_config.get("num_episodes", 1)
        parallel_episodes = (
            base_config.get("parallel_episodes", False) and num_episodes > 1
        )

        if parallel_episodes:
            # Strategy 1: Parallelize episodes across multiple Ray tasks
            episode_tasks = create_episode_tasks(
                game_name, game_specific_config, seed, num_episodes
            )
            pending_game_tasks.append((game_name, episode_tasks))
        else:
            # Strategy 2: Sequential episodes within a single Ray task
            single_game_task = simulate_game_ray.remote(
                game_name, game_specific_config, seed
            )
            pending_game_tasks.append((game_name, [single_game_task]))

    return pending_game_tasks


def execute_parallel_simulations(
    game_configs: List[Dict[str, Any]],
    config: Dict[str, Any],
    seed: int
) -> List[Any]:
    """
    Execute simulations using Ray for parallel processing.

    Args:
        game_configs: List of game configurations
        config: Base configuration dictionary
        seed: Random seed

    Returns:
        List of simulation results
    """
    all_results = []

    # Create all Ray tasks
    pending_game_tasks = create_game_tasks(game_configs, config, seed)

    # Collect results from completed tasks
    for game_name, task_futures in pending_game_tasks:
        episode_results = ray.get(task_futures)
        all_results.extend(episode_results)
        logger.info(
            "Parallel simulation results for %s completed (%d episodes)",
            game_name, len(episode_results)
        )

    return all_results


def execute_sequential_simulations(
    game_configs: List[Dict[str, Any]],
    config: Dict[str, Any],
    seed: int
) -> List[Any]:
    """
    Execute simulations sequentially without Ray.

    Args:
        game_configs: List of game configurations
        config: Base configuration dictionary
        seed: Random seed

    Returns:
        List of simulation results
    """
    all_results = []

    for game_config in game_configs:
        game_name = game_config["game_name"]
        game_specific_config = create_game_config(config, game_config)

        result = simulate_game(game_name, game_specific_config, seed)
        all_results.append(result)
        logger.info(
            "Sequential simulation results for %s completed",
            game_name
        )

    return all_results


def create_game_config(
    base_config: Dict[str, Any],
    game_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a game-specific configuration from the base config.

    Args:
        base_config: The main configuration dictionary
        game_config: Game-specific configuration to merge

    Returns:
        Merged configuration dictionary for the specific game
    """
    game_name = game_config["game_name"]
    # Use absolute path to results directory (project root level)
    if "output_path" not in game_config:
        project_root = Path(__file__).resolve().parent.parent
        results_dir = project_root / "results"
        filename = f"{game_name}_simulation_results.json"
        default_output_path = str(results_dir / filename)
    else:
        default_output_path = game_config["output_path"]

    output_path = game_config.get("output_path", default_output_path)
    return {
        **base_config,  # Inherit global settings
        "env_config": game_config,  # Game configuration
        "max_game_rounds": game_config.get("max_game_rounds", None),
        "num_episodes": base_config.get("num_episodes", 1),
        "agents": base_config.get("agents", {}),
        "output_path": output_path,
    }


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

    # Choose execution strategy based on configuration
    use_ray = config.get("use_ray", False)
    should_use_parallel = use_ray and len(game_configs) > 1

    if should_use_parallel:
        logger.info("Using Ray for parallel execution")
        all_results = execute_parallel_simulations(game_configs, config, seed)
    else:
        execution_mode = "Ray disabled" if not use_ray else "single game"
        logger.info("Using sequential execution (%s)", execution_mode)
        all_results = execute_sequential_simulations(
            game_configs, config, seed
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

    # Configure logging level based on config
    log_level = config.get("log_level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Reconfigure logging with the correct level
    logging.basicConfig(
        filename="run_logs.txt",
        filemode="w",
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True  # Force reconfiguration
    )

    logger.info(config)

    try:
        # Run simulation with parsed config
        print("Running simulation...")
        run_simulation(config)

        print("Running post-game processing...")
        current_dir = Path(__file__).parent
        script_path = (
            current_dir / ".." / "analysis" / "post_game_processing.py"
        )
        subprocess.run(["python3", str(script_path)], check=True)

        print("Simulation completed.")

    finally:
        # Clean up resources after simulation
        full_cleanup("auto")


if __name__ == "__main__":
    main()


# start tensorboard from the terminal:
# tensorboard --logdir=runs

# In the browser:
# http://localhost:6006/