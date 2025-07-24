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


def initialize_ray():
    """Initializes Ray if not already initialized."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized.")

@ray.remote
def simulate_game_ray(game_name: str, config: Dict[str, Any], seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ray remote wrapper for parallel game simulation.
    Calls the standard simulate_game function.
    """
    return simulate_game(game_name, config, seed)

def run_simulation(config):
    """
    Orchestrates simulation runs across multiple games and agent matchups.
    Uses the provided configuration, sets up Ray if enabled, and collects simulation results.
    """

    seed = config.get("seed", 42)
    set_seed(seed)

    use_ray = config.get("use_ray", True)
    if use_ray:
        initialize_ray()

    # Extract game configuration
    game_config = config.get("env_config", {})
    game_name = game_config["game_name"]

    # Prepare the game-specific configuration
    game_specific_config = {
        **config,  # Inherit global settings
        "env_config": game_config,  # Game configuration
        "max_game_rounds": game_config.get("max_game_rounds", None),
        "num_episodes": config.get("num_episodes", 1),
        "agents": config.get("agents", {}),
        "output_path": game_config.get("output_path", f"results/{game_name}_simulation_results.json"),
    }

    # Run the simulation (no need for Ray with single game)
    results = [simulate_game(game_name, game_specific_config, seed)]

    logger.info("Simulation results for %s ended", game_name)
    return results


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
        # Clean up resources - backend type no longer needed since it's automatic
        full_cleanup("auto")


if __name__ == "__main__":
    main()
