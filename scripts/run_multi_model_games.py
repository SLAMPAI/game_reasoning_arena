#!/usr/bin/env python3
"""
Multi-Model Game Runner Script

Runs games with different LLM models against random opponents,
then runs analysis scripts to generate plots and reports.

USAGE:
======

Basic usage (uses defaults):
    python3 scripts/run_multi_model_games.py

With a config file:
    python3 scripts/run_multi_model_games.py --config path/to/config.yaml

With CLI overrides:
    python3 scripts/run_multi_model_games.py --override num_episodes=5

Change the model:
    python3 scripts/run_multi_model_games.py \
      --override agents.player_0.model=litellm_groq/llama3-70b-8192

Multiple overrides:
    python3 scripts/run_multi_model_games.py \
      --override num_episodes=3 \
      --override agents.player_0.model=litellm_groq/llama3-70b-8192 \
      --override env_config.game_name=connect_four

Show all available options:
    python3 scripts/run_multi_model_games.py --help

DEFAULTS:
=========
- Game: tic_tac_toe
- Model: litellm_groq/llama-3.1-8b-instant
- Episodes: 1
- Mode: llm_vs_random

CONFIGURATION:
==============
The script uses the centralized config parser from
game_reasoning_arena.configs.config_parser which supports:
- YAML config files (--config)
- Base configs (--base-config)
- Ray configs (--ray-config)
- CLI overrides (--override key=value)
- Model selection (--model)
- Backend selection (--backend)

OUTPUT:
=======
Results are saved to:
- plots/ - Visualizations and analysis charts
- results/ - Raw data and processed results
- runs/ - Individual game run data
"""

import subprocess
import time
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position,import-error
from game_reasoning_arena.configs.config_parser import (  # noqa: E402
    build_cli_parser,
    parse_config
)


def load_config_vars(config):
    """Extract games, models, num_episodes from parsed config."""
    # Extract games from env_configs if present
    games = []
    if "env_configs" in config:
        for env in config["env_configs"]:
            if "game_name" in env:
                games.append(env["game_name"])
    elif "games" in config:
        games = config["games"]
    elif "env_config" in config and "game_name" in config["env_config"]:
        games = [config["env_config"]["game_name"]]

    # Extract models from top-level models if present
    if "models" in config and isinstance(config["models"], list):
        models = config["models"]
    else:
        # Fallback: Extract models from agents
        models = []
        agents = config.get("agents", {})
        for agent in agents.values():
            if agent.get("type") == "llm" and "model" in agent:
                models.append(agent["model"])
        # If no models found in agents, use llm_backend default_model
        if not models:
            models = [config["llm_backend"]["default_model"]]

    num_episodes = config["num_episodes"]
    return games, models, num_episodes


def get_args_and_config():
    """Parse command line arguments and configuration."""
    parser = build_cli_parser()
    args = parser.parse_args()
    config = parse_config(args)
    return args, config


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")

    # Set PYTHONPATH to include project root so subprocesses can import game_reasoning_arena
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.parent.resolve())
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            env=env
        )
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED with exit code {e.returncode}")
        print(f"Error: {e.stderr}")
        return False


def run_analysis():
    """Run all analysis scripts"""
    print("\n" + "="*60)
    print("🔍 STARTING ANALYSIS PHASE")
    print("="*60)

    # Post-game processing
    run_command(
        f"{sys.executable} analysis/post_game_processing.py",
        "Post-game processing - Merging logs"
    )
    # Run full analysis pipeline (comprehensive)
    run_command(
        f"{sys.executable} analysis/run_full_analysis.py",
        "Running full analysis pipeline"
    )

    print("\n" + "="*60)
    print("📊 Analysis complete! Check plots/ and results/ directories")
    print("="*60)


def main():
    """Main execution function"""
    args, config = get_args_and_config()
    games, models, num_episodes = load_config_vars(config)

    print("🎮 Multi-Model Game Reasoning Arena Runner")
    print(f"Games: {', '.join(games)}")
    print(f"Models: {', '.join([m.split('/')[-1] for m in models])}")
    print(f"Episodes per game: {num_episodes}")
    if args.config:
        print(f"Using config file: {args.config}")
    else:
        print("No config file provided. Using defaults and CLI overrides.")

    total_runs = len(games) * len(models)
    current_run = 0
    start_time = time.time()

    # Run all game combinations
    for model in models:
        for game in games:
            current_run += 1
            print(f"\n🎯 Progress: {current_run}/{total_runs}")

            model_short = model.split('/')[-1]

            # Build command to run single game
            command = f"{sys.executable} scripts/runner.py "
            if args.config:
                command += f"--config {args.config} "
            command += (
                f"--override env_config.game_name={game} "
                f"--override agents.player_0.model={model} "
                f"--override num_episodes={num_episodes} "
                f"--override mode=llm_vs_random"
            )

            description = f"Running {game} with {model_short} vs Random"
            success = run_command(command, description)

            if not success:
                print(f"⚠️ Failed {game} with {model_short}, continuing...")

            time.sleep(2)  # Brief pause between runs

    duration = time.time() - start_time
    print(f"\n{'='*60}")
    print("🏁 ALL GAMES COMPLETED!")
    print(f"Total time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    print("="*60)

    # Run analysis
    run_analysis()

    print("\n🎉 EVERYTHING COMPLETE! 🎉")
    print("Check 'plots/' for visualizations and 'results/' for data.")


if __name__ == "__main__":
    main()
