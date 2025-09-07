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
Example:
    python3 scripts/run_multi_model_games.py \
        --config src/game_reasoning_arena/configs/multi_game_multi_model.yaml

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
  - merged_logs_latest.csv - Single consolidated CSV (overwrites on each run)
  - Individual .db files for each agent/model
- runs/ - Individual game run data

NOTE: This script now creates ONE merged CSV file (merged_logs_latest.csv)
instead of multiple timestamped files, preventing data duplication.
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

# Import model validation utility
try:
    from utils.model_validator import validate_models_and_credits
    MODEL_VALIDATOR_AVAILABLE = True
except ImportError:
    MODEL_VALIDATOR_AVAILABLE = False
    print("⚠️  Model validator not available - skipping pre-validation")


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
    elif "env_config" in config and isinstance(config["env_config"], list):
        games = [env["game_name"] for env in config["env_config"]
                 if "game_name" in env]
    elif "env_config" in config and "game_name" in config["env_config"]:
        # fallback for dict (should not be used in new configs)
        games = [config["env_config"]["game_name"]]
    else:
        # Default fallback
        games = ["tic_tac_toe"]

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
        if not models and "llm_backend" in config:
            models = [config["llm_backend"]["default_model"]]
        elif not models:
            # Final fallback
            models = ["litellm_groq/llama-3.1-8b-instant"]

    num_episodes = config.get("num_episodes", 1)
    # Ensure num_episodes is an integer (handle boolean conversion issues)
    if isinstance(num_episodes, bool):
        num_episodes = 1 if num_episodes else 0
    elif isinstance(num_episodes, str):
        try:
            num_episodes = int(num_episodes)
        except ValueError:
            num_episodes = 1
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

    # Set PYTHONPATH to include project root so subprocesses can import
    # game_reasoning_arena
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.parent.resolve())
    src_dir = str(Path(__file__).parent.parent / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{src_dir}{os.pathsep}{current_pythonpath}"
    )
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            env=env,
            timeout=300  # 5 minute timeout
        )
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])
        return True
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - Command took longer than 5 minutes")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED with exit code {e.returncode}")
        print(f"Error: {e.stderr}")
        return False


def run_analysis():
    """Run all analysis scripts"""
    print("\n" + "="*60)
    print("🔍 STARTING ANALYSIS PHASE")
    print("="*60)

    # Post-game processing - creates/overwrites merged_logs_latest.csv
    run_command(
        f"{sys.executable} analysis/post_game_processing.py "
        f"--output merged_logs_latest.csv",
        "Post-game processing - Merging logs into single CSV"
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

    # Validate models and check credits before running
    if MODEL_VALIDATOR_AVAILABLE:
        print("\n🔍 PRE-RUN VALIDATION")
        print("=" * 50)
        validation_result = validate_models_and_credits(models)

        if not validation_result["all_valid"]:
            print("\n❌ VALIDATION FAILED!")
            print("Fix the above issues before running expensive experiments.")
            print("This prevents wasted time and API costs.")
            return

        # Show credit warnings if any
        if validation_result["summary"]["credits_warnings"]:
            print("\n⚠️  Proceeding with credit warnings.")
            print("Monitor your usage during the experiment.")

        print("\n✅ Pre-validation passed! Starting experiments...")

    else:
        print("\n⚠️  Skipping pre-validation (validator not available)")

    print("\n📊 Execution Plan:")
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

    # Get the game configurations from config
    # runner.py expects either env_config (single) or env_configs (multiple)
    game_configs = []
    if "env_configs" in config:
        game_configs = config["env_configs"]
    elif "env_config" in config and isinstance(config["env_config"], dict):
        game_configs = [config["env_config"]]
    else:
        # Default fallback - create basic configs from game names
        game_configs = [{"game_name": game} for game in games]

    # Run all game/model combinations
    for model in models:
        for game_cfg in game_configs:
            current_run += 1
            game = game_cfg.get("game_name", "unknown")
            print(f"\n🎯 Progress: {current_run}/{total_runs}")

            model_short = model.split('/')[-1]

            # Build command to run single game with all game config overrides
            # Since runner.py expects env_config as a single dict, we pass each
            # game separately WITHOUT the base config file to avoid conflicts
            command = f"{sys.executable} scripts/runner.py "

            # Don't use the base config file since it has env_configs (plural)
            # Instead, build individual overrides for a single game

            # Override env_config with the specific game config
            game_name = game_cfg['game_name']
            command += f"--override env_config.game_name={game_name} "
            if 'max_game_rounds' in game_cfg:
                max_rounds = game_cfg['max_game_rounds']
                command += (
                    f"--override env_config.max_game_rounds={max_rounds} "
                )

            # Add other overrides from base config
            # Use config values instead of hardcoded ones
            llm_config = config.get("llm_backend", {})
            log_level = config.get("log_level", "INFO")
            max_tokens = llm_config.get("max_tokens", 250)
            temperature = llm_config.get("temperature", 0.1)
            use_ray = config.get("use_ray", False)
            parallel_episodes = config.get("parallel_episodes", False)

            parallel_str = str(parallel_episodes).lower()

            command += (
                f"--override agents.player_0.model={model} "
                f"--override agents.player_0.type=llm "
                f"--override agents.player_1.type=random "
                f"--override num_episodes={num_episodes} "
                f"--override mode=llm_vs_random "
                f"--override seed=42 "
                f"--override use_ray={str(use_ray).lower()} "
                f"--override parallel_episodes={parallel_str} "
                f"--override log_level={log_level} "
                f"--override llm_backend.max_tokens={max_tokens} "
                f"--override llm_backend.temperature={temperature} "
                f"--override llm_backend.default_model={model}"
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
