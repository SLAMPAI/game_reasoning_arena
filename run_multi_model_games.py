#!/usr/bin/env python3
"""
Multi-Model Game Runner Script

Runs 3 games with 2 different LLM models against random opponents,
then runs analysis scripts.
"""

import subprocess
import time
import os


# Configuration
GAMES = ["kuhn_poker", "connect_four", "tic_tac_toe"]
MODELS = [
    "litellm_groq/llama3-8b-8192",
    "litellm_groq/gemma2-9b-it"
]
NUM_EPISODES = 5
BASE_CONFIG = "src/game_reasoning_arena/configs/three_games.yaml"


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED with exit code {e.returncode}")
        print(f"Error: {e.stderr}")
        return False


def run_single_game(model, game):
    """Run a single game with a specific model"""
    model_short = model.split('/')[-1]

    # Create override for single game
    override_config = f'[{{"game_name": "{game}", "max_game_rounds": null}}]'

    command = (
        f"python3 scripts/runner.py "
        f"--config {BASE_CONFIG} "
        f"--override env_configs='{override_config}' "
        f"--override agents.player_0.model={model} "
        f"--override num_episodes={NUM_EPISODES} "
        f"--override mode=llm_vs_random"
    )

    description = f"Running {game} with {model_short} vs Random"
    return run_command(command, description)


def run_analysis():
    """Run all analysis scripts"""
    print("\n" + "="*60)
    print("🔍 STARTING ANALYSIS PHASE")
    print("="*60)

    # Post-game processing
    run_command(
        "python3 analysis/post_game_processing.py",
        "Post-game processing - Merging logs"
    )

    # Generate plots
    run_command(
        "python3 analysis/generate_reasoning_plots.py",
        "Generating reasoning plots"
    )

    print("\n" + "="*60)
    print("📊 Analysis complete! Check plots/ and results/ directories")
    print("="*60)


def main():
    """Main execution function"""
    print("🎮 Multi-Model Game Reasoning Arena Runner")
    print(f"Games: {', '.join(GAMES)}")
    print(f"Models: {', '.join([m.split('/')[-1] for m in MODELS])}")
    print(f"Episodes per game: {NUM_EPISODES}")

    total_runs = len(GAMES) * len(MODELS)
    current_run = 0
    start_time = time.time()

    # Run all game combinations
    for model in MODELS:
        for game in GAMES:
            current_run += 1
            print(f"\n🎯 Progress: {current_run}/{total_runs}")

            success = run_single_game(model, game)
            if not success:
                model_short = model.split('/')[-1]
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
