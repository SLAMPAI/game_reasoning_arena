#!/usr/bin/env python3
"""
Test script to verify all available games work properly.
Tests each game with a small LLM model against a random player.
"""

import os
import sys
import subprocess
import tempfile
import yaml
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Available games from the README
AVAILABLE_GAMES = [
    "tic_tac_toe",
    "connect_four",
    "kuhn_poker",
    "prisoners_dilemma",
    "matrix_pd",
    "matching_pennies",
    "matrix_rps"
]

# Test configuration template
TEST_CONFIG = {
    "env_config": {
        "game_name": None,  # Will be set for each game
        "max_game_rounds": None
    },
    "num_episodes": 2,  # Small number for quick testing
    "seed": 42,
    "use_ray": False,
    "mode": "llm_vs_random",
    "agents": {
        "player_0": {
            "type": "llm",
            "model": "litellm_groq/llama3-8b-8192"  # Fast, small model
        },
        "player_1": {
            "type": "random"
        }
    },
    "llm_backend": {
        "max_tokens": 100,
        "temperature": 0.1,
        "default_model": "litellm_groq/llama3-8b-8192"
    },
    "log_level": "INFO"
}


def create_temp_config(game_name):
    """Create a temporary config file for the given game."""
    config = TEST_CONFIG.copy()
    config["env_config"]["game_name"] = game_name

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        return f.name


def test_game(game_name):
    """Test a specific game and return success status."""
    print(f"\n{'='*50}")
    print(f"Testing game: {game_name}")
    print(f"{'='*50}")

    # Create temporary config file
    config_file = create_temp_config(game_name)

    try:
        # Run the game test
        cmd = [
            "python3",
            str(project_root / "scripts" / "runner.py"),
            "--config",
            config_file
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print(f"✅ {game_name}: SUCCESS")
            return True
        else:
            print(f"❌ {game_name}: FAILED")
            print(f"Error output: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f" {game_name}: TIMEOUT (took longer than 5 minutes)")
        return False
    except Exception as e:
        print(f" {game_name}: EXCEPTION - {str(e)}")
        return False
    finally:
        # Clean up temporary config file
        try:
            os.unlink(config_file)
        except OSError:
            pass


def main():
    """Test all available games."""
    print("Board Game Arena - Testing All Available Games")
    print("=" * 60)
    print(f"Testing {len(AVAILABLE_GAMES)} games with LLM vs Random")
    print("LLM Model: litellm_groq/llama3-8b-8192")
    print(f"Episodes per game: {TEST_CONFIG['num_episodes']}")
    print("=" * 60)

    results = {}

    for game_name in AVAILABLE_GAMES:
        success = test_game(game_name)
        results[game_name] = success

    # Print final summary
    print(f"\n\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")

    successful_games = []
    failed_games = []

    for game_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{game_name:20} {status}")

        if success:
            successful_games.append(game_name)
        else:
            failed_games.append(game_name)

    print("\n Summary:")
    print(f"   Total games tested: {len(AVAILABLE_GAMES)}")
    print(f"   Successful: {len(successful_games)}")
    print(f"   Failed: {len(failed_games)}")

    if failed_games:
        print(f"\n  Failed games: {', '.join(failed_games)}")
        return 1
    else:
        print("\n All games passed successfully!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
