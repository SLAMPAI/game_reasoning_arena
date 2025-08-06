#!/usr/bin/env python3
"""
Tests each game with a small LLM model against a random player.
"""

import os
import sys
import subprocess
import tempfile
import yaml
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
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

# Test models as requested
TEST_MODELS = [
    "litellm_together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "litellm_together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "litellm_groq/llama3-8b-8192",
    "litellm_groq/llama3-70b-8192"
]

# Test configuration template
TEST_CONFIG = {
    "env_config": {
        "game_name": None,  # Will be set for each game
        "max_game_rounds": None
    },
    "num_episodes": 1,  # Reduced for faster testing
    "seed": 42,
    "use_ray": False,
    "mode": "llm_vs_random",
    "agents": {
        "player_0": {
            "type": "llm",
            "model": None  # Will be set for each model test
        },
        "player_1": {
            "type": "random"
        }
    },
    "llm_backend": {
        "max_tokens": 200,
        "temperature": 0.1,
        "default_model": None  # Will be set for each model test
    },
    "log_level": "INFO"
}


def create_temp_config(game_name, model_name):
    """Create a temporary config file for the given game and model."""
    config = TEST_CONFIG.copy()
    config["env_config"]["game_name"] = game_name
    config["agents"]["player_0"]["model"] = model_name
    config["llm_backend"]["default_model"] = model_name

    # Create temporary file that gets auto-deleted when closed
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        return f.name


def test_game_model_combination(game_name, model_name):
    """Test a specific game with a specific model and return success status."""
    print(f"\n{'='*50}")
    print(f"Testing: {game_name} with {model_name}")
    print(f"{'='*50}")

    # Create temporary config file
    config_file = create_temp_config(game_name, model_name)

    try:
        # Set up environment with proper PYTHONPATH
        env = os.environ.copy()
        src_path = str(project_root / "src")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = src_path

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
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for faster testing
        )

        if result.returncode == 0:
            print(f"✅ {game_name} + {model_name}: SUCCESS")
            return True
        else:
            print(f"❌ {game_name} + {model_name}: FAILED")
            print(f"Error output: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️ {game_name} + {model_name}: "
              f"TIMEOUT (took longer than 10 minutes)")
        return False
    except Exception as e:
        print(f"💥 {game_name} + {model_name}: EXCEPTION - {str(e)}")
        return False
    finally:
        # Clean up temporary config file
        try:
            os.unlink(config_file)
        except OSError:
            pass


def main():
    """Test all available games with all specified models."""
    print("Board Game Arena - Testing All Available Games")
    print("=" * 60)
    print(f"Testing {len(AVAILABLE_GAMES)} games with "
          f"{len(TEST_MODELS)} models")
    print("Mode: LLM vs Random")
    print("Models:")
    for model in TEST_MODELS:
        print(f"  - {model}")
    print(f"Episodes per game: {TEST_CONFIG['num_episodes']}")
    print("=" * 60)

    results = {}
    total_tests = len(AVAILABLE_GAMES) * len(TEST_MODELS)
    test_count = 0

    for game_name in AVAILABLE_GAMES:
        results[game_name] = {}
        for model_name in TEST_MODELS:
            test_count += 1
            print(f"\nProgress: {test_count}/{total_tests}")
            success = test_game_model_combination(game_name, model_name)
            results[game_name][model_name] = success

    # Print final summary
    print(f"\n\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")

    successful_tests = 0
    failed_tests = 0

    for game_name in AVAILABLE_GAMES:
        print(f"\nGame: {game_name}")
        for model_name in TEST_MODELS:
            success = results[game_name][model_name]
            status = "✅ PASS" if success else "❌ FAIL"
            model_short = (model_name.split('/')[-1]
                           if '/' in model_name else model_name)
            print(f"  {model_short:30} {status}")

            if success:
                successful_tests += 1
            else:
                failed_tests += 1

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY:")
    print(f"   Total tests run: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"   Success rate: {successful_tests/total_tests*100:.1f}%")

    if failed_tests > 0:
        print(f"\n❌ {failed_tests} tests failed")
        return 1
    else:
        print("\n✅ All tests passed successfully!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
