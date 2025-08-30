#!/usr/bin/env python3
"""
Ray Multi-Model Game Runner

This script maximizes parallelization by running multiple models simultaneously
using Ray distributed computing. Each model runs all its games in parallel.

Key differences from run_multi_model_games.py:
- Models run in parallel (not sequential)
- Each model gets its own Ray task
- Better resource utilization
- Significantly faster execution

USAGE:
======

Basic usage:
    python3 scripts/run_ray_multi_model.py

With Ray config:
    python3 scripts/run_ray_multi_model.py \
        --config src/game_reasoning_arena/configs/ray_multi_model.yaml

Override settings:
    python3 scripts/run_ray_multi_model.py \
        --override num_episodes=10 \
        --override ray_config.num_cpus=16

ARCHITECTURE:
=============
Main Process
├── Ray Task 1: Model A → All Games (parallel) → All Episodes (parallel)
├── Ray Task 2: Model B → All Games (parallel) → All Episodes (parallel)
├── Ray Task 3: Model C → All Games (parallel) → All Episodes (parallel)
└── ... (all models run simultaneously)

SPEEDUP:
========
Expected speedup: N_models × N_games × N_episodes parallelization
Example: 6 models × 5 games × 5 episodes = 150x theoretical speedup
"""

import subprocess
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import ray

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position,import-error
from game_reasoning_arena.configs.config_parser import (  # noqa: E402
    build_cli_parser,
    parse_config
)


@ray.remote
def run_single_model_all_games(
    model: str,
    config: Dict[str, Any],
    model_index: int,
    total_models: int
) -> Dict[str, Any]:
    """
    Run all games for a single model using the native runner with Ray.

    This function runs as a Ray task, allowing multiple models to execute
    simultaneously. Each model runs all configured games and episodes
    in parallel within its own Ray cluster.

    Args:
        model: Model identifier (e.g., "litellm_groq/llama3-8b-8192")
        config: Full configuration dictionary
        model_index: Index of this model (for progress tracking)
        total_models: Total number of models being run

    Returns:
        Dictionary with execution results and timing information
    """
    start_time = time.time()
    model_short = model.split('/')[-1]

    print(f"🚀 [{model_index+1}/{total_models}] Starting {model_short}")

    # Build command using the native runner with Ray enabled
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.parent.resolve())
    src_dir = str(Path(__file__).parent.parent / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{src_dir}{os.pathsep}{current_pythonpath}"
    )

    # Use a temporary config approach or direct runner invocation
    command_parts = [
        sys.executable, "scripts/runner.py",
        "--config", "src/game_reasoning_arena/configs/ray_multi_model.yaml",
        # Override the model for this specific run
        "--override", f"agents.player_0.model={model}",
        "--override", f"llm_backend.default_model={model}",
        # Ensure Ray is enabled with reduced CPU allocation per model
        "--override", "use_ray=true",
        "--override", "parallel_episodes=true",
        "--override", "ray_config.num_cpus=3",  # Limit per model
    ]

    command = " ".join(command_parts)

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            env=env,
            timeout=1800  # 30 minute timeout per model
        )

        duration = time.time() - start_time
        print(f"✅ [{model_index+1}/{total_models}] {model_short} "
              f"completed in {duration:.1f}s")

        return {
            "model": model,
            "model_short": model_short,
            "success": True,
            "duration": duration,
            "output_lines": (
                len(result.stdout.split('\n')) if result.stdout else 0
            ),
            "error": None
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        error_msg = f"Timeout after {duration:.1f}s (limit: 1800s)"
        print(f"⏰ [{model_index+1}/{total_models}] {model_short} timed out")

        return {
            "model": model,
            "model_short": model_short,
            "success": False,
            "duration": duration,
            "output_lines": 0,
            "error": error_msg
        }

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        error_msg = f"Exit code {e.returncode}: {e.stderr[:200]}..."
        print(f"❌ [{model_index+1}/{total_models}] {model_short} failed")

        return {
            "model": model,
            "model_short": model_short,
            "success": False,
            "duration": duration,
            "output_lines": 0,
            "error": error_msg
        }


def extract_config_info(config: Dict[str, Any]) -> Tuple[List[str], int, int]:
    """Extract models, games count, and episodes from config."""
    # Extract models
    models = config.get("models", [])
    if not models:
        # Fallback to agent configuration
        agents = config.get("agents", {})
        for agent in agents.values():
            if agent.get("type") == "llm" and "model" in agent:
                models = [agent["model"]]
                break
        if not models:
            models = ["litellm_groq/llama3-8b-8192"]  # Default

    # Count games
    env_configs = config.get("env_configs", [])
    num_games = len(env_configs) if env_configs else 1

    # Get episodes
    num_episodes = config.get("num_episodes", 1)

    return models, num_games, num_episodes


def initialize_ray_cluster(config: Dict[str, Any]) -> None:
    """Initialize Ray cluster with configuration."""
    if ray.is_initialized():
        print("📡 Ray already initialized")
        return

    ray_config = config.get("ray_config", {})

    init_params = {
        "ignore_reinit_error": True,
    }

    # Add Ray configuration parameters
    if ray_config.get("num_cpus"):
        init_params["num_cpus"] = ray_config["num_cpus"]
    if ray_config.get("object_store_memory"):
        init_params["object_store_memory"] = ray_config["object_store_memory"]
    if ray_config.get("include_dashboard") is not None:
        init_params["include_dashboard"] = ray_config["include_dashboard"]

    ray.init(**init_params)

    cluster_resources = ray.cluster_resources()
    cpu_count = cluster_resources.get("CPU", "unknown")
    memory_mb = cluster_resources.get("memory", 0) / (1024 * 1024)

    print(f"📡 Ray cluster initialized:")
    print(f"   CPUs: {cpu_count}")
    print(f"   Memory: {memory_mb:.1f} MB")


def run_analysis_phase() -> bool:
    """Run the analysis pipeline after all models complete."""
    print("\n" + "="*60)
    print("🔍 STARTING ANALYSIS PHASE")
    print("="*60)

    try:
        # Run post-game processing
        subprocess.run(
            [sys.executable, "analysis/post_game_processing.py"],
            capture_output=True,
            text=True,
            timeout=300,
            check=True
        )
        print("✅ Post-game processing completed")

        # Run full analysis
        subprocess.run(
            [sys.executable, "analysis/run_full_analysis.py"],
            capture_output=True,
            text=True,
            timeout=600,
            check=True
        )
        print("✅ Full analysis pipeline completed")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e.stderr[:200]}...")
        return False
    except subprocess.TimeoutExpired:
        print("⏰ Analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def main():
    """Main execution with full Ray parallelization."""
    print("🎮 Ray Multi-Model Game Reasoning Arena")
    print("=" * 50)

    # Parse configuration
    parser = build_cli_parser()
    args = parser.parse_args()
    config = parse_config(args)

    # Extract configuration information
    models, num_games, num_episodes = extract_config_info(config)

    print(f"📊 Execution Plan:")
    print(
        f"   Models: {len(models)} "
        f"({', '.join([m.split('/')[-1] for m in models])})"
    )
    print(f"   Games per model: {num_games}")
    print(f"   Episodes per game: {num_episodes}")
    print(f"   Total combinations: {len(models) * num_games * num_episodes}")
    print(f"   Ray enabled: {config.get('use_ray', False)}")

    # Initialize Ray cluster
    initialize_ray_cluster(config)

    # Launch all models in parallel
    print(f"\n🚀 Launching {len(models)} models in parallel...")
    start_time = time.time()

    # Create Ray tasks for all models
    model_tasks = []
    for i, model in enumerate(models):
        task = run_single_model_all_games.remote(
            model, config, i, len(models)
        )
        model_tasks.append(task)

    print("⏳ Waiting for all models to complete...")

    # Wait for completion and collect results
    results = ray.get(model_tasks)

    total_duration = time.time() - start_time

    # Analyze results
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_output_lines = sum(r["output_lines"] for r in results)
    avg_duration = sum(r["duration"] for r in results) / len(results)

    # Print summary
    print(f"\n{'='*60}")
    print("🏁 PARALLEL EXECUTION COMPLETED!")
    print(f"{'='*60}")
    print(
        f"⏱️  Total wall time: {total_duration:.1f}s "
        f"({total_duration/60:.1f} min)"
    )
    print(f"⚡ Average model time: {avg_duration:.1f}s")
    print(
        f"🚀 Speedup vs sequential: ~"
        f"{avg_duration * len(models) / total_duration:.1f}x"
    )
    print(f"✅ Successful: {successful}/{len(models)}")
    print(f"❌ Failed: {failed}/{len(models)}")
    print(f"📝 Total output lines: {total_output_lines}")

    # Detailed results
    print(f"\n📊 Model Results:")
    print("-" * 50)
    for result in sorted(results, key=lambda x: x["duration"]):
        status = "✅" if result["success"] else "❌"
        duration_str = f"{result['duration']:6.1f}s"
        model_short = result["model_short"][:20]

        print(f"{status} {model_short:<20} {duration_str}")

        if not result["success"] and result["error"]:
            print(f"   Error: {result['error'][:60]}...")

    # Run analysis if any models succeeded
    if successful > 0:
        analysis_success = run_analysis_phase()
        if analysis_success:
            print("\n📊 Analysis completed successfully!")
            print("Check 'plots/' and 'results/' directories")
        else:
            print("\n⚠️  Analysis had issues, but model runs completed")

    print("\n🎉 RAY PARALLEL EXECUTION COMPLETE! 🎉")
    print("Effective parallelization: Models + Games + Episodes")


if __name__ == "__main__":
    main()
