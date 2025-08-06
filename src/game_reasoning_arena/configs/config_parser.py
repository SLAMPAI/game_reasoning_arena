"""
Simple configuration system with YAML and key-value CLI overrides.
"""

import os
import argparse
import yaml
from typing import Any, Dict
from ..arena.games.registry import registry


def default_simulation_config() -> Dict[str, Any]:
    """Returns the default simulation configuration."""
    return {
        "env_config": {
            "game_name": "tic_tac_toe",
            "max_game_rounds": None,  # Only use this for iterated games
        },
        "num_episodes": 1,
        "seed": 42,
        "use_ray": False,
        "mode": "llm_vs_random",  # "manual", "llm_vs_llm"
        "agents": {
            "player_0": {
                "type": "llm",
                "model": "litellm_groq/llama-3.1-8b-instant"
            },
            "player_1": {
                "type": "random"
            }
        },
        "llm_backend": {
            "max_tokens": 250,
            "temperature": 0.1,
            "default_model": "litellm_groq/gemma-7b-it",
        },
        "log_level": "INFO",
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """Creates a simple CLI parser for key-value overrides and YAML config."""
    parser = argparse.ArgumentParser(
        description="Game Simulation Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config file.",
    )

    parser.add_argument(
        "--ray-config",
        type=str,
        help="Path to a Ray-specific YAML config file to merge.",
    )

    parser.add_argument(
        "--base-config",
        type=str,
        help="Path to a base YAML config file (merged first).",
    )

    # LLM Backend Configuration
    parser.add_argument(
        "--backend",
        type=str,
        choices=["litellm", "vllm", "hybrid"],
        help="LLM inference backend to use.",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Default LLM model to use (e.g., groq/llama3-8b-8192).",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens for LLM generation.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM generation (0.0-2.0).",
    )

    parser.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        help="Key-value overrides for configuration.",
    )
    return parser


def parse_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Parses the configuration, merging multiple YAML configs and CLI overrides.

    Order of precedence (later overrides earlier):
    1. Default config
    2. Base config file (--base-config)
    3. Main config file (--config)
    4. Ray config file (--ray-config)
    5. CLI overrides (--override)

    Validates the configuration against the game requirements.
    """
    # Start with default config
    config = default_simulation_config()

    # Merge base config first (if provided)
    if hasattr(args, 'base_config') and args.base_config:
        base_config = load_config(args.base_config)
        config = deep_merge_dicts(config, base_config)

    # Merge main config (if provided)
    if args.config:
        main_config = load_config(args.config)
        config = deep_merge_dicts(config, main_config)

    # Merge Ray config (if provided)
    if hasattr(args, 'ray_config') and args.ray_config:
        ray_config = load_config(args.ray_config)
        config = deep_merge_dicts(config, ray_config)

    # Apply specific CLI arguments for LLM backend
    if hasattr(args, 'backend') and args.backend:
        config["llm_backend"]["inference_backend"] = args.backend

    if hasattr(args, 'model') and args.model:
        config["llm_backend"]["default_model"] = args.model

    if hasattr(args, 'max_tokens') and args.max_tokens:
        config["llm_backend"]["max_tokens"] = args.max_tokens

    if hasattr(args, 'temperature') and args.temperature is not None:
        config["llm_backend"]["temperature"] = args.temperature

    # Apply CLI key-value overrides (if provided)
    if hasattr(args, 'override') and args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            config = apply_override(config, key, value)

    # Apply backend configuration to environment variables
    apply_backend_config(config["llm_backend"])

    validate_config(config)
    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML file

    Returns:
        Dictionary containing the configuration

    Raises:
        ValueError: If file format is not supported or parsing fails
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file {config_path}: {e}")
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {config_path}")


def deep_merge_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary to merge into base (takes precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if (key in result and
                isinstance(result[key], dict) and
                isinstance(value, dict)):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # Override or add the value
            result[key] = value

    return result


def apply_override(
        config: Dict[str, Any],
        key: str, value: str
        ) -> Dict[str, Any]:
    """
    Apply a single key-value override to the configuration.

    Supports nested keys using dot notation (e.g., "env_config.game_name").

    Args:
        config: Configuration dictionary to modify
        key: Key path (supports dot notation)
        value: String value to set

    Returns:
        Modified configuration dictionary
    """
    # Parse nested keys (e.g., "env_config.game_name")
    keys = key.split(".")

    # Navigate to the parent dictionary
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set the final value with type conversion
    final_key = keys[-1]
    current[final_key] = convert_value_type(value)

    return config


def convert_value_type(value: str) -> Any:
    """
    Convert string value to appropriate Python type.

    Args:
        value: String representation of the value

    Returns:
        Converted value (bool, int, float, or str)
    """
    # Handle boolean values
    if value.lower() in ["true", "yes", "1"]:
        return True
    elif value.lower() in ["false", "no", "0"]:
        return False
    elif value.lower() in ["none", "null"]:
        return None

    # Try numeric conversion
    try:
        # Try integer first
        if "." not in value:
            return int(value)
        else:
            return float(value)
    except ValueError:
        # Return as string if conversion fails
        return value


def apply_backend_config(llm_config: Dict[str, Any]) -> None:
    """
    Apply LLM backend configuration to environment variables.

    Args:
        llm_config: LLM backend configuration dictionary
    """
    # Set environment variables for backend configuration
    if "inference_backend" in llm_config:
        os.environ["LLM_BACKEND"] = llm_config["inference_backend"]

    if "default_model" in llm_config:
        os.environ["DEFAULT_MODEL"] = llm_config["default_model"]


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration for common issues.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate game name
    game_name = config.get("env_config", {}).get("game_name")
    if not game_name:
        raise ValueError("Game name is required in env_config.game_name")

    # Check if game is registered
    if game_name not in registry._registry:
        available_games = list(registry._registry.keys())
        raise ValueError(
            f"Game '{game_name}' not found. "
            f"Available games: {available_games}"
        )

    # Validate agent configurations
    agents = config.get("agents", {})
    for player_id, agent_config in agents.items():
        if "type" not in agent_config:
            raise ValueError(f"Agent {player_id} missing 'type' field")

        agent_type = agent_config["type"]
        if agent_type == "llm" and "model" not in agent_config:
            raise ValueError(f"LLM agent {player_id} missing 'model' field")

    # Validate episode count
    num_episodes = config.get("num_episodes", 1)
    if not isinstance(num_episodes, int) or num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
