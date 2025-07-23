"""
Simple configuration system with JSON and key-value CLI overrides.
"""

import os
import argparse
import json
from typing import Any, Dict
from board_game_arena.arena.games.registry import registry # Initilizes an empty registry dictionary


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
                "model": "litellm_groq/llama-3.1-8b-instant"  # LLM player needs model
            },
            "player_1": {
                "type": "random"  # Random player doesn't need model
            }
        },
        "llm_backend": {
            "max_tokens": 250,
            "temperature": 0.1,
            "default_model": "litellm_groq/gemma-7b-it",  # Fallback model
        },
        "log_level": "INFO",
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """Creates a simple CLI parser for key-value overrides and JSON config."""
    parser = argparse.ArgumentParser(
        description="Game Simulation Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON config file or raw JSON string.",
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
        help="Key-value overrides for configuration (e.g., game_name=tic_tac_toe).",
    )
    return parser


def parse_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Parses the configuration, merging JSON config and CLI overrides.

       Validates the configuration against the game requirements.
    """
    # Default config
    config = default_simulation_config()

    # Update with JSON config (if provided)
    if args.config:
        if args.config.strip().startswith("{"):
            # Raw JSON string
            json_config = json.loads(args.config)
        else:
            # JSON file
            with open(args.config, "r") as f:
                json_config = json.load(f)
        config.update(json_config)

    # Apply specific CLI arguments for LLM backend
    if args.backend:
        config["llm_backend"]["inference_backend"] = args.backend

    if args.model:
        config["llm_backend"]["default_model"] = args.model

    if args.max_tokens:
        config["llm_backend"]["max_tokens"] = args.max_tokens

    if args.temperature is not None:
        config["llm_backend"]["temperature"] = args.temperature

    # Apply CLI key-value overrides (if provided)
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            config = apply_override(config, key, value)

    # Apply backend configuration to environment variables
    apply_backend_config(config["llm_backend"])

    validate_config(config)
    return config


def apply_backend_config(llm_config: Dict[str, Any]) -> None:
    """Apply LLM backend configuration to environment variables."""
    # Set environment variables for the backend system
    # Note: INFERENCE_BACKEND is no longer needed - automatic detection based on model prefixes
    os.environ["MAX_TOKENS"] = str(llm_config["max_tokens"])
    os.environ["TEMPERATURE"] = str(llm_config["temperature"])


def get_available_models() -> list:
    """Get list of available models from the backend system."""
    try:
        from backends import (initialize_llm_registry,
                              get_available_models as get_models
                              )
        # Initialize the LLM registry to load models
        initialize_llm_registry()
        return get_models()
    except ImportError:
        print("Warning: Backend system not available")
        return []


def apply_override(config: Dict[str, Any], key: str, value: str) -> Dict[str, Any]:
    """Applies a key-value override to the configuration."""
    keys = key.split(".")
    current = config

    for i, k in enumerate(keys[:-1]):
        # Handle dictionary keys
        if k.isdigit():
            k = int(k)  # Convert index to integer
            if not isinstance(current, dict) or k not in current:
                raise ValueError(f"Invalid key '{k}' in override '{key}'")
        current = current.setdefault(k, {}) # type: ignore

    # Handle the final key
    final_key = keys[-1]
    if final_key.isdigit():
        final_key = int(final_key)
        if not isinstance(current, dict) or final_key not in current:
            raise ValueError(f"Invalid key '{final_key}' in override '{key}'")
    current[final_key] = parse_value(value)  # type: ignore

    return config


def parse_value(value: str) -> Any:
    """Converts a string value to the appropriate type (int, float, bool, etc.)."""
    try:
        return json.loads(value)  # Automatically parses JSON types
    except json.JSONDecodeError:
        return value  # Leave as string if parsing fails


def validate_config(config: Dict[str, Any]) -> None:
    """Validates the configuration dict."""
    # Get the single game configuration
    env_config = config.get("env_config", {})
    if not env_config.get("game_name"):
        raise ValueError("Missing 'game_name' in env_config")

    game_name = env_config["game_name"]
    num_players = registry.get_game_loader(game_name)().num_players()

    # Check for agents key
    if "agents" not in config:
        raise ValueError("Missing 'agents' configuration")

    # Check if the number of agents matches the number of players
    if len(config["agents"]) != num_players:
        raise ValueError(
            f"Game '{game_name}' requires {num_players} players, "
            f"but {len(config['agents'])} agents were provided."
        )

    # Validate agent configs and types
    valid_agent_types = {"human", "random", "llm"}
    for i in range(num_players):
        player_key = f"player_{i}"
        agent_config = config["agents"].get(player_key)
        if agent_config is None:
            raise ValueError(f"Missing agent configuration for {player_key}")
        agent_type = agent_config["type"].lower()
        if agent_type not in valid_agent_types:
            raise ValueError(f"Unsupported agent type: '{agent_type}'")
        if agent_type == "llm" and not agent_config.get("model"):
            # Use default model if not specified
            default_model = config.get("llm_backend", {}).get("default_model")
            if default_model:
                agent_config["model"] = default_model
            else:
                raise ValueError(f"LLM agent '{player_key}' must specify a model")
