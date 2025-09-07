"""
Model Filtering Utilities for Game Reasoning Arena

This module provides utilities to filter models for aggregate plots to avoid
clutter and focus on representative models from different families.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_priority_models_config(config_path: str = None) -> Dict[str, Any]:
    """Load priority models configuration from YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses default.

    Returns:
        Dictionary containing the configuration
    """
    if config_path is None:
        # Path to config in src/game_reasoning_arena/configs/
        base_path = Path(__file__).parent.parent
        config_path = (base_path / "src" / "game_reasoning_arena" /
                       "configs" / "priority_models_config.yaml")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Priority models config not found at {config_path}, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration if config file is not available."""
    return {
        "max_models_in_aggregate": 7,
        "priority_models": [
            "openai/gpt-4o-mini", "gpt-4o-mini",
            "meta-llama/llama-3.1-8b-instruct", "llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct", "llama-3.1-70b-instruct",
            "google/gemma-2-9b-it", "gemma-2-9b-it",
            "google/gemini-2.0-flash-exp", "gemini-2.0-flash-exp",
            "qwen/qwen-2.5-72b-instruct", "qwen-2.5-72b-instruct",
            "mistralai/mistral-7b-instruct", "mistral-7b-instruct"
        ],
        "fallback_behavior": "first_n",
        "exclude_patterns": ["random", "human", "test", "debug"]
    }


def clean_model_name_for_matching(model_name: str) -> str:
    """Clean model name for flexible matching."""
    # Import here to avoid circular imports
    try:
        from utils import clean_model_name
        return clean_model_name(model_name).lower()
    except ImportError:
        # Fallback if utils not available
        return model_name.lower().replace('_', '-').replace('/', '-')


def filter_models_for_aggregate_plot(
    available_models: List[str],
    max_models: int = None,
    priority_config: Dict[str, Any] = None,
    force_include: List[str] = None
) -> List[str]:
    """Filter models to show in aggregate plots to avoid clutter.

    Args:
        available_models: List of all available models in the dataset
        max_models: Maximum number of models to include (overrides config)
        priority_config: Priority models configuration (loads default if None)
        force_include: Models to force include regardless of priority

    Returns:
        Filtered list of models for aggregate plotting
    """
    if priority_config is None:
        priority_config = load_priority_models_config()

    if max_models is None:
        max_models = priority_config.get("max_models_in_aggregate", 7)

    if force_include is None:
        force_include = []

    # Filter out excluded patterns
    exclude_patterns = priority_config.get("exclude_patterns", ["random", "human"])
    filtered_models = []
    for model in available_models:
        model_lower = model.lower()
        should_exclude = any(pattern.lower() in model_lower for pattern in exclude_patterns)
        if not should_exclude:
            filtered_models.append(model)

    # If we have few enough models, return all
    if len(filtered_models) <= max_models:
        logger.info(f"Using all {len(filtered_models)} available models (under limit)")
        return filtered_models

    # Start with force_include models
    selected_models = [m for m in force_include if m in filtered_models]
    remaining_slots = max_models - len(selected_models)

    if remaining_slots <= 0:
        logger.info(f"Force included models fill all {max_models} slots")
        return selected_models[:max_models]

    # Find priority models that are available
    priority_models = priority_config.get("priority_models", [])

    for priority_model in priority_models:
        if len(selected_models) >= max_models:
            break

        # Look for matches using flexible matching
        priority_clean = clean_model_name_for_matching(priority_model)

        for available_model in filtered_models:
            if available_model in selected_models:
                continue  # Already selected

            available_clean = clean_model_name_for_matching(available_model)

            # Check for matches with flexible patterns
            if (priority_clean in available_clean or
                available_clean in priority_clean or
                # Remove hyphens/underscores for even more flexible matching
                priority_clean.replace('-', '').replace('_', '') in
                available_clean.replace('-', '').replace('_', '')):

                selected_models.append(available_model)
                logger.debug(f"Matched priority model '{priority_model}' to '{available_model}'")
                break

    # Fill remaining slots with other models if needed
    fallback_behavior = priority_config.get("fallback_behavior", "first_n")
    while len(selected_models) < max_models and len(selected_models) < len(filtered_models):
        for model in filtered_models:
            if model not in selected_models:
                selected_models.append(model)
                break

    logger.info(f"Selected {len(selected_models)} models for aggregate plot: {selected_models}")
    return selected_models[:max_models]


def get_aggregate_plot_title_suffix(
    total_available: int,
    models_shown: int
) -> str:
    """Generate a title suffix indicating model filtering.

    Args:
        total_available: Total number of models available
        models_shown: Number of models actually shown

    Returns:
        Title suffix string
    """
    if models_shown >= total_available:
        return ""  # No filtering applied
    else:
        return f" (showing {models_shown} of {total_available} models)"


def should_filter_aggregate_plots(
    available_models: List[str],
    config_path: str = None
) -> bool:
    """Check if aggregate plots should be filtered based on number of models.

    Args:
        available_models: List of available models
        config_path: Path to configuration file

    Returns:
        True if filtering should be applied
    """
    config = load_priority_models_config(config_path)
    max_models = config.get("max_models_in_aggregate", 7)

    # Filter out excluded models for count
    exclude_patterns = config.get("exclude_patterns", ["random", "human"])
    non_excluded = [m for m in available_models
                   if not any(pattern.lower() in m.lower()
                            for pattern in exclude_patterns)]

    return len(non_excluded) > max_models
