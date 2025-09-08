"""
UI utilities for the Game Reasoning Arena Gradio app.

This module contains utility functions for the Gradio interface,
including model name cleaning and display formatting.
"""

import os
import sqlite3
import glob
from typing import Set
import logging

log = logging.getLogger(__name__)


def get_games_from_databases() -> Set[str]:
    """
    Extract unique game names from all database files in the workspace.

    Returns:
        Set of game names found in database files
    """
    unique_games = set()

    # Find all .db files in the workspace
    db_files = glob.glob("**/*.db", recursive=True)

    for db_file in db_files:
        if not os.path.exists(db_file):
            continue

        try:
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()

                # Check if game_results table exists and get unique game names
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='game_results'"
                )
                if cursor.fetchone():
                    cursor.execute(
                        "SELECT DISTINCT game_name FROM game_results "
                        "WHERE game_name IS NOT NULL"
                    )
                    games_in_db = cursor.fetchall()
                    for (game_name,) in games_in_db:
                        if game_name:  # Skip None/empty values
                            unique_games.add(game_name)

        except Exception as e:
            log.warning("Error reading database %s: %s", db_file, e)

    return unique_games


def clean_model_name(model_name) -> str:
    """
    Clean up long model names to display only the essential model name.

    This function handles various model naming patterns
    from different providers:
    - LiteLLM models with provider prefixes
    - vLLM models with prefixes
    - Models with slash-separated paths
    - GPT model variants

    Args:
        model_name: Full model name from database
                   (e.g., "litellm_together_ai_meta_llama_Meta_Llama_3.1...")

    Returns:
        Cleaned model name (e.g., "Meta-Llama-3.1-8B-Instruct-Turbo")

    Examples:
        >>> clean_model_name(
        "litellm_together_ai/meta-llama/Meta-Llama-3.1-8B"
        )
        "Meta-Llama-3.1-8B"
        >>> clean_model_name(
        "litellm_fireworks_ai/accounts/fireworks/models/glm-4p5-air"
        )
        "glm-4p5-air"
        >>> clean_model_name("vllm_Qwen2-7B-Instruct")
        "Qwen2-7B-Instruct"
        >>> clean_model_name("litellm_gpt-4-turbo")
        "GPT-4-turbo"
    """
    # Handle NaN, None, and float values
    import pandas as pd
    if pd.isna(model_name) or model_name is None:
        return "Unknown"

    # Convert to string if it's not already
    model_name = str(model_name)

    if not model_name or model_name == "Unknown" or model_name == "nan":
        return "Unknown"

    # Handle special cases first
    if model_name == "None" or model_name.lower() == "random":
        return "Random Bot"

    # Handle random_None specifically
    if model_name == "random_None":
        return "Random Bot"

    # GPT models - keep the GPT part
    if "gpt" in model_name.lower():
        # Extract GPT model variants
        if "gpt_3.5" in model_name.lower() or "gpt-3.5" in model_name.lower():
            return "GPT-3.5-turbo"
        elif "4o" in model_name.lower():  # Handle GPT-4o variants first
            if "mini" in model_name.lower():
                return "GPT-4o-mini"
            else:
                return "GPT-4o"
        elif "gpt_4" in model_name.lower() or "gpt-4" in model_name.lower():
            if "turbo" in model_name.lower():
                return "GPT-4-turbo"
            elif "mini" in model_name.lower():
                return "GPT-4-mini"
            else:
                return "GPT-4"
        elif "gpt_5" in model_name.lower() or "gpt-5" in model_name.lower():
            if "mini" in model_name.lower():
                return "GPT-5-mini"
            else:
                return "GPT-5"
        elif "gpt2" in model_name.lower() or "gpt-2" in model_name.lower():
            return "GPT-2"
        elif "distilgpt2" in model_name.lower():
            return "DistilGPT-2"
        elif "gpt-neo" in model_name.lower():
            return "GPT-Neo-125M"

    # For litellm models, extract everything after the last slash
    if "litellm_" in model_name and "/" in model_name:
        # Split by "/" and take the last part
        model_part = model_name.split("/")[-1]
        # Clean up underscores and make it more readable
        cleaned = model_part.replace("_", "-")
        return cleaned

    # For openrouter models, extract everything after the last slash
    if "openrouter_" in model_name and "/" in model_name:
        # Split by "/" and take the last part
        model_part = model_name.split("/")[-1]
        # Clean up underscores and make it more readable
        cleaned = model_part.replace("_", "-")
        return cleaned

    # For vllm models, extract the model name part
    if model_name.startswith("vllm_"):
        # Remove vllm_ prefix
        model_part = model_name[5:]
        # Clean up underscores
        cleaned = model_part.replace("_", "-")
        return cleaned

    # For openrouter models without slashes (from database storage)
    # These correspond to the slash-separated patterns in the YAML
    # Models are stored as llm_openrouter_provider_model_name
    if model_name.startswith("llm_openrouter_"):
        parts = model_name.split("_")

        # Handle OpenAI models via OpenRouter: llm_openrouter_openai_*
        if len(parts) >= 4 and parts[2] == "openai":
            # Everything after "llm_openrouter_openai_"
            model_parts = parts[3:]
            cleaned = "-".join(model_parts)
            # Clean up common GPT model patterns
            if "gpt" in cleaned.lower():
                # Handle GPT-4o variants first (more specific)
                if "4o" in cleaned:
                    # Replace gpt-4o or gpt_4o with GPT-4o
                    cleaned = cleaned.replace("gpt-4o", "GPT-4o")
                    cleaned = cleaned.replace("gpt_4o", "GPT-4o")
                elif "3.5" in cleaned:
                    cleaned = cleaned.replace("gpt-3.5", "GPT-3.5")
                    cleaned = cleaned.replace("gpt_3.5", "GPT-3.5")
                elif "gpt-4" in cleaned:
                    # Only replace if it's not part of "gpt-4o"
                    cleaned = cleaned.replace("gpt-4", "GPT-4")
                    cleaned = cleaned.replace("gpt_4", "GPT-4")
            return cleaned

        # Handle Anthropic models via OpenRouter: llm_openrouter_anthropic_*
        if len(parts) >= 4 and parts[2] == "anthropic":
            # Everything after "llm_openrouter_anthropic_"
            model_parts = parts[3:]
            cleaned = "-".join(model_parts)
            # Capitalize Claude
            if "claude" in cleaned.lower():
                cleaned = cleaned.replace("claude", "Claude")
            return cleaned

        # Handle Meta Llama models via OpenRouter: llm_openrouter_meta_llama_*
        if len(parts) >= 5 and parts[2] == "meta" and parts[3] == "llama":
            # Everything after "llm_openrouter_meta_llama_"
            model_parts = parts[4:]
            cleaned = "-".join(model_parts)
            # Capitalize Llama
            if "llama" in cleaned.lower():
                cleaned = cleaned.replace("llama", "Llama")
            return cleaned

        # Handle Google models via OpenRouter: llm_openrouter_google_*
        if len(parts) >= 4 and parts[2] == "google":
            # Everything after "llm_openrouter_google_"
            model_parts = parts[3:]
            cleaned = "-".join(model_parts)
            # Capitalize common Google model names
            if "gemini" in cleaned.lower():
                cleaned = cleaned.replace("gemini", "Gemini")
            elif "gemma" in cleaned.lower():
                cleaned = cleaned.replace("gemma", "Gemma")
            return cleaned

        # Handle Qwen models via OpenRouter: llm_openrouter_qwen_*
        if len(parts) >= 4 and parts[2] == "qwen":
            # Everything after "llm_openrouter_qwen_"
            model_parts = parts[3:]
            cleaned = "-".join(model_parts)
            # Capitalize Qwen
            cleaned = cleaned.replace("qwen", "Qwen")
            return cleaned

        # Handle MistralAI models via OpenRouter: llm_openrouter_mistralai_*
        if len(parts) >= 4 and parts[2] == "mistralai":
            # Everything after "llm_openrouter_mistralai_"
            model_parts = parts[3:]
            cleaned = "-".join(model_parts)
            # Capitalize Mistral
            if "mistral" in cleaned.lower():
                cleaned = cleaned.replace("mistral", "Mistral")
            return cleaned

        # For other llm_openrouter patterns, skip first three parts
        # (llm_openrouter_provider_)
        if len(parts) >= 4:
            model_parts = parts[3:]  # Everything after provider
            cleaned = "-".join(model_parts)
            return cleaned

    # For litellm models without slashes (from database storage)
    # These correspond to the slash-separated patterns in the YAML
    if model_name.startswith("litellm_"):
        parts = model_name.split("_")

        # Handle Fireworks AI pattern:
        # litellm_fireworks_ai_accounts_fireworks_models_*
        if (
            "fireworks" in model_name
            and "accounts" in model_name
            and "models" in model_name
        ):
            try:
                models_idx = parts.index("models")
                model_parts = parts[models_idx + 1:]
                return "-".join(model_parts)
            except ValueError:
                pass

        # Handle Together AI pattern: litellm_together_ai_meta_llama_*
        # Original:
        # litellm_together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
        # Becomes:
        # litellm_together_ai_meta_llama_Meta_Llama_3.1_8B_Instruct_Turbo
        # We want: Meta-Llama-3.1-8B-Instruct-Turbo
        if (
            "together" in model_name
            and "meta" in model_name
            and "llama" in model_name
        ):
            try:
                # Find "meta" and "llama" -
                # the model name starts after "meta_llama_"
                for i, part in enumerate(parts):
                    if (
                        part == "meta"
                        and i + 1 < len(parts)
                        and parts[i + 1] == "llama"
                    ):
                        # Model name starts after "meta_llama_"
                        model_parts = parts[i + 2:]
                        return "-".join(model_parts)
            except Exception:
                pass

        # Handle Groq pattern: litellm_groq_*
        # These are simpler patterns
        if parts[1] == "groq" and len(parts) >= 3:
            model_parts = parts[2:]  # Everything after "litellm_groq_"
            cleaned = "-".join(model_parts)
            # Special handling for common models
            if "llama3" in cleaned.lower():
                cleaned = cleaned.replace("llama3", "Llama-3")
            elif "qwen" in cleaned.lower():
                cleaned = cleaned.replace("qwen", "Qwen")
            elif "gemma" in cleaned.lower():
                cleaned = cleaned.replace("gemma", "Gemma")
            return cleaned

        # For other patterns, skip first two parts (litellm_provider_)
        if len(parts) >= 3:
            model_parts = parts[2:]  # Everything after provider
            cleaned = "-".join(model_parts)
            return cleaned

    # For models with slashes but not litellm (like direct model paths)
    if "/" in model_name:
        return model_name.split("/")[-1].replace("_", "-")

    # Handle intermediate OpenRouter format: openrouter-provider-model-name
    # Also handle underscore version: openrouter_provider_model_name
    if (model_name.startswith("openrouter-") or
            model_name.startswith("openrouter_")):
        # Normalize to use dashes for consistent processing
        normalized_name = model_name.replace("_", "-")
        parts = normalized_name.split("-")
        if len(parts) >= 3:  # At least openrouter-provider-model
            # Remove "openrouter" prefix
            remaining_parts = parts[1:]

            # Handle specific provider patterns
            if remaining_parts[0] == "openai":
                # openrouter-openai-gpt-4o-mini -> GPT-4o-mini
                model_parts = remaining_parts[1:]
                cleaned = "-".join(model_parts)
                if "gpt" in cleaned.lower():
                    if "4o" in cleaned:
                        cleaned = cleaned.replace("gpt", "GPT")
                        return cleaned.replace("GPT-4o", "GPT-4o")
                    elif "3.5" in cleaned:
                        cleaned = cleaned.replace("gpt", "GPT")
                        return cleaned.replace("GPT-3.5", "GPT-3.5")
                    elif "4" in cleaned:
                        return cleaned.replace("gpt", "GPT")
                return cleaned

            elif remaining_parts[0] == "anthropic":
                # openrouter-anthropic-claude-3.5-sonnet -> Claude-3.5-Sonnet
                model_parts = remaining_parts[1:]
                cleaned = "-".join(model_parts)
                if "claude" in cleaned.lower():
                    cleaned = cleaned.replace("claude", "Claude")
                    return cleaned.replace("sonnet", "Sonnet")
                return cleaned

            elif (remaining_parts[0] == "meta" and len(remaining_parts) >= 2
                  and remaining_parts[1] == "llama"):
                # openrouter-meta-llama-llama-3.1-8b-instruct
                # -> Llama-3.1-8B-Instruct
                model_parts = remaining_parts[2:]  # Skip meta-llama
                cleaned = "-".join(model_parts)
                cleaned = cleaned.replace("llama", "Llama")
                cleaned = cleaned.replace("8b", "8B")
                cleaned = cleaned.replace("70b", "70B")
                cleaned = cleaned.replace("instruct", "Instruct")
                return cleaned

            elif remaining_parts[0] == "google":
                # openrouter-google-gemma-2-9b-it -> Gemma-2-9B-IT
                model_parts = remaining_parts[1:]
                cleaned = "-".join(model_parts)
                cleaned = cleaned.replace("gemma", "Gemma")
                cleaned = cleaned.replace("9b", "9B")
                cleaned = cleaned.replace("it", "IT")
                return cleaned

            elif remaining_parts[0] == "qwen":
                # openrouter-qwen-qwen-2.5-72b-instruct
                # -> Qwen-2.5-72B-Instruct
                model_parts = remaining_parts[1:]
                cleaned = "-".join(model_parts)
                cleaned = cleaned.replace("qwen", "Qwen")
                cleaned = cleaned.replace("72b", "72B")
                cleaned = cleaned.replace("instruct", "Instruct")
                return cleaned

            elif remaining_parts[0] == "mistralai":
                # openrouter-mistralai-mistral-7b-instruct
                # -> Mistral-7B-Instruct
                model_parts = remaining_parts[1:]
                cleaned = "-".join(model_parts)
                cleaned = cleaned.replace("mistral", "Mistral")
                cleaned = cleaned.replace("7b", "7B")
                cleaned = cleaned.replace("instruct", "Instruct")
                return cleaned

            # For other providers, just remove openrouter prefix
            return "-".join(remaining_parts)

    # Default: just replace underscores with dashes
    return model_name.replace("_", "-")
