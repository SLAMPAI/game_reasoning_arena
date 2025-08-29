"""
UI utilities for the Game Reasoning Arena Gradio app.

This module contains utility functions for the Gradio interface,
including model name cleaning and display formatting.
"""


def clean_model_name(model_name: str) -> str:
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
    if not model_name or model_name == "Unknown":
        return model_name

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

    # For vllm models, extract the model name part
    if model_name.startswith("vllm_"):
        # Remove vllm_ prefix
        model_part = model_name[5:]
        # Clean up underscores
        cleaned = model_part.replace("_", "-")
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

    # Default: just replace underscores with dashes
    return model_name.replace("_", "-")
