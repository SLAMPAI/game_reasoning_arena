"""
Central LLM registry for managing multiple backends.

This file provides:

- Centralized registry for available models.
- Functions to detect which backend to use for a model.
- Unified API for loading models and generating responses.
- Convenience functions for listing and describing models.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from .litellm_backend import LiteLLMBackend
from .vllm_backend import VLLMBackend
from .huggingface_backend import HuggingFaceBackend
from .openrouter_backend import OpenRouterBackend

# Get the directory of this file to construct relative paths
_current_dir = Path(__file__).parent
_configs_dir = _current_dir.parent / "configs"


def _load_config_file(file_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {file_path}") from exc
    except yaml.YAMLError as e:
        raise ValueError(
            f"Error parsing YAML config file {file_path}: {e}"
        ) from e


# Model configuration paths from environment with package-relative defaults
LITELLM_MODELS_PATH = os.getenv(
    "LITELLM_MODELS_PATH",
    str(_configs_dir / "litellm_models.yaml")
)
VLLM_MODELS_PATH = os.getenv(
    "VLLM_MODELS_PATH",
    str(_configs_dir / "vllm_models.yaml")
)
OPENROUTER_MODELS_PATH = os.getenv(
    "OPENROUTER_MODELS_PATH",
    str(_configs_dir / "openrouter_models.yaml")
)

# Global registry for loaded models
LLM_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Backend instances
_litellm_backend = None
_vllm_backend = None
_huggingface_backend = None
_openrouter_backend = None


def get_litellm_backend() -> LiteLLMBackend:
    """Get or create LiteLLM backend instance."""
    global _litellm_backend  # noqa
    if _litellm_backend is None:
        _litellm_backend = LiteLLMBackend()  # Uses config automatically
    return _litellm_backend


def get_huggingface_backend() -> HuggingFaceBackend:
    """Get or create HuggingFace backend instance."""
    global _huggingface_backend  # noqa
    if _huggingface_backend is None:
        _huggingface_backend = HuggingFaceBackend()
    return _huggingface_backend


def get_vllm_backend() -> VLLMBackend:
    """Get or create vLLM backend instance."""
    global _vllm_backend  # noqa
    if _vllm_backend is None:
        _vllm_backend = VLLMBackend()
    return _vllm_backend


def get_openrouter_backend() -> OpenRouterBackend:
    """Get or create OpenRouter backend instance."""
    global _openrouter_backend  # noqa
    if _openrouter_backend is None:
        _openrouter_backend = OpenRouterBackend()
    return _openrouter_backend


def extract_model_name(full_model_name: str) -> str:
    """Extract the actual model name from the prefixed version.

    Args:
        full_model_name: Model name with backend prefix (e.g., "litellm_gpt-4")

    Returns:
        str: The actual model name without prefix (e.g., "gpt-4")
    """
    if full_model_name.startswith("litellm_"):
        return full_model_name[8:]  # Remove "litellm_" prefix
    elif full_model_name.startswith("vllm_"):
        return full_model_name[5:]  # Remove "vllm_" prefix
    elif full_model_name.startswith("hf_"):
        return full_model_name[3:]  # Remove "hf_" prefix
    elif full_model_name.startswith("openrouter_"):
        return full_model_name[11:]  # Remove "openrouter_" prefix
    else:
        raise ValueError(
            f"Model name '{full_model_name}' must start with 'litellm_', "
            f"'vllm_', 'hf_', or 'openrouter_' prefix. "
            f"Examples: 'litellm_gpt-4', 'vllm_Qwen2-7B-Instruct', "
            f"'hf_gpt2', 'openrouter_openai/gpt-4'"
        )


def detect_model_backend(model_name: str) -> str:
    """Detect which backend should handle a given model based on naming.

    Models must be named with prefixes:
    - litellm_<model_name> for LiteLLM backend
    - vllm_<model_name> for vLLM backend
    - hf_<model_name> for HuggingFace backend
    - openrouter_<model_name> for OpenRouter backend
    """
    # Check for explicit backend prefix in model name
    if model_name.startswith("litellm_"):
        return "litellm"
    elif model_name.startswith("vllm_"):
        return "vllm"
    elif model_name.startswith("hf_"):
        return "huggingface"
    elif model_name.startswith("openrouter_"):
        return "openrouter"
    else:
        raise ValueError(
            f"Model name '{model_name}' must start with 'litellm_', "
            f"'vllm_', 'hf_', or 'openrouter_' prefix."
        )


def load_llm_model(model_name: str) -> Any:
    """Load a model using the appropriate backend."""
    backend_name = detect_model_backend(model_name)
    actual_model_name = extract_model_name(model_name)

    if backend_name == "litellm":
        return get_litellm_backend().load_model(actual_model_name)
    elif backend_name == "vllm":
        return get_vllm_backend().load_model(actual_model_name)
    elif backend_name == "huggingface":
        return get_huggingface_backend().load_model(actual_model_name)
    elif backend_name == "openrouter":
        return get_openrouter_backend().load_model(actual_model_name)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def generate_response(model_name: str, prompt: str, **kwargs) -> str:
    """Generate response using the appropriate backend."""
    backend_name = detect_model_backend(model_name)
    actual_model_name = extract_model_name(model_name)

    if backend_name == "litellm":
        backend = get_litellm_backend()
        return backend.generate_response(actual_model_name, prompt, **kwargs)
    elif backend_name == "vllm":
        backend = get_vllm_backend()
        return backend.generate_response(actual_model_name, prompt, **kwargs)
    elif backend_name == "huggingface":
        backend = get_huggingface_backend()
        return backend.generate_response(actual_model_name, prompt, **kwargs)
    elif backend_name == "openrouter":
        backend = get_openrouter_backend()
        return backend.generate_response(actual_model_name, prompt, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def initialize_llm_registry() -> None:
    """Initialize the global LLM registry.

    Automatically loads models from both litellm_models.yaml and
    vllm_models.yaml.
    Backend selection is automatic based on model name prefixes.
    """
    global LLM_REGISTRY, _litellm_backend, _vllm_backend, _openrouter_backend  # noqa
    LLM_REGISTRY.clear()

    # Reset backend instances to pick up new config
    _litellm_backend = None
    _vllm_backend = None
    _openrouter_backend = None

    print("Initializing LLM registry with automatic backend detection")

    # Load models from both configuration files
    available_models = []

    # Load LiteLLM models
    try:
        models_data = _load_config_file(LITELLM_MODELS_PATH)
        litellm_models = models_data.get("models", [])
        available_models.extend(litellm_models)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Error: LiteLLM models config not found at {LITELLM_MODELS_PATH}"
        ) from exc

    # Load vLLM models
    try:
        vllm_data = _load_config_file(VLLM_MODELS_PATH)
        vllm_model_configs = vllm_data.get("models", [])

        # Extract just the model names from the configurations
        vllm_models = []
        for model_config in vllm_model_configs:
            if isinstance(model_config, dict):
                model_name = model_config.get("name")
                if model_name:
                    vllm_models.append(model_name)

        available_models.extend(vllm_models)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Error: vLLM models config not found at {VLLM_MODELS_PATH}"
        ) from exc

    # Load OpenRouter models
    try:
        openrouter_data = _load_config_file(OPENROUTER_MODELS_PATH)
        openrouter_models = openrouter_data.get("models", [])
        available_models.extend(openrouter_models)
    except FileNotFoundError:
        # OpenRouter is optional, so just warn but don't fail
        print(f"Warning: OpenRouter models config not found at "
              f"{OPENROUTER_MODELS_PATH}")

    # Register all models
    for model in available_models:
        LLM_REGISTRY[model] = {
            "display_name": model,
            "description": f"LLM model {model}",
            "backend": detect_model_backend(model),
            "model_loader": lambda model_name=model: (
                load_llm_model(model_name)
            ),
        }


def get_available_models() -> List[str]:
    """Get list of all available models across all backends."""
    return list(LLM_REGISTRY.keys())


def get_backend_for_model(model_name: str) -> str:
    """Get the backend name for a specific model."""
    if model_name in LLM_REGISTRY:
        return LLM_REGISTRY[model_name]["backend"]
    else:
        return detect_model_backend(model_name)


def list_models() -> None:
    """Print all available models and their backends."""
    models = get_available_models()
    if not models:
        print("No models registered.")
        return

    print("\nRegistered Models:")
    print("-" * 50)
    for model in sorted(models):
        backend = get_backend_for_model(model)
        print(f"{model:<40} ({backend})")


def is_litellm_model(model_name: str) -> bool:
    """Check if a model is available via LiteLLM backend."""
    try:
        models_data = _load_config_file(LITELLM_MODELS_PATH)
        litellm_models = set(models_data.get("models", []))
        return model_name in litellm_models
    except FileNotFoundError:
        return False


def is_vllm_model(model_name: str) -> bool:
    """Check if a model is available via vLLM backend."""
    try:
        models_data = _load_config_file(VLLM_MODELS_PATH)
        vllm_models = set(models_data.get("models", []))
        return model_name in vllm_models
    except FileNotFoundError:
        return False


def is_openrouter_model(model_name: str) -> bool:
    """Check if a model is available via OpenRouter backend."""
    try:
        models_data = _load_config_file(OPENROUTER_MODELS_PATH)
        openrouter_models = set(models_data.get("models", []))
        return model_name in openrouter_models
    except FileNotFoundError:
        return False
