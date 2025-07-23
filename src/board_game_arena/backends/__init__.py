"""
Backends package for LLM inference.

This package provides different backends for LLM inference:
- LiteLLM: API-based inference (OpenAI, Anthropic, etc.)
- vLLM: Local GPU inference
"""

from .llm_registry import (
    initialize_llm_registry,
    LLM_REGISTRY,
    generate_response,
    get_available_models,
    get_backend_for_model,
    extract_model_name,
    detect_model_backend,
    list_models
)
from .litellm_backend import LiteLLMBackend
from .vllm_backend import VLLMBackend
from .base_backend import BaseLLMBackend

__all__ = [
    "initialize_llm_registry",
    "LLM_REGISTRY",
    "generate_response",
    "get_available_models",
    "get_backend_for_model",
    "extract_model_name",
    "detect_model_backend",
    "list_models",
    "LiteLLMBackend",
    "VLLMBackend",
    "BaseLLMBackend"
]
