"""
Configuration management for LLM backends.
"""

import os
from typing import Dict, Any


class BackendConfig:
    """Configuration class for managing backend settings."""

    def __init__(self):
        # Default paths - can be overridden by environment variables
        self.litellm_config = os.getenv(
            "LITELLM_CONFIG_FILE",
            os.path.join(os.path.dirname(__file__), "..", "configs", "litellm.json")
        )

        self.vllm_config = os.getenv(
            "VLLM_CONFIG_FILE",
            os.path.join(os.path.dirname(__file__), "..", "configs", "vllm_models.json")
        )

        self.models_dir = os.getenv(
            "MODELS_DIR",
        )

        # Backend selection
        self.inference_backend = os.getenv("INFERENCE_BACKEND", "litellm")

        # Inference parameters
        self.max_tokens = int(os.getenv("MAX_TOKENS", "250"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))

    def get_api_key(self, provider: str) -> str:
        """Get API key for a specific provider."""
        key_mapping = {
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "cohere": "COHERE_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "replicate": "REPLICATE_API_KEY"
        }

        env_var = key_mapping.get(provider.lower())
        if not env_var:
            raise ValueError(f"Unknown provider: {provider}")

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found for {provider}. "
                           f"Please set {env_var} in your .env file")

        return api_key

    def get_inference_params(self) -> Dict[str, Any]:
        """Get inference parameters as a dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

    def validate(self) -> bool:
        """Validate the configuration."""
        valid_backends = ["litellm", "vllm", "hybrid"]
        if self.inference_backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.inference_backend}. "
                           f"Must be one of {valid_backends}")

        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, "
                           f"got {self.temperature}")

        if self.max_tokens <= 0:
            raise ValueError(f"Max tokens must be positive, "
                           f"got {self.max_tokens}")

        return True

    def __str__(self) -> str:
        return (f"BackendConfig(backend={self.inference_backend}, "
                f"max_tokens={self.max_tokens}, "
                f"temperature={self.temperature})")


# Global configuration instance
config = BackendConfig()
