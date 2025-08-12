"""
Configuration management for LLM backends.
"""

import os
from pathlib import Path
from typing import Dict, Any


class BackendConfig:
    """Configuration class for managing backend settings."""

    def __init__(self):
        # Default paths - can be overridden by environment variables
        config_dir = Path(__file__).parent / ".." / "configs"

        self.litellm_config = os.getenv(
            "LITELLM_CONFIG_FILE",
            str(config_dir / "litellm_models.yaml")
        )

        self.vllm_config = os.getenv(
            "VLLM_CONFIG_FILE",
            str(config_dir / "vllm_models.yaml")
        )

        # Backend selection
        self.inference_backend = os.getenv("INFERENCE_BACKEND", "litellm")

        # Default inference parameters (can be overridden by YAML config)
        self.max_tokens = 250
        self.temperature = 0.1

    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider using standard naming convention."""
        env_var = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_var)

        if not api_key:
            raise ValueError(f"API key not found for provider '{provider}'. "
                             f"Please set {env_var} in your environment")

        return api_key

    def get_api_key_env_var(self, provider: str) -> str:
        """Get the environment variable name for a specific provider."""
        return f"{provider.upper()}_API_KEY"

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
