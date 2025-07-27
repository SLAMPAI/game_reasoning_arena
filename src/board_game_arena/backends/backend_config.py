"""
Configuration management for LLM backends.
"""

import os
import yaml
from typing import Dict, Any, Optional


class BackendConfig:
    """Configuration class for managing backend settings."""

    def __init__(self):
        # Default paths - can be overridden by environment variables
        self.litellm_config = os.getenv(
            "LITELLM_CONFIG_FILE",
            os.path.join(os.path.dirname(__file__), "..", "configs",
                        "litellm_models.yaml")
        )

        self.vllm_config = os.getenv(
            "VLLM_CONFIG_FILE",
            os.path.join(os.path.dirname(__file__), "..", "configs",
                         "vllm_models.yaml")
        )

        self.api_providers_config = os.getenv(
            "API_PROVIDERS_CONFIG_FILE",
            os.path.join(os.path.dirname(__file__), "..", "configs",
                         "api_providers.yaml")
        )

        # Backend selection
        self.inference_backend = os.getenv("INFERENCE_BACKEND", "litellm")

        # Inference parameters
        self.max_tokens = int(os.getenv("MAX_TOKENS", "250"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))

        # Load provider configuration
        self._provider_config = self._load_provider_config()

    def _load_provider_config(self) -> Dict[str, Any]:
        """Load API provider configuration from YAML file."""
        try:
            with open(self.api_providers_config, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('providers', {})
        except FileNotFoundError:
            raise FileNotFoundError(
                f"API providers configuration file not found: "
                f"{self.api_providers_config}. "
                f"Please ensure the file configs/api_providers.yaml exists."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load provider config: {e}")

    def get_api_key(self, provider: str) -> str:
        """Get API key for a specific provider."""
        provider_lower = provider.lower()

        # Check if provider exists in configuration
        if provider_lower not in self._provider_config:
            raise ValueError(f"Unknown provider: {provider}")

        # Get the environment variable name for this provider
        env_var = self._provider_config[provider_lower].get("api_key_env")
        if not env_var:
            # Fallback to standard naming convention
            env_var = f"{provider.upper()}_API_KEY"

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found for {provider}. "
                             f"Please set {env_var} in your .env file")

        return api_key

    def get_api_key_env_var(self, provider: str) -> str:
        """Get the environment variable name for a specific provider."""
        provider_lower = provider.lower()

        if provider_lower in self._provider_config:
            env_var = self._provider_config[provider_lower].get("api_key_env")
            if env_var:
                return env_var

        # Fallback to standard naming convention
        return f"{provider.upper()}_API_KEY"

    def add_provider(self, provider_name: str, api_key_env: str,
                     description: str = "") -> None:
        """
        Dynamically add a new API provider.

        Args:
            provider_name: Name of the provider (e.g., 'custom_ai')
            api_key_env: Environment variable name for the API key
            description: Optional description of the provider
        """
        self._provider_config[provider_name.lower()] = {
            "api_key_env": api_key_env,
            "description": description
        }

    def get_supported_providers(self) -> list:
        """Get list of all supported providers."""
        return list(self._provider_config.keys())

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
