"""
OpenRouter backend for API-based inference.
"""

import logging
import requests
from typing import Any, Dict, Optional

from .base_backend import BaseLLMBackend

# Suppress verbose logging
logging.getLogger("urllib3").setLevel(logging.WARNING)


class OpenRouterBackend(BaseLLMBackend):
    """Backend for API-based LLM inference using OpenRouter."""

    def __init__(self, config_file: Optional[str] = None,
                 user_config: Optional[Dict[str, Any]] = None):
        """Initialize OpenRouter backend.

        Args:
            config_file: Optional path to OpenRouter configuration file
            user_config: Optional user configuration dict from runner.py
        """
        super().__init__("OpenRouter")

        # Import config here to avoid circular imports
        from .backend_config import config as backend_config

        self.config_file = config_file
        self.backend_config = backend_config
        self.user_config = user_config or {}

        # OpenRouter API endpoint
        self.api_base = "https://openrouter.ai/api/v1"

        # Set API key from environment variables
        self._set_api_key()

    def _set_api_key(self):
        """Set API key for OpenRouter."""
        try:
            self.api_key = self.backend_config.get_api_key("openrouter")
        except ValueError as e:
            print(f"Error: {e}")
            print("OpenRouter models will not be available")
            self.api_key = None

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available by verifying we have an API key."""
        if not self.api_key:
            print(f"Error: Model '{model_name}' not available - "
                  f"no OpenRouter API key")
            return False

        # We could potentially query OpenRouter's models endpoint here,
        # but for now we'll assume any model name is valid if we have an
        # API key
        return True

    def load_model(self, model_name: str) -> Any:
        """Load model (for OpenRouter, this just returns the model name)."""
        return model_name

    def generate_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate response using OpenRouter API."""
        if not self.api_key:
            raise RuntimeError("OpenRouter API key not available")

        try:
            # Use backend config defaults if not provided
            inference_params = self.backend_config.get_inference_params()

            # Extract the actual model name without backend prefix
            if model_name.startswith("openrouter_"):
                # Remove "openrouter_" prefix
                actual_model_name = model_name[11:]
            else:
                actual_model_name = model_name

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": (
                    "https://github.com/lcipolina/game_reasoning_arena"
                ),
                "X-Title": "Game Reasoning Arena"
            }

            # Prepare request payload
            payload = {
                "model": actual_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get(
                    "temperature", inference_params["temperature"]
                ),
                "max_tokens": kwargs.get(
                    "max_tokens", inference_params["max_tokens"]
                )
            }

            # Make API request
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )

            # Check for API errors
            if response.status_code != 200:
                error_msg = (f"OpenRouter API error {response.status_code}: "
                             f"{response.text}")
                raise RuntimeError(error_msg)

            # Parse response
            response_data = response.json()

            if ("choices" not in response_data or
                    len(response_data["choices"]) == 0):
                raise RuntimeError(
                    f"No choices in OpenRouter response: {response_data}"
                )

            return response_data["choices"][0]["message"]["content"]

        except requests.RequestException as e:
            raise RuntimeError(
                f"OpenRouter inference failed for {model_name}: "
                f"Network error - {e}"
            ) from e
        except KeyError as e:
            raise RuntimeError(
                f"OpenRouter inference failed for {model_name}: "
                f"Invalid response format - missing {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"OpenRouter inference failed for {model_name}: {e}"
            ) from e

    def get_available_models(self) -> list:
        """Get list of available models from OpenRouter API."""
        if not self.api_key:
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.get(
                f"{self.api_base}/models",
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                models_data = response.json()
                return [model["id"] for model in models_data.get("data", [])]
            else:
                print(f"Warning: Could not fetch OpenRouter models: "
                      f"{response.status_code}")
                return []

        except Exception as e:
            print(f"Warning: Error fetching OpenRouter models: {e}")
            return []

    def verify_model_exists(self, model_name: str) -> bool:
        """Verify that a specific model exists in OpenRouter's API."""
        if not self.api_key:
            return False

        # Extract the actual model name without backend prefix
        if model_name.startswith("openrouter_"):
            actual_model_name = model_name[11:]
        else:
            actual_model_name = model_name

        available_models = self.get_available_models()
        return actual_model_name in available_models
