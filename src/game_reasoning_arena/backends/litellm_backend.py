"""
LiteLLM backend for API-based inference.
"""

import logging
import os
from typing import Any, Dict, Optional

import litellm

from .base_backend import BaseLLMBackend

# Suppress LiteLLM verbose logging
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class LiteLLMBackend(BaseLLMBackend):
    """Backend for API-based LLM inference using LiteLLM."""

    def __init__(self, config_file: Optional[str] = None,
                 user_config: Optional[Dict[str, Any]] = None):
        """Initialize LiteLLM backend.

        Args:
            config_file: Optional path to LiteLLM configuration file
            user_config: Optional user configuration dict from runner.py
        """
        super().__init__("LiteLLM")

        # Import config here to avoid circular imports
        from .backend_config import config as backend_config

        self.config_file = config_file or backend_config.litellm_config
        self.backend_config = backend_config
        self.user_config = user_config or {}

        # Set API keys from environment variables
        self._set_api_keys()

    def _set_api_keys(self):
        """Set API keys for providers that are actually configured."""
        providers_needed = set()

        # Get providers from user config if available
        if self.user_config:
            llm_backend_config = self.user_config.get("llm_backend", {})

            # Add provider from default model
            default_model = llm_backend_config.get("default_model")
            if default_model and '/' in default_model:
                providers_needed.add(default_model.split('/')[0])

            # Add providers from agents configuration
            agents_config = llm_backend_config.get("agents", {})
            for agent_config in agents_config.values():
                model = agent_config.get("model")
                if model and '/' in model:
                    providers_needed.add(model.split('/')[0])

        # Fallback to backend config if no user config
        if not providers_needed:
            try:
                inference_params = self.backend_config.get_inference_params()
                default_model = inference_params.get(
                    "default_model", "groq/gemma-7b-it")

                if '/' in default_model:
                    providers_needed.add(default_model.split('/')[0])
                else:
                    providers_needed.add('openai')

            except Exception:
                # Ultimate fallback
                providers_needed.add('groq')

        # Set API keys only for needed providers
        for provider in providers_needed:
            self._set_api_key_for_provider(provider)

    def _set_api_key_for_provider(self, provider: str):
        """Set API key for a specific provider."""
        try:
            api_key = self.backend_config.get_api_key(provider)
            env_var = self.backend_config.get_api_key_env_var(provider)
            os.environ[env_var] = api_key
        except ValueError as e:
            # API key not found for this provider
            print(f"Error: {e}")
            print(f"Skipping provider '{provider}' - models from this "
                  f"provider will not be available")

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available by querying LiteLLM's validation."""
        try:
            # Check if we have the required API key for the provider
            if '/' in model_name:
                provider = model_name.split('/')[0]
            else:
                provider = 'openai'

            try:
                self.backend_config.get_api_key(provider)
                # Ensure the API key is set in environment for LiteLLM
                self._set_api_key_for_provider(provider)
                return True
            except ValueError as e:
                # No API key for this provider
                print(f"Error: Model '{model_name}' not available - {e}")
                return False

        except Exception as e:
            # If any error occurs, assume model is not available
            print(f"ERROR: Error checking availability of model "
                  f"'{model_name}': {e}")
            return False

    def load_model(self, model_name: str) -> Any:
        """Load model (for LiteLLM, this just returns the model name)."""
        return model_name

    def generate_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate response using LiteLLM API."""
        try:
            # Use backend config defaults if not provided
            inference_params = self.backend_config.get_inference_params()

            response = litellm.completion(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get(
                    "max_tokens", inference_params["max_tokens"]
                ),
                temperature=kwargs.get(
                    "temperature", inference_params["temperature"]
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(
                f"LiteLLM inference failed for {model_name}: {e}"
            ) from e
