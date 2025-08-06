"""
HuggingFace backend for local transformer models.
Uses transformers.pipeline() for lightweight models that run without API keys.
"""

from typing import Any, Dict, Optional
from transformers import pipeline
from .base_backend import BaseLLMBackend


class HuggingFaceBackend(BaseLLMBackend):
    """Backend for local HuggingFace transformer models using pipeline()."""

    def __init__(self, config_file: Optional[str] = None,
                 user_config: Optional[Dict[str, Any]] = None):
        """Initialize HuggingFace backend.

        Args:
            config_file: Optional path to configuration file (not used)
            user_config: Optional user configuration dict
        """
        super().__init__("HuggingFace")
        self.user_config = user_config or {}
        self.pipelines = {}  # Cache for loaded pipelines

        # Define available lightweight models
        self.available_models = {
            "gpt2": {
                "model_id": "gpt2",
                "pipeline_task": "text-generation",
                "max_new_tokens": 50
            },
            "distilgpt2": {
                "model_id": "distilgpt2",
                "pipeline_task": "text-generation",
                "max_new_tokens": 50
            },
            "google/flan-t5-small": {
                "model_id": "google/flan-t5-small",
                "pipeline_task": "text2text-generation",
                "max_new_tokens": 30
            },
            "EleutherAI/gpt-neo-125M": {
                "model_id": "EleutherAI/gpt-neo-125M",
                "pipeline_task": "text-generation",
                "max_new_tokens": 50
            }
        }

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available in our lightweight model registry."""
        return model_name in self.available_models

    def load_model(self, model_name: str) -> Any:
        """Load model using transformers pipeline."""
        if not self.is_model_available(model_name):
            raise ValueError(
                f"Model {model_name} not available in HuggingFace backend")

        if model_name not in self.pipelines:
            model_config = self.available_models[model_name]

            try:
                print(f"Loading {model_name} with transformers pipeline...")
                self.pipelines[model_name] = pipeline(
                    model_config["pipeline_task"],
                    model=model_config["model_id"]
                )
                print(f"✓ Successfully loaded {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
                raise RuntimeError(f"Failed to load model {model_name}: {e}")

        return self.pipelines[model_name]

    def generate_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate response using HuggingFace transformers pipeline."""
        if not self.is_model_available(model_name):
            raise ValueError(f"Model {model_name} not available")

        # Load model if not already loaded
        pipeline_obj = self.load_model(model_name)
        model_config = self.available_models[model_name]

        try:
            # Get parameters
            max_tokens = kwargs.get(
                "max_tokens", model_config["max_new_tokens"])
            temperature = kwargs.get("temperature", 0.3)

            # Handle tokenizer issues
            try:
                pad_token_id = pipeline_obj.tokenizer.eos_token_id
            except AttributeError:
                pad_token_id = None

            # Generate based on pipeline type
            if model_config["pipeline_task"] == "text2text-generation":
                # For T5/Flan-T5 models
                response = pipeline_obj(
                    prompt,
                    max_length=len(prompt.split()) + max_tokens,
                    do_sample=True,
                    temperature=temperature
                )
            else:
                # For GPT-2 and similar models
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "temperature": temperature
                }
                if pad_token_id is not None:
                    generation_kwargs["pad_token_id"] = pad_token_id

                response = pipeline_obj(prompt, **generation_kwargs)

            # Extract generated text
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')
                # For text-generation models, remove the original prompt
                if model_config["pipeline_task"] == "text-generation":
                    response_text = generated_text.replace(prompt, '').strip()
                else:
                    # For text2text-generation, the response is already clean
                    response_text = generated_text.strip()
            else:
                response_text = str(response)

            return response_text

        except Exception as e:
            raise RuntimeError(
                f"HuggingFace inference failed for {model_name}: {e}") from e
