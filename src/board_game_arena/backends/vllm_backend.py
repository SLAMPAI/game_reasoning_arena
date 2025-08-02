"""
vLLM backend for local GPU LLM inference.
"""

from pathlib import Path
from typing import Any, List, Optional
import yaml
from .base_backend import BaseLLMBackend


class VLLMBackend(BaseLLMBackend):
    """Backend for local GPU LLM inference using vLLM."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize vLLM backend.

        Args:
            config_file: Optional path to vLLM configuration file
        """
        super().__init__("vLLM")

        # Import config here to avoid circular imports
        from .backend_config import config as backend_config

        self.config_file = config_file or backend_config.vllm_config
        self.backend_config = backend_config
        self._models = None
        self._model_paths = {}
        self._loaded_models = {}

    def load_models(self) -> List[str]:
        """Load available models from vLLM configuration."""
        if self._models is None:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                self._models = []

                # Handle the nested structure with "models" key
                models_list = config_data.get("models", [])

                for item in models_list:
                    if isinstance(item, dict):
                        model_name = item.get("name")
                        model_path = item.get("model_path")
                        tokenizer_path = item.get("tokenizer_path", model_path)

                        if model_name and model_path:
                            self._models.append(model_name)
                            self._model_paths[model_name] = {
                                "model_path": model_path,
                                "tokenizer_path": tokenizer_path
                            }
                    else:
                        raise ValueError(
                            f"Invalid configuration format. Expected dict with "
                            f"'name' and 'model_path' fields, got: {type(item)}"
                        )

            except FileNotFoundError:
                print(f"Warning: {self.config_file} not found. "
                      f"No vLLM models available.")
                self._models = []

        return self._models

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available.

        Args:
            model_name: Model name without the backend prefix
        """
        # Check if any available model (with vllm_ prefix) matches
        for available_model in self.load_models():
            if (available_model.startswith("vllm_") and
                available_model[5:] == model_name):
                return True
        return False

    def load_model(self, model_name: str) -> Any:
        """Load a specific model for inference.

        Args:
            model_name: Model name without the backend prefix

        Raises:
            ValueError: If model is not available
        """
        if not self.is_model_available(model_name):
            raise ValueError(f"Model {model_name} not available in vLLM backend")

        # Return cached model if already loaded
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        try:
            # Import vLLM only when needed
            from vllm import LLM as vLLM

            # Get the full model name with prefix to look up the path
            full_model_name = f"vllm_{model_name}"

            # Get the model paths from our stored paths
            if full_model_name not in self._model_paths:
                raise ValueError(
                    f"Model path not found for {full_model_name}. "
                    f"Make sure it's configured in {self.config_file}"
                )

            paths = self._model_paths[full_model_name]
            model_path = paths["model_path"]
            tokenizer_path = paths["tokenizer_path"]

            # Verify the paths exist
            model_path_obj = Path(model_path)
            tokenizer_path_obj = Path(tokenizer_path)

            if not model_path_obj.exists():
                raise FileNotFoundError(
                    f"Model path does not exist: {model_path}"
                )

            if not tokenizer_path_obj.exists():
                raise FileNotFoundError(
                    f"Tokenizer path does not exist: {tokenizer_path}"
                )

            print(f"Loading vLLM model from: {model_path}")
            print(f"Using tokenizer from: {tokenizer_path}")

            # Load model with vLLM
            model = vLLM(
                model=model_path,
                tokenizer=tokenizer_path,
                tensor_parallel_size=1,  # Can be made configurable
                gpu_memory_utilization=0.7,
                trust_remote_code=True,
                dtype="half",
            )

            # Cache the loaded model
            self._loaded_models[model_name] = model
            print(f"Successfully loaded vLLM model: {model_name}")
            return model

        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Please install it to use local "
                "inference."
            ) from exc
        except Exception as e:
            raise RuntimeError(
                f"Failed to load vLLM model {model_name}: {e}") from e

    def generate_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate response using vLLM."""
        try:
            from vllm import SamplingParams

            model = self.load_model(model_name)

            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 512),
                stop=kwargs.get("stop", None),
            )

            # Generate response
            outputs = model.generate([prompt], sampling_params)

            # Extract the generated text
            if outputs and len(outputs) > 0:
                return outputs[0].outputs[0].text.strip()
            else:
                return ""

        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {e}") from e

    def cleanup(self):
        """Clean up loaded models."""
        self._loaded_models.clear()
        print("vLLM backend cleaned up")
