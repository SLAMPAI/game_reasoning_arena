"""
Base backend interface for LLM inference.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class BaseLLMBackend(ABC):
    """Abstract base class for LLM inference backends."""

    def __init__(self, name: str):
        self.name = name

    def load_models(self) -> List[str]:
        """Load and return list of available models.

        Optional method - backends can override if needed.
        Default implementation returns empty list.
        """
        return []

    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in this backend."""
        pass

    @abstractmethod
    def load_model(self, model_name: str) -> Any:
        """Load a specific model."""
        pass

    @abstractmethod
    def generate_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate a response using the specified model."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
