# LLM Inference Backends

This package provides a unified interface for LLM inference with multiple backends.

**Model Naming Convention:**
- LiteLLM models: `litellm_<model_name>` (e.g., `litellm_gpt-4`)
- vLLM models: `vllm_<model_name>` (e.g., `vllm_Qwen2-7B-Instruct`)

This prefix-based naming allows **automatic backend selection** and enables **multi-provider configurations** where different models use different backends within the same simulation.

**Example: Multiple Providers in One Simulation**
```json
{
  "agents": {
    "player_0": {"type": "llm", "model": "litellm_groq/llama3-8b-8192"},
    "player_1": {"type": "llm", "model": "litellm_openai/gpt-4"},
    "player_2": {"type": "llm", "model": "vllm_Qwen2-7B-Instruct"}
  }
}
```

## 🏗️ Architecture

```
src/backends/
├── __init__.py           # Main package interface
├── base_backend.py       # Abstract base class for backends
├── litellm_backend.py    # LiteLLM API backend
├── vllm_backend.py       # vLLM local GPU backend
├── llm_registry.py       # Central model registry
├── config.py             # Configuration management
├── example.py            # Usage example
└── README.md             # This file
```

## 🚀 Quick Start

```python
import os
from src.backends import initialize_llm_registry, generate_response

# Set your preferred backend
# No need to set INFERENCE_BACKEND - automatic detection based on model prefix

# Initialize the system
initialize_llm_registry()

# Generate a response
response = generate_response(
    model_name="litellm_gpt-3.5-turbo",
    prompt="Hello, world!",
    max_tokens=50,
    temperature=0.1
)
print(response)
```

## 1. LiteLLM (API-based) - Default
**Recommended for most users**

- **Easy setup** - No GPU management needed
- **Multiple providers** - OpenAI, Anthropic, Cohere, etc.
- **No local storage** - Models hosted by providers
- **Consistent API** - Same interface for all models

### Configuration
Models are configured in `src/configs/litellm_models.json`:
```json
{
  "models": [
    "litellm_gpt-4",
    "litellm_gpt-3.5-turbo",
    "litellm_groq/llama3-8b-8192",
    "litellm_together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
  ]
}
```

### Usage
```bash
# Default - automatic backend selection based on model prefixes
python scripts/runner.py
```

## 2. vLLM (Local GPU) - Advanced
**For users with local GPUs and models**

- **Advanced setup** - Requires GPU configuration
- **High performance** - Direct GPU inference
- **Local models** - Full control over model files
- **Customizable** - Fine-tune inference parameters

### Prerequisites
```bash
# Install vLLM (requires CUDA)
pip install vllm

# Ensure models are downloaded locally
```

### Configuration
Models are configured in `src/configs/vllm_models.json` with explicit model and tokenizer paths:
```json
{
  "models": [
    {
      "name": "vllm_Qwen2-7B-Instruct",
      "model_path": "/path/to/models/Qwen/Qwen2-7B-Instruct",
      "tokenizer_path": "/path/to/models/Qwen/Qwen2-7B-Instruct",
      "description": "Qwen2 7B Instruct model for local inference"
    },
    {
      "name": "vllm_llama2-7b-chat",
      "model_path": "/path/to/models/llama2-7b-chat",
      "tokenizer_path": "/path/to/models/llama2-7b-chat",
      "description": "Llama2 7B Chat model"
    }
  ]
}
```

**Configuration fields:**
- `name`: Model identifier with `vllm_` prefix for automatic backend detection
- `model_path`: Absolute path to the model files (required)
- `tokenizer_path`: Path to tokenizer files (optional, defaults to model_path)
- `description`: Human-readable description (optional)

### Usage
```bash
# Use vLLM models with automatic detection
python scripts/runner.py --config config_with_vllm_models.json
```

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `MAX_TOKENS` | Integer | `250` | Maximum tokens per response |
| `TEMPERATURE` | Float 0.0-1.0 | `0.1` | Sampling temperature |

## Multi-Provider Configuration
The system automatically uses both backends based on model prefixes:
- Models starting with `litellm_` use the LiteLLM backend
- Models starting with `vllm_` use the vLLM backend

This allows you to mix multiple inference providers in the same configuration.

## Troubleshooting

### LiteLLM Issues
- Ensure API keys are set (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- Check model names match provider documentation
- Verify network connectivity

### vLLM Issues
- Ensure CUDA is properly installed
- Check GPU memory availability
- Verify model files exist in `MODELS_DIR`
- Try reducing `gpu_memory_utilization` in config

## Performance Comparison

| Aspect | LiteLLM | vLLM |
|--------|---------|------|
| Setup Time | Minutes | Hours |
| First Request | Fast | Slow (model loading) |
| Subsequent Requests | Fast | Very Fast |
| GPU Memory | None | High |
| Cost | Pay per token | Hardware cost |
| Model Variety | High | Limited to downloaded |
