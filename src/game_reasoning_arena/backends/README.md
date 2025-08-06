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
Models are configured in `src/configs/vllm_models.yaml` with absolute paths to model directories:

```yaml
models:
  - name: vllm_Qwen2-7B-Instruct
    model_path: /path/to/models/Qwen/Qwen2-7B-Instruct
    tokenizer_path: /path/to/models/Qwen/Qwen2-7B-Instruct  # Optional
    description: Qwen2 7B Instruct model for local inference

  - name: vllm_custom-model
    model_path: /absolute/path/to/custom-model
    description: Custom model
```

**Configuration fields:**
- `name`: Model identifier with `vllm_` prefix for automatic backend detection
- `model_path`: **Absolute path** to model files (required)
- `tokenizer_path`: **Absolute path** to tokenizer files (optional, defaults to model_path)
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

## Adding New Providers and Models

### Adding a New API Provider
You can add new API providers in two ways:

#### Method 1: Configuration File (Recommended)
Edit `src/game_reasoning_arena/configs/api_providers.yaml`:

```yaml
providers:
  # Existing providers...

  # Add your new provider
  my_custom_ai:
    api_key_env: "MY_CUSTOM_AI_API_KEY"
    description: "My Custom AI Provider"
```

Then set the environment variable:
```bash
export MY_CUSTOM_AI_API_KEY="your_api_key_here"
```

#### Method 2: Runtime Registration
Add providers programmatically in your code:

```python
from game_reasoning_arena.backends.backend_config import config

# Add a new provider at runtime
config.add_provider(
    provider_name="custom_ai",
    api_key_env="CUSTOM_AI_API_KEY",
    description="Custom AI Provider"
)

# Now you can use it
api_key = config.get_api_key("custom_ai")
```

### Adding Models to LiteLLM Backend

To add new LiteLLM models, edit `src/game_reasoning_arena/configs/litellm_models.yaml`:

```yaml
models:
  # Add new models with the litellm_ prefix
  - litellm_my_custom_ai/my-model-v1
```

### Adding Models to vLLM Backend

To add new vLLM models, edit `src/game_reasoning_arena/configs/vllm_models.yaml`:

```yaml
models:
  # Existing models...

  # Add new models with absolute paths
  - name: vllm_custom-model-7b
    model_path: /home/user/models/custom-models/my-model-7b
    description: My custom 7B model

  - name: vllm_Llama-2-13b-chat-hf
    model_path: /home/user/models/meta-llama/Llama-2-13b-chat-hf
    description: Llama2 13B Chat model
```

**Important for vLLM models**: Unlike API-based models, vLLM requires the actual model files to be downloaded and stored locally on your machine.

#### Setting up Local Models

1. **Choose a directory** for storing your models:
```bash
mkdir -p /home/user/models
cd /home/user/models
```

2. **Download models using huggingface-hub**:
```bash
pip install huggingface-hub

# Example: Download Llama-2-7b-chat-hf
python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-2-7b-chat-hf', local_dir='./meta-llama/Llama-2-7b-chat-hf')"
```

3. **Update your configuration with the absolute paths**:
```yaml
models:
  - name: vllm_Llama-2-7b-chat-hf
    model_path: /home/user/models/meta-llama/Llama-2-7b-chat-hf
    description: Llama2 7B Chat model
```### Environment Variables for New Providers

4. After adding a provider, set the corresponding environment variable:

```bash
# In your .env file
CUSTOM_AI_API_KEY="your_key_here""
```

### Verifying New Providers
You can check what providers are supported:

```python
from game_reasoning_arena.backends.backend_config import config

# List all supported providers
providers = config.get_supported_providers()
print(f"Supported providers: {providers}")
```

### Model Naming Best Practices

- **LiteLLM models**: Always prefix with `litellm_`
  - Format: `litellm_<provider>/<model_name>`
  - Examples: `litellm_openai/gpt-4`, `litellm_groq/llama3-8b-8192`

- **vLLM models**: Always prefix with `vllm_`
  - Format: `vllm_<model_name>`
  - Examples: `vllm_Llama-2-7b-chat-hf`, `vllm_Qwen2-7B-Instruct`

### Complete Example: Adding a New Provider

1. **Add to configuration** (`configs/api_providers.yaml`):
```yaml
providers:
  perplexity:
    api_key_env: "PERPLEXITY_API_KEY"
    description: "Perplexity AI API"
```

2. **Add models** (`configs/litellm_models.yaml`):
```yaml
models:
  - litellm_perplexity/llama-3.1-sonar-small-128k-online
  - litellm_perplexity/llama-3.1-sonar-large-128k-online
```

3. **Set environment variable**:
```bash
export PERPLEXITY_API_KEY="pplx-your-api-key"
```

4. **Use in your configuration**:
```yaml
agents:
  player_0:
    type: "llm"
    model: "litellm_perplexity/llama-3.1-sonar-small-128k-online"
```

### Configuration Files
- `configs/api_providers.yaml` - API provider settings
- `configs/litellm_models.yaml` - Available LiteLLM models
- `configs/vllm_models.yaml` - Available vLLM models


## Troubleshooting

### LiteLLM Issues
- Ensure API keys are set (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- Check model names match provider documentation

### vLLM Issues
- Ensure CUDA is properly installed
- Check GPU memory availability
- Verify model files exist at the specified absolute paths
- Try reducing `gpu_memory_utilization` in config
