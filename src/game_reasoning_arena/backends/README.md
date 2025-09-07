# LLM Backends

This directory contains the backend implementations for different LLM providers and the unified model management system.

## Overview

The Game Reasoning Arena supports multiple LLM backends with automatic routing based on model name prefixes:

- **OpenRouter**: Access to 300+ models via unified API (`openrouter_*`)
- **LiteLLM**: Integration with 100+ providers (`litellm_*`)
- **vLLM**: Local GPU inference (`vllm_*`)
- **HuggingFace**: Direct transformers access (`hf_*`)

## Backend Files

- `base_backend.py` - Abstract base class for all backends
- `openrouter_backend.py` - OpenRouter API integration with credit management
- `litellm_backend.py` - LiteLLM provider integration
- `vllm_backend.py` - Local vLLM inference
- `huggingface_backend.py` - HuggingFace Transformers integration
- `llm_registry.py` - Central model registry with automatic backend detection
- `backend_config.py` - Configuration and API key management

## 🔍 Model Validation

Before running expensive multi-model experiments, use the model validator to check OpenRouter credits and ensure models are accessible.

### Usage Examples

```bash
# Validate specific models
python3 scripts/utils/model_validator.py --models litellm_groq/llama-3.1-8b-instant openrouter_openai/gpt-4o-mini

# Validate from config file
python3 scripts/utils/model_validator.py --config path/to/config.yaml

# The runner scripts now automatically validate before starting:
python3 scripts/run_ray_multi_model.py --config config.yaml
```

### What the Validator Checks

- **OpenRouter models**: API key existence and credit balance (shows warnings for low credits)
- **Other models** (LiteLLM, vLLM, HuggingFace): Assumes valid (no detailed validation)

### Credit Warnings

The validator will warn you about:
- Balance < $1.00: "Low balance" warning
- Balance < $5.00: "Balance warning"
- No API key: Validation failure

### Pre-run Integration

The validation is automatically integrated into runner scripts:

1. **Ray Multi-Model Runner** (`scripts/run_ray_multi_model.py`)
2. **Sequential Multi-Model Runner** (`scripts/run_multi_model_games.py`)

The runners will:
- ✅ Check OpenRouter credits before starting expensive experiments
- ✅ Show credit balance and warnings
- ❌ Fail fast if OpenRouter has no API key or insufficient access
- ✅ Assume non-OpenRouter models are valid (no time wasted on complex validation)

## 🚀 Quick Start

```python
from game_reasoning_arena.backends.llm_registry import generate_response

# Models are automatically routed to the correct backend
response = generate_response(
    model_name="openrouter_openai/gpt-4o-mini",
    prompt="Explain the rules of tic-tac-toe",
    max_tokens=200,
    temperature=0.1
)
```

## Environment Setup

Create a `.env` file in the project root:

```bash
# OpenRouter (recommended - single API key for 300+ models)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# LiteLLM providers (optional - for direct provider access)
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-key
GROQ_API_KEY=gsk-your-groq-key
TOGETHER_API_KEY=your-together-key
FIREWORKS_API_KEY=fw-your-fireworks-key
GEMINI_API_KEY=your-gemini-key
MISTRAL_API_KEY=your-mistral-key
```

## Backend Details

### 1. OpenRouter Backend (Recommended)
**Single API key, 300+ models, unified billing**

**Features:**
- Access to OpenAI, Anthropic, Google, Meta, Mistral, and more
- Competitive pricing with load balancing
- Credit management with automatic warnings
- Consistent API across all providers

**Configuration:** `../configs/openrouter_models.yaml`
```yaml
models:

```

### 2. LiteLLM Backend
**Direct provider access with individual API keys**

**Features:**
- Direct access to provider APIs
- Fine-grained control over API settings
- Support for provider-specific features

**Configuration:** `../configs/litellm_models.yaml`

### 3. vLLM Backend (Advanced)
**Local GPU inference for self-hosted models**

**Features:**
- Complete control over model deployment
- GPU-optimized inference
- Privacy (no data leaves your infrastructure)
- Cost control for heavy usage

**Requirements:**
- NVIDIA GPU with CUDA support
- Local model files
- vLLM package installation

**Configuration:** `../configs/vllm_models.yaml`
```yaml
models:
  - name: vllm_llama-2-7b-chat
```

### 4. HuggingFace Backend
**Direct access to HuggingFace Transformers**

**Features:**
- No API keys required
- Automatic model downloading
- Good for experimentation

**Supported models:** `hf_gpt2`, `hf_distilgpt2`, etc.

## Model Naming Convention

All models use prefixes for automatic backend routing:

```bash
openrouter_<provider>/<model>    # → OpenRouter backend
litellm_<provider>/<model>       # → LiteLLM backend
vllm_<model>                     # → vLLM backend
hf_<model>                       # → HuggingFace backend
```

**Examples:**
- `openrouter_openai/gpt-4o` → OpenRouter backend
- `litellm_groq/llama-3.1-8b-instant` → LiteLLM backend
- `vllm_llama-2-7b-chat` → vLLM backend
- `hf_gpt2` → HuggingFace backend

## Multi-Backend Configuration

You can mix backends in the same experiment:

```yaml
agents:
  player_0:
    type: llm
    model: openrouter_anthropic/claude-3.5-sonnet  # OpenRouter
  player_1:
    type: llm
    model: litellm_groq/llama-3.1-8b-instant       # LiteLLM
  player_2:
    type: llm
    model: vllm_llama-2-7b-chat                    # Local vLLM
```

## Configuration Files

- `openrouter_models.yaml` - Available OpenRouter models
- `litellm_models.yaml` - Available LiteLLM models
- `vllm_models.yaml` - Local vLLM model configurations

## Adding New Models

### OpenRouter Models
Edit `../configs/openrouter_models.yaml` and add the model:
```yaml
models:
  - openrouter_provider/new-model-name
```

### LiteLLM Models
Edit `../configs/litellm_models.yaml` and add the model:
```yaml
models:
  - litellm_provider/new-model-name
```

Set the provider's API key in `.env`:
```bash
PROVIDER_API_KEY=your-key-here
```

### vLLM Models
1. Download the model locally
2. Edit `../configs/vllm_models.yaml`:
```yaml
models:
  - name: vllm_new-model
    model_path: /absolute/path/to/model
    description: Description of the model
```

## Troubleshooting

### OpenRouter Issues
- Check `OPENROUTER_API_KEY` is set in `.env`
- Use the validator to check credits: `python3 scripts/utils/model_validator.py --models openrouter_openai/gpt-4o-mini`
- Verify model names match the configuration

### LiteLLM Issues
- Ensure provider API keys are set (e.g., `GROQ_API_KEY`, `TOGETHER_API_KEY`)
- Check model names match provider documentation
- Verify the model is listed in `litellm_models.yaml`

### vLLM Issues
- Ensure CUDA is properly installed
- Check GPU memory availability
- Verify model files exist at the specified paths
- Check that vLLM package is installed: `pip install vllm`

### General Issues
- Ensure the `.env` file is in the project root
- Check that model names use the correct prefixes
- Use the model validator to diagnose issues

## Development

To extend the backend system:

1. **Create a new backend class** inheriting from `BaseLLMBackend`
2. **Implement required methods**: `generate_response()`, `is_model_available()`, `get_available_models()`
3. **Add backend detection** in `llm_registry.py`
4. **Update configuration files** as needed
