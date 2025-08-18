# Gradio Interface Components

This directory contains the Gradio interface components for the Board Game Arena.

## Files

- `gradio_config_generator.py` - Configuration generator that bridges Gradio UI with the game infrastructure
- `utils.py` - Utility functions for UI display formatting (e.g., model name cleaning)
- `test_utils.py` - Test cases for the utility functions
- `__init__.py` - Package initialization

## Key Functions

### `clean_model_name(model_name: str) -> str` (from utils.py)

Cleans up long model names from the database to display only the essential model name.

**Purpose**: The database stores model names with full provider paths like:
- `litellm_together_ai_meta_llama_Meta_Llama_3.1_8B_Instruct_Turbo`
- `litellm_fireworks_ai_accounts_fireworks_models_glm_4p5_air`

This function extracts just the model name part for cleaner display:
- `Meta-Llama-3.1-8B-Instruct-Turbo`
- `glm-4p5-air`

**Supported Patterns**:
- LiteLLM models with provider prefixes
- vLLM models with prefixes
- Models with slash-separated paths
- GPT model variants
- Special cases (random bots, etc.)

**Testing**:
Run `python3 test_utils.py` from the `ui/` directory to test all supported patterns.

## Main App

The main Gradio app (`app.py`) is located in the root directory for HuggingFace Spaces compatibility.

## Running the App

From the project root directory:

```bash
python app.py
```

## Architecture

```
app.py (Gradio UI - in root directory for HF Spaces)
    ↓
ui/gradio_config_generator.py (Game configuration bridge)
    ↓
src/game_reasoning_arena/ (Core game library)
```

The Gradio app provides:
- Interactive game interface
- Performance leaderboards
- Metrics dashboards
- LLM reasoning analysis


## Uploading Results

- Go to **Leaderboard** tab → **Upload .db**
- Files are stored in `scripts/results/` inside the Space