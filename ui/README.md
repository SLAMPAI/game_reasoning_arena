# Gradio Interface Components

This directory contains the Gradio interface components for the Board Game Arena.

## Files

- `gradio_config_generator.py` - Configuration generator that bridges Gradio UI with the game infrastructure
- `__init__.py` - Package initialization

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