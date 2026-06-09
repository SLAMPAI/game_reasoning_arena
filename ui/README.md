# Gradio Interface Components

This directory contains the Gradio interface components for the Board Game Arena.

## Files

- `gradio_config_generator.py` - Configuration generator that bridges Gradio UI with the game infrastructure
- `utils.py` - Utility functions for UI display formatting (e.g., model name cleaning)
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

For a frontend that follows the original Game Reasoning Arena methodology
directly, use:

```bash
python3 game_reasoning_arena_frontend.py
```

This UI uses the repository's OpenSpiel registry, policy manager, backend model
prefixes, `LLMAgent` / `RandomAgent`, SQLite logging, CSV/JSON exports, and
frontend charts for rewards, outcomes, and turns per episode.

For the lightweight Arkadium Testing Arena live game viewer:

```bash
python3 live_connect_four_app.py
```

The live viewer requires:

```bash
python3 -m pip install open-spiel litellm gradio python-dotenv
```

Name the game set before running, choose a game and model preset, then paste
the matching provider API key into the password field. OpenRouter Gemini
presets use an OpenRouter key; Google Gemini presets use a Google Gemini/AI
Studio key. The key is used only in the running Python process as the selected
provider's API-key environment variable; it is not written to result logs.

The live viewer supports Connect Four, Tic-Tac-Toe, Gin Rummy, and OpenSpiel
Solitaire. Set `Rounds` to run multiple episodes. The turn log shows player 0's
win/loss/draw outcome per round, and the SQLite `game_results.status` values
include the outcome suffix (for example `terminated_win`).

Keep `Save/export game data` enabled to create a downloadable ZIP file with:

- JSON metadata, round summaries, full turn records, and transcript
- CSV turn-level records for spreadsheet analysis

For a transparent research-oriented explanation of the code path, data flow,
logging, exports, reproducibility steps, and limitations, see:

```text
docs/arkadium_testing_arena_research_notes.md
```

Note: Cursor API keys are shown in the live viewer for clarity, but they are
not playable model keys for this per-turn game loop. Cursor's public API is
for Cursor Cloud Agent/admin operations, not chat-completions responses.

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