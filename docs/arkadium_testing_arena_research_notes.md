# Arkadium Testing Arena: Research Transparency and Reproducibility Notes

This document explains the current research prototype added on the
`cursor/connect-four-baseline-542b` branch. It is intended to make the code
path transparent enough for review, replication, and later extension into a
larger Arkadium-style evaluation framework.

The objective of this prototype is not to build stronger game-playing agents.
The objective is to run controlled game episodes, capture model decisions and
reasoning traces, and preserve enough metadata to analyze reasoning failures.

## 1. Current User-Facing Entry Point

The lightweight frontend is:

```text
live_connect_four_app.py
```

Despite the historical filename, the page title is now:

```text
Arkadium Testing Arena
```

Run from the repository root:

```bash
python3 live_connect_four_app.py
```

Then open:

```text
http://127.0.0.1:7860
```

If Gradio prints a different local URL, use the URL printed in the terminal.

## 2. Required Runtime Dependencies

Install the minimal runtime dependencies:

```bash
python3 -m pip install open-spiel litellm gradio python-dotenv
```

Why each dependency is needed:

| Dependency | Purpose |
|---|---|
| `open-spiel` | Provides the game engines for Connect Four, Tic-Tac-Toe, Gin Rummy, and Solitaire |
| `litellm` | Provides a common API wrapper for hosted LLM providers |
| `gradio` | Provides the local browser frontend |
| `python-dotenv` | Allows optional `.env` loading for API keys |

Some analysis scripts in the original repository use additional packages such
as `pandas`, `torch`, `ray`, or `transformers`. The prototype code was adjusted
so these are not required for the lightweight frontend unless the specific
feature that needs them is used.

## 3. API Keys and Model Access

Hosted frontier models require hosted API access. The game loop sends one
prompt per model move, so these models cannot be evaluated without a usable
provider key.

Currently supported frontend presets:

| Frontend preset | Model string | Required key environment variable |
|---|---|---|
| OpenAI: GPT-4o-mini | `litellm_gpt-4o-mini` | `OPENAI_API_KEY` |
| Groq: Llama 3.1 8B Instant | `litellm_groq/llama-3.1-8b-instant` | `GROQ_API_KEY` |
| OpenRouter: Claude 3.5 Sonnet | `openrouter_anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` |
| OpenRouter: Gemini 2.5 Flash | `openrouter_google/gemini-2.5-flash` | `OPENROUTER_API_KEY` |
| Google Gemini: Gemini 2.5 Flash | `litellm_gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |

There is also a Cursor API key option in the dropdown, but it is intentionally
marked as unsupported for play. Cursor API keys are for Cursor Cloud Agent and
admin endpoints, not per-prompt chat-completion responses. The game loop needs
a provider that can answer a prompt for each move.

## 4. Supported Games in the Prototype

The frontend currently exposes:

| Display name | Internal game name | Source |
|---|---|---|
| Connect Four | `connect_four` | OpenSpiel |
| Tic-Tac-Toe | `tic_tac_toe` | OpenSpiel |
| Gin Rummy | `gin_rummy` | OpenSpiel |
| Solitaire | `solitaire` | OpenSpiel |

Important caveat:

- The Solitaire game is OpenSpiel's `solitaire`, not a confirmed Arkadium
  Klondike implementation.
- Gin Rummy and Solitaire use a generic text-rendered environment wrapper.
  Their state strings are OpenSpiel observations, not custom Arkadium UI
  renderings.

## 5. Relevant Code Files

### Frontend and experiment runner

```text
live_connect_four_app.py
```

Responsibilities:

- Builds the Gradio UI.
- Lets the user choose:
  - game set name
  - game
  - model/provider preset
  - API key
  - seed
  - number of rounds
  - maximum turns per round
  - whether to save/export game data
- Runs one or more rounds.
- Streams:
  - board state
  - turn log
  - legal actions
  - chosen actions
  - LLM reasoning traces
  - per-round outcome summaries
- Writes SQLite logs through the existing repo logger.
- Writes a downloadable ZIP export containing JSON and CSV files.

### Game registry

```text
src/game_reasoning_arena/arena/games/loaders.py
src/game_reasoning_arena/arena/games/registry.py
```

Responsibilities:

- `loaders.py` maps internal game names to OpenSpiel games.
- `registry.py` builds environment instances from those registrations.
- This branch registers `gin_rummy` and `solitaire` in addition to the games
  already supported by the repository.

### Game environments

```text
src/game_reasoning_arena/arena/envs/open_spiel_env.py
src/game_reasoning_arena/arena/envs/connect_four_env.py
src/game_reasoning_arena/arena/envs/tic_tac_toe_env.py
src/game_reasoning_arena/arena/envs/generic_text_env.py
```

Responsibilities:

- `open_spiel_env.py` implements the shared reset, step, observation, prompt,
  chance-node handling, and reward logic.
- `connect_four_env.py` renders Connect Four boards and legal actions.
- `tic_tac_toe_env.py` renders Tic-Tac-Toe boards and legal actions.
- `generic_text_env.py` renders OpenSpiel observation strings for games without
  custom UI renderers, currently Gin Rummy and Solitaire.

### Agents

```text
src/game_reasoning_arena/arena/agents/llm_agent.py
src/game_reasoning_arena/arena/agents/random_agent.py
src/game_reasoning_arena/arena/agents/policy_manager.py
src/game_reasoning_arena/arena/agents/agent_registry.py
```

Responsibilities:

- `LLMAgent` calls a model backend and extracts:
  - action
  - reasoning
- `RandomAgent` chooses uniformly from legal actions.
- `policy_manager.py` creates one agent per OpenSpiel player.
- `agent_registry.py` maps configured agent types to classes.

### Model/provider routing

```text
src/game_reasoning_arena/backends/llm_registry.py
src/game_reasoning_arena/backends/litellm_backend.py
src/game_reasoning_arena/backends/openrouter_backend.py
src/game_reasoning_arena/configs/litellm_models.yaml
src/game_reasoning_arena/configs/openrouter_models.yaml
```

Responsibilities:

- `llm_registry.py` selects a backend based on the model prefix.
- `litellm_backend.py` calls models through LiteLLM.
- `openrouter_backend.py` calls models through OpenRouter.
- `litellm_models.yaml` lists direct LiteLLM model names.
- `openrouter_models.yaml` lists OpenRouter model names.

### Logging

```text
src/game_reasoning_arena/arena/utils/loggers.py
```

Responsibilities:

- Writes SQLite database files under:

```text
results/
```

Tables:

| Table | Purpose |
|---|---|
| `moves` | One row per move |
| `rewards` | Final reward rows |
| `illegal_moves` | Illegal move records |
| `game_results` | Round/game result rows |

This branch also suffixes `game_results.status` with player outcome, for
example:

```text
terminated_win
terminated_loss
terminated_draw
truncated_loss
```

## 6. Current Frontend Control Flow

High-level flow inside `live_connect_four_app.py`:

1. User opens Arkadium Testing Arena.
2. User enters a game set name.
3. User selects game, model preset, rounds, seed, and max turns.
4. User pastes the selected provider's API key.
5. User clicks:

```text
Run selected model vs Random
```

6. `run_live_match(...)` validates:
   - game set name
   - selected provider/model
   - required API key
   - game availability
7. The app builds a config dictionary for the selected game and model.
8. For each round:
   - seed is set to `base_seed + round_index`
   - policies are initialized
   - the OpenSpiel environment is reset
   - turns run until terminal, truncated, illegal move, or error
   - move data is streamed to the UI
   - move data is written to SQLite
   - turn-level records are accumulated for export
9. At the end of each round:
   - final rewards are read from OpenSpiel
   - player 0 outcome is classified as win/loss/draw
   - result rows are written to SQLite
10. At the end of the game set:
    - summary counts are shown in the transcript
    - if save/export is enabled, JSON and CSV files are written and zipped
    - the ZIP is made downloadable in the browser

## 7. Exported Data

Exports are written under:

```text
results/live_exports/
```

Each saved game set produces a ZIP file named approximately:

```text
<game_set_name>_<timestamp>.zip
```

The ZIP contains:

| File | Contents |
|---|---|
| JSON | Metadata, round summaries, full turn records, transcript |
| CSV | One row per turn for spreadsheet/statistical analysis |

Turn-level CSV columns include:

```text
game_set_name
game
game_display_name
model_preset
model_name
round
turn
player_id
agent_type
agent_model
legal_actions
chosen_action
reasoning
outcome
round_status
rewards
state_before
state_after
board_before
board_after
seed
```

These fields are intended to support later analysis of:

- illegal moves
- state tracking failures
- memory failures
- turn-to-failure
- per-round win/loss/draw
- model reasoning traces
- long-horizon degradation across turns

## 8. Outcome Classification

Outcome classification is currently simple and reward-based.

For two-player games:

```text
player 0 reward > opponent reward => win
player 0 reward < opponent reward => loss
otherwise => draw
```

For single-player games:

```text
player 0 reward > 0 => win
otherwise => loss
```

For truncated games:

```text
truncated => loss
```

This is intentionally transparent, but it is not yet a complete game-specific
success metric for all future Arkadium games. Future game adapters should
define game-specific scoring semantics.

## 9. Reasoning Trace Collection

The prompt is generated by:

```text
src/game_reasoning_arena/arena/envs/open_spiel_env.py
```

The prompt includes:

- game name
- player marker
- move number
- board or observation string
- legal actions
- instruction to return JSON-like reasoning and action

The LLM call is made by:

```text
src/game_reasoning_arena/arena/agents/llm_agent.py
```

The returned text is parsed into:

- `action`
- `reasoning`

The reasoning string is:

- streamed in the frontend transcript
- written into SQLite `moves.reasoning`
- written into the JSON/CSV export

## 10. Reproducing a Run

To reproduce a run:

1. Pull the branch:

```bash
git pull origin cursor/connect-four-baseline-542b
```

2. Install dependencies:

```bash
python3 -m pip install open-spiel litellm gradio python-dotenv
```

3. Start the app:

```bash
python3 live_connect_four_app.py
```

4. Record the following run parameters:

```text
git commit hash
game set name
game
model preset
model string
provider
base seed
rounds
max turns per round
timestamp
```

5. Run the game set.
6. Download the ZIP export.
7. Preserve the SQLite database files under `results/`.

For strict reproducibility, use the same:

- branch/commit
- model provider
- model name
- seed
- number of rounds
- max turns
- dependency versions

Hosted frontier models may still produce different outputs over time because
providers can update serving infrastructure or model snapshots.

## 11. Known Limitations

Current limitations:

- Hosted model calls require working API keys and provider quota.
- Hosted APIs can fail with rate limits, quota errors, or temporary 503
  availability errors.
- Gin Rummy and Solitaire currently use generic OpenSpiel text observations,
  not custom human-optimized renderers.
- OpenSpiel Solitaire is not guaranteed to match Arkadium Klondike Solitaire.
- Outcome classification is reward-based and should be refined per game.
- Failure taxonomy labeling is not automated yet.
- The current frontend compares player 0 model against RandomAgent, not yet
  model-vs-model.
- API keys are not written to logs or exports, but users should still avoid
  pasting keys into chat transcripts or committed files.

## 12. Suggested Next Research Extensions

Recommended next steps:

1. Add explicit failure taxonomy labels to each turn:
   - Myopic Trap Commitment
   - Irreversible Commitment Error
   - Multi-Constraint Collapse
   - Long-Horizon Decoupling
   - Visuospatial Blind Spot
   - Illegal Move
   - Memory Failure
   - State Tracking Failure
2. Add model-vs-model match support.
3. Add direct OpenRouter presets for GPT and Grok if those models are available
   on the research account.
4. Add game-specific adapters for Arkadium games.
5. Add richer metrics:
   - illegal move rate
   - turn-to-failure
   - failure type distribution
   - win rate by model/game
   - degradation curves by turn number
6. Add deterministic export manifests that include package versions and git
   commit hashes automatically.
