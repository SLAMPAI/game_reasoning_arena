#!/usr/bin/env python3
"""Arkadium Testing Arena local frontend.

This file is intentionally written as a transparent, single-file experiment
runner. It is more verbose than a production web app because the project is a
research workflow: reviewers should be able to inspect how games, models,
rounds, logs, and exports are created without chasing hidden framework magic.

Run from the repository root:

    python3 live_connect_four_app.py

Then open the printed local URL, choose a game and provider/model preset,
paste the matching API key, and click "Run selected model vs Random".
"""

from __future__ import annotations

# Standard-library modules only. Keeping these imports explicit makes it clear
# which parts of the app touch files (`csv`, `json`, `zipfile`, `Path`), process
# environment variables (`os`), or time/metadata (`datetime`, `time`).
import csv
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple


# Resolve paths relative to this file so the app works when launched from the
# repository root in VS Code. `SRC_DIR` is inserted into `sys.path` because the
# package is not always installed with `pip install -e .` on student machines.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
EXPORT_DIR = ROOT_DIR / "results" / "live_exports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# `.env` support is optional. If python-dotenv is not installed, the app still
# runs; users can paste keys into the UI or set environment variables manually.
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


# ---------------------------------------------------------------------------
# Frontend presets
# ---------------------------------------------------------------------------
#
# These dictionaries are the source of truth for what appears in the Gradio
# dropdowns. Display labels are intentionally human-readable; values map to the
# internal OpenSpiel game names and LiteLLM/OpenRouter model identifiers used by
# the repository.
DEFAULT_PRESET = "OpenAI: GPT-4o-mini"
DEFAULT_GAME = "connect_four"
GAME_PRESETS = {
    "Connect Four": "connect_four",
    "Tic-Tac-Toe": "tic_tac_toe",
    "Gin Rummy": "gin_rummy",
    "Solitaire": "solitaire",
}
MODEL_PRESETS = {
    "OpenAI: GPT-4o-mini": {
        "model": "litellm_gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
    },
    "Groq: Llama 3.1 8B Instant": {
        "model": "litellm_groq/llama-3.1-8b-instant",
        "env_var": "GROQ_API_KEY",
    },
    "OpenRouter: Claude 3.5 Sonnet": {
        "model": "openrouter_anthropic/claude-3.5-sonnet",
        "env_var": "OPENROUTER_API_KEY",
    },
    "OpenRouter: Gemini 2.5 Flash": {
        "model": "openrouter_google/gemini-2.5-flash",
        "env_var": "OPENROUTER_API_KEY",
    },
    "Google Gemini: Gemini 2.5 Flash": {
        "model": "litellm_gemini/gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY",
    },
    "Cursor API key (not a playable chat model)": {
        "model": "cursor-api",
        "env_var": "CURSOR_API_KEY",
        "unsupported_reason": (
            "Cursor API keys are for Cursor Cloud Agent/admin endpoints, "
            "not a chat-completions model endpoint. This live game needs a "
            "provider that can answer one prompt per move, such as OpenRouter "
            "or Groq."
        ),
    },
}

# Custom CSS keeps the frontend readable for research sessions without adding a
# separate JavaScript/React application. The class names are attached to Gradio
# components below via `elem_classes`.
APP_CSS = """
:root {
  --ark-bg: #0f172a;
  --ark-panel: #111827;
  --ark-panel-soft: #1f2937;
  --ark-card: #ffffff;
  --ark-text: #0f172a;
  --ark-muted: #64748b;
  --ark-blue: #2563eb;
  --ark-cyan: #06b6d4;
  --ark-green: #16a34a;
  --ark-yellow: #ca8a04;
  --ark-red: #dc2626;
  --ark-border: #dbe3ef;
}

.ark-page {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 32rem),
    linear-gradient(135deg, #f8fbff 0%, #eef4ff 48%, #f8fafc 100%);
}

.ark-hero {
  padding: 1.25rem 1.5rem;
  border-radius: 22px;
  color: white;
  background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 58%, #0891b2 100%);
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.24);
}

.ark-hero h1 {
  margin-bottom: 0.2rem;
  letter-spacing: -0.035em;
}

.ark-sidebar {
  min-width: 260px;
  padding: 1rem;
  border: 1px solid var(--ark-border);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}

.setup-modal {
  padding: 1rem;
  border: 1px solid rgba(37, 99, 235, 0.22);
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 20px 60px rgba(15, 23, 42, 0.18);
}

.setup-modal::before {
  content: "Run setup";
  display: inline-block;
  margin-bottom: 0.75rem;
  padding: 0.25rem 0.65rem;
  border-radius: 999px;
  color: #1d4ed8;
  background: #dbeafe;
  font-size: 0.78rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.board-card, .status-card, .log-card {
  padding: 1rem;
  border: 1px solid var(--ark-border);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
}

.board-card pre {
  font-family: "SFMono-Regular", "Cascadia Code", "Roboto Mono", Menlo, monospace;
  font-size: 0.95rem;
  line-height: 1.35;
  color: #e5f0ff;
  background: #0b1220;
  border-radius: 14px;
  padding: 1rem;
}

.log-card {
  max-height: 620px;
  overflow-y: auto;
}

.status-card {
  font-size: 0.95rem;
}

.status-pill {
  display: inline-block;
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  color: white;
  font-weight: 800;
}

.status-running { background: var(--ark-blue); }
.status-finished { background: var(--ark-green); }
.outcome-win { color: var(--ark-green); font-weight: 800; }
.outcome-loss { color: var(--ark-red); font-weight: 800; }
.outcome-draw { color: var(--ark-yellow); font-weight: 800; }

.primary-run button {
  min-height: 3rem;
  border-radius: 14px !important;
  font-weight: 800 !important;
  letter-spacing: 0.01em;
}

.secondary-action button {
  border-radius: 12px !important;
}
"""


def _resolve_preset(model_preset: str) -> Tuple[str, str, str]:
    """Map a UI model label to backend model name, API env var, and caveat.

    Returns:
        model_name: The string consumed by the existing LLM backend registry.
        env_var: The provider-specific environment variable expected by the
            backend (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.).
        unsupported_reason: Non-empty when the dropdown option exists only to
            explain why it cannot be used as a playable model.
    """
    preset = MODEL_PRESETS.get(model_preset) or MODEL_PRESETS[DEFAULT_PRESET]
    return (
        preset["model"],
        preset["env_var"],
        preset.get("unsupported_reason", ""),
    )


def _sanitize_name(value: str) -> str:
    """Convert a user-provided game set name into a safe file stem.

    The raw game set name remains in the JSON/CSV metadata. This sanitized
    version is only used for filenames so spaces, slashes, and punctuation do
    not create invalid paths.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "unnamed_game_set"


def _write_export_bundle(
    game_set_name: str,
    metadata: Dict[str, Any],
    turn_records: List[Dict[str, Any]],
    round_records: List[Dict[str, Any]],
    transcript: List[str],
) -> str:
    """Write a complete downloadable export for one named game set.

    The export intentionally contains both JSON and CSV:
    - JSON preserves nested metadata, round summaries, transcript text, and
      full turn records without flattening.
    - CSV gives researchers a spreadsheet-friendly turn-level table.

    Returns:
        Absolute path to the ZIP file that Gradio exposes through `gr.File`.
    """
    # Ensure the export directory exists before writing files. The directory is
    # under `results/` so generated artifacts stay with the rest of the run data.
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Add UTC time to avoid overwriting repeated runs with the same set name.
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_name(game_set_name)
    base_path = EXPORT_DIR / f"{safe_name}_{timestamp}"
    json_path = base_path.with_suffix(".json")
    csv_path = base_path.with_suffix(".csv")
    zip_path = base_path.with_suffix(".zip")

    payload = {
        "metadata": metadata,
        "rounds": round_records,
        "turns": turn_records,
        "transcript": transcript,
    }
    # The JSON file is the highest-fidelity artifact. `ensure_ascii=False`
    # preserves card glyphs from OpenSpiel games like Solitaire.
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CSV columns are fixed so downstream scripts can rely on a stable schema.
    # Values not present in a record are written as empty strings.
    fieldnames = [
        "game_set_name",
        "game",
        "game_display_name",
        "model_preset",
        "model_name",
        "round",
        "turn",
        "player_id",
        "agent_type",
        "agent_model",
        "legal_actions",
        "chosen_action",
        "reasoning",
        "outcome",
        "round_status",
        "rewards",
        "state_before",
        "state_after",
        "board_before",
        "board_after",
        "seed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in turn_records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})

    # Bundle both files so the browser exposes a single download per game set.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname=json_path.name)
        zf.write(csv_path, arcname=csv_path.name)

    return str(zip_path)


def _list_previous_game_sets(limit: int = 12) -> str:
    """Return Markdown for the sidebar list of saved game-set exports.

    The sidebar reads ZIP files created by `_write_export_bundle`. It does not
    parse or mutate them; it simply gives users a lightweight audit trail of
    recent saved runs.
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(
        EXPORT_DIR.glob("*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not zip_files:
        return (
            "### Previous game sets\n"
            "No saved exports yet. Run a game set with **Save/export game data** "
            "enabled."
        )

    lines = ["### Previous game sets"]
    for path in zip_files[:limit]:
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )
        size_kb = path.stat().st_size / 1024
        lines.append(f"- `{path.name}`  \n  {modified} UTC-ish, {size_kb:.1f} KB")

    if len(zip_files) > limit:
        lines.append(f"\n_Showing latest {limit} of {len(zip_files)} exports._")

    return "\n".join(lines)


def _prepare_setup_panel_for_run(
    api_key: str,
    game_set_name: str,
    model_preset: str,
) -> Any:
    """Hide setup only when the basic required run inputs are present.

    This keeps the modal-style setup card visible for correctable validation
    errors such as missing game set name or missing API key. The full run still
    performs its own validation in `run_live_match`; this helper only controls
    the UI transition.
    """
    import gradio as gr

    _, api_key_env_var, unsupported_reason = _resolve_preset(model_preset)
    if unsupported_reason:
        return gr.update(visible=True)
    if not (game_set_name or "").strip():
        return gr.update(visible=True)

    load_dotenv()
    pasted_key = (api_key or "").strip()
    if not pasted_key and not os.getenv(api_key_env_var):
        return gr.update(visible=True)

    return gr.update(visible=False)


def _show_setup_panel() -> Any:
    """Show the setup panel so users can configure another named game set."""
    import gradio as gr

    return gr.update(visible=True)


def _build_config(
    game_name: str,
    seed: int,
    model_name: str,
    num_players: int,
) -> Dict[str, Any]:
    """Build the same kind of config dictionary used by `scripts/runner.py`.

    Player 0 is always the model under evaluation. For two-player OpenSpiel
    games, player 1 is the repo's `RandomAgent`. For single-player games like
    OpenSpiel Solitaire, only player 0 is configured.
    """
    agents = {
        "player_0": {
            "type": "llm",
            "model": model_name,
        },
    }
    if num_players > 1:
        agents["player_1"] = {
            "type": "random",
        }

    return {
        "env_config": {
            "game_name": game_name,
            "max_game_rounds": None,
        },
        "num_episodes": 1,
        "seed": seed,
        "use_ray": False,
        "mode": "llm_vs_random",
        "agents": agents,
        "llm_backend": {
            "max_tokens": 250,
            "temperature": 0.1,
            "default_model": model_name,
        },
        "tensorboard_logging": False,
        "run_post_processing": False,
    }


def _agent_metadata(config: Dict[str, Any], player_id: int) -> Tuple[str, str]:
    """Return normalized type/model metadata for logs and exports.

    Non-LLM agents deliberately report model `None`; this prevents leftover
    model defaults from being attached to random-agent output.
    """
    agent_config = config["agents"].get(f"player_{player_id}", {})
    agent_type = agent_config.get("type", "unknown")
    model_name = agent_config.get("model", "None") if agent_type == "llm" else "None"
    return agent_type, model_name


def _opponent_label(config: Dict[str, Any], player_id: int) -> str:
    """Build a compact opponent label for SQLite result rows."""
    labels = []
    for key, agent_config in config["agents"].items():
        if key == f"player_{player_id}":
            continue
        agent_type = agent_config.get("type", "unknown")
        model_name = agent_config.get("model", "None") if agent_type == "llm" else "None"
        labels.append(f"{agent_type}_{model_name.replace('-', '_')}")
    return ", ".join(labels) if labels else "single_player"


def _extract_action_and_reasoning(response: Any) -> Tuple[int, str]:
    """Normalize agent return values into `(action, reasoning)`.

    The repo agents generally return `{"action": int, "reasoning": str}`.
    This fallback also accepts a raw action for compatibility with simpler
    agents.
    """
    if isinstance(response, dict):
        return response.get("action", -1), response.get("reasoning", "None")
    return response, "None"


def _resolve_game(game_preset: str) -> str:
    """Map the UI game label to the internal OpenSpiel registry name."""
    return GAME_PRESETS.get(game_preset) or DEFAULT_GAME


def _render_board(env: Any, player_id: int = 0) -> str:
    """Render the current environment state as a Markdown code block."""
    return f"```text\n{env.render_board(player_id)}\n```"


def _format_status(
    round_number: int,
    total_rounds: int,
    turn: int,
    rewards: Dict[int, float],
    done: bool,
    summary_counts: Dict[str, int],
) -> str:
    """Create the color-coded status panel shown beside the live board."""
    status = "finished" if done else "running"
    status_class = "status-finished" if done else "status-running"
    return (
        f"<span class='status-pill {status_class}'>● {status.upper()}</span>\n\n"
        f"**Round:** {round_number}/{total_rounds}  \n"
        f"**Turn:** {turn}  \n"
        f"**Rewards:** Player 0 = `{rewards.get(0, 0)}`, "
        f"Player 1 = `{rewards.get(1, 'n/a')}`  \n\n"
        "**Player 0 outcomes**  \n"
        f"<span class='outcome-win'>Wins: {summary_counts.get('win', 0)}</span> · "
        f"<span class='outcome-loss'>Losses: {summary_counts.get('loss', 0)}</span> · "
        f"<span class='outcome-draw'>Draws: {summary_counts.get('draw', 0)}</span>"
    )


def _classify_player_outcome(
    rewards: Dict[int, float],
    player_id: int,
    num_players: int,
    truncated: bool,
) -> str:
    """Classify one player's round outcome from OpenSpiel rewards.

    This deliberately simple reward-based classifier is documented because it
    is part of the research method. Future Arkadium-specific game adapters can
    replace it with richer game-specific success criteria.
    """
    if truncated:
        return "loss"

    player_reward = rewards.get(player_id, 0)
    if num_players == 1:
        return "win" if player_reward > 0 else "loss"

    opponent_rewards = [
        reward for pid, reward in rewards.items() if pid != player_id
    ]
    best_opponent_reward = max(opponent_rewards) if opponent_rewards else 0
    if player_reward > best_opponent_reward:
        return "win"
    if player_reward < best_opponent_reward:
        return "loss"
    return "draw"


def run_live_match(
    api_key: str,
    game_set_name: str,
    seed: int,
    game_preset: str,
    model_preset: str,
    rounds: int,
    save_game_data: bool,
    delay_seconds: float,
    max_turns: int,
) -> Generator[Tuple[str, str, str, Any], None, None]:
    """Run one or more game rounds and stream board/log/status updates.

    Gradio supports streaming by consuming yielded tuples from this generator.
    Every `yield` returns the same four outputs in order:
    1. board Markdown
    2. transcript Markdown
    3. status textbox text
    4. optional export ZIP path for the download component
    """
    # A named game set is required because the export filename and metadata use
    # this value to group multiple rounds into a single research unit.
    game_set_name = (game_set_name or "").strip()
    if not game_set_name:
        yield (
            "No board yet.",
            "Name this set of games before running it.",
            "Status: not started",
            None,
        )
        return

    # Gradio numeric widgets can pass numbers as floats; cast them before using
    # them as seeds, loop bounds, or max-turn cutoffs.
    seed = int(seed)
    rounds = max(1, int(rounds))
    max_turns = int(max_turns)
    delay_seconds = float(delay_seconds)

    # Convert UI dropdown labels into internal game/model identifiers.
    game_name = _resolve_game(game_preset)
    model_name, api_key_env_var, unsupported_reason = _resolve_preset(
        model_preset
    )

    # Some dropdown choices are intentionally explanatory rather than playable.
    # Cursor API keys fall into this category.
    if unsupported_reason:
        yield (
            "No board yet.",
            f"{model_preset} cannot run this match.\n\n{unsupported_reason}",
            "Status: unsupported provider",
            None,
        )
        return

    # Load `.env` first, then let a pasted key override the environment for this
    # Python process. The key is never written to SQLite, JSON, CSV, or logs by
    # this app.
    load_dotenv()
    api_key = (api_key or "").strip()
    if api_key:
        os.environ[api_key_env_var] = api_key

    # Stop before importing/running provider code when the selected provider key
    # is absent. This gives users a clear UI error instead of a backend stack.
    if not os.getenv(api_key_env_var):
        yield (
            "No board yet.",
            f"Missing API key for {model_preset}. Paste it in the password "
            f"box or set {api_key_env_var}.",
            "Status: not started",
            None,
        )
        return

    # Import heavy repo components lazily so opening the UI does not require the
    # whole experiment stack until a run actually starts.
    try:
        from game_reasoning_arena.arena.agents.policy_manager import (
            initialize_policies,
        )
        from game_reasoning_arena.arena.games.registry import registry
        from game_reasoning_arena.arena.utils.loggers import SQLiteLogger
        from game_reasoning_arena.arena.utils.seeding import set_seed
        from game_reasoning_arena.backends import initialize_llm_registry
    except ImportError as exc:
        yield (
            "No board yet.",
            "Missing dependency while starting the live app:\n"
            f"{exc}\n\n"
            "Install the UI/runtime dependencies, for example:\n"
            "python3 -m pip install open-spiel litellm gradio python-dotenv",
            "Status: dependency error",
            None,
        )
        return

    # Ask OpenSpiel how many players the selected game has. This determines
    # whether we configure player 1 as RandomAgent or run a single-player task.
    try:
        num_players = registry.get_game_loader(game_name)().num_players()
    except Exception as exc:
        yield (
            "No board yet.",
            f"Failed to load game `{game_name}`:\n{type(exc).__name__}: {exc}",
            "Status: game load error",
            None,
        )
        return

    # Build one config object for the selected game/model. The config is reused
    # across rounds, while the seed changes per round for reproducibility.
    config = _build_config(
        game_name=game_name,
        seed=seed,
        model_name=model_name,
        num_players=num_players,
    )
    transcript: List[str] = [
        "# Arkadium Testing Arena Run",
        "",
        f"- Game set: {game_set_name}",
        f"- Game: {game_preset} (`{game_name}`)",
        f"- Preset: {model_preset}",
        f"- Player 0: LLM `{model_name}`",
        "- Player 1: repo RandomAgent" if num_players > 1 else "- Single-player game",
        f"- Rounds: {rounds}",
        f"- Base seed: {seed}",
        "",
    ]

    # Initialize global seeding and model registry once per game set. Actual
    # agent instances and environments are recreated each round below.
    try:
        set_seed(seed)
        initialize_llm_registry()
    except Exception as exc:
        yield (
            "No board yet.",
            "Failed to initialize the match:\n"
            f"{type(exc).__name__}: {exc}",
            "Status: initialization error",
            None,
        )
        return

    # Create one SQLite logger per player. The existing repository logger writes
    # to `results/<agent_type>_<model>.db`.
    loggers = {}
    for player_id in range(num_players):
        agent_type, agent_model = _agent_metadata(config, player_id)
        sanitized_model = agent_model.replace("-", "_").replace("/", "_")
        loggers[player_id] = SQLiteLogger(agent_type, sanitized_model)

    # These in-memory structures feed the live transcript and downloadable
    # JSON/CSV export. SQLite logging still happens independently.
    summary_counts = {"win": 0, "loss": 0, "draw": 0}
    round_summaries: List[str] = []
    round_records: List[Dict[str, Any]] = []
    turn_records: List[Dict[str, Any]] = []
    last_board = "No board yet."
    export_path = None
    metadata = {
        "game_set_name": game_set_name,
        "game": game_name,
        "game_display_name": game_preset,
        "model_preset": model_preset,
        "model_name": model_name,
        "api_key_env_var": api_key_env_var,
        "rounds_requested": rounds,
        "base_seed": seed,
        "max_turns_per_round": max_turns,
        "num_players": num_players,
        "started_at_utc": datetime.utcnow().isoformat(),
    }

    # Main experiment loop: one iteration equals one complete game episode.
    for round_index in range(rounds):
        round_number = round_index + 1
        round_seed = seed + round_index

        # Recreate policies and environment each round so episodes are isolated.
        # The deterministic seed progression lets researchers rerun the same
        # named set with the same seeds.
        try:
            set_seed(round_seed)
            policies = initialize_policies(config, game_name, round_seed)
            player_to_agent = {
                player_id: policy
                for player_id, policy in enumerate(policies.values())
            }
            env = registry.make_env(game_name, config)
            observations, _ = env.reset(seed=round_seed)
        except Exception as exc:
            yield (
                last_board,
                "\n".join(transcript)
                + "\n\n"
                + f"Failed to initialize round {round_number}:\n"
                + f"`{type(exc).__name__}: {exc}`",
                "Status: round initialization error",
                export_path,
            )
            return

        # Rewards are tracked separately from OpenSpiel step rewards so the UI
        # can display the latest known reward values at every yield.
        rewards = {player_id: 0.0 for player_id in range(num_players)}
        terminated = truncated = False
        turn = 0

        # Yield the starting state before any model call. This makes the browser
        # visibly update as soon as a round begins.
        transcript.append(f"\n# Round {round_number}/{rounds}")
        transcript.append(f"Seed: `{round_seed}`")
        transcript.append("Initial board:")
        last_board = _render_board(env)
        yield (
            last_board,
            "\n".join(transcript),
            _format_status(
                round_number, rounds, turn, rewards, False, summary_counts
            ),
            export_path,
        )

        # Inner turn loop: ask the active player for one action, validate it,
        # apply it to OpenSpiel, stream the update, and repeat.
        while not (terminated or truncated):
            # A max-turn cutoff protects research runs from hanging on games
            # with very long or pathological trajectories.
            if turn >= max_turns:
                truncated = True
                transcript.append(f"\nStopped after max_turns={max_turns}.")
                break

            # This framework currently handles turn-based OpenSpiel games. The
            # active player is the only player with an observation/action here.
            current_player = env.state.current_player()
            observation = observations[current_player]
            legal_actions = observation["legal_actions"]
            agent = player_to_agent[current_player]

            # Capture pre-action state for export before the environment mutates.
            state_before = observation.get("state_string", "")
            board_before = env.render_board(0)

            transcript.append(f"\n## Round {round_number}, Turn {turn}: Player {current_player}")
            transcript.append(f"Legal actions: `{legal_actions}`")

            # Agent invocation is the only place where a model API call happens
            # for LLM players. RandomAgent returns immediately without API use.
            try:
                response = agent(observation)
                action, reasoning = _extract_action_and_reasoning(response)
            except Exception as exc:
                transcript.append(
                    "Action generation failed:\n"
                    f"`{type(exc).__name__}: {exc}`"
                )
                yield (
                    _render_board(env),
                    "\n".join(transcript),
                    "Status: action generation error",
                    export_path,
                )
                return

            # Add the action and reasoning trace to the live transcript before
            # applying the move, so failed/illegal choices remain visible.
            agent_type, agent_model = _agent_metadata(config, current_player)
            transcript.append(f"Chosen action: `{action}`")
            if agent_type == "llm":
                transcript.append("Reasoning trace:")
                transcript.append(f"> {reasoning}")

            # Illegal moves are logged and treated as terminal failures for the
            # run. This is one of the failure categories the project wants to
            # measure explicitly.
            if action not in legal_actions:
                loggers[current_player].log_illegal_move(
                    game_name=game_name,
                    episode=round_number,
                    turn=turn,
                    agent_id=current_player,
                    illegal_action=action,
                    reason="Illegal action",
                    board_state=observation["state_string"],
                )
                transcript.append(f"Illegal move detected: `{action}`")
                yield (
                    _render_board(env),
                    "\n".join(transcript),
                    "Status: illegal move",
                    export_path,
                )
                return

            # Persist the move through the repository's existing SQLite schema.
            # The richer JSON/CSV export is built separately below.
            loggers[current_player].log_move(
                game_name=game_name,
                episode=round_number,
                turn=turn,
                action=action,
                reasoning=reasoning,
                opponent=_opponent_label(config, current_player),
                generation_time=0.0,
                agent_type=agent_type,
                agent_model=agent_model,
                seed=round_seed,
                board_state=observation["state_string"],
            )

            # Apply the action to OpenSpiel. The environment returns next
            # observations, current rewards, and termination/truncation flags.
            observations, step_rewards, terminated, truncated, _ = env.step(
                {current_player: action}
            )
            rewards.update(step_rewards)

            # Capture post-action state for export. Terminal states do not have
            # a next active-player observation, so `state_after` is blank there.
            state_after = "" if terminated or truncated else observations[
                env.state.current_player()
            ].get("state_string", "")
            board_after = env.render_board(0)

            # Store a full turn record for the downloadable export. The outcome
            # fields are filled in after the round ends and rewards are final.
            turn_records.append({
                "game_set_name": game_set_name,
                "game": game_name,
                "game_display_name": game_preset,
                "model_preset": model_preset,
                "model_name": model_name,
                "round": round_number,
                "turn": turn,
                "player_id": current_player,
                "agent_type": agent_type,
                "agent_model": agent_model,
                "legal_actions": json.dumps(legal_actions),
                "chosen_action": action,
                "reasoning": reasoning,
                "outcome": "",
                "round_status": "",
                "rewards": json.dumps(rewards),
                "state_before": state_before,
                "state_after": state_after,
                "board_before": board_before,
                "board_after": board_after,
                "seed": round_seed,
            })
            turn += 1
            last_board = _render_board(env)

            # Stream the latest board/transcript/status back to Gradio.
            yield (
                last_board,
                "\n".join(transcript),
                _format_status(
                    round_number,
                    rounds,
                    turn,
                    rewards,
                    terminated or truncated,
                    summary_counts,
                ),
                export_path,
            )

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        # Once the round exits, classify the model player's result and record a
        # compact round summary for both the UI and the export.
        final_status = "truncated" if truncated else "terminated"
        player0_outcome = _classify_player_outcome(
            rewards, player_id=0, num_players=num_players, truncated=truncated
        )
        summary_counts[player0_outcome] += 1
        round_summary = (
            f"Round {round_number}: {player0_outcome.upper()} "
            f"(status={final_status}, rewards={rewards})"
        )
        round_summaries.append(round_summary)
        round_records.append({
            "game_set_name": game_set_name,
            "game": game_name,
            "game_display_name": game_preset,
            "round": round_number,
            "seed": round_seed,
            "status": final_status,
            "player0_outcome": player0_outcome,
            "rewards": rewards.copy(),
            "turns": turn,
        })

        # Attach round-level outcome/status to each turn from this round. This
        # makes per-turn CSV analysis possible without joining separate tables.
        for record in turn_records:
            if record["round"] == round_number:
                record["outcome"] = player0_outcome
                record["round_status"] = final_status
        transcript.append(f"\n## {round_summary}")

        # Persist final rewards/results to SQLite for every player. The status
        # suffix (`_win`, `_loss`, `_draw`) makes round outcomes queryable.
        for player_id, reward in rewards.items():
            player_outcome = _classify_player_outcome(
                rewards,
                player_id=player_id,
                num_players=num_players,
                truncated=truncated,
            )
            loggers[player_id].log_rewards(game_name, round_number, reward)
            loggers[player_id].log_game_result(
                game_name=game_name,
                episode=round_number,
                status=f"{final_status}_{player_outcome}",
                reward=reward,
                opponent=_opponent_label(config, player_id),
            )

        # Stream the end-of-round summary before starting the next round.
        transcript.append("\nRound summaries so far:")
        transcript.extend(f"- {summary}" for summary in round_summaries)
        yield (
            last_board,
            "\n".join(transcript),
            _format_status(
                round_number, rounds, turn, rewards, True, summary_counts
            ),
            export_path,
        )

    # After all requested rounds finish, optionally produce the downloadable
    # research artifact bundle.
    transcript.append("\n# Final summary")
    transcript.append(f"Player 0 outcomes: `{summary_counts}`")
    if save_game_data:
        metadata["completed_at_utc"] = datetime.utcnow().isoformat()
        metadata["summary_counts"] = summary_counts.copy()
        export_path = _write_export_bundle(
            game_set_name=game_set_name,
            metadata=metadata,
            turn_records=turn_records,
            round_records=round_records,
            transcript=transcript,
        )
        transcript.append(f"Download export: `{export_path}`")
    else:
        transcript.append("Export saving was disabled for this run.")

    # Final yield exposes the ZIP path to Gradio's file download component.
    yield (
        last_board,
        "\n".join(transcript),
        _format_status(
            rounds,
            rounds,
            0,
            {0: 0, 1: "done" if num_players > 1 else "n/a"},
            True,
            summary_counts,
        ),
        export_path,
    )


def build_app() -> Any:
    """Construct the Gradio interface for the local research app.

    The UI code is kept thin on purpose: widgets collect parameters, and the
    `run_live_match` generator above performs the experiment. This separation
    makes the research control flow easier to audit.
    """
    # Import Gradio lazily so this module can still be imported for tests or
    # export helpers in environments that have not installed UI dependencies.
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required for the live frontend. Install it with:\n"
            "python3 -m pip install gradio"
        ) from exc

    # `Blocks` is Gradio's simple layout API. It avoids a custom JavaScript
    # frontend while still giving us dropdowns, live streaming, and downloads.
    with gr.Blocks(
        title="Arkadium Testing Arena",
        elem_classes=["ark-page"],
    ) as demo:
        # Inject CSS inside the document instead of relying on constructor
        # parameters that differ across Gradio versions.
        gr.HTML(f"<style>{APP_CSS}</style>")

        # Header text defines the expected privacy boundary for API keys and
        # reminds users that keys are not included in research artifacts.
        gr.Markdown(
            "# Arkadium Testing Arena\n"
            "Transparent multi-game LLM evaluation for game-based reasoning "
            "research. Name a set of games, configure the run, then watch the "
            "board, model reasoning, outcomes, and export artifact update live.",
            elem_classes=["ark-hero"],
        )

        with gr.Row():
            # Sidebar: persistent run-management area. It remains visible while
            # the setup panel hides, so users can find prior saved game sets.
            with gr.Column(scale=1, min_width=260, elem_classes=["ark-sidebar"]):
                previous_sets = gr.Markdown(value=_list_previous_game_sets())
                refresh_sets = gr.Button(
                    "Refresh saved sets",
                    elem_classes=["secondary-action"],
                )
                new_game_set = gr.Button(
                    "New game set",
                    elem_classes=["secondary-action"],
                )

            with gr.Column(scale=4):
                # Modal-style setup panel. It is visible on load and hidden as
                # soon as a run starts, approximating a modal flow while staying
                # compatible with Gradio's Python-only UI model.
                with gr.Column(visible=True, elem_classes=["setup-modal"]) as setup_panel:
                    # This name groups multiple rounds into one named research
                    # run. It becomes part of export filenames and metadata.
                    game_set_name = gr.Textbox(
                        label="Game set name",
                        placeholder="Example: connect4_gemini_10_rounds_trial_1",
                    )

                    # Provider/model and game choices are dropdowns to reduce
                    # typo-driven failures and make the condition explicit.
                    with gr.Row():
                        api_key = gr.Textbox(
                            label="Provider API key",
                            type="password",
                            placeholder="OpenAI, Groq, OpenRouter, or Google Gemini key",
                        )
                        model_preset = gr.Dropdown(
                            label="Model preset",
                            choices=list(MODEL_PRESETS.keys()),
                            value=DEFAULT_PRESET,
                        )
                        game_preset = gr.Dropdown(
                            label="Game",
                            choices=list(GAME_PRESETS.keys()),
                            value="Connect Four",
                        )

                    # Reproducibility and runtime controls. Rounds are separate
                    # OpenSpiel episodes; max turns is a safety cutoff.
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=42, precision=0)
                        rounds = gr.Number(label="Rounds", value=1, precision=0)
                        save_game_data = gr.Checkbox(
                            label="Save/export game data",
                            value=True,
                        )
                        max_turns = gr.Number(
                            label="Max turns per round",
                            value=80,
                            precision=0,
                        )

                    delay_seconds = gr.Slider(
                        label="Delay between turns",
                        minimum=0.0,
                        maximum=5.0,
                        value=0.5,
                        step=0.25,
                    )

                    run_button = gr.Button(
                        "Run selected model vs Random",
                        variant="primary",
                        elem_classes=["primary-run"],
                    )

                # Live outputs are segmented into distinct cards so board state,
                # run status, and logs are visually separate.
                with gr.Row():
                    board = gr.Markdown(
                        value="### Board\nConfigure a game set and click **Run**.",
                        label="Board",
                        elem_classes=["board-card"],
                    )
                    status = gr.Markdown(
                        value="<span class='status-pill status-running'>● READY</span>",
                        label="Status",
                        elem_classes=["status-card"],
                    )

                transcript = gr.Markdown(
                    value=(
                        "### Round log and reasoning trace\n"
                        "The setup panel will disappear when a run starts. "
                        "Click **New game set** in the sidebar to configure "
                        "another run."
                    ),
                    label="Turn log and reasoning trace",
                    elem_classes=["log-card"],
                )

                # This remains empty until `run_live_match` writes an export ZIP
                # at the end of a saved run.
                export_file = gr.File(label="Download game set export")

        # Sidebar actions are intentionally independent of the run button.
        refresh_sets.click(fn=_list_previous_game_sets, outputs=previous_sets)
        new_game_set.click(fn=_show_setup_panel, outputs=setup_panel)

        # First decide whether the setup panel should stay visible or disappear,
        # then run the streaming generator, then refresh the saved-set sidebar
        # after the generator completes.
        run_event = run_button.click(
            fn=_prepare_setup_panel_for_run,
            inputs=[api_key, game_set_name, model_preset],
            outputs=setup_panel,
        )
        run_event.then(
            fn=run_live_match,
            inputs=[
                api_key,
                game_set_name,
                seed,
                game_preset,
                model_preset,
                rounds,
                save_game_data,
                delay_seconds,
                max_turns,
            ],
            outputs=[board, transcript, status, export_file],
        ).then(fn=_list_previous_game_sets, outputs=previous_sets)

    return demo


if __name__ == "__main__":
    # Bind to all interfaces for compatibility with local VS Code, Codespaces,
    # and cloud/remote development. Users should open the URL Gradio prints.
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
