#!/usr/bin/env python3
"""Minimal frontend for the original Game Reasoning Arena workflow.

This frontend deliberately follows the methodology of the upstream
Game Reasoning Arena repository:

- games come from the OpenSpiel-backed registry
- agents come from the repository policy manager
- model names use the existing backend prefixes (`litellm_`, `openrouter_`,
  `hf_`, `vllm_`)
- moves, reasoning traces, rewards, illegal moves, and game results are written
  through `SQLiteLogger`
- CSV/JSON exports are derived from the same per-move/per-result fields used by
  the repository's logging and analysis workflow

Run from the repository root:

    python3 game_reasoning_arena_frontend.py
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


MAX_LLM_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = [2, 5]


# ---------------------------------------------------------------------------
# Repository path setup
# ---------------------------------------------------------------------------
#
# The original README assumes users run scripts from the repository root after
# `pip install -e .`. In student/local VS Code environments that editable
# install is easy to miss, so the frontend adds `src/` to Python's import path.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
EXPORT_DIR = ROOT_DIR / "results" / "frontend_exports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


# ---------------------------------------------------------------------------
# UI choices that mirror the upstream README/model naming conventions
# ---------------------------------------------------------------------------

GAME_CHOICES = {
    "Tic-Tac-Toe": "tic_tac_toe",
    "Connect Four": "connect_four",
    "Kuhn Poker": "kuhn_poker",
    "Matrix Prisoner's Dilemma": "matrix_pd",
    "Matching Pennies": "matching_pennies",
    "Matrix Rock-Paper-Scissors": "matrix_rps",
    "Hex": "hex",
}

MODEL_CHOICES = {
    # Remote/API models.
    "Remote / OpenAI GPT-4o-mini": {
        "model": "litellm_gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
    },
    "Remote / Groq Llama 3.1 8B Instant": {
        "model": "litellm_groq/llama-3.1-8b-instant",
        "env_var": "GROQ_API_KEY",
    },
    "Remote / OpenRouter Claude 3.5 Sonnet": {
        "model": "openrouter_anthropic/claude-3.5-sonnet",
        "env_var": "OPENROUTER_API_KEY",
    },
    "Remote / OpenRouter Gemini 2.5 Flash": {
        "model": "openrouter_google/gemini-2.5-flash",
        "env_var": "OPENROUTER_API_KEY",
    },
    "Remote / OpenRouter Grok 4": {
        "model": "openrouter_x-ai/grok-4",
        "env_var": "OPENROUTER_API_KEY",
    },
    "Remote / Google Gemini 2.5 Flash": {
        "model": "litellm_gemini/gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY",
    },
    # Local models. These do not need hosted API keys, but they do need local
    # runtime dependencies/models.
    "Local / HuggingFace distilgpt2": {
        "model": "hf_distilgpt2",
        "env_var": "",
    },
    "Local / HuggingFace FLAN-T5 Small": {
        "model": "hf_google/flan-t5-small",
        "env_var": "",
    },
    "Local / vLLM Qwen2-7B-Instruct": {
        "model": "vllm_Qwen2-7B-Instruct",
        "env_var": "",
    },
}

AGENT_TYPES = ["llm", "strong_bot", "openspiel_bot", "random"]

EVAL_GAME_CHOICES = {
    "Tic-Tac-Toe": "tic_tac_toe",
    "Connect Four": "connect_four",
}

FAILURE_MECHANISMS = [
    "Visual Processing",
    "Physical Reasoning",
    "Social Reasoning",
    "World Model Learning",
    "Memory",
    "Spatial Temporal",
    "Long Horizon Planning",
    "Strategic Decision Making",
]

APP_CSS = """
body, .gradio-container {
  background: #ffffff !important;
  color: #111111 !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif !important;
}
.gra-hero {
  border: 1px solid #111111;
  padding: 1.25rem 1.5rem;
  background: #ffffff;
}
.gra-hero h1 {
  color: #000000;
  letter-spacing: -0.04em;
  margin-bottom: 0.25rem;
}
.gra-panel {
  border: 1px solid #111111;
  padding: 1rem;
  background: #ffffff;
}
.gra-board pre {
  background: #000000;
  color: #ffffff;
  padding: 1rem;
  border-radius: 0;
  font-family: "SFMono-Regular", "Cascadia Code", "Roboto Mono", Menlo, monospace;
  line-height: 1.35;
}
.board-title {
  margin: 0 0 0.75rem 0;
  font-size: 1.05rem;
  font-weight: 800;
}
.ttt-board {
  display: grid;
  grid-template-columns: repeat(3, minmax(64px, 88px));
  gap: 0;
  width: max-content;
  border: 2px solid #000;
  background: #000;
}
.ttt-cell {
  width: 82px;
  height: 82px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #fff;
  border: 1px solid #000;
  font-size: 2.1rem;
  font-weight: 900;
}
.connect4-board {
  display: grid;
  grid-template-columns: repeat(7, 46px);
  gap: 7px;
  width: max-content;
  padding: 12px;
  border: 2px solid #000;
  background: #f7f7f7;
}
.connect4-cell {
  width: 46px;
  height: 46px;
  border: 2px solid #000;
  border-radius: 50%;
  background: #fff;
}
.piece-x {
  background: #111;
}
.piece-o {
  background: #fff;
  box-shadow: inset 0 0 0 9px #d1d5db;
}
.column-labels {
  display: grid;
  grid-template-columns: repeat(7, 46px);
  gap: 7px;
  width: max-content;
  margin: 0 0 0.35rem 14px;
  font-size: 0.75rem;
  font-weight: 800;
  text-align: center;
}
.gra-log {
  max-height: 560px;
  overflow-y: auto;
}
.gra-button button {
  background: #000000 !important;
  color: #ffffff !important;
  border: 2px solid #000000 !important;
  border-radius: 0 !important;
  font-weight: 800 !important;
}
.gra-secondary button {
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid #111111 !important;
  border-radius: 0 !important;
}
"""


# ---------------------------------------------------------------------------
# Small helpers for model/game mapping and logging metadata
# ---------------------------------------------------------------------------

def _resolve_game(label: str) -> str:
    return GAME_CHOICES.get(label, "tic_tac_toe")


def _resolve_model(label: str) -> Tuple[str, str]:
    preset = MODEL_CHOICES.get(label) or MODEL_CHOICES["Remote / OpenAI GPT-4o-mini"]
    return preset["model"], preset["env_var"]


def _agent_config(agent_type: str, model_label: str) -> Dict[str, str]:
    if agent_type == "llm":
        model_name, _ = _resolve_model(model_label)
        return {"type": "llm", "model": model_name}
    if agent_type == "strong_bot":
        return {"type": "strong_bot", "model": "search_heuristic"}
    if agent_type == "openspiel_bot":
        return {"type": "openspiel_bot", "model": "uniform_random"}
    return {"type": "random"}


def _agent_metadata(config: Dict[str, Any], player_id: int) -> Tuple[str, str]:
    agent = config["agents"].get(f"player_{player_id}", {"type": "unknown"})
    agent_type = agent.get("type", "unknown")
    model = agent.get("model", "None") if agent_type in {"llm", "strong_bot", "openspiel_bot"} else "None"
    return agent_type, model


def _opponent_label(config: Dict[str, Any], player_id: int) -> str:
    labels = []
    for key, agent in config["agents"].items():
        if key == f"player_{player_id}":
            continue
        agent_type = agent.get("type", "unknown")
        model = agent.get("model", "None") if agent_type in {"llm", "strong_bot", "openspiel_bot"} else "None"
        labels.append(f"{agent_type}_{model.replace('-', '_')}")
    return ", ".join(labels) if labels else "single_player"


def _extract_action_and_reasoning(response: Any) -> Tuple[int, str]:
    if isinstance(response, dict):
        return response.get("action", -1), response.get("reasoning", "None")
    return int(response), "None"


def _symbols_from_observation(env: Any, player_id: int, expected_count: int) -> List[str]:
    """Extract x/o/dot board symbols from an OpenSpiel observation string."""
    raw = env.state.observation_string(player_id)
    symbols = [char for char in raw if char in ("x", "o", ".")]
    return symbols if len(symbols) == expected_count else []


def _render_tic_tac_toe_html(env: Any, player_id: int) -> str:
    """Render Tic-Tac-Toe as a 3x3 visual grid with position labels."""
    symbols = _symbols_from_observation(env, player_id, 9)
    if not symbols:
        return ""
    cells = []
    for idx, symbol in enumerate(symbols):
        display = "" if symbol == "." else symbol.upper()
        hint = idx if symbol == "." else ""
        cells.append(
            "<div class='ttt-cell'>"
            f"<span>{display}</span><small>{hint}</small>"
            "</div>"
        )
    return (
        "<div class='board-title'>Tic-Tac-Toe board</div>"
        "<div class='ttt-board'>"
        + "".join(cells)
        + "</div>"
        "<p><strong>Empty cell labels show legal board indices.</strong></p>"
    )


def _render_connect_four_html(env: Any, player_id: int) -> str:
    """Render Connect Four as a 6x7 disc board with column labels."""
    symbols = _symbols_from_observation(env, player_id, 42)
    if not symbols:
        return ""
    labels = "".join(f"<div>{idx}</div>" for idx in range(7))
    cells = []
    for symbol in symbols:
        piece_class = "piece-x" if symbol == "x" else "piece-o" if symbol == "o" else ""
        cells.append(f"<div class='connect4-cell {piece_class}'></div>")
    return (
        "<div class='board-title'>Connect Four board</div>"
        f"<div class='column-labels'>{labels}</div>"
        "<div class='connect4-board'>"
        + "".join(cells)
        + "</div>"
        "<p><strong>Column labels are action numbers.</strong></p>"
    )


def _render_board(env: Any, player_id: int = 0) -> str:
    game_name = getattr(env, "game_name", "")
    if game_name == "tic_tac_toe":
        html = _render_tic_tac_toe_html(env, player_id)
        if html:
            return html
    if game_name == "connect_four":
        html = _render_connect_four_html(env, player_id)
        if html:
            return html
    try:
        rendered = env.render_board(player_id)
    except Exception:
        rendered = str(env.state)
    return f"```text\n{rendered}\n```"


def _status_text(
    episode: int,
    num_episodes: int,
    turn: int,
    rewards: Dict[int, float],
    complete: bool,
) -> str:
    state = "COMPLETE" if complete else "RUNNING"
    return (
        f"**Status:** {state}\n\n"
        f"**Episode:** {episode}/{num_episodes}  \n"
        f"**Turn:** {turn}  \n"
        f"**Rewards:** `{rewards}`"
    )


def _build_config(
    game_name: str,
    seed: int,
    num_players: int,
    player0_type: str,
    player0_model_label: str,
    player1_type: str,
    player1_model_label: str,
) -> Dict[str, Any]:
    agents = {
        "player_0": _agent_config(player0_type, player0_model_label),
    }
    if num_players > 1:
        agents["player_1"] = _agent_config(player1_type, player1_model_label)

    return {
        "env_config": {"game_name": game_name, "max_game_rounds": None},
        "num_episodes": 1,
        "seed": seed,
        "use_ray": False,
        "mode": "frontend_live",
        "agents": agents,
        "llm_backend": {
            "max_tokens": 250,
            "temperature": 0.1,
            "default_model": agents["player_0"].get("model", "None"),
        },
        "tensorboard_logging": False,
        "run_post_processing": False,
    }


def _safe_clone_apply(state: Any, action: int) -> Any:
    """Clone an OpenSpiel state and apply one action."""
    next_state = state.clone()
    next_state.apply_action(action)
    return next_state


def _connect_four_window_score(
    window: List[str],
    player_symbol: str,
    opponent_symbol: str,
) -> int:
    """Score one four-cell Connect Four window for a simple heuristic."""
    player_count = window.count(player_symbol)
    opponent_count = window.count(opponent_symbol)
    empty_count = window.count(".")

    if player_count == 4:
        return 100000
    if opponent_count == 4:
        return -100000
    if player_count == 3 and empty_count == 1:
        return 100
    if player_count == 2 and empty_count == 2:
        return 10
    if opponent_count == 3 and empty_count == 1:
        return -120
    if opponent_count == 2 and empty_count == 2:
        return -8
    return 0


def _connect_four_heuristic(state: Any, player_id: int) -> float:
    """Evaluate non-terminal Connect Four states from one player's view.

    OpenSpiel already provides exact rewards at terminal states. For depth
    cutoffs, this heuristic prefers center control and potential four-in-a-row
    windows while penalizing opponent threats.
    """
    if state.is_terminal():
        return state.player_reward(player_id) * 100000

    raw = state.observation_string(player_id)
    cells = [char for char in raw if char in ("x", "o", ".")]
    if len(cells) != 42:
        return 0

    board = [cells[i:i + 7] for i in range(0, 42, 7)]
    player_symbol = "x" if player_id == 0 else "o"
    opponent_symbol = "o" if player_id == 0 else "x"
    score = 0

    center_column = [row[3] for row in board]
    score += center_column.count(player_symbol) * 6
    score -= center_column.count(opponent_symbol) * 6

    # Horizontal windows.
    for row in range(6):
        for col in range(4):
            score += _connect_four_window_score(
                board[row][col:col + 4],
                player_symbol,
                opponent_symbol,
            )

    # Vertical windows.
    for col in range(7):
        for row in range(3):
            window = [board[row + i][col] for i in range(4)]
            score += _connect_four_window_score(
                window,
                player_symbol,
                opponent_symbol,
            )

    # Down-right diagonal windows.
    for row in range(3):
        for col in range(4):
            window = [board[row + i][col + i] for i in range(4)]
            score += _connect_four_window_score(
                window,
                player_symbol,
                opponent_symbol,
            )

    # Up-right diagonal windows.
    for row in range(3, 6):
        for col in range(4):
            window = [board[row - i][col + i] for i in range(4)]
            score += _connect_four_window_score(
                window,
                player_symbol,
                opponent_symbol,
            )

    return score


def _state_heuristic(game_name: str, state: Any, player_id: int) -> float:
    """Dispatch game-specific heuristic values for search cutoffs."""
    if state.is_terminal():
        return state.player_reward(player_id) * 100000
    if game_name == "connect_four":
        return _connect_four_heuristic(state, player_id)
    return 0


def _alpha_beta_value(
    game_name: str,
    state: Any,
    root_player: int,
    depth: int,
    alpha: float,
    beta: float,
) -> float:
    """Minimax/alpha-beta value for deterministic turn-based OpenSpiel games."""
    if state.is_terminal() or depth <= 0:
        return _state_heuristic(game_name, state, root_player)

    current_player = state.current_player()
    if current_player < 0:
        return _state_heuristic(game_name, state, root_player)

    legal_actions = state.legal_actions(current_player)
    if not legal_actions:
        return _state_heuristic(game_name, state, root_player)

    maximizing = current_player == root_player
    if maximizing:
        value = float("-inf")
        for action in legal_actions:
            child = _safe_clone_apply(state, action)
            value = max(
                value,
                _alpha_beta_value(
                    game_name,
                    child,
                    root_player,
                    depth - 1,
                    alpha,
                    beta,
                ),
            )
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = float("inf")
    for action in legal_actions:
        child = _safe_clone_apply(state, action)
        value = min(
            value,
            _alpha_beta_value(
                game_name,
                child,
                root_player,
                depth - 1,
                alpha,
                beta,
            ),
        )
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def _find_immediate_tactical_action(
    state: Any,
    player_id: int,
    legal_actions: List[int],
) -> Optional[int]:
    """Return a winning move, or a blocking move against opponent win."""
    # First, win immediately if possible.
    for action in legal_actions:
        child = _safe_clone_apply(state, action)
        if child.is_terminal() and child.player_reward(player_id) > 0:
            return action

    # Then, block any opponent action that would immediately win.
    opponent = 1 - player_id
    for opponent_action in state.legal_actions(opponent):
        child = _safe_clone_apply(state, opponent_action)
        if child.is_terminal() and child.player_reward(opponent) > 0:
            return opponent_action if opponent_action in legal_actions else None

    return None


def _strong_bot_action(
    game_name: str,
    state: Any,
    player_id: int,
    legal_actions: List[int],
) -> Tuple[int, str]:
    """Choose a stronger game-specific action for Tic-Tac-Toe/Connect Four."""
    if not legal_actions:
        return 0, "No legal actions were available."

    if game_name not in {"tic_tac_toe", "connect_four"}:
        return (
            legal_actions[0],
            "Strong bot is only implemented for Tic-Tac-Toe and Connect Four; "
            "used the first legal action as a deterministic fallback.",
        )

    tactical_action = _find_immediate_tactical_action(
        state,
        player_id,
        legal_actions,
    )
    if tactical_action is not None:
        return tactical_action, "Strong bot found an immediate win or block."

    if game_name == "tic_tac_toe":
        search_depth = 9
    else:
        search_depth = 5

    preferred_order = legal_actions
    if game_name == "connect_four":
        center_order = [3, 2, 4, 1, 5, 0, 6]
        preferred_order = [a for a in center_order if a in legal_actions]

    best_action = preferred_order[0]
    best_value = float("-inf")
    for action in preferred_order:
        child = _safe_clone_apply(state, action)
        value = _alpha_beta_value(
            game_name,
            child,
            player_id,
            search_depth - 1,
            float("-inf"),
            float("inf"),
        )
        if value > best_value:
            best_value = value
            best_action = action

    return (
        best_action,
        f"Strong bot used alpha-beta search depth {search_depth} "
        f"and selected action {best_action} with value {best_value:.1f}.",
    )


def _initialize_frontend_agents(
    config: Dict[str, Any],
    game_name: str,
    seed: int,
    num_players: int,
) -> Dict[int, Any]:
    """Create player agents for the live frontend.

    This mirrors the repository policy mapping for `llm` and `random`, while
    adding an explicit OpenSpiel bot path. OpenSpiel bots need access to the
    current OpenSpiel state (`env.state`), so the live loop handles them
    directly with `bot.step(env.state)`.
    """
    import pyspiel
    from game_reasoning_arena.arena.agents.llm_agent import LLMAgent
    from game_reasoning_arena.arena.agents.random_agent import RandomAgent

    player_to_agent = {}
    for player_id in range(num_players):
        agent_config = config["agents"][f"player_{player_id}"]
        agent_type = agent_config["type"]
        if agent_type == "llm":
            player_to_agent[player_id] = LLMAgent(
                model_name=agent_config["model"],
                game_name=game_name,
            )
        elif agent_type == "openspiel_bot":
            player_to_agent[player_id] = pyspiel.make_uniform_random_bot(
                player_id,
                seed,
            )
        elif agent_type == "strong_bot":
            player_to_agent[player_id] = None
        elif agent_type == "random":
            player_to_agent[player_id] = RandomAgent(seed=seed)
        else:
            raise ValueError(f"Unsupported frontend agent type: {agent_type}")

    return player_to_agent


def _required_env_vars_for_config(config: Dict[str, Any]) -> List[str]:
    """Return provider API-key env vars needed by selected LLM agents."""
    required = []
    for agent in config["agents"].values():
        if agent.get("type") != "llm":
            continue

        model_name = agent.get("model", "")
        for preset in MODEL_CHOICES.values():
            if preset["model"] == model_name and preset["env_var"]:
                required.append(preset["env_var"])
                break

    return sorted(set(required))


def _set_api_key_for_config(api_key: str, config: Dict[str, Any]) -> None:
    """Apply one pasted key to the provider env vars required by this run."""
    load_dotenv()
    pasted_key = (api_key or "").strip()
    if not pasted_key:
        return

    for env_var in _required_env_vars_for_config(config):
        os.environ[env_var] = pasted_key


def _key_shape_warning(env_var: str, api_key: str) -> Optional[str]:
    """Return a warning when a pasted key does not resemble its provider."""
    key = (api_key or "").strip()
    if not key:
        return None
    if env_var == "OPENROUTER_API_KEY" and not key.startswith("sk-or-"):
        return (
            "The selected model uses OpenRouter, but the pasted key does not "
            "look like an OpenRouter key. OpenRouter keys usually start with "
            "`sk-or-`. If you have a Google Gemini key, select "
            "`Remote / Google Gemini 2.5 Flash` instead."
        )
    if env_var == "GEMINI_API_KEY" and key.startswith("sk-or-"):
        return (
            "The selected model uses Google Gemini directly, but the pasted key "
            "looks like an OpenRouter key. Select an OpenRouter model preset "
            "instead, or paste a Google AI Studio/Gemini key."
        )
    if env_var == "GROQ_API_KEY" and not key.startswith("gsk_"):
        return (
            "The selected model uses Groq, but the pasted key does not look "
            "like a Groq key. Groq keys usually start with `gsk_`."
        )
    return None


def _validate_pasted_key_shape(config: Dict[str, Any], api_key: str) -> Optional[str]:
    """Validate the single pasted key against selected LLM provider(s)."""
    pasted_key = (api_key or "").strip()
    if not pasted_key:
        return None
    warnings = [
        warning
        for env_var in _required_env_vars_for_config(config)
        if (warning := _key_shape_warning(env_var, pasted_key))
    ]
    if warnings:
        return "\n".join(warnings)
    return None


def _validate_keys_for_agents(config: Dict[str, Any]) -> Optional[str]:
    missing = [
        env_var
        for env_var in _required_env_vars_for_config(config)
        if not os.getenv(env_var)
    ]

    if missing:
        return "Missing API key(s): " + ", ".join(sorted(set(missing)))
    return None


def _classify_provider_error(exc: Exception) -> str:
    """Map raw provider exceptions into a short research-friendly category."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "rate" in text or "quota" in text or "429" in text:
        return "rate_limit_or_quota"
    if "503" in text or "unavailable" in text or "overload" in text or "demand" in text:
        return "provider_unavailable"
    if "api key" in text or "auth" in text or "unauthorized" in text or "401" in text:
        return "authentication"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "model" in text and ("not found" in text or "does not exist" in text):
        return "model_unavailable"
    return "provider_error"


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for failures that often succeed after a short retry."""
    return _classify_provider_error(exc) in {
        "rate_limit_or_quota",
        "provider_unavailable",
        "timeout",
    }


def _llm_action_with_retries(
    agent: Any,
    observation: Dict[str, Any],
    episode: int,
    turn: int,
    player_id: int,
    agent_model: str,
) -> Tuple[int, str, List[Dict[str, Any]]]:
    """Call an LLM agent with bounded retries and return retry metadata."""
    failures = []
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            response = agent(observation)
            action, reasoning = _extract_action_and_reasoning(response)
            if failures:
                reasoning = (
                    f"{reasoning}\n\n"
                    f"[Recovered after {len(failures)} failed attempt(s).]"
                )
            return action, reasoning, failures
        except Exception as exc:
            category = _classify_provider_error(exc)
            failure = {
                "episode": episode,
                "turn": turn,
                "player_id": player_id,
                "agent_model": agent_model,
                "attempt": attempt,
                "error_type": category,
                "error_message": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now().isoformat(),
            }
            failures.append(failure)
            if attempt >= MAX_LLM_ATTEMPTS or not _is_retryable_error(exc):
                raise RuntimeError(json.dumps(failures, ensure_ascii=False)) from exc

            time.sleep(RETRY_DELAYS_SECONDS[min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)])

    raise RuntimeError(json.dumps(failures, ensure_ascii=False))


def _preflight_llm_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate selected LLM models with a tiny test call before gameplay."""
    from game_reasoning_arena.backends import generate_response

    checked_models = []
    failures = []
    for player_key, agent in config["agents"].items():
        if agent.get("type") != "llm":
            continue
        model_name = agent.get("model", "")
        if model_name in checked_models:
            continue
        checked_models.append(model_name)
        try:
            response = generate_response(
                model_name=model_name,
                prompt=(
                    "Reply with exactly this JSON and no extra text: "
                    "{\"reasoning\":\"preflight ok\",\"action\":0}"
                ),
                max_tokens=30,
                temperature=0.0,
            )
            if not response or not str(response).strip():
                raise RuntimeError("Provider returned an empty preflight response")
        except Exception as exc:
            failures.append({
                "player": player_key,
                "agent_model": model_name,
                "error_type": _classify_provider_error(exc),
                "error_message": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now().isoformat(),
            })

    return failures


def _list_previous_frontend_runs(limit: int = 12) -> str:
    """Return Markdown listing recent CSV/JSON exports from this frontend."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(
        EXPORT_DIR.glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        return (
            "### Previous runs\n"
            "No frontend exports yet. Completed runs will appear here."
        )

    lines = ["### Previous runs"]
    for csv_path in csv_files[:limit]:
        json_path = csv_path.with_suffix(".json")
        timestamp = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )
        json_marker = "JSON available" if json_path.exists() else "CSV only"
        lines.append(f"- `{csv_path.name}`  \n  {timestamp} · {json_marker}")

    if len(csv_files) > limit:
        lines.append(f"\n_Showing latest {limit} of {len(csv_files)} exports._")

    return "\n".join(lines)


def _model_dropdown_update(agent_type: str) -> Any:
    """Only show a model dropdown when the selected player type is `llm`."""
    import gradio as gr

    return gr.update(visible=(agent_type == "llm"))


def _player_type_ui_updates(changed_type: str, other_type: str) -> Tuple[Any, Any]:
    """Update one model dropdown and the single API-key field visibility."""
    import gradio as gr

    model_update = gr.update(visible=(changed_type == "llm"))
    api_key_update = gr.update(visible=(changed_type == "llm" or other_type == "llm"))
    return model_update, api_key_update


def _show_setup_panel() -> Any:
    """Reopen the setup panel for another run."""
    import gradio as gr

    return gr.update(visible=True)


def _prepare_setup_panel_for_run(
    run_name: str,
    player0_type: str,
    player0_model_label: str,
    player1_type: str,
    player1_model_label: str,
    provider_api_key: str,
) -> Any:
    """Hide setup only when the minimum required run inputs are present."""
    import gradio as gr

    if not (run_name or "").strip():
        return gr.update(visible=True)

    temp_config = {
        "agents": {
            "player_0": _agent_config(player0_type, player0_model_label),
            "player_1": _agent_config(player1_type, player1_model_label),
        }
    }
    required_env_vars = _required_env_vars_for_config(temp_config)
    if required_env_vars:
        load_dotenv()
        pasted_key = (provider_api_key or "").strip()
        has_existing_key = any(os.getenv(env_var) for env_var in required_env_vars)
        if not pasted_key and not has_existing_key:
            return gr.update(visible=True)

    return gr.update(visible=False)


# ---------------------------------------------------------------------------
# Export and chart helpers
# ---------------------------------------------------------------------------

def _write_exports(
    run_name: str,
    metadata: Dict[str, Any],
    move_records: List[Dict[str, Any]],
    result_records: List[Dict[str, Any]],
    failure_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_name)
    safe_name = safe_name.strip("._-") or "game_reasoning_run"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = EXPORT_DIR / f"{safe_name}_{timestamp}"
    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")

    payload = {
        "metadata": metadata,
        "moves": move_records,
        "game_results": result_records,
        "failures": failure_records or [],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "run_name",
        "game_name",
        "episode",
        "turn",
        "player_id",
        "action",
        "reasoning",
        "opponent",
        "generation_time",
        "agent_type",
        "agent_model",
        "timestamp",
        "run_id",
        "seed",
        "board_state",
        "legal_actions",
        "error_type",
        "error_message",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in move_records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})
        for failure in failure_records or []:
            writer.writerow({
                "run_name": run_name,
                "game_name": metadata.get("game_name", ""),
                "episode": failure.get("episode", ""),
                "turn": failure.get("turn", ""),
                "player_id": failure.get("player_id", ""),
                "agent_model": failure.get("agent_model", ""),
                "timestamp": failure.get("timestamp", ""),
                "error_type": failure.get("error_type", ""),
                "error_message": failure.get("error_message", ""),
            })

    return str(csv_path), str(json_path)


def _utc_now() -> str:
    """Return an ISO UTC timestamp with a Z suffix for export files."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _empty_failure_counts() -> Dict[str, int]:
    """Return all supported failure mechanisms initialized to zero."""
    return {mechanism: 0 for mechanism in FAILURE_MECHANISMS}


def _summarize_human_eval(
    turns: List[Dict[str, Any]],
    result: str,
) -> Dict[str, Any]:
    """Build the summary object in the user-requested JSON schema."""
    failure_counts = _empty_failure_counts()
    first_failure_turn = None
    for turn in turns:
        mechanisms = turn.get("failure_mechanism") or []
        if mechanisms and first_failure_turn is None:
            first_failure_turn = turn["turn"]
        for mechanism in mechanisms:
            if mechanism in failure_counts:
                failure_counts[mechanism] += 1

    legal_moves = sum(1 for turn in turns if turn.get("move_is_legal") is True)
    illegal_moves = sum(1 for turn in turns if turn.get("move_is_legal") is False)
    optimal_moves = sum(1 for turn in turns if turn.get("move_is_optimal") is True)
    suboptimal_moves = sum(1 for turn in turns if turn.get("move_is_optimal") is False)

    return {
        "result": result,
        "total_turns": len(turns),
        "legal_moves": legal_moves,
        "illegal_moves": illegal_moves,
        "optimal_moves": optimal_moves,
        "suboptimal_moves": suboptimal_moves,
        "failure_counts": failure_counts,
        "first_failure_turn": first_failure_turn,
    }


def _write_human_eval_exports(state: Dict[str, Any]) -> Tuple[str, str]:
    """Write JSON and CSV files for the human move-evaluation workflow."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_id = state["experiment_id"]
    safe_name = "".join(
        ch if ch.isalnum() or ch in "-_." else "_"
        for ch in experiment_id
    ).strip("._-") or "human_eval"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = EXPORT_DIR / f"{safe_name}_{timestamp}"
    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")

    turns = state.get("turns", [])
    result = state.get("result", "incomplete")
    payload = {
        "game": state["game_name"],
        "episode": state["episode_id"],
        "started_at": state["started_at"],
        "turns": [
            {
                "turn": turn["turn"],
                "timestamp": turn["timestamp"],
                "move_is_legal": turn["move_is_legal"],
                "move_is_optimal": turn["move_is_optimal"],
                "failure_mechanism": turn["failure_mechanism"],
            }
            for turn in turns
        ],
        "summary": _summarize_human_eval(turns, result),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "experiment_id",
        "timestamp",
        "game_name",
        "episode_id",
        "turn_number",
        "model_name",
        "provider",
        "board_state",
        "legal_actions",
        "chosen_action",
        "raw_model_response",
        "reasoning_trace",
        "human_legality_label",
        "move_is_legal",
        "move_is_optimal",
        "failure_mechanism",
        "evaluator_notes",
        "prompt_version",
        "stage_history",
        "result",
        "total_turns",
        "legal_moves",
        "illegal_moves",
        "optimal_moves",
        "suboptimal_moves",
        "first_failure_turn",
    ]
    summary = payload["summary"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for turn in turns:
            writer.writerow({
                "experiment_id": experiment_id,
                "timestamp": turn["timestamp"],
                "game_name": state["game_name"],
                "episode_id": state["episode_id"],
                "turn_number": turn["turn"],
                "model_name": state["model_name"],
                "provider": state["provider"],
                "board_state": turn.get("board_state", ""),
                "legal_actions": json.dumps(turn.get("legal_actions", [])),
                "chosen_action": turn.get("chosen_action"),
                "raw_model_response": turn.get("raw_model_response", ""),
                "reasoning_trace": turn.get("reasoning_trace", ""),
                "human_legality_label": turn.get("human_legality_label"),
                "move_is_legal": turn.get("move_is_legal"),
                "move_is_optimal": turn.get("move_is_optimal"),
                "failure_mechanism": json.dumps(turn.get("failure_mechanism")),
                "evaluator_notes": turn.get("evaluator_notes", ""),
                "prompt_version": turn.get("prompt_version", ""),
                "stage_history": json.dumps(turn.get("stage_history", [])),
                "result": summary["result"],
                "total_turns": summary["total_turns"],
                "legal_moves": summary["legal_moves"],
                "illegal_moves": summary["illegal_moves"],
                "optimal_moves": summary["optimal_moves"],
                "suboptimal_moves": summary["suboptimal_moves"],
                "first_failure_turn": summary["first_failure_turn"],
            })

    return str(json_path), str(csv_path)


def _collect_sqlite_run_records(
    run_name: str,
    loggers: Dict[int, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read persisted move/result rows back from SQLiteLogger databases.

    This mirrors the original reasoning-trace workflow: once gameplay has
    written SQLite rows, final frontend exports/charts read the persisted data
    instead of trusting a separate in-memory data model.
    """
    move_records: List[Dict[str, Any]] = []
    result_records: List[Dict[str, Any]] = []

    for player_id, logger in loggers.items():
        conn = sqlite3.connect(logger.db_path)
        conn.row_factory = sqlite3.Row
        try:
            moves = conn.execute(
                """
                SELECT game_name, episode, turn, action, reasoning, opponent,
                       generation_time, agent_type, agent_model, timestamp,
                       run_id, seed, board_state
                FROM moves
                WHERE run_id = ?
                ORDER BY game_name, episode, turn, id
                """,
                (logger.run_id,),
            ).fetchall()
            for row in moves:
                record = dict(row)
                record["run_name"] = run_name
                record["player_id"] = player_id
                record.setdefault("legal_actions", "")
                move_records.append(record)

            results = conn.execute(
                """
                SELECT game_name, episode, status, reward, opponent, timestamp,
                       run_id
                FROM game_results
                WHERE run_id = ?
                ORDER BY game_name, episode, id
                """,
                (logger.run_id,),
            ).fetchall()
            for row in results:
                record = dict(row)
                record["run_name"] = run_name
                record["player_id"] = player_id
                result_records.append(record)
        finally:
            conn.close()

    move_records.sort(
        key=lambda r: (
            r.get("game_name", ""),
            int(r.get("episode", 0) or 0),
            int(r.get("turn", 0) or 0),
            int(r.get("player_id", 0) or 0),
        )
    )
    result_records.sort(
        key=lambda r: (
            r.get("game_name", ""),
            int(r.get("episode", 0) or 0),
            int(r.get("player_id", 0) or 0),
        )
    )
    return move_records, result_records


def _format_reasoning_traces_from_moves(move_records: List[Dict[str, Any]]) -> str:
    """Format SQLite `moves` rows like the original trace extractor."""
    if not move_records:
        return "### Reasoning traces\nNo persisted move rows found for this run."

    lines = ["### Reasoning traces from SQLite `moves` table"]
    trace_number = 1
    for record in move_records:
        reasoning = (record.get("reasoning") or "").strip()
        if not reasoning or reasoning == "None":
            continue

        lines.extend([
            "",
            f"#### Trace {trace_number}",
            f"- Game: `{record.get('game_name')}`",
            f"- Episode: `{record.get('episode')}`, Turn: `{record.get('turn')}`",
            f"- Agent: `{record.get('agent_type')} ({record.get('agent_model')})`",
            f"- Action: `{record.get('action')}`",
            "",
            "**Board state at decision time**",
            "```text",
            str(record.get("board_state") or "[not available]"),
            "```",
            "",
            "**Reasoning**",
            str(reasoning),
        ])
        trace_number += 1

    if trace_number == 1:
        lines.append("No non-empty reasoning strings were found.")

    return "\n".join(lines)


def _scale_points(values: List[Tuple[int, float]], width: int, height: int) -> str:
    if not values:
        return ""
    xs = [episode for episode, _ in values]
    ys = [reward for _, reward in values]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_span = max(max_x - min_x, 1)
    y_span = max(max_y - min_y, 1)
    points = []
    for episode, reward in values:
        x = 40 + ((episode - min_x) / x_span) * (width - 80)
        y = height - 35 - ((reward - min_y) / y_span) * (height - 70)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _bar_svg(title: str, labels: List[str], values: List[float]) -> str:
    width, height = 520, 260
    max_value = max(values) if values else 1
    max_value = max(max_value, 1)
    bar_width = 70
    gap = 35
    start_x = 55
    bars = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = start_x + idx * (bar_width + gap)
        bar_height = (value / max_value) * 150
        y = 190 - bar_height
        bars.append(
            f"<rect x='{x}' y='{y:.1f}' width='{bar_width}' height='{bar_height:.1f}' "
            "fill='white' stroke='black' stroke-width='2'/>"
            f"<text x='{x + bar_width / 2}' y='215' text-anchor='middle' "
            "font-size='12'>{label}</text>"
            f"<text x='{x + bar_width / 2}' y='{y - 8:.1f}' text-anchor='middle' "
            "font-size='12' font-weight='700'>{value:g}</text>"
        )
    return (
        "<div class='gra-panel'>"
        f"<h3>{title}</h3>"
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='260'>"
        "<line x1='35' y1='190' x2='490' y2='190' stroke='black'/>"
        + "".join(bars)
        + "</svg></div>"
    )


def _make_charts(result_records: List[Dict[str, Any]], move_records: List[Dict[str, Any]]):
    """Create dependency-free SVG/HTML charts for the frontend.

    The upstream project has richer matplotlib/seaborn analysis scripts. This
    live frontend provides lightweight summary charts without requiring those
    plotting dependencies at runtime.
    """
    # Chart 1: reward by episode/player.
    by_player: Dict[int, List[Tuple[int, float]]] = {}
    for record in result_records:
        by_player.setdefault(record["player_id"], []).append(
            (record["episode"], record["reward"])
        )
    width, height = 640, 260
    polylines = []
    for player_id, values in sorted(by_player.items()):
        points = _scale_points(sorted(values), width, height)
        dash = " stroke-dasharray='6 4'" if player_id else ""
        polylines.append(
            f"<polyline points='{points}' fill='none' stroke='black' "
            f"stroke-width='2'{dash}/>"
        )
    reward_chart = (
        "<div class='gra-panel'><h3>Rewards by Episode</h3>"
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='260'>"
        "<line x1='40' y1='225' x2='600' y2='225' stroke='black'/>"
        "<line x1='40' y1='25' x2='40' y2='225' stroke='black'/>"
        + "".join(polylines)
        + "<text x='42' y='18' font-size='12'>reward</text>"
        + "<text x='560' y='245' font-size='12'>episode</text>"
        + "</svg><p>Solid line = Player 0; dashed line = Player 1.</p></div>"
    )

    # Chart 2: player 0 outcome counts.
    counts = {"win": 0, "loss": 0, "draw": 0}
    for record in result_records:
        if record["player_id"] != 0:
            continue
        reward = record["reward"]
        if reward > 0:
            counts["win"] += 1
        elif reward < 0:
            counts["loss"] += 1
        else:
            counts["draw"] += 1
    outcome_chart = _bar_svg(
        "Player 0 Outcomes",
        list(counts.keys()),
        [counts["win"], counts["loss"], counts["draw"]],
    )

    # Chart 3: moves per episode.
    turns_by_episode: Dict[int, int] = {}
    for record in move_records:
        episode = int(record["episode"])
        turns_by_episode[episode] = max(turns_by_episode.get(episode, 0), int(record["turn"]) + 1)
    sorted_episodes = sorted(turns_by_episode.keys())
    turns_chart = _bar_svg(
        "Turns per Episode",
        [str(episode) for episode in sorted_episodes],
        [turns_by_episode[episode] for episode in sorted_episodes],
    )

    return reward_chart, outcome_chart, turns_chart


# ---------------------------------------------------------------------------
# Human-in-the-loop move evaluation flow
# ---------------------------------------------------------------------------

def _eval_stage_updates(stage: str) -> Tuple[Any, Any, Any]:
    """Return visibility updates for legality/optimality/attribution panels."""
    import gradio as gr

    return (
        gr.update(visible=(stage == "legality")),
        gr.update(visible=(stage == "optimality")),
        gr.update(visible=stage in {"attribution_illegal", "attribution_suboptimal"}),
    )


def _provider_from_model_label(model_label: str) -> str:
    if model_label.startswith("Remote / OpenRouter"):
        return "openrouter"
    if model_label.startswith("Remote / Google Gemini"):
        return "google_gemini"
    if model_label.startswith("Remote / OpenAI"):
        return "openai"
    if model_label.startswith("Remote / Groq"):
        return "groq"
    if model_label.startswith("Local /"):
        return "local"
    return "unknown"


def _human_eval_status(state: Dict[str, Any]) -> str:
    stage = state.get("stage", "not_started")
    labels = {
        "legality": "Legality check",
        "optimality": "Optimality judgment",
        "attribution_illegal": "Failure attribution for illegal move",
        "attribution_suboptimal": "Failure attribution for suboptimal move",
        "complete": "Completed",
        "error": "Error",
    }
    return (
        f"**Stage:** {labels.get(stage, stage)}  \n"
        f"**Turn:** {state.get('turn_number', 0)}  \n"
        f"**Saved labels:** {len(state.get('turns', []))}"
    )


def _pending_move_markdown(state: Dict[str, Any]) -> str:
    pending = state.get("pending")
    if not pending:
        return "No pending model move."
    return (
        "### Current model move\n"
        f"**Chosen action:** `{pending['chosen_action']}`  \n"
        f"**Legal actions:** `{pending['legal_actions']}`  \n\n"
        "**Model reasoning**\n\n"
        f"{pending['reasoning_trace']}"
    )


def _finalize_human_eval(state: Dict[str, Any], result: str) -> Tuple[str, str]:
    if result == "complete" and state.get("env") is not None:
        reward = state["env"].state.player_reward(0)
        if reward > 0:
            result = "win"
        elif reward < 0:
            result = "loss"
        else:
            result = "draw"
    state["result"] = result
    json_path, csv_path = _write_human_eval_exports(state)
    state["json_path"] = json_path
    state["csv_path"] = csv_path
    return json_path, csv_path


def _auto_play_until_model_turn(state: Dict[str, Any]) -> Dict[str, Any]:
    """Let the opponent move until it is player 0/model turn or game ends."""
    env = state["env"]
    observations = state["observations"]
    opponent_agent = state["opponent_agent"]

    while not env.state.is_terminal() and env.state.current_player() != 0:
        current_player = env.state.current_player()
        legal_actions = observations[current_player]["legal_actions"]
        action, _reasoning = _strong_bot_action(
            state["game_name"],
            env.state,
            current_player,
            legal_actions,
        )
        observations, _rewards, _terminated, _truncated, _ = env.step({current_player: action})
        state["observations"] = observations

    return state


def _prepare_next_human_eval_turn(state: Dict[str, Any]) -> Dict[str, Any]:
    """Advance to the next model move and store it as `state['pending']`."""
    state = _auto_play_until_model_turn(state)
    env = state["env"]
    if env.state.is_terminal():
        _finalize_human_eval(state, "complete")
        state["stage"] = "complete"
        state["pending"] = None
        return state

    if state["turn_number"] >= state["max_turns"]:
        _finalize_human_eval(state, "max_turns")
        state["stage"] = "complete"
        state["pending"] = None
        return state

    observation = state["observations"][0]
    action, reasoning, failures = _llm_action_with_retries(
        state["model_agent"],
        observation,
        episode=state["episode_id"],
        turn=state["turn_number"] + 1,
        player_id=0,
        agent_model=state["model_name"],
    )
    if failures:
        state.setdefault("provider_failures", []).extend(failures)

    state["turn_number"] += 1
    state["pending"] = {
        "turn": state["turn_number"],
        "timestamp": _utc_now(),
        "board_state": observation.get("state_string", ""),
        "legal_actions": observation.get("legal_actions", []),
        "chosen_action": action,
        "raw_model_response": json.dumps({"action": action, "reasoning": reasoning}, ensure_ascii=False),
        "reasoning_trace": reasoning,
        "prompt_version": "game_reasoning_arena_frontend_v1",
        "stage_history": ["legality"],
    }
    state["stage"] = "legality"
    return state


def start_human_evaluation(
    experiment_id: str,
    game_label: str,
    model_label: str,
    provider_api_key: str,
    seed: int,
    max_turns: int,
) -> Tuple[str, str, str, Dict[str, Any], Any, Any, Any, Any, Any]:
    """Start a human-labeled game for Tic-Tac-Toe or Connect Four."""
    if not (experiment_id or "").strip():
        empty_state = {"stage": "error", "turns": []}
        legality, optimality, attribution = _eval_stage_updates("error")
        return (
            "```text\nNo board yet.\n```",
            "Name the experiment before starting.",
            "Status: missing experiment id",
            empty_state,
            legality,
            optimality,
            attribution,
            None,
            None,
        )

    game_name = EVAL_GAME_CHOICES.get(game_label, "connect_four")
    model_name, env_var = _resolve_model(model_label)
    config = {
        "agents": {
            "player_0": {"type": "llm", "model": model_name},
            "player_1": {"type": "strong_bot", "model": "search_heuristic"},
        }
    }
    _set_api_key_for_config(provider_api_key, config)
    key_shape_warning = _validate_pasted_key_shape(config, provider_api_key)
    if key_shape_warning:
        state = {"stage": "error", "turns": []}
        legality, optimality, attribution = _eval_stage_updates("error")
        return (
            "```text\nNo board yet.\n```",
            key_shape_warning,
            "Status: API key/provider mismatch",
            state,
            legality,
            optimality,
            attribution,
            None,
            None,
        )

    missing_key = _validate_keys_for_agents(config)
    if missing_key:
        state = {"stage": "error", "turns": []}
        legality, optimality, attribution = _eval_stage_updates("error")
        return (
            "```text\nNo board yet.\n```",
            missing_key,
            "Status: missing API key",
            state,
            legality,
            optimality,
            attribution,
            None,
            None,
        )

    from game_reasoning_arena.arena.games.registry import registry
    from game_reasoning_arena.arena.agents.llm_agent import LLMAgent
    from game_reasoning_arena.backends import initialize_llm_registry

    initialize_llm_registry()
    env_config = _build_config(
        game_name=game_name,
        seed=int(seed),
        num_players=2,
        player0_type="llm",
        player0_model_label=model_label,
        player1_type="strong_bot",
        player1_model_label=model_label,
    )
    env = registry.make_env(game_name, env_config)
    observations, _ = env.reset(seed=int(seed))
    state = {
        "experiment_id": experiment_id.strip(),
        "game_name": game_name,
        "episode_id": 1,
        "started_at": _utc_now(),
        "model_name": model_name,
        "provider": _provider_from_model_label(model_label),
        "env": env,
        "observations": observations,
        "model_agent": LLMAgent(model_name=model_name, game_name=game_name),
        "opponent_agent": "strong_bot",
        "turn_number": 0,
        "max_turns": int(max_turns),
        "turns": [],
        "stage": "legality",
        "pending": None,
        "result": "incomplete",
    }
    try:
        state = _prepare_next_human_eval_turn(state)
    except Exception as exc:
        state["stage"] = "error"
        legality, optimality, attribution = _eval_stage_updates("error")
        return (
            _render_board(state["env"], 0),
            f"Model move generation failed: {type(exc).__name__}: {exc}",
            "Status: model error",
            state,
            legality,
            optimality,
            attribution,
            None,
            None,
        )

    legality, optimality, attribution = _eval_stage_updates(state["stage"])
    return (
        _render_board(state["env"], 0),
        _pending_move_markdown(state),
        _human_eval_status(state),
        state,
        legality,
        optimality,
        attribution,
        None,
        None,
    )


def _save_pending_human_eval(
    state: Dict[str, Any],
    move_is_legal: bool,
    move_is_optimal: Optional[bool],
    failure_mechanism: Optional[List[str]],
    notes: str,
) -> Dict[str, Any]:
    pending = state["pending"]
    pending["stage_history"].append("completed_turn")
    state["turns"].append({
        **pending,
        "human_legality_label": "Legal" if move_is_legal else "Illegal",
        "move_is_legal": move_is_legal,
        "move_is_optimal": move_is_optimal,
        "failure_mechanism": failure_mechanism,
        "evaluator_notes": notes or "",
    })

    env = state["env"]
    action = pending["chosen_action"]
    if action in pending["legal_actions"]:
        observations, _rewards, _terminated, _truncated, _ = env.step({0: action})
        state["observations"] = observations
    else:
        _finalize_human_eval(state, "illegal_move_unapplied")
        state["stage"] = "complete"
        state["pending"] = None
        return state

    try:
        state = _prepare_next_human_eval_turn(state)
    except Exception as exc:
        state["stage"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"
    return state


def mark_move_legal(state: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any], Any, Any, Any, Any, Any]:
    state["pending"]["stage_history"].append("marked_legal")
    state["stage"] = "optimality"
    legality, optimality, attribution = _eval_stage_updates(state["stage"])
    return (
        _render_board(state["env"], 0),
        _pending_move_markdown(state),
        _human_eval_status(state),
        state,
        legality,
        optimality,
        attribution,
        state.get("json_path"),
        state.get("csv_path"),
    )


def mark_move_illegal(state: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any], Any, Any, Any, Any, Any]:
    state["pending"]["stage_history"].append("marked_illegal")
    state["stage"] = "attribution_illegal"
    legality, optimality, attribution = _eval_stage_updates(state["stage"])
    return (
        _render_board(state["env"], 0),
        _pending_move_markdown(state),
        _human_eval_status(state),
        state,
        legality,
        optimality,
        attribution,
        state.get("json_path"),
        state.get("csv_path"),
    )


def mark_move_optimal(state: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any], Any, Any, Any, Any, Any]:
    state["pending"]["stage_history"].append("marked_optimal")
    state = _save_pending_human_eval(state, True, True, None, "")
    legality, optimality, attribution = _eval_stage_updates(state["stage"])
    return (
        _render_board(state["env"], 0) if state.get("env") else "```text\nComplete\n```",
        _pending_move_markdown(state) if state.get("pending") else "Game complete. Download JSON/CSV below.",
        _human_eval_status(state),
        state,
        legality,
        optimality,
        attribution,
        state.get("json_path"),
        state.get("csv_path"),
    )


def mark_move_suboptimal(state: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any], Any, Any, Any, Any, Any]:
    state["pending"]["stage_history"].append("marked_suboptimal")
    state["stage"] = "attribution_suboptimal"
    legality, optimality, attribution = _eval_stage_updates(state["stage"])
    return (
        _render_board(state["env"], 0),
        _pending_move_markdown(state),
        _human_eval_status(state),
        state,
        legality,
        optimality,
        attribution,
        state.get("json_path"),
        state.get("csv_path"),
    )


def submit_failure_attribution(
    mechanisms: List[str],
    notes: str,
    state: Dict[str, Any],
) -> Tuple[str, str, str, Dict[str, Any], Any, Any, Any, Any, Any]:
    stage = state.get("stage")
    move_is_legal = stage != "attribution_illegal"
    move_is_optimal = False if move_is_legal else None
    state = _save_pending_human_eval(
        state,
        move_is_legal=move_is_legal,
        move_is_optimal=move_is_optimal,
        failure_mechanism=mechanisms or [],
        notes=notes,
    )
    legality, optimality, attribution = _eval_stage_updates(state["stage"])
    return (
        _render_board(state["env"], 0) if state.get("env") else "```text\nComplete\n```",
        _pending_move_markdown(state) if state.get("pending") else "Game complete. Download JSON/CSV below.",
        _human_eval_status(state),
        state,
        legality,
        optimality,
        attribution,
        state.get("json_path"),
        state.get("csv_path"),
    )


# ---------------------------------------------------------------------------
# Live experiment runner
# ---------------------------------------------------------------------------

def run_experiment_live(
    run_name: str,
    game_label: str,
    num_episodes: int,
    seed: int,
    player0_type: str,
    player0_model_label: str,
    player1_type: str,
    player1_model_label: str,
    provider_api_key: str,
    max_turns: int,
    delay_seconds: float,
) -> Generator[Tuple[str, str, str, str, Any, Any, Any, Any, Any], None, None]:
    """Run an original-methodology Game Reasoning Arena experiment live."""
    if not run_name.strip():
        yield (
            "```text\nNo board yet.\n```",
            "Name the run before starting.",
            "Status: not started",
            "",
            None,
            None,
            None,
            None,
            None,
        )
        return

    try:
        from game_reasoning_arena.arena.games.registry import registry
        from game_reasoning_arena.arena.utils.loggers import SQLiteLogger
        from game_reasoning_arena.arena.utils.seeding import set_seed
        from game_reasoning_arena.backends import initialize_llm_registry
    except ImportError as exc:
        yield (
            "```text\nNo board yet.\n```",
            f"Missing dependency: {exc}",
            "Status: dependency error",
            "",
            None,
            None,
            None,
            None,
            None,
        )
        return

    game_name = _resolve_game(game_label)
    game = registry.get_game_loader(game_name)()
    num_players = game.num_players()
    num_episodes = max(1, int(num_episodes))
    seed = int(seed)
    max_turns = max(1, int(max_turns))
    delay_seconds = float(delay_seconds)

    config = _build_config(
        game_name=game_name,
        seed=seed,
        num_players=num_players,
        player0_type=player0_type,
        player0_model_label=player0_model_label,
        player1_type=player1_type,
        player1_model_label=player1_model_label,
    )

    _set_api_key_for_config(provider_api_key, config)

    key_shape_warning = _validate_pasted_key_shape(config, provider_api_key)
    if key_shape_warning:
        yield (
            "```text\nNo board yet.\n```",
            key_shape_warning,
            "Status: API key/provider mismatch",
            "",
            None,
            None,
            None,
            None,
            None,
        )
        return

    missing_key_message = _validate_keys_for_agents(config)
    if missing_key_message:
        yield (
            "```text\nNo board yet.\n```",
            missing_key_message,
            "Status: missing API key",
            "",
            None,
            None,
            None,
            None,
            None,
        )
        return

    if any(agent.get("type") == "llm" for agent in config["agents"].values()):
        initialize_llm_registry()

    loggers = {}
    for player_id in range(num_players):
        agent_type, model_name = _agent_metadata(config, player_id)
        sanitized_model = model_name.replace("-", "_").replace("/", "_")
        loggers[player_id] = SQLiteLogger(agent_type, sanitized_model)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "run_name": run_name,
        "run_id": run_id,
        "game_name": game_name,
        "game_label": game_label,
        "num_episodes": num_episodes,
        "seed": seed,
        "agents": config["agents"],
        "methodology": "Game Reasoning Arena live frontend using OpenSpiel registry, LLMAgent/RandomAgent/OpenSpiel bots/strong search bots, and SQLiteLogger",
    }
    move_records: List[Dict[str, Any]] = []
    result_records: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []
    transcript = [
        "# Game Reasoning Arena Live Run",
        f"- Run: `{run_name}`",
        f"- Game: `{game_name}`",
        f"- Episodes: `{num_episodes}`",
        f"- Seed: `{seed}`",
        f"- Agents: `{config['agents']}`",
        "",
    ]

    if any(agent.get("type") == "llm" for agent in config["agents"].values()):
        transcript.append("\n## Preflight model/API validation")
        preflight_failures = _preflight_llm_models(config)
        if preflight_failures:
            failure_records.extend(preflight_failures)
            metadata["completed_at"] = datetime.utcnow().isoformat()
            metadata["preflight_status"] = "failed"
            csv_path, json_path = _write_exports(
                run_name,
                metadata,
                move_records,
                result_records,
                failure_records,
            )
            failure_text = json.dumps(preflight_failures, indent=2, ensure_ascii=False)
            transcript.append("Preflight failed before gameplay started.")
            transcript.append(f"```json\n{failure_text}\n```")
            yield (
                "```text\nNo board yet.\n```",
                "\n".join(transcript),
                "Status: preflight failed",
                "",
                csv_path,
                json_path,
                None,
                None,
                None,
            )
            return
        transcript.append("Preflight passed.")

    last_board = "```text\nNo board yet.\n```"

    for episode_index in range(num_episodes):
        episode = episode_index + 1
        episode_seed = seed + episode_index
        set_seed(episode_seed)

        player_to_agent = _initialize_frontend_agents(
            config,
            game_name,
            episode_seed,
            num_players,
        )
        env = registry.make_env(game_name, config)
        observations, _ = env.reset(seed=episode_seed)
        rewards = {player_id: 0.0 for player_id in range(num_players)}
        terminated = truncated = False
        turn = 0

        transcript.append(f"\n## Episode {episode}/{num_episodes}")
        transcript.append(f"Seed: `{episode_seed}`")
        last_board = _render_board(env, 0)
        yield (
            last_board,
            "\n".join(transcript),
            _status_text(episode, num_episodes, turn, rewards, False),
            "",
            None,
            None,
            None,
            None,
            None,
        )

        while not (terminated or truncated):
            if turn >= max_turns:
                truncated = True
                transcript.append(f"Episode truncated at max_turns={max_turns}.")
                break

            if env.state.is_simultaneous_node():
                active_players = list(range(num_players))
            else:
                active_players = [env.state.current_player()]

            action_dict = {}
            for player_id in active_players:
                observation = observations[player_id]
                legal_actions = observation["legal_actions"]
                agent_type, agent_model = _agent_metadata(config, player_id)
                try:
                    if agent_type == "strong_bot":
                        action, reasoning = _strong_bot_action(
                            game_name,
                            env.state,
                            player_id,
                            legal_actions,
                        )
                    elif agent_type == "openspiel_bot":
                        action = player_to_agent[player_id].step(env.state)
                        reasoning = "OpenSpiel uniform random bot selected the action."
                    elif agent_type == "llm":
                        action, reasoning, retry_failures = _llm_action_with_retries(
                            player_to_agent[player_id],
                            observation,
                            episode,
                            turn,
                            player_id,
                            agent_model,
                        )
                        failure_records.extend(retry_failures)
                    else:
                        response = player_to_agent[player_id](observation)
                        action, reasoning = _extract_action_and_reasoning(response)
                except Exception as exc:
                    parsed_failures = []
                    try:
                        parsed_failures = json.loads(str(exc))
                    except Exception:
                        parsed_failures = [{
                            "episode": episode,
                            "turn": turn,
                            "player_id": player_id,
                            "agent_model": agent_model,
                            "error_type": _classify_provider_error(exc),
                            "error_message": f"{type(exc).__name__}: {exc}",
                            "timestamp": datetime.now().isoformat(),
                        }]
                    failure_records.extend(parsed_failures)
                    metadata["completed_at"] = datetime.utcnow().isoformat()
                    metadata["failure_status"] = "action_generation_error"
                    csv_path, json_path = _write_exports(
                        run_name,
                        metadata,
                        move_records,
                        result_records,
                        failure_records,
                    )
                    error_message = (
                        f"Action generation failed in episode {episode}, "
                        f"turn {turn}, player {player_id} "
                        f"({agent_type}/{agent_model}): "
                        f"{parsed_failures[-1].get('error_type')}: "
                        f"{parsed_failures[-1].get('error_message')}"
                    )
                    transcript.append(f"\n## Error\n{error_message}")
                    transcript.append(
                        "Failure details were written to the CSV/JSON export."
                    )
                    yield (
                        last_board,
                        "\n".join(transcript),
                        f"Status: action generation error\n{error_message}",
                        _format_reasoning_traces_from_moves(move_records),
                        csv_path,
                        json_path,
                        None,
                        None,
                        None,
                    )
                    return

                transcript.append(
                    f"\nEpisode {episode}, Turn {turn}, Player {player_id}"
                )
                transcript.append(f"Legal actions: `{legal_actions}`")
                transcript.append(f"Chosen action: `{action}`")
                if agent_type in {"llm", "strong_bot", "openspiel_bot"}:
                    transcript.append(f"Reasoning: {reasoning}")

                if action not in legal_actions:
                    loggers[player_id].log_illegal_move(
                        game_name=game_name,
                        episode=episode,
                        turn=turn,
                        agent_id=player_id,
                        illegal_action=action,
                        reason="Illegal action",
                        board_state=observation.get("state_string", ""),
                    )
                    truncated = True
                    transcript.append("Illegal move detected; episode truncated.")
                    break

                action_dict[player_id] = action
                record = {
                    "run_name": run_name,
                    "game_name": game_name,
                    "episode": episode,
                    "turn": turn,
                    "player_id": player_id,
                    "action": action,
                    "reasoning": reasoning,
                    "opponent": _opponent_label(config, player_id),
                    "generation_time": 0.0,
                    "agent_type": agent_type,
                    "agent_model": agent_model,
                    "timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "seed": episode_seed,
                    "board_state": observation.get("state_string", ""),
                    "legal_actions": json.dumps(legal_actions),
                }
                move_records.append(record)
                loggers[player_id].log_move(
                    game_name=game_name,
                    episode=episode,
                    turn=turn,
                    action=action,
                    reasoning=reasoning,
                    opponent=record["opponent"],
                    generation_time=0.0,
                    agent_type=agent_type,
                    agent_model=agent_model,
                    seed=episode_seed,
                    board_state=record["board_state"],
                )

            if truncated:
                break

            try:
                observations, step_rewards, terminated, truncated, _ = env.step(action_dict)
            except Exception as exc:
                failure_records.append({
                    "episode": episode,
                    "turn": turn,
                    "player_id": "",
                    "agent_model": "",
                    "error_type": "environment_step_error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "timestamp": datetime.now().isoformat(),
                })
                metadata["completed_at"] = datetime.utcnow().isoformat()
                metadata["failure_status"] = "environment_step_error"
                csv_path, json_path = _write_exports(
                    run_name,
                    metadata,
                    move_records,
                    result_records,
                    failure_records,
                )
                error_message = (
                    f"OpenSpiel step failed in episode {episode}, turn {turn}: "
                    f"{type(exc).__name__}: {exc}"
                )
                transcript.append(f"\n## Error\n{error_message}")
                transcript.append(
                    "Failure details were written to the CSV/JSON export."
                )
                yield (
                    last_board,
                    "\n".join(transcript),
                    f"Status: environment step error\n{error_message}",
                    _format_reasoning_traces_from_moves(move_records),
                    csv_path,
                    json_path,
                    None,
                    None,
                    None,
                )
                return
            rewards.update(step_rewards)
            turn += 1
            last_board = _render_board(env, 0)

            yield (
                last_board,
                "\n".join(transcript),
                _status_text(episode, num_episodes, turn, rewards, terminated or truncated),
                "",
                None,
                None,
                None,
                None,
                None,
            )

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        status = "truncated" if truncated else "terminated"
        transcript.append(f"Episode {episode} finished with status `{status}` and rewards `{rewards}`.")

        for player_id, reward in rewards.items():
            opponent = _opponent_label(config, player_id)
            loggers[player_id].log_rewards(game_name, episode, reward)
            loggers[player_id].log_game_result(
                game_name=game_name,
                episode=episode,
                status=status,
                reward=reward,
                opponent=opponent,
            )
            result_records.append({
                "run_name": run_name,
                "game_name": game_name,
                "episode": episode,
                "player_id": player_id,
                "status": status,
                "reward": reward,
                "opponent": opponent,
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
            })

    sqlite_move_records, sqlite_result_records = _collect_sqlite_run_records(
        run_name,
        loggers,
    )
    if sqlite_move_records:
        move_records = sqlite_move_records
    if sqlite_result_records:
        result_records = sqlite_result_records

    metadata["completed_at"] = datetime.utcnow().isoformat()
    metadata["export_source"] = (
        "SQLiteLogger moves/game_results tables"
        if sqlite_move_records or sqlite_result_records
        else "in-memory fallback"
    )
    csv_path, json_path = _write_exports(
        run_name,
        metadata,
        move_records,
        result_records,
        failure_records,
    )
    reasoning_traces = _format_reasoning_traces_from_moves(move_records)
    reward_fig, outcome_fig, turns_fig = _make_charts(result_records, move_records)
    transcript.append("\n## Outputs")
    transcript.append(f"- CSV: `{csv_path}`")
    transcript.append(f"- JSON: `{json_path}`")

    yield (
        last_board,
        "\n".join(transcript),
        _status_text(num_episodes, num_episodes, 0, {}, True),
        reasoning_traces,
        csv_path,
        json_path,
        reward_fig,
        outcome_fig,
        turns_fig,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required for the frontend. Install it with: python3 -m pip install gradio"
        ) from exc

    with gr.Blocks(title="Game Reasoning Arena Frontend") as demo:
        gr.HTML(f"<style>{APP_CSS}</style>")
        gr.Markdown(
            "# Game Reasoning Arena Frontend\n"
            "OpenSpiel-based LLM game evaluation following the original repository methodology.",
            elem_classes=["gra-hero"],
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=260, elem_classes=["gra-panel"]):
                previous_runs = gr.Markdown(value=_list_previous_frontend_runs())
                refresh_runs = gr.Button("Refresh previous runs", elem_classes=["gra-secondary"])
                new_run = gr.Button("New experiment", elem_classes=["gra-secondary"])

            with gr.Column(scale=4):
                with gr.Column(visible=True, elem_classes=["gra-panel"]) as setup_panel:
                    with gr.Row():
                        run_name = gr.Textbox(
                            label="Run name",
                            placeholder="example_connect_four_gemini_10eps",
                        )
                        provider_api_key = gr.Textbox(
                            label="API key for selected LLM provider",
                            type="password",
                            placeholder="Paste one key only. Hidden if no LLM is selected.",
                        )

                    with gr.Row():
                        game = gr.Dropdown(
                            choices=list(GAME_CHOICES.keys()),
                            value="Tic-Tac-Toe",
                            label="OpenSpiel game",
                        )
                        num_episodes = gr.Number(label="Episodes", value=1, precision=0)
                        seed = gr.Number(label="Seed", value=42, precision=0)
                        max_turns = gr.Number(label="Max turns per episode", value=80, precision=0)

                    with gr.Row():
                        player0_type = gr.Dropdown(AGENT_TYPES, value="llm", label="Player 0 type")
                        player0_model = gr.Dropdown(
                            list(MODEL_CHOICES.keys()),
                            value="Remote / OpenRouter Gemini 2.5 Flash",
                            label="Player 0 model",
                            visible=True,
                        )
                        player1_type = gr.Dropdown(AGENT_TYPES, value="strong_bot", label="Player 1 type")
                        player1_model = gr.Dropdown(
                            list(MODEL_CHOICES.keys()),
                            value="Remote / Groq Llama 3.1 8B Instant",
                            label="Player 1 model",
                            visible=False,
                        )

                    delay_seconds = gr.Slider(
                        label="Live delay between turns",
                        minimum=0,
                        maximum=3,
                        value=0.25,
                        step=0.25,
                    )

                    run_button = gr.Button("Run experiment", variant="primary", elem_classes=["gra-button"])

                with gr.Row():
                    board = gr.Markdown("```text\nNo game running.\n```", elem_classes=["gra-board"])
                    status = gr.Markdown("Status: idle", elem_classes=["gra-panel"])

                transcript = gr.Markdown(
                    "Reasoning traces and episode logs will appear here.",
                    elem_classes=["gra-panel", "gra-log"],
                )
                reasoning_traces = gr.Markdown(
                    "SQLite reasoning traces will appear here after the run.",
                    elem_classes=["gra-panel", "gra-log"],
                )

                with gr.Row():
                    csv_file = gr.File(label="Download CSV")
                    json_file = gr.File(label="Download JSON")

                with gr.Row():
                    reward_plot = gr.HTML(label="Rewards by episode")
                    outcome_plot = gr.HTML(label="Player 0 outcomes")
                    turns_plot = gr.HTML(label="Turns per episode")

        refresh_runs.click(fn=_list_previous_frontend_runs, outputs=previous_runs)
        new_run.click(fn=_show_setup_panel, outputs=setup_panel)
        player0_type.change(
            fn=_player_type_ui_updates,
            inputs=[player0_type, player1_type],
            outputs=[player0_model, provider_api_key],
        )
        player1_type.change(
            fn=_player_type_ui_updates,
            inputs=[player1_type, player0_type],
            outputs=[player1_model, provider_api_key],
        )

        run_event = run_button.click(
            fn=_prepare_setup_panel_for_run,
            inputs=[
                run_name,
                player0_type,
                player0_model,
                player1_type,
                player1_model,
                provider_api_key,
            ],
            outputs=setup_panel,
        )
        run_event.then(
            fn=run_experiment_live,
            inputs=[
                run_name,
                game,
                num_episodes,
                seed,
                player0_type,
                player0_model,
                player1_type,
                player1_model,
                provider_api_key,
                max_turns,
                delay_seconds,
            ],
            outputs=[
                board,
                transcript,
                status,
                reasoning_traces,
                csv_file,
                json_file,
                reward_plot,
                outcome_plot,
                turns_plot,
            ],
        ).then(fn=_list_previous_frontend_runs, outputs=previous_runs)

        gr.Markdown(
            "## Human move evaluation mode\n"
            "Use this section when the research task is to label each model "
            "move manually. This mode is limited to Tic-Tac-Toe and Connect "
            "Four and exports JSON/CSV in the requested evaluation schema.",
            elem_classes=["gra-hero"],
        )

        human_eval_state = gr.State({})
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["gra-panel"]):
                eval_experiment_id = gr.Textbox(
                    label="Experiment ID",
                    placeholder="connect_four_gemini_episode_1",
                )
                eval_game = gr.Dropdown(
                    choices=list(EVAL_GAME_CHOICES.keys()),
                    value="Connect Four",
                    label="Game",
                )
                eval_model = gr.Dropdown(
                    list(MODEL_CHOICES.keys()),
                    value="Remote / OpenRouter Gemini 2.5 Flash",
                    label="Model",
                )
                eval_api_key = gr.Textbox(
                    label="API key for selected model",
                    type="password",
                )
                eval_seed = gr.Number(label="Seed", value=42, precision=0)
                eval_max_turns = gr.Number(label="Max model turns", value=42, precision=0)
                start_eval = gr.Button(
                    "Start human evaluation",
                    variant="primary",
                    elem_classes=["gra-button"],
                )

            with gr.Column(scale=2):
                eval_board = gr.HTML(
                    value="<div class='gra-panel'>Start an evaluation to see the board.</div>",
                    label="Current game state",
                )
                eval_move = gr.Markdown(
                    value="Current model action and reasoning will appear here.",
                    elem_classes=["gra-panel"],
                )
                eval_status = gr.Markdown(
                    value="Status: waiting to start",
                    elem_classes=["gra-panel"],
                )

                with gr.Group(visible=False) as legality_panel:
                    gr.Markdown("### Stage 1 — Human legality judgment")
                    legal_btn = gr.Button("Legal", elem_classes=["gra-secondary"])
                    illegal_btn = gr.Button("Illegal", elem_classes=["gra-secondary"])

                with gr.Group(visible=False) as optimality_panel:
                    gr.Markdown("### Stage 2 — Human optimality judgment")
                    optimal_btn = gr.Button("Optimal / competent", elem_classes=["gra-secondary"])
                    suboptimal_btn = gr.Button("Suboptimal", elem_classes=["gra-secondary"])

                with gr.Group(visible=False) as attribution_panel:
                    gr.Markdown("### Stage 3 — Failure attribution")
                    failure_checks = gr.CheckboxGroup(
                        choices=FAILURE_MECHANISMS,
                        label="Failure mechanism(s)",
                    )
                    evaluator_notes = gr.Textbox(
                        label="Evaluator notes",
                        lines=3,
                    )
                    save_failure_btn = gr.Button(
                        "Save labels + next turn",
                        variant="primary",
                        elem_classes=["gra-button"],
                    )

                with gr.Row():
                    eval_json = gr.File(label="Download evaluation JSON")
                    eval_csv = gr.File(label="Download evaluation CSV")

        eval_outputs = [
            eval_board,
            eval_move,
            eval_status,
            human_eval_state,
            legality_panel,
            optimality_panel,
            attribution_panel,
            eval_json,
            eval_csv,
        ]
        start_eval.click(
            fn=start_human_evaluation,
            inputs=[
                eval_experiment_id,
                eval_game,
                eval_model,
                eval_api_key,
                eval_seed,
                eval_max_turns,
            ],
            outputs=eval_outputs,
        )
        legal_btn.click(fn=mark_move_legal, inputs=human_eval_state, outputs=eval_outputs)
        illegal_btn.click(fn=mark_move_illegal, inputs=human_eval_state, outputs=eval_outputs)
        optimal_btn.click(fn=mark_move_optimal, inputs=human_eval_state, outputs=eval_outputs)
        suboptimal_btn.click(fn=mark_move_suboptimal, inputs=human_eval_state, outputs=eval_outputs)
        save_failure_btn.click(
            fn=submit_failure_attribution,
            inputs=[failure_checks, evaluator_notes, human_eval_state],
            outputs=eval_outputs,
        )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7861)
