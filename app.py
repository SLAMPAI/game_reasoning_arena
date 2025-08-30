#!/usr/bin/env python3
"""
Game Reasoning Arena — Hugging Face Spaces Gradio App

This module provides a web interface for playing games between humans and AI agents,
analyzing LLM performance, and visualizing game statistics.

Pipeline:
User clicks "Start Game" in Gradio
    ↓
app.py (play_game)
    ↓
ui/gradio_config_generator.py (run_game_with_existing_infrastructure)
    ↓
src/game_reasoning_arena/ (core game infrastructure)
    ↓
Game results + metrics displayed in Gradio

Features:
- Interactive human vs AI gameplay
- LLM leaderboards and performance metrics
- Real-time game visualization
- Database management for results
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import sqlite3
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Generator, TypedDict

# Third-party imports
import pandas as pd
import gradio as gr

# Logging configuration
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("arena_space")

# Optional transformers import
try:
    from transformers import pipeline  # noqa: F401
except Exception:
    pass

# =============================================================================
# PATH SETUP & CORE IMPORTS
# =============================================================================

# Make sure src is on PYTHONPATH
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Game arena core imports
from game_reasoning_arena.arena.games.registry import (
    registry as games_registry
)
from game_reasoning_arena.backends.huggingface_backend import (
    HuggingFaceBackend,
)
from game_reasoning_arena.backends import (
    initialize_llm_registry, LLM_REGISTRY,
)

# UI utilities
from ui.utils import clean_model_name, get_games_from_databases

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Backend availability flag
BACKEND_SYSTEM_AVAILABLE = True

# HuggingFace demo-safe tiny models (CPU friendly)
HUGGINGFACE_MODELS: Dict[str, str] = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "google/flan-t5-small": "google/flan-t5-small",
    "EleutherAI/gpt-neo-125M": "EleutherAI/gpt-neo-125M",
}

# Global registries
GAMES_REGISTRY: Dict[str, Any] = {}

# Database configuration
db_dir = Path(__file__).resolve().parent / "results"

# Leaderboard display columns
LEADERBOARD_COLUMNS = [
    "agent_name", "agent_type", "# game instances", "total rewards",
    # "avg_generation_time (sec)",  # Commented out - needs fixing
    "win-rate", "win vs_random (%)",
]

# =============================================================================
# BACKEND INITIALIZATION
# =============================================================================

# Initialize HuggingFace backend and register models
huggingface_backend = None
if BACKEND_SYSTEM_AVAILABLE:
    try:
        huggingface_backend = HuggingFaceBackend()
        initialize_llm_registry()

        # Register available HuggingFace models
        for model_name in HUGGINGFACE_MODELS.keys():
            if huggingface_backend.is_model_available(model_name):
                registry_key = f"hf_{model_name}"
                LLM_REGISTRY[registry_key] = {
                    "backend": huggingface_backend,
                    "model_name": model_name,
                }
                log.info("Registered HuggingFace model: %s", registry_key)
    except Exception as e:
        log.error("Failed to initialize HuggingFace backend: %s", e)
        huggingface_backend = None

# =============================================================================
# GAMES REGISTRY SETUP
# =============================================================================

# Load available games from the registry
try:
    if games_registry is not None:
        GAMES_REGISTRY = {
            name: cls for name, cls in games_registry._registry.items()
        }
        log.info("Successfully imported full arena - games are playable.")
        log.info(
            "Available games in registry: %s", list(GAMES_REGISTRY.keys())
        )

        # Debug: Check if hex is specifically missing and why
        if "hex" not in GAMES_REGISTRY:
            log.warning("HEX GAME NOT FOUND! Investigating...")
            try:
                # Try to manually import hex components
                from game_reasoning_arena.arena.envs.hex_env import (  # noqa: F401
                    HexEnv
                )
                log.info("✅ HexEnv import successful")
            except Exception as hex_env_error:
                log.error("❌ HexEnv import failed: %s", hex_env_error)

            try:
                import pyspiel
                test_hex = pyspiel.load_game("hex")
                log.info("✅ PySpiel hex game load successful")
            except Exception as pyspiel_error:
                log.error("❌ PySpiel hex game load failed: %s", pyspiel_error)
        else:
            log.info("✅ Hex game found in registry")

    else:
        GAMES_REGISTRY = {}
        log.warning("games_registry is None - using fallback games")
except Exception as e:
    log.warning("Failed to load games registry: %s", e)
    GAMES_REGISTRY = {}


# Debug: Log games available in database files
try:
    db_games = get_games_from_databases()
    log.info("Games found in database files: %s", sorted(list(db_games)))

    if "hex" in db_games:
        log.info("✅ Hex data found in databases")
    else:
        log.warning("❌ No hex data found in databases")

    # Check for registry vs database mismatch
    registry_games = set(GAMES_REGISTRY.keys())
    missing_in_registry = db_games - registry_games
    missing_in_db = registry_games - db_games

    if missing_in_registry:
        log.warning(
            "Games in DB but missing from registry: %s",
            sorted(list(missing_in_registry))
        )
    if missing_in_db:
        log.info(
            "Games in registry but missing from DB: %s",
            sorted(list(missing_in_db))
        )

except Exception as e:
    log.error("Error checking database games: %s", e)


def _get_game_display_mapping() -> Dict[str, str]:
    """
    Build a mapping from internal game keys to their human-friendly
    display names. If the registry is not available or a game has no
    explicit display_name, fall back to a title-cased version of the
    internal key.

    Returns:
        Dict mapping internal game keys to display names
    """
    mapping: Dict[str, str] = {}
    if games_registry is not None and hasattr(games_registry, "_registry"):
        for key, info in games_registry._registry.items():
            if isinstance(info, dict):
                display = info.get("display_name")
            else:
                display = None
            if not display:
                display = key.replace("_", " ").title()
            mapping[key] = display
    return mapping


# =============================================================================
# DATABASE HELPER FUNCTIONS
# =============================================================================

def ensure_results_dir() -> None:
    """Create the results directory if it doesn't exist."""
    db_dir.mkdir(parents=True, exist_ok=True)


def iter_agent_databases() -> Generator[Tuple[str, str, str], None, None]:
    """
    Yield (db_file, agent_type, model_name) for non-random agents.

    Yields:
        Tuple of (database file path, agent type, model name)
    """
    for db_file in find_or_download_db():
        agent_type, model_name = extract_agent_info(db_file)
        if agent_type != "random":
            yield db_file, agent_type, model_name


def find_or_download_db() -> List[str]:
    """
    Return .db files; ensure random_None.db exists with minimal schema.

    Returns:
        List of database file paths
    """
    ensure_results_dir()

    random_db_path = db_dir / "random_None.db"
    if not random_db_path.exists():
        conn = sqlite3.connect(str(random_db_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY,
                    game_name TEXT,
                    player1 TEXT,
                    player2 TEXT,
                    winner INTEGER,
                    timestamp TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    return [str(p) for p in db_dir.glob("*.db")]


def extract_agent_info(filename: str) -> Tuple[str, str]:
    """
    Extract agent type and model name from database filename.

    Args:
        filename: Database filename (e.g., "llm_gpt2.db")

    Returns:
        Tuple of (agent_type, model_name)
    """
    base_name = Path(filename).stem
    parts = base_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], "Unknown"


def get_available_games(include_aggregated: bool = True) -> List[str]:
    """
    Return only games from the registry.

    Args:
        include_aggregated: Whether to include "Aggregated Performance" option

    Returns:
        List of available game names
    """
    if GAMES_REGISTRY:
        game_list = sorted(GAMES_REGISTRY.keys())
    else:
        game_list = ["tic_tac_toe", "kuhn_poker", "connect_four"]
    if include_aggregated:
        game_list.insert(0, "Aggregated Performance")
    return game_list


def extract_illegal_moves_summary() -> pd.DataFrame:
    """
    Extract summary of illegal moves per agent.

    Returns:
        DataFrame with agent names and illegal move counts
    """
    summary = []
    for db_file, agent_type, model_name in iter_agent_databases():
        conn = sqlite3.connect(db_file)
        try:
            df = pd.read_sql_query(
                "SELECT COUNT(*) AS illegal_moves FROM illegal_moves", conn
            )
            count = int(df["illegal_moves"].iloc[0]) if not df.empty else 0
        except Exception:
            count = 0
        finally:
            conn.close()
        clean_name = clean_model_name(model_name)
        summary.append({"agent_name": clean_name, "illegal_moves": count})
    return pd.DataFrame(summary)


# =============================================================================
# PLAYER CONFIGURATION & TYPE DEFINITIONS
# =============================================================================


class PlayerConfigData(TypedDict, total=False):
    """Type definition for player configuration data."""
    player_types: List[str]
    player_type_display: Dict[str, str]
    available_models: List[str]


class GameArenaConfig(TypedDict, total=False):
    """Type definition for game arena configuration."""
    available_games: List[str]
    player_config: PlayerConfigData
    model_info: str
    backend_available: bool


def setup_player_config(
    player_type: str, player_model: str, player_id: str
) -> Dict[str, Any]:
    """
    Map dropdown selection to agent config for the runner.

    Args:
        player_type: Display label for player type
        player_model: Model name if LLM type
        player_id: Player identifier

    Returns:
        Agent configuration dictionary
    """
    # Create a temporary config to get the display-to-key mapping
    temp_config = create_player_config()
    display_to_key = {
        v: k for k, v in
        temp_config["player_config"]["player_type_display"].items()
    }

    # Map display label back to internal key
    internal_key = display_to_key.get(player_type, player_type)

    if internal_key == "random_bot":
        return {"type": "random"}

    if internal_key == "human":
        return {"type": "human"}

    if (
        internal_key
        and (
            internal_key.startswith("llm_")
            or internal_key.startswith("hf_")
        )
    ):
        model_id = internal_key.split("_", 1)[1]
        if BACKEND_SYSTEM_AVAILABLE and model_id in HUGGINGFACE_MODELS:
            return {"type": "llm", "model": model_id}

    if (
        internal_key == "llm"
        and player_model in HUGGINGFACE_MODELS
        and BACKEND_SYSTEM_AVAILABLE
    ):
        return {"type": "llm", "model": player_model}

    return {"type": "random"}


def create_player_config(include_aggregated: bool = False) -> GameArenaConfig:
    """
    Create player and game configuration for the arena.

    Args:
        include_aggregated: Whether to include aggregated stats option

    Returns:
        Complete game arena configuration
    """
    # Internal names for arena dropdown
    available_keys = get_available_games(include_aggregated=include_aggregated)

    # Map internal names to display names
    key_to_display = _get_game_display_mapping()
    mapped_games = [
        key_to_display.get(key, key.replace("_", " ").title())
        for key in available_keys
    ]
    # Deduplicate while preserving order
    seen = set()
    available_games = []
    for name in mapped_games:
        if name not in seen:
            available_games.append(name)
            seen.add(name)

    # Define available player types
    player_types = ["human", "random_bot"]
    player_type_display = {
        "human": "Human Player",
        "random_bot": "Random Bot"
    }

    # Add HuggingFace models if backend is available
    if BACKEND_SYSTEM_AVAILABLE:
        for model_key in HUGGINGFACE_MODELS.keys():
            key = f"hf_{model_key}"
            player_types.append(key)
            # Clean up model names for display
            tag = model_key.split("/")[-1]
            if tag == "gpt2":
                display_name = "GPT-2"
            elif tag == "distilgpt2":
                display_name = "DistilGPT-2"
            elif tag == "flan-t5-small":
                display_name = "FLAN-T5 Small"
            elif tag == "gpt-neo-125M":
                display_name = "GPT-Neo 125M"
            else:
                # Fallback for any new models
                display_name = tag.replace("-", " ").title()
            player_type_display[key] = display_name

    all_models = list(HUGGINGFACE_MODELS.keys())
    model_info = (
        "HuggingFace transformer models integrated with backend system."
        if BACKEND_SYSTEM_AVAILABLE
        else "Backend system not available - limited functionality."
    )

    # Build display→key mapping for games
    display_to_key = {}
    for key in available_keys:
        display = key_to_display.get(key, key.replace("_", " ").title())
        if display not in display_to_key:
            display_to_key[display] = key

    return {
        "available_games": available_games,
        "game_display_to_key": display_to_key,
        "player_config": {
            "player_types": player_types,
            "player_type_display": player_type_display,
            "available_models": all_models,
        },
        "model_info": model_info,
        "backend_available": BACKEND_SYSTEM_AVAILABLE,
    }


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def play_game(
    game_name: str,
    player1_type: str,
    player2_type: str,
    rounds: int = 1,
    seed: int | None = None,
) -> str:
    """
    Execute a complete game simulation between two players.

    Args:
        game_name: Name of the game to play
        player1_type: Type of player 1 (display name like "Human Player", "GPT-2")
        player2_type: Type of player 2 (display name like "Human Player", "GPT-2")
        rounds: Number of rounds to play
        seed: Random seed for reproducibility

    Returns:
        Game result log as string
    """
    if game_name == "No Games Found":
        return "No games available. Please add game databases."

    log.info(
        "Starting game: %s | P1=%s P2=%s rounds=%d",
        game_name,
        player1_type,
        player2_type,
        rounds,
    )

    # Map human‑friendly game name back to internal key if needed
    config = create_player_config()
    if ("game_display_to_key" in config and
            game_name in config["game_display_to_key"]):
        game_name = config["game_display_to_key"][game_name]

    # Map display labels for player types back to keys
    display_to_key = {
        v: k for k, v in config["player_config"]["player_type_display"].items()
    }

    # Extract internal keys and models
    p1_key = display_to_key.get(player1_type, player1_type)
    p2_key = display_to_key.get(player2_type, player2_type)

    player1_model = None
    player2_model = None
    if p1_key.startswith("hf_"):
        player1_model = p1_key.split("_", 1)[1]
    if p2_key.startswith("hf_"):
        player2_model = p2_key.split("_", 1)[1]

    import time
    try:
        from ui.gradio_config_generator import (
            run_game_with_existing_infrastructure,
        )
        # Use a random seed if not provided
        if seed is None:
            seed = int(time.time() * 1000) % (2**31 - 1)
        result = run_game_with_existing_infrastructure(
            game_name=game_name,
            player1_type=p1_key,
            player2_type=p2_key,
            player1_model=player1_model,
            player2_model=player2_model,
            rounds=rounds,
            seed=seed,
        )
        return result
    except Exception as e:
        return f"Error during game simulation: {e}"


# =============================================================================
# LEADERBOARD & ANALYTICS
# =============================================================================

def extract_leaderboard_stats(game_name: str) -> pd.DataFrame:
    """
    Extract leaderboard statistics for a specific game or all games.

    Args:
        game_name: Name of the game or "Aggregated Performance"

    Returns:
        DataFrame with leaderboard statistics
    """
    all_stats = []
    for db_file, agent_type, model_name in iter_agent_databases():
        conn = sqlite3.connect(db_file)
        try:
            if game_name == "Aggregated Performance":
                # Get totals across all games in this DB
                df = pd.read_sql_query(
                    "SELECT COUNT(*) AS total_games, SUM(reward) AS total_rewards "
                    "FROM game_results",
                    conn,
                )
                # Each row represents a game instance
                games_played = int(df["total_games"].iloc[0] or 0)
                # avg_time = conn.execute(
                #     "SELECT AVG(generation_time) FROM moves"
                # ).fetchone()[0] or 0 # to fix later
                wins_vs_random = conn.execute(
                    "SELECT COUNT(*) FROM game_results "
                    "WHERE opponent = 'random_None' AND reward > 0",
                ).fetchone()[0] or 0
                total_vs_random = conn.execute(
                    "SELECT COUNT(*) FROM game_results "
                    "WHERE opponent = 'random_None'",
                ).fetchone()[0] or 0
            else:
                # Filter by the selected game
                df = pd.read_sql_query(
                    "SELECT COUNT(*) AS total_games, SUM(reward) AS total_rewards "
                    "FROM game_results WHERE game_name = ?",
                    conn,
                    params=(game_name,),
                )
                # Each row represents a game instance
                games_played = int(df["total_games"].iloc[0] or 0)
                # avg_time = conn.execute(
                #     "SELECT AVG(generation_time) FROM moves "
                #     "WHERE game_name = ?", (game_name,),
                # ).fetchone()[0] or 0
                wins_vs_random = conn.execute(
                    "SELECT COUNT(*) FROM game_results "
                    "WHERE opponent = 'random_None' AND reward > 0 "
                    "AND game_name = ?",
                    (game_name,),
                ).fetchone()[0] or 0
                total_vs_random = conn.execute(
                    "SELECT COUNT(*) FROM game_results "
                    "WHERE opponent = 'random_None' AND game_name = ?",
                    (game_name,),
                ).fetchone()[0] or 0

            # If there were no results for this game, df will be empty or NaNs.
            if df.empty or df["total_games"].iloc[0] is None:
                games_played = 0
                total_rewards = 0.0
            else:
                total_rewards = float(df["total_rewards"].iloc[0] or 0) / 2.0

            vs_random_rate = (
                (wins_vs_random / total_vs_random) * 100.0
                if total_vs_random > 0
                else 0.0
            )

            # Build a single-row DataFrame for this agent
            row = {
                "agent_name": clean_model_name(model_name),
                "agent_type": agent_type,
                "# game instances": games_played,
                "total rewards": total_rewards,
                # "avg_generation_time (sec)": round(float(avg_time), 3),
                "win-rate": round(vs_random_rate, 2),
                "win vs_random (%)": round(vs_random_rate, 2),
            }
            all_stats.append(pd.DataFrame([row]))
        finally:
            conn.close()

    # Concatenate all rows; if all_stats is empty, return an empty DataFrame
    # with columns.
    if not all_stats:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    leaderboard_df = pd.concat(all_stats, ignore_index=True)
    return leaderboard_df[LEADERBOARD_COLUMNS]


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def create_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    horizontal: bool = False,
) -> gr.BarPlot:
    """
    Create a bar plot with optional horizontal orientation.

    Args:
        data: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        horizontal: Whether to create horizontal bars

    Returns:
        Gradio BarPlot component
    """
    if horizontal:
        # Swap x and y for horizontal bars
        return gr.BarPlot(
            value=data,
            x=y_col,  # metrics on x-axis
            y=x_col,  # model names on y-axis
            title=title,
            x_label=y_label,  # swap labels too
            y_label=x_label,
        )
    else:
        return gr.BarPlot(
            value=data,
            x=x_col,
            y=y_col,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )


# =============================================================================
# FILE UPLOAD HANDLERS
# =============================================================================

def handle_db_upload(files: list[gr.File]) -> str:
    """
    Handle upload of database files to the results directory.

    Args:
        files: List of uploaded files

    Returns:
        Status message about upload success
    """
    ensure_results_dir()
    saved = []
    for f in files or []:
        dest = db_dir / Path(f.name).name
        Path(f.name).replace(dest)
        saved.append(dest.name)
    return (
        f"Uploaded: {', '.join(saved)}" if saved else "No files uploaded."
    )


# =============================================================================
# GRADIO USER INTERFACE
# =============================================================================

"""
This section defines the complete Gradio web interface with the following tabs:
1. Game Arena: Interactive gameplay between humans and AI
2. Leaderboard: Performance statistics and rankings
3. Metrics Dashboard: Visual analytics and charts
4. Analysis of LLM Reasoning: Illegal moves and behavior analysis
5. About: Documentation and information

The interface supports:
- Real-time human vs AI gameplay
- Automatic AI move processing
- Dynamic dropdown population
- State management for interactive games
- File upload for database results
- Interactive visualizations
"""

with gr.Blocks() as interface:
    # =========================================================================
    # TAB 1: GAME ARENA
    # =========================================================================

    with gr.Tab("Game Arena"):
        config = create_player_config(include_aggregated=False)

        # Header and introduction
        gr.Markdown("# Interactive Game Reasoning Arena")
        gr.Markdown("Play games against LLMs, a random bot or watch LLMs compete!")
        gr.Markdown(
            f"> **🤖 Available AI Players**: {config['model_info']}\n"
            "> Local transformer models run with Hugging Face transformers. "
            "No API tokens required!\n\n"
            "> **⚠️ Note on Reasoning Quality**: The available models are "
            "relatively basic (GPT-2, DistilGPT-2, etc.) and may produce "
            "limited or nonsensical reasoning. They are suitable for "
            "demonstration purposes but don't expect sophisticated "
            "strategic thinking or coherent explanations."
        )

        # Game selection and configuration
        with gr.Row():
            game_dropdown = gr.Dropdown(
                choices=config["available_games"],
                label="Select a Game",
                value=(
                    config["available_games"][0]
                    if config["available_games"]
                    else "No Games Found"
                ),
            )
            rounds_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                label="Number of Rounds",
            )

        def player_selector_block(label: str):
            """Create player selection UI block."""
            gr.Markdown(f"### {label}")
            # Create display choices (what user sees)
            display_choices = [
                config["player_config"]["player_type_display"][key]
                for key in config["player_config"]["player_types"]
            ]
            # Set default to first display choice
            default_choice = display_choices[0] if display_choices else None

            dd_type = gr.Dropdown(
                choices=display_choices,
                label=f"{label}",  # Just "Player 0" or "Player 1"
                value=default_choice,
            )
            return dd_type

        # Player configuration
        with gr.Row():
            p1_type = player_selector_block("Player 0")
            p2_type = player_selector_block("Player 1")

        # Validation error message
        validation_error = gr.Markdown(visible=False)

        # Game state management
        game_state = gr.State(value=None)
        human_choices_p0 = gr.State([])
        human_choices_p1 = gr.State([])

        # Interactive game components (initially hidden)
        with gr.Column(visible=False) as interactive_panel:
            gr.Markdown("## Interactive Game")

            with gr.Row():
                with gr.Column(scale=2):
                    board_display = gr.Textbox(
                        label="Game Board",
                        lines=10,
                        placeholder="Board state will appear here...",
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    # Human move controls
                    gr.Markdown("### Your Move")

                    # Player 0 move selection
                    human_move_p0 = gr.Dropdown(
                        choices=[],
                        label="Your move (Player 0)",
                        visible=False,
                        interactive=True,
                    )

                    # Player 1 move selection
                    human_move_p1 = gr.Dropdown(
                        choices=[],
                        label="Your move (Player 1)",
                        visible=False,
                        interactive=True,
                    )

                    submit_btn = gr.Button(
                        "Submit Move",
                        variant="primary",
                        visible=False
                    )

                    reset_game_btn = gr.Button(
                        "Reset Game",
                        visible=False
                    )

        # Game control buttons
        play_button = gr.Button("🎮 Start Game", variant="primary")
        start_btn = gr.Button(
            "🎯 Start Interactive Game",
            variant="secondary",
            visible=False
        )

        # Game output display
        game_output = gr.Textbox(
            label="Game Log",
            lines=20,
            placeholder="Game results will appear here...",
        )

        def check_for_human_players(p1_type, p2_type):
            """Show/hide interactive controls based on player types."""
            # Map display labels back to internal keys
            display_to_key = {
                v: k for k, v in
                config["player_config"]["player_type_display"].items()
            }
            p1_key = display_to_key.get(p1_type, p1_type)
            p2_key = display_to_key.get(p2_type, p2_type)

            has_human = (p1_key == "human" or p2_key == "human")
            return (
                gr.update(visible=has_human),  # interactive_panel
                gr.update(visible=has_human),  # start_btn
                gr.update(visible=not has_human),  # play_button (single-shot)
            )

        def validate_player_selection(p1_type, p2_type):
            """Validate players and update dropdown choices accordingly."""
            # Map display labels back to internal keys
            display_to_key = {
                v: k for k, v in
                config["player_config"]["player_type_display"].items()
            }
            p1_key = display_to_key.get(p1_type, p1_type)
            p2_key = display_to_key.get(p2_type, p2_type)

            # Check if both players are human
            both_human = (p1_key == "human" and p2_key == "human")

            # Create display choices for dropdowns
            display_choices = [
                config["player_config"]["player_type_display"][key]
                for key in config["player_config"]["player_types"]
            ]

            # Filter choices based on current selection
            p1_choices = display_choices.copy()
            p2_choices = display_choices.copy()

            # If Player 0 is human, remove "Human Player" from Player 1 choices
            if p1_key == "human":
                human_display = config["player_config"][
                    "player_type_display"
                ]["human"]
                if human_display in p2_choices:
                    p2_choices.remove(human_display)

            # If Player 1 is human, remove "Human Player" from Player 0 choices
            if p2_key == "human":
                human_display = config["player_config"][
                    "player_type_display"
                ]["human"]
                if human_display in p1_choices:
                    p1_choices.remove(human_display)

            # Generate error message if both are human
            error_msg = ""
            if both_human:
                error_msg = ("⚠️ **Cannot have Human vs Human games!** "
                             "Please select an AI player for one side.")

            # Return updated dropdown choices and error message
            return (
                gr.update(choices=p1_choices),  # p1_type dropdown
                gr.update(choices=p2_choices),  # p2_type dropdown
                error_msg  # validation error message
            )

        # Update UI when player types change
        def update_validation_and_ui(p1_type, p2_type):
            """Update validation, player choices, and UI visibility."""
            # First update validation and dropdowns
            p1_update, p2_update, error_msg = validate_player_selection(
                p1_type, p2_type
            )

            # Then update UI visibility
            vis_update = check_for_human_players(p1_type, p2_type)

            # Show/hide error message
            error_visible = bool(error_msg)
            error_update = gr.update(
                value=error_msg,
                visible=error_visible
            )

            return (
                p1_update,      # p1_type choices
                p2_update,      # p2_type choices
                error_update,   # validation_error
                vis_update[0],  # interactive_panel
                vis_update[1],  # start_btn
                vis_update[2],  # play_button
            )

        # Wire up change handlers for both player dropdowns
        for player_dropdown in [p1_type, p2_type]:
            player_dropdown.change(
                update_validation_and_ui,
                inputs=[p1_type, p2_type],
                outputs=[
                    p1_type, p2_type, validation_error,
                    interactive_panel, start_btn, play_button
                ],
            )

        # Standard single-shot game
        def start_game_with_validation(
            game_name, p1_type, p2_type, rounds
        ):
            """Start game only if validation passes."""
            # Map display labels back to internal keys
            display_to_key = {
                v: k for k, v in
                config["player_config"]["player_type_display"].items()
            }
            p1_key = display_to_key.get(p1_type, p1_type)
            p2_key = display_to_key.get(p2_type, p2_type)

            # Check if both players are human
            if p1_key == "human" and p2_key == "human":
                return ("⚠️ **Cannot start Human vs Human game!** "
                        "Please select an AI player for one side.")

            # If validation passes, start the game
            return play_game(game_name, p1_type, p2_type, rounds)

        play_button.click(
            start_game_with_validation,
            inputs=[
                game_dropdown,
                p1_type,
                p2_type,
                rounds_slider,
            ],
            outputs=[game_output],
        )

        # Interactive game functions
        def start_interactive_game(
            game_name, p1_type, p2_type, rounds
        ):
            """Initialize an interactive game session."""
            try:
                # Map display labels back to internal keys
                display_to_key = {
                    v: k for k, v in
                    config["player_config"]["player_type_display"].items()
                }
                p1_key = display_to_key.get(p1_type, p1_type)
                p2_key = display_to_key.get(p2_type, p2_type)

                # Check if both players are human
                if p1_key == "human" and p2_key == "human":
                    return (
                        None,   # game_state
                        [],     # human_choices_p0
                        [],     # human_choices_p1
                        ("⚠️ **Cannot start Human vs Human game!** "
                         "Please select an AI player for one side."),
                        gr.update(choices=[], visible=False),  # human_move_p0
                        gr.update(choices=[], visible=False),  # human_move_p1
                        gr.update(visible=False),              # submit_btn
                        gr.update(visible=False),              # reset_game_btn
                    )

                from ui.gradio_config_generator import start_game_interactive
                import time

                # Map display game name back to internal key if needed
                game_display_to_key = config.get("game_display_to_key", {})
                internal_game = game_display_to_key.get(game_name, game_name)

                # Extract model from player type if it's an LLM
                p1_model = None
                p2_model = None
                if p1_key.startswith("hf_"):
                    p1_model = p1_key.split("_", 1)[1]
                if p2_key.startswith("hf_"):
                    p2_model = p2_key.split("_", 1)[1]

                # Use timestamp as seed
                seed = int(time.time() * 1000) % (2**31 - 1)

                log, state, legal_p0, legal_p1 = start_game_interactive(
                    game_name=internal_game,
                    player1_type=p1_key,
                    player2_type=p2_key,
                    player1_model=p1_model,
                    player2_model=p2_model,
                    rounds=rounds,
                    seed=seed,
                )

                # Store choices in state for reliable mapping
                # [(action_id, label), ...] from _legal_actions_with_labels()
                p0_choices = legal_p0
                p1_choices = legal_p1

                # Create Gradio dropdown choices: user sees OpenSpiel action
                # labels, selects action IDs
                p0_dropdown_choices = [
                    (label, action_id) for action_id, label in p0_choices
                ]
                p1_dropdown_choices = [
                    (label, action_id) for action_id, label in p1_choices
                ]

                # Show/hide dropdowns based on whether each player is human
                p0_is_human = (p1_key == "human")
                p1_is_human = (p2_key == "human")

                return (
                    state,  # game_state
                    p0_choices,  # human_choices_p0
                    p1_choices,  # human_choices_p1
                    log,    # board_display
                    gr.update(
                        choices=p0_dropdown_choices,
                        visible=p0_is_human,
                        value=None
                    ),  # human_move_p0
                    gr.update(
                        choices=p1_dropdown_choices,
                        visible=p1_is_human,
                        value=None
                    ),  # human_move_p1
                    gr.update(visible=True),  # submit_btn
                    gr.update(visible=True),  # reset_game_btn
                )
            except Exception as e:
                return (
                    None,   # game_state
                    [],     # human_choices_p0
                    [],     # human_choices_p1
                    f"Error starting interactive game: {e}",  # board_display
                    gr.update(choices=[], visible=False),     # human_move_p0
                    gr.update(choices=[], visible=False),     # human_move_p1
                    gr.update(visible=False),                 # submit_btn
                    gr.update(visible=False),                 # reset_game_btn
                )

        def submit_human_move_handler(p0_action, p1_action, state, choices_p0, choices_p1):
            """Process human moves and advance the game."""
            try:
                from ui.gradio_config_generator import submit_human_move

                if not state:
                    return (
                        state, [], [], "No game running.",
                        gr.update(choices=[], visible=False),
                        gr.update(choices=[], visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )

                # The submit_human_move function already handles:
                # 1. Taking human actions for human players
                # 2. Computing AI actions for AI players
                # 3. Advancing the game with both actions
                # 4. Returning the next legal moves
                log_append, new_state, next_p0, next_p1 = submit_human_move(
                    action_p0=p0_action,  # None if P0 is AI, action_id if P0 is human
                    action_p1=p1_action,  # None if P1 is AI, action_id if P1 is human
                    state=state,
                )

                # next_p0 and next_p1 are from _legal_actions_with_labels()
                # Format: [(action_id, label), ...] where label comes from OpenSpiel
                new_choices_p0 = next_p0
                new_choices_p1 = next_p1

                # Create Gradio dropdown choices: user sees OpenSpiel labels, selects action IDs
                p0_dropdown_choices = [(label, action_id) for action_id, label in new_choices_p0]
                p1_dropdown_choices = [(label, action_id) for action_id, label in new_choices_p1]

                # Check if game is finished
                game_over = (new_state.get("terminated", False) or
                           new_state.get("truncated", False))

                return (
                    new_state,  # game_state
                    new_choices_p0,  # human_choices_p0
                    new_choices_p1,  # human_choices_p1
                    log_append,  # board_display (append to current)
                    gr.update(choices=p0_dropdown_choices, visible=len(p0_dropdown_choices) > 0 and not game_over, value=None),
                    gr.update(choices=p1_dropdown_choices, visible=len(p1_dropdown_choices) > 0 and not game_over, value=None),
                    gr.update(visible=not game_over),  # submit_btn
                    gr.update(visible=True),           # reset_game_btn
                )
            except Exception as e:
                return (
                    state, choices_p0, choices_p1, f"Error processing move: {e}",
                    gr.update(), gr.update(), gr.update(), gr.update()
                )

        def reset_interactive_game():
            """Reset the interactive game state."""
            return (
                None,  # game_state
                [],    # human_choices_p0
                [],    # human_choices_p1
                "Game reset. Click 'Start Interactive Game' to begin a new game.",  # board_display
                gr.update(choices=[], visible=False),  # human_move_p0
                gr.update(choices=[], visible=False),  # human_move_p1
                gr.update(visible=False),              # submit_btn
                gr.update(visible=False),              # reset_game_btn
            )

        # Wire up interactive game handlers
        start_btn.click(
            start_interactive_game,
            inputs=[game_dropdown, p1_type, p2_type, rounds_slider],
            outputs=[game_state, human_choices_p0, human_choices_p1, board_display, human_move_p0, human_move_p1, submit_btn, reset_game_btn],
        )

        submit_btn.click(
            submit_human_move_handler,
            inputs=[human_move_p0, human_move_p1, game_state, human_choices_p0, human_choices_p1],
            outputs=[game_state, human_choices_p0, human_choices_p1, board_display, human_move_p0, human_move_p1, submit_btn, reset_game_btn],
        )

        reset_game_btn.click(
            reset_interactive_game,
            outputs=[game_state, human_choices_p0, human_choices_p1, board_display, human_move_p0, human_move_p1, submit_btn, reset_game_btn],
        )

    with gr.Tab("Leaderboard"):
        gr.Markdown(
            "# LLM Model Leaderboard\n"
            "Track performance across different games!"
        )
        # Use the same display logic as Game Arena
        leaderboard_config = create_player_config(include_aggregated=True)

        # Debug: Log the available games for troubleshooting
        log.info(
            "Leaderboard available games: %s",
            leaderboard_config["available_games"]
        )

        leaderboard_game_dropdown = gr.Dropdown(
            choices=leaderboard_config["available_games"],
            label="Select Game",
            value=(
                leaderboard_config["available_games"][0]
                if leaderboard_config["available_games"]
                else "No Games Found"
            ),
        )
        leaderboard_table = gr.Dataframe(
            value=extract_leaderboard_stats("Aggregated Performance"),
            headers=LEADERBOARD_COLUMNS,
            interactive=False,
        )
        refresh_btn = gr.Button("🔄 Refresh")

        def _update_leaderboard(game: str) -> pd.DataFrame:
            # Map display name back to internal key
            display_to_key = leaderboard_config.get("game_display_to_key", {})
            internal_game = display_to_key.get(game, game)
            return extract_leaderboard_stats(internal_game)

        leaderboard_game_dropdown.change(
            _update_leaderboard,
            inputs=[leaderboard_game_dropdown],
            outputs=[leaderboard_table],
        )
        refresh_btn.click(
            _update_leaderboard,
            inputs=[leaderboard_game_dropdown],
            outputs=[leaderboard_table],
        )

        gr.Markdown("### Upload new `.db` result files")
        db_files = gr.Files(file_count="multiple", file_types=[".db"])
        upload_btn = gr.Button("⬆️ Upload to results/")
        upload_status = gr.Markdown()

        upload_btn.click(
            handle_db_upload, inputs=[db_files], outputs=[upload_status]
        )

    with gr.Tab("Metrics Dashboard"):
        gr.Markdown(
            "# 📊 Metrics Dashboard\n"
            "Visual summaries of LLM performance across games."
        )
        metrics_df = extract_leaderboard_stats("Aggregated Performance")

        with gr.Row():
            create_bar_plot(
                data=metrics_df,
                x_col="agent_name",
                y_col="win vs_random (%)",
                title="Win Rate vs Random Bot",
                x_label="LLM Model",
                y_label="Win Rate (%)",
                horizontal=True,
            )

        with gr.Row():
            # Commented out - avg_generation_time needs fixing
            # create_bar_plot(
            #     data=metrics_df,
            #     x_col="agent_name",
            #     y_col="avg_generation_time (sec)",
            #     title="Average Generation Time",
            #     x_label="LLM Model",
            #     y_label="Time (sec)",
            # )
            pass

        with gr.Row():
            gr.Dataframe(
                value=metrics_df,
                label="Performance Summary",
                interactive=False,
            )

    with gr.Tab("Analysis of LLM Reasoning"):
        gr.Markdown(
            "# 🧠 Analysis of LLM Reasoning\n"
            "Insights into move legality and decision behavior."
        )
        illegal_df = extract_illegal_moves_summary()

        with gr.Row():
            create_bar_plot(
                data=illegal_df,
                x_col="agent_name",
                y_col="illegal_moves",
                title="Illegal Moves by Model",
                x_label="LLM Model",
                y_label="# of Illegal Moves",
                horizontal=True,
            )

        with gr.Row():
            gr.Dataframe(
                value=illegal_df,
                label="Illegal Move Summary",
                interactive=False,
            )

    with gr.Tab("About"):
        gr.Markdown(
            """
            # About Game Reasoning Arena

            This app analyzes and visualizes LLM performance in games.

            - **Game Arena**: Play games vs. LLMs or watch LLM vs. LLM
            - **Leaderboard**: Performance statistics across games
            - **Metrics Dashboard**: Visual summaries
            - **Reasoning Analysis**: Illegal moves & behavior

            **Data**: SQLite databases in `/results/`.
            """
        )

# Local run only. On Spaces, the runtime will serve `interface` automatically.
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=None, show_api=False)