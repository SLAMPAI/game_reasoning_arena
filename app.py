#!/usr/bin/env python3
"""
Game Reasoning Arena — Hugging Face Spaces Gradio App

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
"""

from __future__ import annotations

import sqlite3

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Generator

import pandas as pd
import gradio as gr

# Logging (optional)
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("arena_space")

# Optional transformers import (only needed if your backend uses it here)
try:
    from transformers import pipeline  # noqa: F401
except Exception:
    pass

# Make sure src is on PYTHONPATH
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Try to import game registry
try:
    from src.game_reasoning_arena.arena.games.registry import (
        registry as games_registry
    )
except Exception as e:
    log.warning("Game registry not available: %s", e)
    games_registry = None

# Optional: import backend & LLM registry
try:
    from src.game_reasoning_arena.backends.huggingface_backend import (
        HuggingFaceBackend,
    )
    from src.game_reasoning_arena.backends import (
        initialize_llm_registry, LLM_REGISTRY,
    )
    BACKEND_SYSTEM_AVAILABLE = True
    log.info("Backend system available - using proper LLM infrastructure.")
except Exception as e:
    BACKEND_SYSTEM_AVAILABLE = False
    log.warning("Backend system not available: %s", e)

# -----------------------------------------------------------------------------
# Config & constants
# -----------------------------------------------------------------------------

# HF demo-safe tiny models (CPU friendly)
HUGGINGFACE_MODELS: Dict[str, str] = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "google/flan-t5-small": "google/flan-t5-small",
    "EleutherAI/gpt-neo-125M": "EleutherAI/gpt-neo-125M",
}

GAMES_REGISTRY: Dict[str, Any] = {}
db_dir = Path(__file__).resolve().parent / "scripts" / "results"

LEADERBOARD_COLUMNS = [
    "agent_name", "agent_type", "# games", "total rewards",
    "avg_generation_time (sec)", "win-rate", "win vs_random (%)",
]

# -----------------------------------------------------------------------------
# Init backend + register models (optional)
# -----------------------------------------------------------------------------

huggingface_backend = None
if BACKEND_SYSTEM_AVAILABLE:
    try:
        huggingface_backend = HuggingFaceBackend()
        initialize_llm_registry()

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

# -----------------------------------------------------------------------------
# Load games registry
# -----------------------------------------------------------------------------

try:
    if games_registry is not None:
        GAMES_REGISTRY = {
            name: cls for name, cls in games_registry._registry.items()
        }
        log.info("Successfully imported full arena - games are playable.")
    else:
        GAMES_REGISTRY = {}
except Exception as e:
    log.warning("Failed to load games registry: %s", e)
    GAMES_REGISTRY = {}

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------


def ensure_results_dir() -> None:
    db_dir.mkdir(parents=True, exist_ok=True)


def iter_agent_databases() -> Generator[Tuple[str, str, str], None, None]:
    """Yield (db_file, agent_type, model_name) for non-random agents."""
    for db_file in find_or_download_db():
        agent_type, model_name = extract_agent_info(db_file)
        if agent_type != "random":
            yield db_file, agent_type, model_name


def find_or_download_db() -> List[str]:
    """Return .db files; ensure random_None.db exists with minimal schema."""
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
    base_name = Path(filename).stem
    parts = base_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], "Unknown"


def get_available_games(include_aggregated: bool = True) -> List[str]:
    """Union of games seen in DBs and in registry."""
    game_names = set()

    # From DBs
    for db_file in find_or_download_db():
        conn = sqlite3.connect(db_file)
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT game_name FROM moves", conn
            )
            game_names.update(df["game_name"].tolist())
        except Exception:
            pass
        finally:
            conn.close()

    # From registry
    if GAMES_REGISTRY:
        game_names.update(GAMES_REGISTRY.keys())

    if not game_names:
        game_names.update(["tic_tac_toe", "kuhn_poker", "connect_four"])

    game_list = sorted(game_names)
    if include_aggregated:
        game_list.insert(0, "Aggregated Performance")
    return game_list


def extract_illegal_moves_summary() -> pd.DataFrame:
    """# illegal moves per agent."""
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
        summary.append({"agent_name": model_name, "illegal_moves": count})
    return pd.DataFrame(summary)

# -----------------------------------------------------------------------------
# Player config
# -----------------------------------------------------------------------------


class PlayerConfigData(gr.TypedDict, total=False):
    player_types: List[str]
    player_type_display: Dict[str, str]
    available_models: List[str]


class GameArenaConfig(gr.TypedDict, total=False):
    available_games: List[str]
    player_config: PlayerConfigData
    model_info: str
    backend_available: bool


def setup_player_config(
    player_type: str, player_model: str, player_id: str
) -> Dict[str, Any]:
    """Map dropdown selection to agent config for the runner."""
    if player_type == "random_bot":
        return {"type": "random"}

    if (
        player_type
        and (
            player_type.startswith("llm_")
            or player_type.startswith("hf_")
        )
    ):
        model_id = player_type.split("_", 1)[1]
        if BACKEND_SYSTEM_AVAILABLE and model_id in HUGGINGFACE_MODELS:
            return {"type": "llm", "model": model_id}

    if (
        player_type == "llm"
        and player_model in HUGGINGFACE_MODELS
        and BACKEND_SYSTEM_AVAILABLE
    ):
        return {"type": "llm", "model": player_model}

    return {"type": "random"}


def create_player_config() -> GameArenaConfig:
    available_games = get_available_games(include_aggregated=False)

    # Collect models seen in DBs (for charts/labels)
    database_models = [model for _, _, model in iter_agent_databases()]

    player_types = ["random_bot"]
    player_type_display = {"random_bot": "Random Bot"}

    if BACKEND_SYSTEM_AVAILABLE:
        for model_key in HUGGINGFACE_MODELS.keys():
            key = f"hf_{model_key}"
            player_types.append(key)
            tag = model_key.split("/")[-1]
            player_type_display[key] = f"HuggingFace: {tag}"

    all_models = list(HUGGINGFACE_MODELS.keys()) + database_models

    model_info = (
        "HuggingFace transformer models integrated with backend system."
        if BACKEND_SYSTEM_AVAILABLE
        else "Backend system not available - limited functionality."
    )

    return {
        "available_games": available_games,
        "player_config": {
            "player_types": player_types,
            "player_type_display": player_type_display,
            "available_models": all_models,
        },
        "model_info": model_info,
        "backend_available": BACKEND_SYSTEM_AVAILABLE,
    }

# -----------------------------------------------------------------------------
# Main game entry
# -----------------------------------------------------------------------------


def play_game(
    game_name: str,
    player1_type: str,
    player2_type: str,
    player1_model: str | None = None,
    player2_model: str | None = None,
    rounds: int = 1,
) -> str:
    if game_name == "No Games Found":
        return "No games available. Please add game databases."

    log.info(
        "Starting game: %s | P1=%s(%s) P2=%s(%s) rounds=%d",
        game_name,
        player1_type,
        player1_model,
        player2_type,
        player2_model,
        rounds,
    )

    # Gradio passes display labels sometimes—map back to keys
    config = create_player_config()
    display_to_key = {
        v: k for k, v in config["player_config"]["player_type_display"].items()
    }
    if player1_type in display_to_key:
        player1_type = display_to_key[player1_type]
    if player2_type in display_to_key:
        player2_type = display_to_key[player2_type]

    try:
        # IMPORTANT: rename your local folder to 'ui/'
        from ui.gradio_config_generator import (
            run_game_with_existing_infrastructure,
        )

        result = run_game_with_existing_infrastructure(
            game_name=game_name,
            player1_type=player1_type,
            player2_type=player2_type,
            player1_model=player1_model,
            player2_model=player2_model,
            rounds=rounds,
            seed=42,
        )
        return result
    except Exception as e:
        return f"Error during game simulation: {e}"


def extract_leaderboard_stats(game_name: str) -> pd.DataFrame:
    all_stats = []

    for db_file, agent_type, model_name in iter_agent_databases():
        conn = sqlite3.connect(db_file)
        try:
            if game_name == "Aggregated Performance":
                q = (
                    "SELECT COUNT(DISTINCT episode) AS games_played, "
                    "SUM(reward) AS total_rewards FROM game_results"
                )
                df = pd.read_sql_query(q, conn)
                avg_time = conn.execute(
                    "SELECT AVG(generation_time) FROM moves "
                    "WHERE game_name = 'kuhn_poker'"
                ).fetchone()[0] or 0
            else:
                q = (
                    "SELECT COUNT(DISTINCT episode) AS games_played, "
                    "SUM(reward) AS total_rewards "
                    "FROM game_results WHERE game_name = ?"
                )
                df = pd.read_sql_query(q, conn, params=(game_name,))
                avg_time = conn.execute(
                 "SELECT AVG(generation_time) FROM moves WHERE game_name = ?",
                    (game_name,),
                ).fetchone()[0] or 0

            df["total_rewards"] = (
                df["total_rewards"].fillna(0).astype(float) / 2
            )
            avg_time = round(float(avg_time), 3)

            wins_vs_random = conn.execute(
                "SELECT COUNT(*) FROM game_results "
                "WHERE opponent = 'random_None' AND reward > 0"
            ).fetchone()[0] or 0
            total_vs_random = conn.execute(
                "SELECT COUNT(*) FROM game_results "
                "WHERE opponent = 'random_None'"
            ).fetchone()[0] or 0
            vs_random_rate = (
                wins_vs_random / total_vs_random * 100
                if total_vs_random > 0
                else 0
            )

            df.insert(0, "agent_name", model_name)
            df.insert(1, "agent_type", agent_type)
            df["avg_generation_time (sec)"] = avg_time
            df["win vs_random (%)"] = round(vs_random_rate, 2)
            # Optional: derive win-rate from rewards/games if you wish
            df["# games"] = df["games_played"]
            df["win-rate"] = df["win vs_random (%)"]  # simple proxy for table

            all_stats.append(df)
        finally:
            conn.close()

    leaderboard_df = (
        pd.concat(all_stats, ignore_index=True)
        if all_stats
        else pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    )
    # Reorder columns to match LEADERBOARD_COLUMNS (ignore missing)
    cols = [c for c in LEADERBOARD_COLUMNS if c in leaderboard_df.columns]
    leaderboard_df = leaderboard_df[cols]
    return leaderboard_df

# -----------------------------------------------------------------------------
# Simple plotting helpers
# -----------------------------------------------------------------------------


def create_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> gr.BarPlot:
    return gr.BarPlot(
        value=data,
        x=x_col,
        y=y_col,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

# -----------------------------------------------------------------------------
# Upload handler (save .db files to scripts/results/)
# -----------------------------------------------------------------------------


def handle_db_upload(files: list[gr.File]) -> str:
    ensure_results_dir()
    saved = []
    for f in files or []:
        dest = db_dir / Path(f.name).name
        Path(f.name).replace(dest)
        saved.append(dest.name)
    return (
        f"Uploaded: {', '.join(saved)}" if saved else "No files uploaded."
    )

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
with gr.Blocks() as interface:
    pass
with gr.Blocks() as interface:
    with gr.Tab("Game Arena"):
        config = create_player_config()

        gr.Markdown("# LLM Game Arena")
        gr.Markdown("Play games against LLMs or watch LLMs compete!")
        gr.Markdown(
            f"> **🤖 Available AI Players**: {config['model_info']}\n"
            "> Local transformer models run with Hugging Face transformers. "
            "No API tokens required!"
        )

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
            gr.Markdown(f"### {label}")
            choices_pairs = [
                (key, config["player_config"]["player_type_display"][key])
                for key in config["player_config"]["player_types"]
            ]
            dd_type = gr.Dropdown(
                choices=choices_pairs,
                label=f"{label} Type",
                value=choices_pairs[0][0],
            )
            dd_model = gr.Dropdown(
                choices=config["player_config"]["available_models"],
                label=f"{label} Model (if LLM)",
                visible=False,
            )
            return dd_type, dd_model

        with gr.Row():
            p1_type, p1_model = player_selector_block("Player 1")
            p2_type, p2_model = player_selector_block("Player 2")

        def _vis(player_type: str):
            is_llm = (
                player_type == "llm"
                or (
                    player_type
                    and (
                        player_type.startswith("llm_")
                        or player_type.startswith("hf_")
                    )
                )
            )
            return gr.update(visible=is_llm)

        p1_type.change(_vis, inputs=p1_type, outputs=p1_model)
        p2_type.change(_vis, inputs=p2_type, outputs=p2_model)

        play_button = gr.Button("🎮 Start Game", variant="primary")
        game_output = gr.Textbox(
            label="Game Log",
            lines=20,
            placeholder="Game results will appear here...",
        )

        play_button.click(
            play_game,
            inputs=[
                game_dropdown,
                p1_type,
                p2_type,
                p1_model,
                p2_model,
                rounds_slider,
            ],
            outputs=[game_output],
        )

    with gr.Tab("Leaderboard"):
        gr.Markdown(
            "# LLM Model Leaderboard\n"
            "Track performance across different games!"
        )
        leaderboard_game_dropdown = gr.Dropdown(
            choices=get_available_games(),
            label="Select Game",
            value="Aggregated Performance",
        )
        leaderboard_table = gr.Dataframe(
            value=extract_leaderboard_stats("Aggregated Performance"),
            headers=LEADERBOARD_COLUMNS,
            interactive=False,
        )
        refresh_btn = gr.Button("🔄 Refresh")

        def _update_leaderboard(game: str) -> pd.DataFrame:
            return extract_leaderboard_stats(game)

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
            )

        with gr.Row():
            create_bar_plot(
                data=metrics_df,
                x_col="agent_name",
                y_col="avg_generation_time (sec)",
                title="Average Generation Time",
                x_label="LLM Model",
                y_label="Time (sec)",
            )

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

            **Data**: SQLite databases in `scripts/results/`.
            """
        )

# Local run only. On Spaces, the runtime will serve `interface` automatically.
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=None, show_api=False)
