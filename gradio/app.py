# # !/usr/bin/env python3
"""
Board Game Arena - HuggingFace Gradio App

A G# Local HuggingFace models - these will be integrated with the backend system
HUGGINGFACE_MODELS = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "google/flan-t5-small": "google/flan-t5-small",
    "EleutherAI/gpt-neo-125M": "EleutherAI/gpt-neo-125M"
}nterface for playing board games with LLM agents and analyzing
their performance. Optimized for deployment on HuggingFace Spaces.

Features:
- Play games against LLM agents (free HuggingFace models supported)
- View leaderboards and performance metrics
- Analyze LLM reasoning and decision-making
- Interactive visualizations and charts

User clicks "Start Game" in Gradio
        ↓
    app.py (play_game function)
        ↓
gradio_config_generator.py (run_game_with_existing_infrastructure)
        ↓
src/board_game_arena/ (core game infrastructure)
        ↓
    Game results displayed in Gradio
"""

import sqlite3
import pandas as pd
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Tuple, Generator
from transformers import pipeline
import logging
from dataclasses import dataclass

# Add the src directory to Python path
import sys
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Try importing from the board game arena package
# Prioritize src.board_game_arena path for HuggingFace Spaces deployment

from src.board_game_arena.arena.games.registry import registry as games_registry

# Import backend system for proper LLM handling
try:
    from src.board_game_arena.backends.huggingface_backend import (
        HuggingFaceBackend)
    from src.board_game_arena.backends import (
        initialize_llm_registry, LLM_REGISTRY)
    BACKEND_SYSTEM_AVAILABLE = True
    print("✅ Backend system available - using proper LLM infrastructure")
except ImportError as e:
    print(f"⚠️  Backend system not available: {e}")
    BACKEND_SYSTEM_AVAILABLE = False


# ============================================================================
# INITIALIZATION - Try to import game modules if available
# ============================================================================

# Initialize games registry - will be populated from the game registry module
GAMES_REGISTRY = {}


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class PlayerConfigData:
    """Data class for player configuration."""
    player_types: List[str]
    player_type_display: Dict[str, str]
    available_models: List[str]


@dataclass
class GameArenaConfig:
    """Configuration for Game Arena interface."""
    available_games: List[str]
    player_config: PlayerConfigData
    model_info: str
    backend_available: bool


# Local HuggingFace models - these will be integrated with the backend system
HUGGINGFACE_MODELS = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "google/flan-t5-small": "google/flan-t5-small",
    "EleutherAI/gpt-neo-125M": "EleutherAI/gpt-neo-125M"
}

# Initialize HuggingFace backend and register models if available
huggingface_backend = None
if BACKEND_SYSTEM_AVAILABLE:
    try:
        huggingface_backend = HuggingFaceBackend()

        # Initialize the LLM registry
        initialize_llm_registry()

        # Register HuggingFace models in the LLM registry with hf_ prefix
        for model_name in HUGGINGFACE_MODELS.keys():
            if huggingface_backend.is_model_available(model_name):
                # Register with hf_ prefix to distinguish from LiteLLM models
                registry_key = f"hf_{model_name}"
                LLM_REGISTRY[registry_key] = {
                    "backend": huggingface_backend,
                    "model_name": model_name
                }
                print(f"✓ Registered HuggingFace model: {registry_key}")
    except Exception as e:
        print(f"Failed to initialize HuggingFace backend: {e}")
        huggingface_backend = None

# Initialize games registry from the imported registry
try:
    GAMES_REGISTRY = {name: cls for name, cls in games_registry._registry.items()}
    print("✅ Successfully imported full board game arena - GAMES ARE PLAYABLE!")
except Exception as e:
    print(f"⚠️ Failed to load games registry: {e}")
    GAMES_REGISTRY = {}

# Directory to store SQLite results
db_dir = Path("../results")

# Constants for column headers
LEADERBOARD_COLUMNS = [
    "agent_name", "agent_type", "# games", "total rewards",
    "avg_generation_time (sec)", "win-rate", "win vs_random (%)"
]


# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================

def iter_agent_databases() -> Generator[Tuple[str, str, str], None, None]:
    """Iterate through database files for non-random agents.

    Yields:
        Tuple of (db_file, agent_type, model_name)
    """
    db_files = find_or_download_db()
    for db_file in db_files:
        agent_type, model_name = extract_agent_info(db_file)
        if agent_type != "random":
            yield db_file, agent_type, model_name


def find_or_download_db():
    """Check if SQLite .db files exist; create empty random.db if needed."""
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)

    db_files = list(db_dir.glob("*.db"))

    # Ensure the random bot database exists
    random_db_path = db_dir / "random_None.db"
    if not random_db_path.exists():
        try:
            conn = sqlite3.connect(str(random_db_path))
            conn.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY,
                    game_name TEXT,
                    player1 TEXT,
                    player2 TEXT,
                    winner INTEGER,
                    timestamp TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            raise FileNotFoundError(
                "Please upload results for the random agent in a file named "
                "'random_None.db'.")

    db_files = list(db_dir.glob("*.db"))
    return [str(f) for f in db_files]


def extract_agent_info(filename: str):
    """Extract agent type and model name from the filename."""
    base_name = Path(filename).stem
    parts = base_name.split("_", 1)
    if len(parts) == 2:
        agent_type, model_name = parts
    else:
        agent_type, model_name = parts[0], "Unknown"
    return agent_type, model_name


def get_available_games(include_aggregated=True) -> List[str]:
    """Extract all unique game names from SQLite databases and registry."""
    db_files = find_or_download_db()
    game_names = set()

    # First, add games from database files
    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        try:
            query = "SELECT DISTINCT game_name FROM moves"
            df = pd.read_sql_query(query, conn)
            game_names.update(df["game_name"].tolist())
        except Exception:
            pass  # Ignore errors if table doesn't exist
        finally:
            conn.close()

    # Add games from registry if available
    if GAMES_REGISTRY:
        game_names.update(GAMES_REGISTRY.keys())

    # Fallback to default games if nothing found
    if not game_names:
        fallback_games = ["tic_tac_toe", "kuhn_poker", "connect_four"]
        game_names.update(fallback_games)

    game_list = sorted(game_names) if game_names else ["No Games Found"]
    if include_aggregated:
        game_list.insert(0, "Aggregated Performance")
    return game_list


def extract_illegal_moves_summary() -> pd.DataFrame:
    """Extract number of illegal moves made by each LLM agent."""
    summary = []
    for db_file, agent_type, model_name in iter_agent_databases():
        conn = sqlite3.connect(db_file)
        try:
            query = "SELECT COUNT(*) AS illegal_moves FROM illegal_moves"
            df = pd.read_sql_query(query, conn)
            count = int(df["illegal_moves"].iloc[0]) if not df.empty else 0
        except Exception:
            count = 0
        finally:
            conn.close()
        summary.append({"agent_name": model_name, "illegal_moves": count})
    return pd.DataFrame(summary)


# ============================================================================
# HELPER FUNCTIONS FOR PLAYER CONFIGURATION
# ============================================================================

def setup_player_config(player_type: str, player_model: str,
                        player_id: str) -> dict:
    """Configure a single player for the game environment."""
    if player_type == "random_bot":
        return {"type": "random"}
    elif player_type.startswith("llm_"):
        # Extract model ID from the player type (format: llm_model_id)
        model_id = player_type[4:]  # Remove 'llm_' prefix
        if (model_id in HUGGINGFACE_MODELS and
            BACKEND_SYSTEM_AVAILABLE and
            model_id in LLM_REGISTRY):
            # Use proper LLM agent with HuggingFace backend
            return {
                "type": "llm",
                "model": model_id
            }
    elif (player_type == "llm" and
          player_model in HUGGINGFACE_MODELS and
          BACKEND_SYSTEM_AVAILABLE):
        return {
            "type": "llm",
            "model": player_model
        }

    return {"type": "random"}  # Fallback


# ============================================================================
# MAIN GAME FUNCTION (REFACTORED TO USE EXISTING INFRASTRUCTURE)
# ============================================================================

def play_game(game_name: str, player1_type: str, player2_type: str,
              player1_model: str = None, player2_model: str = None,
              rounds: int = 1) -> str:
    """Play the selected game with specified players.

    This function has been refactored to reuse the existing runner.py and
    simulate.py infrastructure instead of duplicating code.

    Args:
        game_name: Name of the game to play
        player1_type: Type of player 1 (human, random_bot, llm_*)
        player2_type: Type of player 2 (human, random_bot, llm_*)
        player1_model: LLM model for player 1 (if applicable)
        player2_model: LLM model for player 2 (if applicable)
        rounds: Number of rounds to play

    Returns:
        str: Game log/output to display
    """
    if game_name == "No Games Found":
        return "No games available. Please add game databases."

    # Debug: Print what the user selected
    print("🎮 GRADIO DEBUG: Starting game with parameters:")
    print(f"   Game: {game_name}")
    print(f"   Player 1 Type: {player1_type}")
    print(f"   Player 1 Model: {player1_model}")
    print(f"   Player 2 Type: {player2_type}")
    print(f"   Player 2 Model: {player2_model}")
    print(f"   Rounds: {rounds}")

    # ISSUE FIX: Convert display names back to keys
    # Gradio is passing display names instead of keys
    config = create_player_config()
    display_to_key = {
        v: k for k, v in config.player_config.player_type_display.items()
    }

    # Convert display names to actual keys
    if player1_type in display_to_key:
        player1_type = display_to_key[player1_type]
        print(f"   → Converted Player 1 Type to: {player1_type}")

    if player2_type in display_to_key:
        player2_type = display_to_key[player2_type]
        print(f"   → Converted Player 2 Type to: {player2_type}")

    try:
        # Import the gradio config generator
        from gradio_config_generator import (
            run_game_with_existing_infrastructure
        )

        # Use the existing infrastructure instead of reimplementing
        result = run_game_with_existing_infrastructure(
            game_name=game_name,
            player1_type=player1_type,
            player2_type=player2_type,
            player1_model=player1_model,
            player2_model=player2_model,
            rounds=rounds,
            seed=42  # Could make this configurable via UI
        )

        return result

    except Exception as e:
        return f"Error during game simulation: {str(e)}"


def extract_leaderboard_stats(game_name: str) -> pd.DataFrame:
    """Extract and aggregate leaderboard stats from all SQLite databases."""
    all_stats = []

    for db_file, agent_type, model_name in iter_agent_databases():
        conn = sqlite3.connect(db_file)
        try:
            if game_name == "Aggregated Performance":
                query = ("SELECT COUNT(DISTINCT episode) AS games_played, "
                         "SUM(reward) AS total_rewards FROM game_results")
                df = pd.read_sql_query(query, conn)

                # Use avg_generation_time from a specific game
                game_query = ("SELECT AVG(generation_time) FROM moves "
                              "WHERE game_name = 'kuhn_poker'")
                avg_gen_time = conn.execute(game_query).fetchone()[0] or 0
            else:
                query = ("SELECT COUNT(DISTINCT episode) AS games_played, "
                         "SUM(reward) AS total_rewards FROM game_results "
                         "WHERE game_name = ?")
                df = pd.read_sql_query(query, conn, params=(game_name,))

                # Fetch average generation time from moves table
                gen_time_query = ("SELECT AVG(generation_time) FROM moves "
                                  "WHERE game_name = ?")
                result = conn.execute(gen_time_query, (game_name,)).fetchone()
                avg_gen_time = result[0] or 0

            # Keep division by 2 for total rewards
            df["total_rewards"] = (df["total_rewards"]
                                   .fillna(0).astype(float) / 2)

            # Ensure avg_gen_time has decimals
            avg_gen_time = round(avg_gen_time, 3)

            # Calculate win rate against random bot using opponent column
            vs_random_query = """
                SELECT COUNT(*) FROM game_results
                WHERE opponent = 'random_None' AND reward > 0
            """
            total_vs_random_query = """
                SELECT COUNT(*) FROM game_results
                WHERE opponent = 'random_None'
            """
            wins_vs_random = conn.execute(vs_random_query).fetchone()[0] or 0
            result = conn.execute(total_vs_random_query).fetchone()
            total_vs_random = result[0] or 0
            if total_vs_random > 0:
                vs_random_rate = (wins_vs_random / total_vs_random * 100)
            else:
                vs_random_rate = 0

            # Ensure agent_name is the first column
            df.insert(0, "agent_name", model_name)
            # Ensure agent_type is second column
            df.insert(1, "agent_type", agent_type)
            df["avg_generation_time (sec)"] = avg_gen_time
            df["win vs_random (%)"] = round(vs_random_rate, 2)

            all_stats.append(df)
        finally:
            conn.close()

    if all_stats:
        leaderboard_df = pd.concat(all_stats, ignore_index=True)
    else:
        leaderboard_df = pd.DataFrame()

    if leaderboard_df.empty:
        leaderboard_df = pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    return leaderboard_df


# ============================================================================
# GRADIO UI COMPONENTS
# ============================================================================

def create_bar_plot(data: pd.DataFrame, x_col: str, y_col: str,
                    title: str, x_label: str, y_label: str) -> gr.BarPlot:
    """Create a standardized bar plot component."""
    return gr.BarPlot(
        value=data,
        x=x_col,
        y=y_col,
        title=title,
        x_label=x_label,
        y_label=y_label
    )


# ============================================================================
# GRADIO INTERFACE SETUP
# ============================================================================

def create_player_config():
    """Create player configuration for the interface."""
    # Get available games
    available_games = get_available_games(include_aggregated=False)

    # Get models from database files using the new iterator
    database_models = []
    for _, _, model_name in iter_agent_databases():
        database_models.append(model_name)

    # Setup player types and display names
    player_types = ["random_bot"]
    player_type_display = {"random_bot": "Random Bot"}

    # Add HuggingFace models as player types if available
    if BACKEND_SYSTEM_AVAILABLE:
        for model_key in HUGGINGFACE_MODELS.keys():
            llm_type_key = f"hf_{model_key}"  # Use 'hf_' prefix
            player_types.append(llm_type_key)
            model_display_name = (
                model_key.split('/')[-1] if '/' in model_key
                else model_key
            )
            display_name = f"HuggingFace: {model_display_name}"
            player_type_display[llm_type_key] = display_name

    # Combine all available models
    all_models = list(HUGGINGFACE_MODELS.keys()) + database_models

    model_info = (
        "HuggingFace transformer models integrated with backend system."
        if BACKEND_SYSTEM_AVAILABLE
        else "Backend system not available - limited functionality."
    )

    return GameArenaConfig(
        available_games=available_games,
        player_config=PlayerConfigData(
            player_types=player_types,
            player_type_display=player_type_display,
            available_models=all_models
        ),
        model_info=model_info,
        backend_available=BACKEND_SYSTEM_AVAILABLE
    )


def create_player_selector(config: GameArenaConfig, player_name: str):
    """Create UI components for a single player configuration."""
    with gr.Column():
        gr.Markdown(f"### {player_name}")

        player_choices = [
            (key, config.player_config.player_type_display[key])
            for key in config.player_config.player_types
        ]

        # Debug: Print the player choices to see what's configured
        print(f"🎯 DROPDOWN DEBUG: {player_name} choices:")
        for key, display in player_choices:
            print(f"   Key: '{key}' → Display: '{display}'")

        player_type = gr.Dropdown(
            choices=player_choices,
            label=f"{player_name} Type",
            value=(player_choices[0][0] if player_choices else None)
        )

        player_model = gr.Dropdown(
            choices=config.player_config.available_models,
            label=f"{player_name} Model (if LLM)",
            visible=False
        )

        return player_type, player_model


def setup_event_handlers(components: Dict[str, Any]):
    """Setup event handlers for UI interactions."""
    def update_model_visibility(player_type):
        """Show/hide model dropdown based on player type."""
        is_llm = (player_type == "llm" or
                  (player_type and (player_type.startswith("llm_") or
                                    player_type.startswith("hf_"))))
        return gr.update(visible=is_llm)

    # Model visibility handlers
    components['player1_type'].change(
        update_model_visibility,
        inputs=components['player1_type'],
        outputs=components['player1_model']
    )

    components['player2_type'].change(
        update_model_visibility,
        inputs=components['player2_type'],
        outputs=components['player2_model']
    )

    # Play button handler
    components['play_button'].click(
        play_game,
        inputs=[
            components['game_dropdown'],
            components['player1_type'],
            components['player2_type'],
            components['player1_model'],
            components['player2_model'],
            components['rounds_slider']
        ],
        outputs=[components['game_output']]
    )


def create_game_arena_tab():
    """Create the Game Arena tab interface."""
    config = create_player_config()

    # Header and description
    gr.Markdown("# LLM Game Arena")
    gr.Markdown("Play games against LLMs or watch LLMs compete!")

    # Model availability info
    gr.Markdown(f"""
    > **🤖 Available AI Players**: {config.model_info}
    >
    > Local transformer models run directly using HuggingFace transformers.
    > No API tokens required!
    """)

    # Game selection and configuration
    with gr.Row():
        game_dropdown = gr.Dropdown(
            choices=config.available_games,
            label="Select a Game",
            value=(config.available_games[0]
                   if config.available_games
                   else "No Games Found")
        )

        rounds_slider = gr.Slider(
            minimum=1, maximum=10, value=1, step=1,
            label="Number of Rounds"
        )

    # Player configuration
    with gr.Row():
        player1_type, player1_model = create_player_selector(
            config, "Player 1"
        )
        player2_type, player2_model = create_player_selector(
            config, "Player 2"
        )

    # Game controls and output
    play_button = gr.Button("🎮 Start Game", variant="primary")
    game_output = gr.Textbox(
        label="Game Log",
        lines=20,
        placeholder="Game results will appear here..."
    )

    # Setup event handlers
    components = {
        'game_dropdown': game_dropdown,
        'rounds_slider': rounds_slider,
        'player1_type': player1_type,
        'player1_model': player1_model,
        'player2_type': player2_type,
        'player2_model': player2_model,
        'play_button': play_button,
        'game_output': game_output
    }

    setup_event_handlers(components)


# ============================================================================
# MAIN GRADIO INTERFACE
# ============================================================================

with gr.Blocks() as interface:
    # Game Arena Tab
    with gr.Tab("Game Arena"):
        create_game_arena_tab()

    # Tab for leaderboard and performance tracking
    with gr.Tab("Leaderboard"):
        gr.Markdown("# LLM Model Leaderboard\n"
                    "Track performance across different games!")
        # Dropdown to select a game, including 'Aggregated Performance'
        leaderboard_game_dropdown = gr.Dropdown(
            choices=get_available_games(),
            label="Select Game",
            value="Aggregated Performance"
        )
        # Table to display leaderboard statistics
        leaderboard_table = gr.Dataframe(
            value=extract_leaderboard_stats("Aggregated Performance"),
            headers=LEADERBOARD_COLUMNS,
            every=5
        )
        # Update the leaderboard when a new game is selected
        leaderboard_game_dropdown.change(
            fn=extract_leaderboard_stats,
            inputs=[leaderboard_game_dropdown],
            outputs=[leaderboard_table]
        )

    # Tab for visual insights and performance metrics
    with gr.Tab("Metrics Dashboard"):
        gr.Markdown("# 📊 Metrics Dashboard\n"
                    "Visual summaries of LLM performance across games.")

        # Extract data for visualizations
        metrics_df = extract_leaderboard_stats("Aggregated Performance")

        with gr.Row():
            create_bar_plot(
                data=metrics_df,
                x_col="agent_name",
                y_col="win vs_random (%)",
                title="Win Rate vs Random Bot",
                x_label="LLM Model",
                y_label="Win Rate (%)"
            )

        with gr.Row():
            create_bar_plot(
                data=metrics_df,
                x_col="agent_name",
                y_col="avg_generation_time (sec)",
                title="Average Generation Time",
                x_label="LLM Model",
                y_label="Time (sec)"
            )

        with gr.Row():
            gr.Dataframe(value=metrics_df, label="Performance Summary")

    # Tab for LLM reasoning and illegal move analysis
    with gr.Tab("Analysis of LLM Reasoning"):
        gr.Markdown("# 🧠 Analysis of LLM Reasoning\n"
                    "Insights into move legality and decision behavior.")

        # Load illegal move stats using global function
        illegal_df = extract_illegal_moves_summary()

        with gr.Row():
            create_bar_plot(
                data=illegal_df,
                x_col="agent_name",
                y_col="illegal_moves",
                title="Illegal Moves by Model",
                x_label="LLM Model",
                y_label="# of Illegal Moves"
            )

        with gr.Row():
            gr.Dataframe(value=illegal_df, label="Illegal Move Summary")

    # Add an "About" tab with information about the project
    with gr.Tab("About"):
        gr.Markdown("""
        # About Board Game Arena

        This application provides analysis and visualization of LLM performance
        playing various board games. It includes:

        - **Game Arena**: Play games against LLMs or watch LLMs compete
        - **Leaderboard**: View performance statistics across different games
        - **Metrics Dashboard**: Visual insights into LLM performance
        - **Reasoning Analysis**: Analyze LLM decision-making and errors

        ## Data Sources

        Game results are stored in SQLite databases in the `results` directory.
        Each database contains game logs, moves, and performance metrics for a
        specific LLM agent.

        ## Repository

        This project is part of research on LLM reasoning and decision-making
        strategic game environments.
        """)

    # Launch the Gradio interface with parameters for HuggingFace Spaces
    interface.launch(
        share=False,  # Do not create a public link
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=None,  # Let Gradio find an available port
        show_api=False,  # Hide API docs
        favicon_path=None,  # Use default favicon
        auth=None,  # No authentication required
        # Allow uploads of .db files for database analysis
        allowed_paths=["*.db"]
    )
