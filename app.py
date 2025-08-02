
''' Used by the Gradio app to display game results, leaderboards,
and performance metrics for LLM agents.
'''

import sqlite3
import pandas as pd
import gradio as gr
from pathlib import Path
from typing import List

# Directory to store SQLite results
db_dir = Path("results")


def find_or_download_db():
    """Check if SQLite .db files exist; if not, attempt to download from cloud storage."""
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
    db_files = list(db_dir.glob("*.db"))

    # Ensure the random bot database exists
    random_db_path = db_dir / "random_None.db"
    if random_db_path not in db_files:
        raise FileNotFoundError("Please upload results for the random agent in a file named 'random_None.db'.")

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
    """Extracts all unique game names from all SQLite databases. Includes 'Aggregated Performance' only when required."""
    db_files = find_or_download_db()
    game_names = set()

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

    game_list = sorted(game_names) if game_names else ["No Games Found"]
    if include_aggregated:
        game_list.insert(0, "Aggregated Performance")  # Ensure 'Aggregated Performance' is always first
    return game_list


def extract_illegal_moves_summary()-> pd.DataFrame:
    """Extracts the number of illegal moves made by each LLM agent.

    Returns:
        pd.DataFrame: DataFrame with columns [agent_name, illegal_moves].
    """
    db_files = find_or_download_db()
    summary = []
    for db_file in db_files:
        agent_type, model_name = extract_agent_info(db_file)
        if agent_type == "random":
            continue # Skip the random agent from this analysis
        conn = sqlite3.connect(db_file)
        try:
            # Count number of illegal moves from the illegal_moves table
            df = pd.read_sql_query("SELECT COUNT(*) AS illegal_moves FROM illegal_moves", conn)
            count = int(df["illegal_moves"].iloc[0]) if not df.empty else 0
        except Exception:
            count = 0 # If the table does not exist or error occurs
        summary.append({"agent_name": model_name, "illegal_moves": count})
        conn.close()
    return pd.DataFrame(summary)


def extract_leaderboard_stats(game_name: str) -> pd.DataFrame:
    """Extract and aggregate leaderboard stats from all SQLite databases."""
    db_files = find_or_download_db()
    all_stats = []

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        agent_type, model_name = extract_agent_info(db_file)

        # Skip random agent rows
        if agent_type == "random":
            conn.close()
            continue

        if game_name == "Aggregated Performance":
            query = "SELECT COUNT(DISTINCT episode) AS games_played, " \
                    "SUM(reward) AS total_rewards " \
                    "FROM game_results"
            df = pd.read_sql_query(query, conn)

            # Use avg_generation_time from a specific game (e.g., Kuhn Poker)
            game_query = "SELECT AVG(generation_time) FROM moves WHERE game_name = 'kuhn_poker'"
            avg_gen_time = conn.execute(game_query).fetchone()[0] or 0
        else:
            query = "SELECT COUNT(DISTINCT episode) AS games_played, " \
                    "SUM(reward) AS total_rewards " \
                    "FROM game_results WHERE game_name = ?"
            df = pd.read_sql_query(query, conn, params=(game_name,))

            # Fetch average generation time from moves table
            gen_time_query = "SELECT AVG(generation_time) FROM moves WHERE game_name = ?"
            avg_gen_time = conn.execute(gen_time_query, (game_name,)).fetchone()[0] or 0

        # Keep division by 2 for total rewards
        df["total_rewards"] = df["total_rewards"].fillna(0).astype(float) / 2

        # Ensure avg_gen_time has decimals
        avg_gen_time = round(avg_gen_time, 3)

        # Calculate win rate against random bot using moves table
        vs_random_query = """
            SELECT COUNT(DISTINCT gr.episode) FROM game_results gr
            JOIN moves m ON gr.game_name = m.game_name AND gr.episode = m.episode
            WHERE m.opponent = 'random_None' AND gr.reward > 0
        """
        total_vs_random_query = """
            SELECT COUNT(DISTINCT gr.episode) FROM game_results gr
            JOIN moves m ON gr.game_name = m.game_name AND gr.episode = m.episode
            WHERE m.opponent = 'random_None'
        """
        wins_vs_random = conn.execute(vs_random_query).fetchone()[0] or 0
        total_vs_random = conn.execute(total_vs_random_query).fetchone()[0] or 0
        vs_random_rate = (wins_vs_random / total_vs_random * 100) if total_vs_random > 0 else 0

        df.insert(0, "agent_name", model_name)  # Ensure agent_name is the first column
        df.insert(1, "agent_type", agent_type)  # Ensure agent_type is second column
        df["avg_generation_time (sec)"] = avg_gen_time
        df["win vs_random (%)"] = round(vs_random_rate, 2)

        all_stats.append(df)
        conn.close()

    leaderboard_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    if leaderboard_df.empty:
        leaderboard_df = pd.DataFrame(columns=["agent_name", "agent_type", "# games", "total rewards", "avg_generation_time (sec)", "win-rate", "win vs_random (%)"])

    return leaderboard_df


##########################################################
with gr.Blocks() as interface:
    # Tab for playing games against LLMs
    with gr.Tab("Game Arena"):
        gr.Markdown("# Play Against LLMs\nChoose a game and an opponent to play!")
        # Dropdown to select a game, excluding 'Aggregated Performance'
        game_dropdown = gr.Dropdown(get_available_games(include_aggregated=False), label="Select a Game")
        # Dropdown to choose an opponent (Random Bot or LLM)
        opponent_dropdown = gr.Dropdown(["Random Bot", "LLM"], label="Choose Opponent")
        # Button to start the game
        play_button = gr.Button("Start Game")
        # Textbox to display the game log
        game_output = gr.Textbox(label="Game Log")

        # Event to start the game when the button is clicked
        play_button.click(lambda game, opponent: f"Game {game} started against {opponent}", inputs=[game_dropdown, opponent_dropdown], outputs=[game_output])

    # Tab for leaderboard and performance tracking
    with gr.Tab("Leaderboard"):
        gr.Markdown("# LLM Model Leaderboard\nTrack performance across different games!")
        # Dropdown to select a game, including 'Aggregated Performance'
        leaderboard_game_dropdown = gr.Dropdown(choices=get_available_games(), label="Select Game", value="Aggregated Performance")
        # Table to display leaderboard statistics
        leaderboard_table = gr.Dataframe(value=extract_leaderboard_stats("Aggregated Performance"), headers=["agent_name", "agent_type", "# games", "total rewards", "avg_generation_time (sec)", "win-rate", "win vs_random (%)"], every=5)
        # Update the leaderboard when a new game is selected
        leaderboard_game_dropdown.change(fn=extract_leaderboard_stats, inputs=[leaderboard_game_dropdown], outputs=[leaderboard_table])

    # Tab for visual insights and performance metrics
    with gr.Tab("Metrics Dashboard"):
        gr.Markdown("# 📊 Metrics Dashboard\nVisual summaries of LLM performance across games.")

        # Extract data for visualizations
        metrics_df = extract_leaderboard_stats("Aggregated Performance")

        with gr.Row():
            gr.BarPlot(
                value=metrics_df,
                x="agent_name",
                y="win vs_random (%)",
                title="Win Rate vs Random Bot",
                x_label="LLM Model",
                y_label="Win Rate (%)"
            )

        with gr.Row():
            gr.BarPlot(
                value=metrics_df,
                x="agent_name",
                y="avg_generation_time (sec)",
                title="Average Generation Time",
                x_label="LLM Model",
                y_label="Time (sec)"
            )

        with gr.Row():
            gr.Dataframe(value=metrics_df, label="Performance Summary")

    # Tab for LLM reasoning and illegal move analysis
    with gr.Tab("Analysis of LLM Reasoning"):
        gr.Markdown("# 🧠 Analysis of LLM Reasoning\nInsights into move legality and decision behavior.")

        # Load illegal move stats using global function
        illegal_df = extract_illegal_moves_summary()

        with gr.Row():
            gr.BarPlot(
                value=illegal_df,
                x="agent_name",
                y="illegal_moves",
                title="Illegal Moves by Model",
                x_label="LLM Model",
                y_label="# of Illegal Moves"
            )

        with gr.Row():
            gr.Dataframe(value=illegal_df, label="Illegal Move Summary")

    # Launch the Gradio interface
    interface.launch()
