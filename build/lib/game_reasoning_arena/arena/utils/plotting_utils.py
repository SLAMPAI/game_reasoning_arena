# utils/plotting_utils.py
"""Plotting utility functions for the OpenSpiel LLM Arena project.

This module provides plotting and visualization functions for game analysis.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt


# Note: JSON export functions removed - the project uses SQLite for storage
# If JSON export is needed, it can be implemented in the analysis module


def print_total_scores(game_name: str, summary: Dict[str, Any]):
    """
    Prints the total scores summary for a game.

    Args:
        game_name: The name of the game being summarized.
        summary: The dictionary containing the game summary.
    """

    print(f"\nTotal scores across all episodes for game {game_name}:")

    # 🔹 Ensure we loop through actual player summaries
    if game_name in summary:
        print(f"Game summary: {summary[game_name]}")
    else:
        for player_id, stats in summary.items():
            print(f"Player {player_id}: {stats}")


def get_win_rate(db_conn, agent_model: str) -> float:
    """Calculates the win rate of an agent from logged game results.

    Args:
        db_conn: SQLite database connection
        agent_model: The model name to calculate win rate for

    Returns:
        float: Win rate as a percentage (0-100)
    """
    cursor = db_conn.cursor()

    # Get total games and wins from game_results table
    cursor.execute("""
        SELECT
            COUNT(*) AS total_games,
            SUM(CASE WHEN reward > 0 THEN 1 ELSE 0 END) AS wins
        FROM game_results
        WHERE run_id IN (
            SELECT DISTINCT run_id FROM moves WHERE agent_model = ?
        )
    """, (agent_model,))

    result = cursor.fetchone()
    total_games, wins = result if result else (0, 0)

    win_rate = (wins / total_games) * 100 if total_games > 0 else 0
    return win_rate


def plot_action_distribution(db_conn, agent_model: str,
                             game_name: str = None, save_path: str = None):
    """Plots the distribution of an agent's chosen actions.

    Args:
        db_conn: SQLite database connection
        agent_model: The model name to analyze
        game_name: Optional game name filter
        save_path: Optional path to save the plot
    """
    cursor = db_conn.cursor()

    # Build query with optional game filter
    query = """
        SELECT action, COUNT(*) as count
        FROM moves
        WHERE agent_model = ?
    """
    params = [agent_model]

    if game_name:
        query += " AND game_name = ?"
        params.append(game_name)

    query += " GROUP BY action ORDER BY action"

    cursor.execute(query, params)
    results = cursor.fetchall()

    if not results:
        print(f"No action data found for {agent_model}")
        return

    actions = [str(r[0]) for r in results]
    counts = [r[1] for r in results]

    plt.figure(figsize=(10, 6))
    plt.bar(actions, counts)
    plt.xlabel("Action")
    plt.ylabel("Frequency")

    title = f"Action Distribution for {agent_model}"
    if game_name:
        title += f" ({game_name})"
    plt.title(title)

    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
