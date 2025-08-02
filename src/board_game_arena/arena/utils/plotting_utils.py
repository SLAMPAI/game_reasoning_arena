# utils/common_utils.py
"""Common utility functions for the OpenSpiel LLM Arena project.

This module provides shared utility functions for logging, configuration,
and other cross-cutting concerns.
"""

import json
from pathlib import Path
from tabulate import tabulate
from games.registry import registry
from typing import Dict, Any, List
import matplotlib.pyplot as plt


#TODO: there 3 functions are not used but useful! it contains the state history!

def save_results(game_name: str, final_scores: List[float], state: Any):
    results = prepare_results(game_name, final_scores, state)
    filename = get_results_filename(game_name)
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")


def prepare_results(game_name: str, final_scores: List[float], state: Any) -> Dict[str, Any]:
    return {
        "game_name": game_name,
        "final_state": str(state),
        "returns": list(final_scores),  # Ensure it's JSON-serializable
        "history": state.history_str(),
    }


def get_results_filename(game_name: str) -> str:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    filename = f"{game_name.lower().replace(' ', '_')}_results.json"
    return str(results_dir / filename)


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


def get_win_rate(db_conn, llm_name):  #TODO: use this!
    """Calculates the win rate of an LLM from logged games."""
    cursor = db_conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS total_games,
               SUM(CASE WHEN action = 1 THEN 1 ELSE 0 END) AS wins
        FROM moves WHERE llm_name = %s
    """, (llm_name,))

    total_games, total_wins = cursor.fetchone()

    win_rate = (total_wins / total_games) * 100 if total_games > 0 else 0
    return win_rate


def plot_action_distribution(db_conn, llm_name): #TODO: use this!
    """Plots the distribution of LLM's chosen actions."""
    cursor = db_conn.cursor()

    cursor.execute("""
        SELECT action, COUNT(*) FROM moves WHERE llm_name = %s GROUP BY action
    """, (llm_name,))

    results = cursor.fetchall()
    actions = [r[0] for r in results]
    counts = [r[1] for r in results]

    plt.bar(actions, counts, tick_label=["Fold", "Call"])
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title(f"Action Distribution for {llm_name}")
    plt.show()
