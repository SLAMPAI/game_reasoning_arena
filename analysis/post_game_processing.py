#!/usr/bin/env python3
"""
post_match_processing.py

Merges all agent-specific SQLite logs, computes summary statistics,
and stores results in 'results/'.
"""

import sqlite3
import json
from pathlib import Path
import pandas as pd
from datetime import datetime


def merge_sqlite_logs(log_dir: str = "results/") -> pd.DataFrame:
    """
    Merges all SQLite log files in the specified directory into a
    single DataFrame.

    Args:
        log_dir (str): Directory where agent-specific SQLite logs are stored.

    Returns:
        pd.DataFrame: Merged DataFrame containing all moves, rewards,
        and game outcomes.
    """
    all_moves = []
    all_results = []

    # Find all SQLite files in the specified directory
    log_path = Path(log_dir)
    sqlite_files = list(log_path.glob("*.db"))
    if not sqlite_files:
        print(f"No SQLite files found in {log_dir}")
        return pd.DataFrame()

    for db_file in sqlite_files:

        # Extract agent name from the SQLite file name
        agent_name = db_file.stem

        # Connect to the SQLite database
        conn = sqlite3.connect(db_file)

        # Retrieve move logs and save as DataFrame
        try:
            df_moves = pd.read_sql_query(
                """SELECT game_name, episode, turn, action, reasoning,
                          generation_time, opponent, timestamp, run_id,
                          seed, agent_type, agent_model, board_state
                   FROM moves""",
                conn
            )
            df_moves["agent_name"] = agent_name  # Add agent name as a column
            # Append to list of DataFrames
            #  Remove duplicates and  store agent's moves
            all_moves.append(df_moves.drop_duplicates())
        except Exception as e:
            print(f"No moves table in {db_file}: {e}")

        # Retrieve game results (includes rewards)
        try:
            df_results = pd.read_sql_query(
                """SELECT game_name, episode, status, reward,
                          timestamp AS result_timestamp, run_id
                   FROM game_results""",
                conn
            )
            df_results["agent_name"] = agent_name
            # Remove duplicates
            all_results.append(df_results.drop_duplicates())
        except Exception as e:
            print(f"No game_results table in {db_file}: {e}")

        conn.close()

    # Merge the same table across all agents
    if all_moves:
        df_moves = pd.concat(all_moves, ignore_index=True)
    else:
        df_moves = pd.DataFrame()
    if all_results:
        df_results = pd.concat(all_results, ignore_index=True)
    else:
        df_results = pd.DataFrame()

    # Convert `opponent` lists into hashable strings before merging
    if "opponent" in df_moves.columns:
        df_moves["opponent"] = df_moves["opponent"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    # Merge moves with game results (which includes rewards)
    if not df_results.empty:
        df_full = df_moves.merge(
            df_results,
            on=["game_name", "episode", "agent_name", "run_id"],
            how="left"
        )
    else:
        # If no results exist, just return the moves alone
        df_full = df_moves.copy()

    # Drop duplicates before returning (final safeguard)
    df_full = df_full.drop_duplicates()

    return df_full


def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Computes summary statistics from the merged results DataFrame.

    Returns:
        dict: Summary statistics keyed by game and agent.
    """
    summary = {}
    if df.empty:
        return summary

    for (game, agent), group in df.groupby(["game_name", "agent_name"]):
        total_moves = group.shape[0]
        if not group["generation_time"].empty:
            avg_gen_time = group["generation_time"].mean()
        else:
            avg_gen_time = None
        total_rewards = group["reward"].sum() if "reward" in group else None

        games_played = group["episode"].nunique()
        if "status" in group:
            terminated_games = group[group["status"] == "terminated"].shape[0]
            truncated_games = group[group["status"] == "truncated"].shape[0]
        else:
            terminated_games = 0
            truncated_games = 0

        summary.setdefault(game, {})[agent] = {
            "games_played": games_played,
            "total_moves": total_moves,
            "average_generation_time": avg_gen_time,
            "total_rewards": total_rewards,
            "games_terminated": terminated_games,
            "games_truncated": truncated_games
        }
    return summary


def save_summary(summary: dict, output_dir: str = "results/") -> str:
    """
    Saves the summary statistics to a JSON file.

    Returns:
        str: The path to the saved JSON file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = Path(output_dir) / f"merged_game_results_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {file_path}")
    return str(file_path)


###########################################################
# Main entry point
###########################################################
def main():
    """Main function to process game logs and compute statistics."""
    print("Starting post-game processing...")

    #  Merge all SQLite logs
    merged_df = merge_sqlite_logs(log_dir="results")
    if merged_df.empty:
        print("No log files found or merged.")
        return

    # Compute statistics - Not needed for now
    # summary = compute_summary_statistics(merged_df)

    # Ensure `agent_name` is the second column
    column_order = ["game_name", "agent_name"] + [
        col for col in merged_df.columns
        if col not in ["game_name", "agent_name"]
    ]
    merged_df = merged_df[column_order]

    # Save logs for review - saves the reasoning behind each move !
    # Use absolute path to scripts/results directory
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "scripts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    merged_csv = results_dir / f"merged_logs_{timestamp}.csv"
    # Drop verbose or unnecessary fields
    merged_df = merged_df.drop(columns=["result_timestamp"], errors="ignore")
    merged_df.to_csv(merged_csv, index=False)
    print(f"Merged logs saved as CSV to {merged_csv}")

    # Save summary results into a JSON file - not needed for now
    # save_summary(summary, output_dir="results")

    # Show how games ended
    print("Game Outcomes Summary:")
    if "status" in merged_df:
        game_end_counts = merged_df["status"].value_counts()
        print(game_end_counts)

    # Display first 5 moves
    print("\nMerged Log DataFrame (First 5 Rows):")
    print(merged_df.head())


if __name__ == "__main__":
    main()
    print("Done.")
