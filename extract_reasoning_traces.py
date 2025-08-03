#!/usr/bin/env python3
"""
Reasoning Traces Extractor

Utility to extract and analyze reasoning traces with board states from game logs.
This script can be used to analyze LLM decision-making patterns across games.
"""

import sys
import os
import sqlite3
import pandas as pd
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def list_available_databases():
    """List all available database files in the results directory"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        return []

    db_files = list(results_dir.glob("*.db"))
    return db_files


def extract_reasoning_traces(db_path, game_name=None, episode=None, output_format="text"):
    """Extract reasoning traces from a database file"""

    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return None

    conn = sqlite3.connect(db_path)

    # Build query with optional filters
    query = """
        SELECT game_name, episode, turn, action, reasoning, board_state,
               timestamp, agent_type, agent_model
        FROM moves
        WHERE 1=1
    """
    params = []

    if game_name:
        query += " AND game_name = ?"
        params.append(game_name)

    if episode:
        query += " AND episode = ?"
        params.append(episode)

    query += " ORDER BY game_name, episode, turn"

    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        print(f"Error reading from database: {e}")
        return None
    finally:
        conn.close()

    if df.empty:
        print("No reasoning traces found with the specified filters.")
        return None

    return df


def display_traces_text(df):
    """Display reasoning traces in text format"""

    current_game = None
    current_episode = None

    for _, row in df.iterrows():
        # Print game/episode headers
        if current_game != row['game_name'] or current_episode != row['episode']:
            print(f"\n{'='*60}")
            print(f"🎮 Game: {row['game_name']} | Episode: {row['episode']}")
            print(f"🤖 Agent: {row['agent_type']} ({row['agent_model']})")
            print(f"{'='*60}")
            current_game = row['game_name']
            current_episode = row['episode']

        print(f"\n🔄 Turn {row['turn']}:")

        # Display board state
        if pd.notna(row['board_state']) and row['board_state'].strip():
            print("📋 Board State:")
            # Handle both escaped and non-escaped newlines
            board_display = row['board_state'].replace('\\n', '\n')
            for line in board_display.split('\n'):
                print(f"   {line}")
        else:
            print("📋 Board State: [Not available]")

        print(f"🎯 Action: {row['action']}")

        if pd.notna(row['reasoning']) and row['reasoning'].strip() and row['reasoning'] != 'None':
            print(f"🧠 Reasoning: {row['reasoning']}")
        else:
            print("🧠 Reasoning: [No reasoning provided]")

        print(f"⏰ Timestamp: {row['timestamp']}")
        print("-" * 40)


def export_traces_csv(df, output_file):
    """Export reasoning traces to CSV format"""

    # Clean up board state formatting for CSV
    df_export = df.copy()
    df_export['board_state'] = df_export['board_state'].str.replace('\\n', '\n')

    df_export.to_csv(output_file, index=False)
    print(f"✅ Reasoning traces exported to: {output_file}")


def analyze_reasoning_patterns(df):
    """Analyze reasoning patterns in the traces"""

    print(f"\n📊 Reasoning Analysis")
    print("=" * 40)

    # Basic statistics
    total_moves = len(df)
    games_analyzed = df['game_name'].nunique()
    episodes_analyzed = df.groupby('game_name')['episode'].nunique().sum()

    print(f"Total moves analyzed: {total_moves}")
    print(f"Games: {games_analyzed}")
    print(f"Episodes: {episodes_analyzed}")

    # Reasoning availability
    has_reasoning = df['reasoning'].notna() & (df['reasoning'] != 'None') & (df['reasoning'].str.strip() != '')
    reasoning_coverage = has_reasoning.sum() / total_moves * 100
    print(f"Moves with reasoning: {has_reasoning.sum()}/{total_moves} ({reasoning_coverage:.1f}%)")

    # Board state availability
    has_board_state = df['board_state'].notna() & (df['board_state'].str.strip() != '')
    board_state_coverage = has_board_state.sum() / total_moves * 100
    print(f"Moves with board state: {has_board_state.sum()}/{total_moves} ({board_state_coverage:.1f}%)")

    # Agent types
    print(f"\nAgent types:")
    for agent_type, count in df['agent_type'].value_counts().items():
        print(f"  - {agent_type}: {count} moves")

    # Games breakdown
    print(f"\nGames breakdown:")
    game_stats = df.groupby('game_name').agg({
        'episode': 'nunique',
        'turn': 'count'
    }).round(1)
    game_stats.columns = ['Episodes', 'Total Moves']
    print(game_stats.to_string())


def main():
    """Main function with command line interface"""

    parser = argparse.ArgumentParser(description="Extract and analyze reasoning traces from Board Game Arena logs")
    parser.add_argument("--db", help="Database file path (if not specified, will list available databases)")
    parser.add_argument("--game", help="Filter by game name")
    parser.add_argument("--episode", type=int, help="Filter by episode number")
    parser.add_argument("--export-csv", help="Export to CSV file")
    parser.add_argument("--analyze-only", action="store_true", help="Only show analysis, not full traces")

    args = parser.parse_args()

    print("🔍 Board Game Arena - Reasoning Traces Extractor")
    print("=" * 55)

    # If no database specified, list available ones
    if not args.db:
        db_files = list_available_databases()
        if not db_files:
            print("No database files found in results/ directory.")
            return

        print("Available database files:")
        for i, db_file in enumerate(db_files, 1):
            print(f"  {i}. {db_file.name}")

        print(f"\nUsage: python3 {sys.argv[0]} --db results/your_database.db")
        return

    # Extract traces
    df = extract_reasoning_traces(args.db, args.game, args.episode)
    if df is None:
        return

    # Show analysis
    analyze_reasoning_patterns(df)

    # Export to CSV if requested
    if args.export_csv:
        export_traces_csv(df, args.export_csv)

    # Display traces unless analyze-only mode
    if not args.analyze_only:
        display_traces_text(df)

    print(f"\n✨ Extraction complete! Found {len(df)} reasoning traces.")


if __name__ == "__main__":
    main()
