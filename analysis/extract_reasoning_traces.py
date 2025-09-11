#!/usr/bin/env python3
"""
Reasoning Traces Extractor

Utility to extract and analyze reasoning traces with board states from
game logs. This script can be used to analyze LLM decision-making patterns
across games.

Usage Examples:

    # List available databases (run without --db argument)
    python3 analysis/extract_reasoning_traces.py

    # Extract all traces from a specific database
    python3 analysis/extract_reasoning_traces.py --db results/model.db

    # Filter by specific game and episode
    python3 analysis/extract_reasoning_traces.py --db results/model.db \\
           --game tic_tac_toe --episode 1

    # Export to CSV for further analysis
    python3 analysis/extract_reasoning_traces.py --db results/model.db \\
           --export-csv traces.csv

    # Export formatted traces to text file (perfect for academic papers)
    python3 analysis/extract_reasoning_traces.py --db results/model.db \\
           --game tic_tac_toe --export-txt report.txt

    # View analysis summary only (no detailed traces)
    python3 analysis/extract_reasoning_traces.py --db results/model.db \\
           --analyze-only

Key Features:
    - Database Discovery: Automatically finds available database files in
      results/
    - Flexible Filtering: Extract by game type, episode, or custom criteria
    - Multiple Output Formats: Text display, CSV export, formatted text export
    - Pattern Analysis: Built-in statistics and reasoning pattern detection
    - Comprehensive Analysis: Move counts, reasoning coverage, agent breakdowns

Arguments:
    --db: Database file path (if not specified, will list available databases)
    --game: Filter by game name (e.g., tic_tac_toe, connect_four, hex)
    --episode: Filter by episode number
    --export-csv: Export traces to CSV file
    --export-txt: Export formatted traces to text file
    --analyze-only: Show analysis summary without detailed traces

Dependencies: sqlite3, pandas, pathlib, argparse
"""

import sys
import os
import sqlite3
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def create_timestamped_filename(filename):
    """Create a timestamped filename to avoid overwriting existing files"""
    path_obj = Path(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = path_obj.stem
    suffix = path_obj.suffix
    return f"{stem}_{timestamp}{suffix}"


def list_available_databases():
    """List all available database files in the results directory"""
    # Use absolute path to results directory
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    if not results_dir.exists():
        print("No results directory found.")
        return []

    db_files = list(results_dir.glob("*.db"))
    # Filter out human player databases
    db_files = [db for db in db_files if not db.stem.startswith("human")]
    return db_files


def extract_reasoning_traces(db_path,
                             game_name=None,
                             episode=None):
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
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
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
        current_game_changed = current_game != row['game_name']
        current_episode_changed = current_episode != row['episode']
        if current_game_changed or current_episode_changed:
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

        reasoning_valid = (pd.notna(row['reasoning']) and
                           row['reasoning'].strip() and
                           row['reasoning'] != 'None')
        if reasoning_valid:
            print(f"🧠 Reasoning: {row['reasoning']}")
        else:
            print("🧠 Reasoning: [No reasoning provided]")

        print(f"⏰ Timestamp: {row['timestamp']}")
        print("-" * 40)


def export_traces_csv(df, output_file):
    """Export reasoning traces to CSV format"""

    # Create reasoning_exports directory if it doesn't exist
    project_root = Path(__file__).resolve().parent.parent
    export_dir = project_root / "results" / "reasoning_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename to avoid overwriting
    timestamped_filename = create_timestamped_filename(Path(output_file).name)
    final_output_path = export_dir / timestamped_filename

    # Clean up board state formatting for CSV
    df_export = df.copy()
    df_export['board_state'] = (df_export['board_state']
                                .str.replace('\\n', '\n', regex=False))

    df_export.to_csv(final_output_path, index=False)
    print(f"✅ Reasoning traces exported to: {final_output_path}")


def export_traces_txt(df, output_file):
    """Export formatted reasoning traces to text file"""

    from contextlib import redirect_stdout

    # Create reasoning_exports directory if it doesn't exist
    project_root = Path(__file__).resolve().parent.parent
    export_dir = project_root / "results" / "reasoning_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename to avoid overwriting
    timestamped_filename = create_timestamped_filename(Path(output_file).name)
    final_output_path = export_dir / timestamped_filename

    with open(final_output_path, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            # Print header
            print("🔍 Board Game Arena - Reasoning Traces Extractor")
            print("=" * 55)

            # Print analysis
            analyze_reasoning_patterns(df)

            # Print detailed traces
            display_traces_text(df)

            print(f"\n✨ Extraction complete! Found {len(df)} reasoning "
                  f"traces.")

    print(f"✅ Formatted traces exported to: {final_output_path}")


def analyze_reasoning_patterns(df):
    """Analyze reasoning patterns in the traces"""

    print("\n📊 Reasoning Analysis")
    print("=" * 40)

    # Basic statistics
    total_moves = len(df)
    games_analyzed = df['game_name'].nunique()
    episodes_analyzed = df.groupby('game_name')['episode'].nunique().sum()

    print(f"Total moves analyzed: {total_moves}")
    print(f"Games: {games_analyzed}")
    print(f"Episodes: {episodes_analyzed}")

    # Reasoning availability
    has_reasoning = (df['reasoning'].notna() &
                     (df['reasoning'] != 'None') &
                     (df['reasoning'].str.strip() != ''))
    reasoning_coverage = has_reasoning.sum() / total_moves * 100
    print(f"Moves with reasoning: {has_reasoning.sum()}/{total_moves} "
          f"({reasoning_coverage:.1f}%)")

    # Board state availability
    has_board_state = (df['board_state'].notna() &
                       (df['board_state'].str.strip() != ''))
    board_state_coverage = has_board_state.sum() / total_moves * 100
    print(f"Moves with board state: {has_board_state.sum()}/{total_moves} "
          f"({board_state_coverage:.1f}%)")

    # Agent types and models
    print("\nAgent types:")
    for agent_type, count in df['agent_type'].value_counts().items():
        print(f"  - {agent_type}: {count} moves")

    # Specific models used
    if 'agent_model' in df.columns:
        print("\nSpecific models:")
        for model, count in df['agent_model'].value_counts().items():
            print(f"  - {model}: {count} moves")

    # Games breakdown
    print("\nGames breakdown:")
    game_stats = df.groupby('game_name').agg({
        'episode': 'nunique',
        'turn': 'count'
    }).round(1)
    game_stats.columns = ['Episodes', 'Total Moves']
    print(game_stats.to_string())


def main():
    """Main function with command line interface"""

    parser = argparse.ArgumentParser(
        description="Extract and analyze reasoning traces from Board Game "
                    "Arena logs")
    parser.add_argument("--db", help="Database file path (if not specified, "
                                     "will list available databases)")
    parser.add_argument("--game", help="Filter by game name")
    parser.add_argument("--episode", type=int, help="Filter by episode number")
    parser.add_argument("--export-csv", help="Export to CSV file")
    parser.add_argument("--export-txt",
                        help="Export formatted traces to text file")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only show analysis, not full traces")

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

    # Export to TXT if requested (includes analysis and traces)
    if args.export_txt:
        export_traces_txt(df, args.export_txt)
        return  # Don't print to console if saving to file

    # Export to CSV if requested
    if args.export_csv:
        # Show analysis summary but don't print detailed traces
        analyze_reasoning_patterns(df)
        export_traces_csv(df, args.export_csv)
        print(f"\n✨ Extraction complete! Found {len(df)} reasoning traces.")
        return  # Don't print detailed traces to console

    # Default behavior: export to TXT with auto-generated filename
    if not args.analyze_only:
        # Generate default filename based on filters
        game_part = f"_{args.game}" if args.game else ""
        episode_part = f"_ep{args.episode}" if args.episode else ""
        default_filename = f"reasoning_traces{game_part}{episode_part}.txt"

        export_traces_txt(df, default_filename)
        return

    # Show analysis only (when --analyze-only is used)
    analyze_reasoning_patterns(df)
    print(f"\n✨ Analysis complete! Found {len(df)} reasoning traces.")


if __name__ == "__main__":
    main()
