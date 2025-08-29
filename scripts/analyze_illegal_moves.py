#!/usr/bin/env python3
"""
Script to analyze illegal moves across all existing game databases.
This script checks all SQLite databases in the results directory for illegal moves.
"""

import sqlite3
import os
from pathlib import Path
import pandas as pd


def check_illegal_moves_in_db(db_path: str):
    """Check for illegal moves in a single database."""
    try:
        conn = sqlite3.connect(db_path)

        # Get illegal moves count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM illegal_moves")
        illegal_count = cursor.fetchone()[0]

        if illegal_count > 0:
            # Get details of illegal moves
            cursor.execute("""
                SELECT game_name, episode, turn, agent_id, illegal_action,
                       reason, timestamp
                FROM illegal_moves
                ORDER BY timestamp
            """)
            illegal_moves = cursor.fetchall()

            print(f"\n🚨 Found {illegal_count} illegal moves in {os.path.basename(db_path)}:")
            for move in illegal_moves:
                game, episode, turn, agent_id, action, reason, timestamp = move
                print(f"  - Game: {game}, Episode: {episode}, Turn: {turn}")
                print(f"    Agent {agent_id} tried action {action}")
                print(f"    Reason: {reason}")
                print(f"    Time: {timestamp}")

        # Check total moves for context
        cursor.execute("SELECT COUNT(*) FROM moves")
        total_moves = cursor.fetchone()[0]

        conn.close()

        return {
            'database': os.path.basename(db_path),
            'illegal_moves': illegal_count,
            'total_moves': total_moves,
            'illegal_rate': illegal_count / total_moves if total_moves > 0 else 0
        }

    except Exception as e:
        print(f"Error checking {db_path}: {e}")
        return None


def analyze_all_databases():
    """Analyze all databases in the results directory."""
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        print("No results directory found!")
        return

    # Find all .db files
    db_files = list(results_dir.glob("*.db"))

    if not db_files:
        print("No database files found in results directory!")
        return

    print(f"Found {len(db_files)} database files to analyze...")

    all_results = []
    total_illegal = 0
    total_moves = 0

    for db_path in sorted(db_files):
        result = check_illegal_moves_in_db(str(db_path))
        if result:
            all_results.append(result)
            total_illegal += result['illegal_moves']
            total_moves += result['total_moves']

            if result['illegal_moves'] == 0:
                print(f"✅ {result['database']}: No illegal moves "
                      f"({result['total_moves']} total moves)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total databases analyzed: {len(all_results)}")
    print(f"Total illegal moves found: {total_illegal}")
    print(f"Total moves across all games: {total_moves}")

    if total_moves > 0:
        illegal_rate = (total_illegal / total_moves) * 100
        print(f"Overall illegal move rate: {illegal_rate:.4f}%")

    # Databases with illegal moves
    problematic_dbs = [r for r in all_results if r['illegal_moves'] > 0]
    if problematic_dbs:
        print(f"\nDatabases with illegal moves: {len(problematic_dbs)}")
        for db in problematic_dbs:
            rate = db['illegal_rate'] * 100
            print(f"  - {db['database']}: {db['illegal_moves']} illegal "
                  f"out of {db['total_moves']} total ({rate:.4f}%)")
    else:
        print("\n🎉 No illegal moves found in any database!")

    return all_results


def check_simulation_logs():
    """Check run logs for illegal move mentions."""
    log_files = [
        "run_logs.txt",
        Path(__file__).parent / "run_logs.txt"
    ]

    for log_path in log_files:
        if Path(log_path).exists():
            print(f"\n📋 Checking {log_path} for illegal move mentions...")

            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                illegal_mentions = content.lower().count('illegal')
                if illegal_mentions > 0:
                    print(f"Found {illegal_mentions} mentions of 'illegal' in logs")

                    # Show first few lines with 'illegal'
                    lines = content.split('\n')
                    illegal_lines = [line for line in lines if 'illegal' in line.lower()]

                    print("Sample illegal move log entries:")
                    for line in illegal_lines[:5]:  # Show first 5
                        print(f"  {line.strip()}")

                    if len(illegal_lines) > 5:
                        print(f"  ... and {len(illegal_lines) - 5} more")
                else:
                    print("No mentions of 'illegal' found in logs")

            except Exception as e:
                print(f"Error reading log file: {e}")
            break


if __name__ == "__main__":
    print("🔍 Analyzing illegal moves in Game Reasoning Arena")
    print("="*60)

    # Check databases
    results = analyze_all_databases()

    # Check logs
    check_simulation_logs()

    print("\n📝 Analysis complete!")
    print("\nInterpretation:")
    print("- If no illegal moves are found, it means LLM agents are")
    print("  generally following the rules correctly")
    print("- The illegal move recording system is working (confirmed by our test)")
    print("- You can create agents that make illegal moves for testing purposes")
