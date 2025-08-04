#!/usr/bin/env python3
"""
Extract and display reasoning traces from the successful game run
"""

import sqlite3
import pandas as pd

def analyze_reasoning_traces():
    """Analyze the reasoning traces from the recent successful run"""

    print("🔍 Analyzing Reasoning Traces from Recent Game Run")
    print("=" * 60)

    # Check the LLM database
    try:
        conn = sqlite3.connect('results/llm_litellm_groq_llama3_8b_8192.db')

        print("📊 Database Analysis:")

        # Get all moves with detailed info
        df = pd.read_sql_query("""
            SELECT game_name, episode, turn, action, reasoning,
                   agent_type, agent_model, board_state, timestamp
            FROM moves
            ORDER BY game_name, episode, turn
        """, conn)

        print(f"Total moves in database: {len(df)}")

        # Show detailed breakdown
        if len(df) > 0:
            print(f"\nAgent types breakdown:")
            print(df['agent_type'].value_counts())

            print(f"\nGames breakdown:")
            print(df['game_name'].value_counts())

            print(f"\n🎮 Detailed Move Analysis:")
            print("-" * 50)

            for i, row in df.iterrows():
                print(f"\nMove {i+1}:")
                print(f"  Game: {row['game_name']}")
                print(f"  Episode: {row['episode']}")
                print(f"  Turn: {row['turn']}")
                print(f"  Agent Type: {row['agent_type']}")
                print(f"  Agent Model: {row['agent_model']}")
                print(f"  Action: {row['action']}")
                print(f"  Reasoning: {repr(row['reasoning'])}")
                print(f"  Board State:")
                if row['board_state']:
                    # Handle both escaped and non-escaped newlines
                    board_clean = row['board_state'].replace('\\n', '\n')
                    for line in board_clean.split('\n'):
                        print(f"    {line}")
                else:
                    print("    [No board state]")
                print(f"  Timestamp: {row['timestamp']}")

        conn.close()

    except Exception as e:
        print(f"Error reading LLM database: {e}")

    # Also check the random agent database
    try:
        print(f"\n" + "=" * 60)
        print("🎲 Random Agent Database Analysis:")

        conn = sqlite3.connect('results/random_None.db')

        df_random = pd.read_sql_query("""
            SELECT game_name, episode, turn, action, reasoning,
                   agent_type, agent_model, board_state
            FROM moves
            ORDER BY game_name, episode, turn
        """, conn)

        print(f"Random agent moves: {len(df_random)}")

        if len(df_random) > 0:
            print("Sample random agent moves:")
            for i, row in df_random.head(2).iterrows():
                print(f"  Turn {row['turn']}: Action {row['action']}, Agent Type: {row['agent_type']}")

        conn.close()

    except Exception as e:
        print(f"Error reading random database: {e}")

    # Check the merged CSV
    try:
        print(f"\n" + "=" * 60)
        print("📋 Merged CSV Analysis:")

        df_csv = pd.read_csv('results/merged_logs_20250803_221432.csv')
        print(f"Total moves in CSV: {len(df_csv)}")

        if len(df_csv) > 0:
            print(f"\nAgent names in CSV:")
            print(df_csv['agent_name'].value_counts())

            print(f"\nAgent types in CSV:")
            print(df_csv['agent_type'].value_counts())

            # Look for LLM moves in CSV
            llm_moves = df_csv[df_csv['agent_name'].str.contains('llm_', na=False)]
            print(f"\nMoves from LLM agent (by name): {len(llm_moves)}")

            if len(llm_moves) > 0:
                print("Sample LLM moves from CSV:")
                for i, row in llm_moves.head(2).iterrows():
                    print(f"  Turn {row['turn']}: Action {row['action']}")
                    print(f"    Agent Name: {row['agent_name']}")
                    print(f"    Agent Type: {row['agent_type']}")
                    print(f"    Reasoning: {repr(row['reasoning'])}")
                    print(f"    Board State Present: {pd.notna(row['board_state'])}")

    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    analyze_reasoning_traces()
