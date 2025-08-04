#!/usr/bin/env python3
"""
Demonstrate the enhanced reasoning traces with board states
"""

import sqlite3
import pandas as pd


def display_reasoning_traces():
    """Display the complete reasoning traces with board states"""

    print("🎮 Board Game Arena - Enhanced Reasoning Traces Demonstration")
    print("=" * 70)

    # Read from LLM database
    conn = sqlite3.connect('results/llm_litellm_groq_llama3_8b_8192.db')

    df = pd.read_sql_query("""
        SELECT game_name, episode, turn, action, reasoning,
               agent_type, agent_model, board_state, timestamp
        FROM moves
        ORDER BY game_name, episode, turn
    """, conn)

    conn.close()

    if len(df) == 0:
        print("❌ No reasoning traces found.")
        return

    print(f"✅ Found {len(df)} reasoning traces with complete information!")
    print()

    for i, row in df.iterrows():
        print(f"🧠 Reasoning Trace #{i+1}")
        print("-" * 40)
        print(f"🎯 Game: {row['game_name']}")
        print(f"📅 Episode: {row['episode']}, Turn: {row['turn']}")
        print(f"🤖 Agent: {row['agent_model']}")
        print(f"🎲 Action Chosen: {row['action']}")

        print("📋 Board State at Decision Time:")
        if row['board_state']:
            board_lines = row['board_state'].split('\n')
            for line in board_lines:
                print(f"     {line}")
        else:
            print("     [No board state recorded]")

        print("🧠 Agent's Reasoning:")
        if row['reasoning'] and row['reasoning'] != 'None':
            # Word wrap the reasoning for better display
            reasoning = row['reasoning']
            words = reasoning.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= 60:
                    current_line += (" " + word if current_line else word)
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            for line in lines:
                print(f"     {line}")
        else:
            print("     [No reasoning provided]")

        print(f"⏰ Timestamp: {row['timestamp']}")
        print()

    # Summary
    print("📊 Summary")
    print("-" * 20)
    board_states_captured = (df['board_state'].notna() &
                             (df['board_state'] != '')).sum()
    print(
        f"✅ Board states captured: "
        f"{board_states_captured}/{len(df)}"
    )
    reasoning_captured = (df['reasoning'].notna() &
                          (df['reasoning'] != 'None') &
                          (df['reasoning'] != '')).sum()
    print(f"✅ Reasoning captured: {reasoning_captured}/{len(df)}")
    print(f"🎮 Games analyzed: {df['game_name'].nunique()}")
    print(f"📈 Episodes analyzed: {df['episode'].nunique()}")


if __name__ == "__main__":
    display_reasoning_traces()
