#!/usr/bin/env python3
"""
Script to demonstrate reasoning traces with board states
Shows how to extract and display reasoning traces from the database
"""

import sys
import os
import sqlite3
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from board_game_arena.arena.utils.loggers import SQLiteLogger


def create_sample_reasoning_traces():
    """Create sample reasoning traces with board states for demonstration"""

    logger = SQLiteLogger("llm", "demo_model")

    # Simulate a tic-tac-toe game with reasoning traces
    game_states = [
        ("...\n...\n...", "I'll take the center position for strategic advantage", 4),
        ("...\n.x.\n...", "Opponent took corner, I'll block potential diagonal", 0),
        ("o..\n.x.\n...", "I need to create a threat while blocking opponent", 8),
        ("o..\n.x.\no..", "Opponent blocked me, I'll create a fork opportunity", 2),
        ("o.o\n.x.\no..", "I can win by completing the top row!", 1),
    ]

    for turn, (board_state, reasoning, action) in enumerate(game_states):
        logger.log_move(
            game_name="tic_tac_toe",
            episode=1,
            turn=turn,
            action=action,
            reasoning=reasoning,
            opponent="random_agent",
            generation_time=0.5,
            agent_type="llm",
            agent_model="demo_model",
            seed=42,
            board_state=board_state
        )

    return logger.db_path


def display_reasoning_traces(db_path):
    """Display reasoning traces with board states"""

    conn = sqlite3.connect(db_path)

    # Query all moves with board states
    df = pd.read_sql_query("""
        SELECT turn, action, reasoning, board_state, timestamp
        FROM moves
        WHERE game_name = 'tic_tac_toe' AND episode = 1
        ORDER BY turn
    """, conn)

    print("🎮 Reasoning Traces with Board States")
    print("=" * 50)

    for _, row in df.iterrows():
        print(f"\nTurn {row['turn']}:")
        print("Board State:")
        print(row['board_state'].replace('\\n', '\n'))
        print(f"Action: {row['action']}")
        print(f"Reasoning: {row['reasoning']}")
        print(f"Timestamp: {row['timestamp']}")
        print("-" * 30)

    conn.close()


def analyze_reasoning_patterns(db_path):
    """Analyze reasoning patterns across turns"""

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
        SELECT turn, action, reasoning, board_state,
               LENGTH(reasoning) as reasoning_length,
               CASE
                   WHEN reasoning LIKE '%block%' THEN 'Defensive'
                   WHEN reasoning LIKE '%win%' THEN 'Offensive'
                   WHEN reasoning LIKE '%strategic%' THEN 'Strategic'
                   WHEN reasoning LIKE '%threat%' THEN 'Tactical'
                   ELSE 'Other'
               END as reasoning_category
        FROM moves
        WHERE game_name = 'tic_tac_toe' AND episode = 1
        ORDER BY turn
    """, conn)

    print("\n📊 Reasoning Analysis")
    print("=" * 30)

    print(f"Average reasoning length: {df['reasoning_length'].mean():.1f} characters")
    print(f"Reasoning categories:")
    for category, count in df['reasoning_category'].value_counts().items():
        print(f"  - {category}: {count}")

    print(f"\nReasoning evolution:")
    for _, row in df.iterrows():
        print(f"Turn {row['turn']}: {row['reasoning_category']} ({row['reasoning_length']} chars)")

    conn.close()


def main():
    """Main demonstration function"""

    print("🚀 Board Game Arena - Enhanced Reasoning Traces Demo")
    print("=" * 60)

    # Create sample data
    db_path = create_sample_reasoning_traces()
    print(f"✅ Created sample reasoning traces in: {db_path}")

    # Display the traces
    display_reasoning_traces(db_path)

    # Analyze patterns
    analyze_reasoning_patterns(db_path)

    print(f"\n🗄️  Database saved at: {db_path}")
    print("You can now use this data for further analysis!")


if __name__ == "__main__":
    main()
