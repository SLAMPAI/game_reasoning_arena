#!/usr/bin/env python3
"""
Test script to verify LLM endpoint error handling.
This script creates a scenario that triggers an LLM endpoint error
to ensure proper termination and logging behavior.
"""

import os
import sys
import tempfile
import sqlite3
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / ".." / "src"
sys.path.insert(0, str(src_dir.resolve()))

from game_reasoning_arena.arena.agents.llm_agent import LLMEndpointError, batch_llm_decide_moves


def test_llm_endpoint_error():
    """Test that LLMEndpointError is raised when endpoint fails."""
    print("Testing LLM endpoint error handling...")

    # Create a test scenario with invalid model
    model_names = {0: "invalid_model_that_does_not_exist"}
    prompts = {0: "Test prompt"}
    legal_actions_dict = {0: [0, 1, 2]}

    try:
        # This should raise LLMEndpointError
        result = batch_llm_decide_moves(model_names, prompts, legal_actions_dict)
        print("❌ ERROR: Expected LLMEndpointError but got result:", result)
        return False
    except LLMEndpointError as e:
        print("✅ SUCCESS: LLMEndpointError raised correctly")
        print(f"   Model: {e.model_name}")
        print(f"   Message: {str(e)}")
        print(f"   Original error type: {type(e.original_error).__name__}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Wrong exception type raised: {type(e).__name__}: {e}")
        return False


def test_database_structure():
    """Test that the database can be queried for endpoint errors."""
    print("\nTesting database structure for error logging...")

    # Create a temporary database to test structure
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        # Test the database structure
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create the illegal_moves table (simulate logger creation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS illegal_moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                turn INTEGER,
                agent_id INTEGER,
                illegal_action INTEGER,
                reason TEXT,
                board_state TEXT,
                timestamp TEXT,
                run_id TEXT
            )
        """)

        # Insert a test LLM endpoint error
        cursor.execute("""
            INSERT INTO illegal_moves
            (game_name, episode, turn, agent_id, illegal_action, reason, board_state, timestamp, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "test_game", 1, 0, 0, -999,
            "LLM endpoint failure: Test error",
            "test board state",
            "2025-01-01T00:00:00",
            "test_run"
        ))

        conn.commit()

        # Query for endpoint errors
        cursor.execute("""
            SELECT * FROM illegal_moves
            WHERE illegal_action = -999 AND reason LIKE 'LLM endpoint failure:%'
        """)

        results = cursor.fetchall()
        conn.close()

        if results:
            print("✅ SUCCESS: Database can store and query LLM endpoint errors")
            print(f"   Found {len(results)} endpoint error record(s)")
            return True
        else:
            print("❌ ERROR: No endpoint error records found")
            return False

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    """Run all tests."""
    print("🧪 Testing LLM Endpoint Error Handling")
    print("=" * 50)

    test1_success = test_llm_endpoint_error()
    test2_success = test_database_structure()

    print("\n" + "=" * 50)
    if test1_success and test2_success:
        print("✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Run a simulation with an invalid LLM model to test end-to-end")
        print("2. Check the results database for endpoint error records")
        print("3. Verify that games terminate early instead of using fallback moves")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
