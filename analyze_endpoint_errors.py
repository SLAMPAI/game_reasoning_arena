#!/usr/bin/env python3
"""
Script to analyze LLM endpoint errors in Game Reasoning Arena databases.
This helps identify when games were terminated due to LLM endpoint failures
rather than actual gameplay.
"""

import sqlite3
import os
import pandas as pd
from pathlib import Path


def analyze_endpoint_errors(db_path):
    """Analyze a single database for endpoint errors."""
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)

    # Query for endpoint errors (marked with illegal_action = -999)
    endpoint_errors = pd.read_sql_query("""
        SELECT
            game_name,
            episode,
            turn,
            agent_id,
            reason,
            timestamp
        FROM illegal_moves
        WHERE illegal_action = -999
        AND reason LIKE 'LLM endpoint failure:%'
        ORDER BY timestamp
    """, conn)

    # Query for total games
    total_games = pd.read_sql_query("""
        SELECT COUNT(DISTINCT episode) as total_episodes
        FROM game_results
    """, conn)

    # Query for move counts
    move_counts = pd.read_sql_query("""
        SELECT COUNT(*) as total_moves
        FROM moves
    """, conn)

    conn.close()

    return {
        'database': os.path.basename(db_path),
        'endpoint_errors': endpoint_errors,
        'total_episodes': total_games.iloc[0]['total_episodes'] if not total_games.empty else 0,
        'total_moves': move_counts.iloc[0]['total_moves'] if not move_counts.empty else 0,
        'error_rate': len(endpoint_errors) / total_games.iloc[0]['total_episodes']
                     if not total_games.empty and total_games.iloc[0]['total_episodes'] > 0 else 0
    }


def main():
    """Analyze all databases in the results directory."""
    results_dir = Path("results")

    if not results_dir.exists():
        print("❌ Results directory not found")
        return

    print("🔍 Analyzing LLM Endpoint Errors in Game Reasoning Arena")
    print("=" * 60)

    db_files = list(results_dir.glob("*.db"))
    total_errors = 0
    total_episodes = 0
    databases_with_errors = 0

    for db_file in sorted(db_files):
        result = analyze_endpoint_errors(db_file)
        if result is None:
            continue

        print(f"\n📊 Database: {result['database']}")
        print(f"   Episodes: {result['total_episodes']}")
        print(f"   Total moves: {result['total_moves']}")
        print(f"   Endpoint errors: {len(result['endpoint_errors'])}")
        print(f"   Error rate: {result['error_rate']:.2%}")

        if len(result['endpoint_errors']) > 0:
            databases_with_errors += 1
            total_errors += len(result['endpoint_errors'])

            print("   🚨 Error details:")
            for _, error in result['endpoint_errors'].iterrows():
                error_msg = error['reason'].replace('LLM endpoint failure: ', '')[:50]
                print(f"      Episode {error['episode']}, Turn {error['turn']}: {error_msg}...")

        total_episodes += result['total_episodes']

    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print(f"Total databases analyzed: {len(db_files)}")
    print(f"Databases with endpoint errors: {databases_with_errors}")
    print(f"Total episodes across all databases: {total_episodes}")
    print(f"Total endpoint errors: {total_errors}")
    if total_episodes > 0:
        print(f"Overall endpoint error rate: {total_errors/total_episodes:.2%}")

    print("\n💡 INTERPRETATION")
    if total_errors == 0:
        print("✅ No endpoint errors found - all LLM calls succeeded")
    else:
        print("⚠️  Endpoint errors detected. Common causes:")
        print("   - API quota exceeded")
        print("   - Invalid model names")
        print("   - Network connectivity issues")
        print("   - Authentication problems")
        print("   - Model service downtime")

    print(f"\n🎯 IMPACT ON DATA QUALITY")
    if total_errors == 0:
        print("✅ Clean data - all recorded moves are genuine LLM decisions")
    elif total_errors / total_episodes < 0.05:  # Less than 5%
        print("✅ Minimal impact - endpoint errors affect <5% of episodes")
    elif total_errors / total_episodes < 0.20:  # Less than 20%
        print("⚠️  Moderate impact - endpoint errors affect <20% of episodes")
    else:
        print("🚨 High impact - endpoint errors affect >20% of episodes")
        print("   Consider investigating LLM service reliability")


if __name__ == "__main__":
    main()
