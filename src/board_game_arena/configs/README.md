# Game Testing Script

This directory contains a comprehensive test script to verify that all available games in the Board Game Arena work properly.

## test_all_games.py

A automated test script that:
- Tests all 7 available games listed in the main README
- Uses a fast LLM model (`litellm_groq/llama3-8b-8192`) vs Random player
- Runs 2 episodes per game for quick validation
- Provides detailed output and summary statistics

### Available Games Tested:
- `tic_tac_toe` - Classic 3x3 grid game
- `connect_four` - Drop pieces to connect four
- `kuhn_poker` - Simple poker with hidden information
- `prisoners_dilemma` - Cooperation vs defection (matrix form)
- `matrix_pd` - Matrix form prisoner's dilemma
- `matching_pennies` - Zero-sum matching game
- `matrix_rps` - Rock-paper-scissors matrix game

### Usage:

```bash
# From the project root directory
cd /path/to/board_game_arena
python3 src/board_game_arena/configs/test_all_games.py
```

### Features:
- **Automatic Configuration**: Creates temporary YAML configs for each game
- **Timeout Protection**: 5-minute timeout per game to prevent hanging
- **Clean Logging**: Clear success/failure indicators with emojis
- **Summary Report**: Final statistics showing which games passed/failed
- **Error Handling**: Graceful handling of timeouts and exceptions
- **Cleanup**: Automatically removes temporary config files

### Sample Output:
```
Board Game Arena - Testing All Available Games
============================================================
Testing 7 games with LLM vs Random
LLM Model: litellm_groq/llama3-8b-8192
Episodes per game: 2
============================================================

==================================================
Testing game: tic_tac_toe
==================================================
✅ tic_tac_toe: SUCCESS

[... other games ...]

============================================================
FINAL RESULTS SUMMARY
============================================================
tic_tac_toe          ✅ PASS
connect_four         ✅ PASS
kuhn_poker           ✅ PASS
[... etc ...]

📊 Summary:
   Total games tested: 7
   Successful: 7
   Failed: 0

🎉 All games passed successfully!
```

### Exit Codes:
- `0`: All games passed successfully
- `1`: One or more games failed

This makes it easy to use in CI/CD pipelines or automated testing workflows.
