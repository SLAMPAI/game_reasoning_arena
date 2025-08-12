"""
OpenSpiel Game List

This script lists all available games in the OpenSpiel framework.
Useful for discovering which games can be used in the board game arena.
"""

import pyspiel

# List all available games in OpenSpiel
available_games = pyspiel.registered_names()
print("\n".join(available_games))
