"""
Analysis utilities: shared helpers for plotting and naming.
"""
from typing import Dict

# Human-friendly game name mapping for plot labels/titles
GAME_DISPLAY_NAMES: Dict[str, str] = {
    "matrix_pd": "Prisoner's Dilemma",
    "matrix_rps": "Rock Paper Scissors",
    "tic_tac_toe": "Tic Tac Toe",
    "connect_four": "Connect Four",
    "matching_pennies": "Matching Pennies",
    "kuhn_poker": "Kuhn Poker",
}


def display_game_name(game_name: str) -> str:
    """Return a reader-friendly name for an internal game code.

    Falls back to a prettified version if not in the mapping.
    """
    if not isinstance(game_name, str):
        return str(game_name)
    return GAME_DISPLAY_NAMES.get(
        game_name,
        game_name.replace("_", " ").title()
    )
