"""Simulator for Tic-Tac-Toe.

This module implements the TicTacToeEnv class, which simulates games of
Tic-Tac-Toe using the OpenSpiel framework.
"""

from typing import Any, Dict, Optional
from .open_spiel_env import OpenSpielEnv

class TicTacToeEnv(OpenSpielEnv):
    """Environment Simulator for Tic-Tac-Toe."""

    def __init__(self, game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None,
                 seed: Optional[int] = None):
        """
        Args:
            game: The OpenSpiel game object.
            game_name: A string representing the name of the game.
            player_types: A dictionary mapping player IDs to their types (e.g., human, random).
            max_game_rounds: Maximum number of rounds
                             for iterated games (optional, default is None).
        """
        super().__init__(game, game_name, player_types, max_game_rounds, seed)


    def get_player_symbol(self, agent_id: int) -> str:
        """Returns the symbol used by a Tic Tac Toe player.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: 'x' for player 0, 'o' for player 1.
        """
        return "x" if agent_id == 0 else "o"

    def describe_legal_actions(self, agent_id: int) -> str:
        """Describes legal actions as board positions in a 3x3 grid.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A list of legal action numbers with positional meaning.
        """
        legal = self.state.legal_actions(agent_id)
        mapping_grid = (
            " 0 | 1 | 2\n"
            "-----------\n"
            " 3 | 4 | 5\n"
            "-----------\n"
            " 6 | 7 | 8\n"
        )
        #return f"{legal} (cell indices)\n\nCell layout:\n{mapping_grid}"
        return f"{legal} (cell indices)\n" #TODO: this is a test to see what works better for the agent

    def render_board(self, agent_id: int) -> str:
        """Renders the Tic-Tac-Toe board with separators and legend.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A board using 'x', 'o', '.', with a readable layout.
        """
        legend = "Legend: x = Player 0, o = Player 1, . = empty cell\n"
        raw = self.state.observation_string(agent_id)
        # Flatten and extract only meaningful symbols
        flat = [char for char in raw if char in ("x", "o", ".")]

        rows = []
        for i in range(0, 9, 3):
            rows.append(" " + " | ".join(flat[i:i+3]))
        board = "\n-----------\n".join(rows)
    #    return f"\n{board} \n{legend}"
        return f"\n{raw} \n" # TODO: this is a test to see what works better for the agent
