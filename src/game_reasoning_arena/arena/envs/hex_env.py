"""Simulator for Hex.

This module implements the TicTacToeSimulator class, which simulates games of
Hex using the OpenSpiel framework.
"""

from typing import Any, Dict, Optional
from .open_spiel_env import OpenSpielEnv


class HexEnv(OpenSpielEnv):
    """Environment Simulator for Hex."""

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
        """Describes legal actions as indices on a hex board.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: Legal action numbers and a flattened board index layout.
        """
        legal = self.state.legal_actions(agent_id)
        size = self.game.board_size  # Usually 11

        # Create a flat index grid (diagonal shape)
        grid = []
        idx = 0
        for r in range(size):
            indent = "  " * r  # diagonal effect
            row = " ".join(f"{idx + c:2}" for c in range(size))
            grid.append(f"{indent}{row}")
            idx += size

        index_grid = "\n".join(grid)
        return f"{legal} (board indices)\n\nBoard index layout:\n{index_grid}"

    def render_board(self, agent_id: int) -> str:
        """Renders the Hex board with diagonal layout and legend.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A diagonally-aligned board with y/o/., and a legend.
        """
        legend = "Legend: y = Player 0, o = Player 1, . = empty cell\n"
        raw = self.state.observation_string(agent_id)
        symbols = [char for char in raw if char in ("y", "o", ".")]

        size = self.game.board_size  # typically 11
        rows = []
        idx = 0
        for row in range(size):
            indent = "  " * row
            row_cells = symbols[idx:idx+size]
            rows.append(indent + " ".join(row_cells))
            idx += size

        board = "\n".join(rows)
        return f"{legend}\n{board}"
