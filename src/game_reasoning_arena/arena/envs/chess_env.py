"""Environment Simulator for the game of chess.

This module implements the ChessEnv class, which simulates the game of
cgess using the OpenSpiel framework.
"""

from typing import Any, Dict, Optional
from envs.open_spiel_env import OpenSpielEnv

class ChessEnv(OpenSpielEnv):
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
        """Returns the color associated with the player.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: 'White' for player 0, 'Black' for player 1.
        """
        return "White" if agent_id == 0 else "Black"

    def render_board(self, agent_id: int) -> str:
            """Renders the board in a readable ASCII format.

            Args:
                agent_id (int): The player's ID (ignored for now).

            Returns:
                str: A human-friendly string representation of the board.
            """
            fen = self.state.to_string()
            return self.render_board_from_fen(fen)

    def render_board_from_fen(self, fen: str) -> str:
        """Converts a FEN string to an ASCII board layout.

        Args:
            fen (str): The board in FEN notation.

        Returns:
            str: The rendered board with rows and pieces.
        """
        board, *_ = fen.split(" ")
        rows = board.split("/")
        display = ""
        for r in rows:
            row = ""
            for char in r:
                if char.isdigit():
                    row += "." * int(char)
                else:
                    row += char
            display += " ".join(row) + "\n"
        return display

    def explain_uci_moves(self, agent_id: int) -> List[str]:
        """Returns UCI moves with natural language explanations.

        Args:
            agent_id (int): The player's ID.

        Returns:
            List[str]: Descriptions like 'e2e4 → Move pawn from e2 to e4'.
        """
        explanations = []
        legal = self.state.legal_actions(agent_id)
        for a in legal:
            uci = self.game.action_to_string(self.state, a)
            if len(uci) >= 4:
                from_sq = uci[:2]
                to_sq = uci[2:4]
                explanations.append(f"{uci} → Move from {from_sq} to {to_sq}")
            else:
                explanations.append(uci)
        return explanations

    def describe_legal_actions(self, agent_id: int) -> str:
        """Describes legal moves using UCI format (e.g., e2e4).

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A list of UCI-encoded legal moves.
        """
        legal = self.state.legal_actions(agent_id)
        return ", ".join(
            self.game.action_to_string(self.state, a) for a in legal
        )

    def parse_llm_action(self, llm_reply: str) -> int:
        """Parses a UCI move string into an OpenSpiel action index.

        Args:
            llm_reply (str): A UCI move string like 'e2e4'.

        Returns:
            int: The corresponding action index for OpenSpiel.

        Raises:
            ValueError: If the UCI move is not legal in the current state.
        """
        uci = llm_reply.strip().lower()

        try:
            action_id = self.game.string_to_action(self.state, uci)
        except Exception as e:
            raise ValueError(f"Could not parse UCI move '{uci}': {e}")

        if action_id not in self.state.legal_actions():
            raise ValueError(f"Illegal move: '{uci}' is not a legal action.")

        return action_id


    def apply_llm_move(self, llm_reply: str) -> None:
        """Applies the LLM's chosen move to the game state.

        Args:
            llm_reply (str): The UCI move string from the LLM.

        Raises:
            ValueError: If the move is not valid or legal.
        """
        action = self.parse_llm_action(llm_reply)
        self.state.apply_action(action)
