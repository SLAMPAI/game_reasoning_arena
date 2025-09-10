"""Simulator for Matrix Games.

This module implements the MatrixGameEnvclass, which handles various
matrix games like Rock-Paper-Scissors and Prisoner's Dilemma using
the OpenSpiel framework.
"""

from typing import Any, Dict, List, Optional
from .open_spiel_env import OpenSpielEnv

class MatrixGameEnv(OpenSpielEnv):
    """Environment Simulator for Matrix Games."""

    def __init__(self, game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None,
                 seed: Optional[int] = None):
        """
        Args:
            game: The OpenSpiel game object.
            game_name: A string representing the name of the game.
            player_types: A dictionary mapping player IDs to their types
            (e.g., human, random).
            max_game_rounds: Maximum number of rounds
                             for iterated games (optional, default is None).
        """
        super().__init__(game, game_name, player_types, max_game_rounds, seed)


    def apply_action(self, action: List[int]):
        """Applies the given list of actions to the environment.

        Args:
            action: If the game is simultaneous-move,
                it is a list of actions (one for each player).
        """
        self.state.apply_actions(action)

    def _state_to_observation(self) -> Dict[int, Dict[str, Any]]:
        """
        Generate the observation for the matrix game.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping from agent ID to observations.
        """

        # Create observations for each player
        observations = {}
        for player_id in range(self.state.num_players()):
            observations[player_id] = {
                "state_string": f"Matrix game - Player {player_id}",
                "legal_actions": self.state.legal_actions(player_id),
                "prompt": self._generate_prompt(player_id)
            }

        return observations

    def _generate_prompt(self, agent_id: int) -> str:
        """Generate a prompt for the matrix game.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A formatted prompt for the matrix game.
        """
        if self.state.is_terminal():
            return ""

        # Get action descriptions
        legal_actions = self.state.legal_actions(agent_id)
        action_descriptions = []
        for action in legal_actions:
            action_name = self.state.action_to_string(agent_id, action)
            action_descriptions.append(f"{action}: {action_name}")

        action_list = "\n".join(action_descriptions)

        prompt = f"""You are Player {agent_id} in the game: {self.game_name}
                This is not a repeated/iterated game.

                Available actions:
                {action_list}

                What action do you choose? Reply only with the action number.

                First, think through the game strategy
                and explain your reasoning.
                Only after that, decide on the best action to take.

                Reply only in the following JSON format:
                {{
                'reasoning': <str>,
                'action': <int>
                }}"""

        return prompt

    def render_board(self, agent_id: int) -> str:
        # Matrix games have no spatial board; return a basic description.
        return "Matrix game – no board representation available"
