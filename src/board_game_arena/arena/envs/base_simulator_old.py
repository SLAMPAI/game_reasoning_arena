"""
Base Game Simulator (Legacy)

Legacy base class for simulating games. Contains core simulation logic
and player management functionality for OpenSpiel-based games.
"""

import os
import json
from typing import Dict, Any, List
from abc import ABC
import random
from board_game_arena.arena.agents.llm_utils import generate_prompt, llm_decide_move
from enum import Enum, unique


@unique
class PlayerId(Enum):
    CHANCE = -1
    SIMULTANEOUS = -2
    INVALID = -3
    TERMINAL = -4
    MEAN_FIELD = -5

    @classmethod
    def from_value(cls, value: int):
        """Returns the PlayerId corresponding to a given integer value.

        Args:
            value (int): The numerical value to map to a PlayerId.

        Returns:
            PlayerId: The matching enum member, or raises a ValueError if invalid.
        """
        for member in cls:
            if member.value == value:
                return member
        if value >= 0:  # Positive integers represent default players
            return None  # No enum corresponds to these values directly
        raise ValueError(f"Unknown player ID value: {value}")


class PlayerType(Enum):
    HUMAN = "human"
    RANDOM_BOT = "random_bot"
    LLM = "llm"
    SELF_PLAY = "self_play"


class GameSimulator(ABC):
    """Base class for simulating games with LLMs.

    Handles common functionality like state transitions, scoring, and logging.
    """

    def __init__(self, game: Any, game_name: str, llms: Dict[str, Any],
                 player_type: Dict[str, str], max_game_rounds: int = None):
        """
        Args:
            game (Any): The OpenSpiel game object being simulated.
            game_name (str): A human-readable name for the game (for logging and reporting).
            llms (Dict[str, Any]): A dictionary mapping player names (e.g., "Player 1")
                to their corresponding LLM instances. Can be empty if no LLMs are used.
            player_type (Dict[str, str]): A dictionary mapping player names to their types.
            max_game_rounds (int): Maximum number of rounds for iterated games. Ignored by single-shot games.
        """
        self.game = game
        self.game_name = game_name
        self.llms = llms
        self.player_type = player_type
        self.max_game_rounds = max_game_rounds  # For iterated games
        self.scores = {name: 0 for name in self.llms.keys()}  # Initialize scores

    def simulate(self, rounds: int = 1, log_fn=None) -> Dict[str, Any]:
        """Simulates a game for multiple rounds and computes metrics .

        Args:
            rounds: Number of times the game should be played.
            log_fn: Optional function to log intermediate states.

        Returns:
            Dict[str, Any]: Summary of results for all rounds.
        """
        outcomes = self._initialize_outcomes() # Reset the outcomes dictionary

        for _ in range(rounds):
            self.scores = {name: 0 for name in self.llms.keys()}  # Reset scores
            state = self.game.new_initial_state()

            while not state.is_terminal():
                if self.max_game_rounds is not None and state.move_number() >= self.max_game_rounds:
                    # If max_game_rounds is specified, terminate the game after the maximum number of rounds.
                    # The state.move_number() method tracks the number of moves (or rounds) within the game.
                    # This ensures that iterated games, such as the Iterated Prisoner's Dilemma,
                    # stop after the specified number of rounds, even if the game would naturally continue.
                    break
                if log_fn:
                    log_fn(state)

                # Collect actions
                current_player = state.current_player()
                player_id = self.normalize_player_id(current_player)

                if player_id == PlayerId.CHANCE.value:
                    # Handle chance nodes where the environment acts randomly.
                    self._handle_chance_node(state)
                elif player_id == PlayerId.SIMULTANEOUS.value:
                     # Handle simultaneous moves for all players.
                    actions = self._collect_actions(state)
                    state.apply_actions(actions)
                elif player_id == PlayerId.TERMINAL.value:
                    break
                elif current_player >= 0:  # Default players (turn-based)
                    legal_actions = state.legal_actions(current_player)
                    action = self._get_action(current_player, state, legal_actions)
                    state.apply_action(action)
                else:
                    raise ValueError(f"Unexpected player ID: {current_player}")

            # Record outcomes
            final_scores = state.returns()
            self._record_outcomes(final_scores, outcomes)

        return outcomes

    def _handle_chance_node(self, state: Any):
        """Handle chance nodes. Default behavior raises an error."""
        raise NotImplementedError("Chance node handling not implemented for this game.")


    def _collect_actions(self, state: Any) -> List[int]:
        """Collects actions for all players in a simultaneous-move game.

        Args:
            state: The current game state.

        Returns:
            List[int]: Actions chosen by all players.
        """
        return [
            self._get_action(player, state, state.legal_actions(player))
            for player in range(self.game.num_players())
        ]

    def _initialize_outcomes(self) -> Dict[str, Any]:
        """Initializes the outcomes dictionary."""
        return {"wins": {name: 0 for name in self.llms.keys()},
                "losses": {name: 0 for name in self.llms.keys()},
                "ties": 0
                }


    def _get_action(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Gets the action for the current player.

        Args:
            player: The index of the current player.
            state: The current game state.
            legal_actions: The legal actions available for the player.

        Returns:
            int: The action selected by the player.
        """
        player_name = f"Player {player + 1}"  # Map index to player name
        player_type = self.player_type.get(player_name)

        if player_type == PlayerType.HUMAN.value:
            return self._get_human_action(state, legal_actions)
        if player_type == PlayerType.RANDOM_BOT.value:
            return random.choice(legal_actions)
        if player_type == PlayerType.LLM.value:
            return self._get_llm_action(player, state, legal_actions)

        raise ValueError(f"Unknown player type for {player_name}: {player_type}")


    def _get_human_action(self, state: Any, legal_actions: List[int]) -> int:
        """Handles input for human players."""
        print(f"Current state of {self.game_name}:\n{state}")
        print(f"Your options: {legal_actions}") # Display legal moves to the user
        while True:
            try:
                action = int(input("Enter your action (number): "))
                if action in legal_actions: # Validate the move
                    return action
            except ValueError:
                pass
            print("Invalid action. Please choose from:", legal_actions)

    def _get_llm_action(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Handles LLM-based decisions."""
        player_name = f"Player {player + 1}"
        llm = self.llms[player_name]
        prompt = generate_prompt(self.game_name, str(state), legal_actions)
        return llm_decide_move(llm, prompt, tuple(legal_actions))

    def _apply_default_action(self, state):
        """
        Applies a default action when the current player is invalid.
        """
        state.apply_action(random.choice(state.legal_actions()))

    def _record_outcomes(self, final_scores: List[float], outcomes: Dict[str, Any]) -> str:
        """Records the outcome of a single game round.

        Args:
            final_scores (List[float]): Final cumulative scores of all players.
            outcomes (Dict[str, Any]): Dictionary to record wins, losses, and ties.

        Returns:
            str: Name of the winner or "tie" if there is no single winner.
        """
        # Check if all scores are equal (a tie)
        if all(score == final_scores[0] for score in final_scores):
            outcomes["ties"] += 1
            return "tie"

        # Find the maximum score and determine winners
        max_score = max(final_scores)
        winners = [name for i, name in enumerate(self.llms.keys()) if final_scores[i] == max_score]

        # Track losers as players who do not have the maximum score
        losers = [name for i, name in enumerate(self.llms.keys()) if final_scores[i] != max_score]

        # If there is one winner, record it; otherwise, record as a tie
        if len(winners) == 1:
            outcomes["wins"][winners[0]] += 1
            for loser in losers:
                outcomes["losses"][loser] += 1
            return winners[0]
        else:
            outcomes["ties"] += 1
            return "tie"


    def save_results(self, state: Any, final_scores: List[float]) -> None:
        """Save simulation results to a JSON file."""
        results = self._prepare_results(state, final_scores)
        filename = self._get_results_filename()

        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {filename}")

    def _prepare_results(self, state: Any, final_scores: List[float]) -> Dict[str, Any]:
        """Prepares the results dictionary for JSON serialization."""
        final_scores = final_scores.tolist() if hasattr(final_scores, "tolist") else final_scores
        return {
            "game_name": self.game_name,
            "final_state": str(state),
            "scores": self.scores,
            "returns": final_scores,
            "history": state.history_str(),
        }

    def _get_results_filename(self) -> str:
        """Generates the filename for saving results."""
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"{self.game_name.lower().replace(' ', '_')}_results.json")

    def log_progress(self, state: Any) -> None:
        """Log the current game state."""
        print(f"Current state of {self.game_name}:\n{state}")

    def normalize_player_id(self,player_id):
        """Normalize player_id to its integer value for consistent comparisons.

           This is needed as OpenSpiel has ambiguous representation of the playerID

        Args:
            player_id (Union[int, PlayerId]): The player ID, which can be an
                integer or a PlayerId enum instance.

        Returns:
            int: The integer value of the player ID.
        """
        if isinstance(player_id, PlayerId):
            return player_id.value  # Extract the integer value from the enum
        return player_id  # If already an integer, return it as is
