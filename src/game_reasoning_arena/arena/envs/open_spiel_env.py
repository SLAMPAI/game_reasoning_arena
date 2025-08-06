"""
open_spiel_env.py

Implements a Gymnasium-like environment on top of an OpenSpiel game.
"""

from typing import Optional, Tuple, Dict, Any
import random
from abc import ABC
from game_reasoning_arena.arena.agents.llm_utils import format_prompt


class OpenSpielEnv(ABC):
    """Environment for OpenSpiel.

    Handles common functionality like state transitions, outcomes recording.
    """

    def __init__(self,
                 game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None,
                 seed: Optional[int] = None
                 ):
        """
        Args:
            game (Any): The OpenSpiel game object being simulated.
            game_name (str): A human-readable name for the game.
            player_type (Dict[str, str]): Maps "Player 1", "Player 2", ... to their types (human, random, llm, etc.).
            max_game_rounds (int): Maximum number of rounds for iterated games. Ignored by single-shot games.
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.game = game
        self.game_name = game_name
        self.player_types = player_types # List of strings
        self.max_game_rounds = max_game_rounds  # For iterated games only
        self.state = None
        self.info = {}
        self.terminated, self.truncated = False, False

        # Set game seed if supported by OpenSpiel
        if hasattr(game, "set_seed"):
            game.set_seed(seed)

        self.state = None

    def reset(self, seed: Optional[int]=None) -> Tuple[str, Dict[str, Any]]:
        """
        Resets the environment to an initial state and returns an initial observation.

        Args:
        seed (Optional[int]): Seed for environment randomization.

        Returns:
            Tuple[str, Dict[str, Any]]:
                - str: A string representation of the initial state.
                - Dict[str, Any]: Additional info
        """
        if seed is not None:
            self.set_seed(seed)
        if hasattr(self.game, "set_seed"):
            self.game.set_seed(seed)

        self.state = self.game.new_initial_state() # Instantiates the pyspiel game state
        self.terminated = False
        self.truncated = False
        self.info = {}

        # Handle chance nodes first (e.g., dealing cards in Kuhn Poker)
        if self.state.is_chance_node():
            self._solve_chance_nodes()

        return self._state_to_observation(), self.info

    def step(self, action_dict: Dict[int, int]) -> Tuple[Any, float, bool,bool, Dict[str, Any]]:
        """Applies the given action(s) to the environment and returns the new state.

        Args:
            action_dict (Dict[int, int]): A dictionary mapping agent IDs to actions.
                - For turn-based games: {current_player: action}
                - For simultaneous games: {player_0: action_0, player_1: action_1, ...}

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: A tuple containing:
                - observation (Any): The resulting state after the action.
                - reward (float): The reward obtained from this step.
                - terminated (bool): Whether the episode has ended normally.
                - truncated (bool): Whether the episode ended due to `max_game_rounds`.
                - info (Dict[str, Any]): Additional diagnostic information (e.g., final scores if done).
        """

        # Handle chance nodes
        if self.state.is_chance_node():
            self._solve_chance_nodes()
            return self._state_to_observation(), {}, False, False, {}

        # Validate actions and track illegal moves for metrics
        illegal_move_count = self.detect_illegal_moves(action_dict)
        if illegal_move_count > 0:
            self.info['illegal_moves_detected'] = illegal_move_count
            # Note: The simulation layer handles termination for illegal moves

        # Move environment to the next state
        if self.state.is_simultaneous_node():
            actions = [action_dict[player] for player in sorted(action_dict.keys())]
            self.state.apply_actions(actions)  # Multi-agent moves
        else:
            current_player = list(action_dict.keys())[0]
            self.state.apply_action(action_dict[current_player]) # Single action

        # Stepwise reward for each OpenSpiel-indexed agent
        reward_dict = self._compute_reward()

        # Check termination due to game end
        self.terminated = self.state.is_terminal()

        # Check truncation due to max rounds (condition for iterated games)
        self.truncated = (
            self.max_game_rounds is not None
             and self.state.move_number() >= self.max_game_rounds
        )

        # If the game is finished, store final scores; otherwise, update current player
        if self.terminated or self.truncated:
            print("game terminated" if self.terminated else "game truncated")
            # Note: final rewards are corectly updated by the OpenSpiel rewards tracker.
            observation_dict = {agentID: None for agentID in list(action_dict.keys())} # No observation when the game ends
        else:
            observation_dict = self._state_to_observation() # Get next observation for all agents

        return observation_dict, reward_dict, self.terminated, self.truncated, self.info

    def render(self, mode: str = 'human'):
        """Print out the current state of the game."""
        if mode == 'human':
            print(f"Current state of {self.game_name}:\n{self.state}")

    def set_seed(self, seed: int = None):
        """
        Sets the random seed for the environment.

        Args:
            seed (int): The random seed.
        """
        self.random_generator = random.Random(seed)  # Ensure Python's RNG is seeded

        # Set game seed if OpenSpiel supports it
        if hasattr(self.game, "set_seed"):
            self.game.set_seed(seed)

        self.seed_value = seed  # Store the seed for tracking

    def detect_illegal_moves(self, actions_dict: Dict[int, int]) -> int:
        """
        Detects illegal moves by comparing chosen actions with OpenSpiel's legal actions.

        Args:
            actions_dict: Dictionary mapping player IDs to chosen actions.

        Returns:
            int: The number of illegal moves detected.
        """
        return sum(
            1 for player, action in actions_dict.items()
            if action not in self.state.legal_actions(player)
        )



    def close(self):
        """Cleanup."""
        pass

    # ----------------------------------------------------------------
    # Additional methods
    # ----------------------------------------------------------------

    def _state_to_observation(self) -> Dict[int, Dict[str, Any]]:
        """Returns the observation for each agent in the game.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping from agent ID to their respective observations.
        """

        agent_id = self.state.current_player()
        return {
            agent_id: {
                "state_string": self.state.observation_string(agent_id),
                "legal_actions": self.state.legal_actions(agent_id),
                "prompt": self._generate_prompt(agent_id) # Overriden in some child classes
            }
        }

    def get_player_symbol(self, agent_id: int) -> str:
        """Returns the symbol or marker used by a player in the current game.

        Args:
            agent_id (int): The player's ID (usually 0 or 1).

        Returns:
            str: A symbol, label, or description associated with the player.
            This defaults to 'Player {id}'
        """
        return f"Player {agent_id}"

    def render_board(self, agent_id: int) -> str:
        """Renders the game board with formatting.
        Should be implemented in the child class. Defaults to a simple board

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A string representation of the game board with action indices.
        """
        raise self.state.observation_string(agent_id)

    def describe_legal_actions(self, agent_id: int) -> str:
        """Returns a human-readable description of legal actions.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A textual description of the available actions.
        """
        legal = self.state.legal_actions(agent_id)
        return f"{legal}"  # default raw list

    def _generate_prompt(self, agent_id: int) -> str:
        """Generates a structured prompt for chat-based or non-chat models.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A formatted prompt.
        """

        if self.state.is_chance_node():
            return ""

        prompt_string = (
            f"You are playing the game: {self.game_name}\n"
            # f"You are player: {agent_id}\n"   # Confuses player ID
            f"and you are playing with the: "
            f"{self.get_player_symbol(agent_id)}.\n\n"
            f"the current move number is: {self.state.move_number()}\n"
            f"Board state:\n{self.render_board(agent_id)}\n"
            f"Available actions:{self.describe_legal_actions(agent_id)}\n\n"
            "What action do you choose? Reply only with the available "
            "action number."
        )
        return format_prompt(prompt_string)

    def _solve_chance_nodes(self) -> None:
        """Automatically plays chance nodes by selecting outcomes.

        Many OpenSpiel games involve chance nodes (e.g., dealing cards).
        This method ensures that chance nodes are resolved before player
        actions.
        """
        while self.state.is_chance_node():
            # List of (outcome, probability)
            outcomes, probs = zip(*self.state.chance_outcomes())
            # Pick a random outcome
            action = random.choices(outcomes, probs)[0]
            self.state.apply_action(action)  # Apply the chosen chance action

    def _compute_reward(self) -> Dict[int, float]:
        """Returns rewards indexed by OpenSpiel player indices (0, 1, ...)."""
        return {
            player: self.state.player_reward(player)
            for player in range(self.state.num_players())
        }
