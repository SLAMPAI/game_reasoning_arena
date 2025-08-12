"""
random_agent.py

Implements an agent that selects a random action.
"""

import random
from typing import Dict, Any
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Agent that selects an action uniformly at random from the legal actions.
    """

    def __init__(self, seed: int = None, *args, **kwargs):
        """
        Args:
            llm (Any): The LLM instance (e.g., an OpenAI API wrapper, or any callable).
            game_name (str): The game's name for context in the prompt.
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.
         """
        super().__init__(agent_type="random")
        self.random_generator = random.Random(seed)

    def compute_action(self, observation: Dict[str,Any]) -> Dict[str, Any]:
        """
        Randomly picks a legal action.

        Args:
            observation (Dict[str, Any]): The observation dictionary with:
                - legal_actions: List of legal actions for current player.

        Returns:
            Dict[str, Any]: Dictionary with "action" and "reasoning" keys.
        """

        #TODO: see if we can change to OS' native bot:
        '''
        game = pyspiel.load_game("kuhn_poker")  # Example: Tic-Tac-Toe
        state = game.new_initial_state()
        # Create a uniform random bot (player 0) with a fixed seed
        bot = pyspiel.make_uniform_random_bot(0, 42)
        action = bot.step(state) --> chooses a random action
        print(action)


        '''

        action = self.random_generator.choice(observation["legal_actions"])
        return {"action": action, "reasoning": "random choice"}
