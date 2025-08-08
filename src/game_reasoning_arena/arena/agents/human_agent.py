"""
human_agent.py

Implements an agent that asks the user for input to choose an action.
"""

from typing import Any, Dict, Optional, List
from .base_agent import BaseAgent
#from agents.llm_utils import generate_prompt #TODO: fix this - the prompt for human agents!!
#TODO: the human agents also need a prompt! but not on the HTML format!

class HumanAgent(BaseAgent):
    """An agent that queries the user for an action."""

    def __init__(self, game_name: str):
        """
        Args:
            game_name (str): A human-readable name for the game,
            used for prompting.
        """
        super().__init__(agent_type="human")
        self.game_name = game_name

    def compute_action(self, observation: Dict[str,Any]) -> Dict[str, Any]:
        """
        Prompts the user for a legal move.

        Args:
            observation (Dict[str, Any]): The observation dictionary with:
                - legal_actions: List of legal actions for current player.
                - state_string: Current game state.
                - info: Additional information.

        Returns:
            Dict[str, Any]: Dictionary with "action" and "reasoning" keys.
        """
        legal_actions=observation["legal_actions"]
        state=observation.get("state_string")
        info = observation.get("info",None)
       # prompt= observation.get("prompt",None)  #TODO: fix this for human agents!!

        # Same prompt as the LLM agent
        #if prompt is None:
        prompt = generate_prompt(self.game_name, str(state), legal_actions, info = info)

        print(prompt)
        while True:
            try:
                action = int(input("Enter your action (number): "))
                if action in legal_actions:
                    return {
                        "action": action,
                        "reasoning": "human selected"
                    }
            except ValueError:
                pass
            print("Invalid action. Please choose from:", legal_actions)


def generate_prompt(game_name: str,
                    state: str,
                    legal_actions: List[int],
                    info: Optional[str] = None) -> str:
    """Generate a natural language prompt for the LLM to decide the next move.

    Args:
        game_name (str): The name of the game.
        state (str): The current game state as a string.
        legal_actions (List[int]): The list of legal actions available to the player.
        info (Optional[str]): Additional information to include in the prompt (optional).

    Returns:
        str: A prompt string for the LLM.
    """
    info_text = f"{info}\n" if info else ""

    return (
        f"You are playing the Game: {game_name}\n"
        f"State:\n{state}\n"
        f"Legal actions: {legal_actions}\n"
        f"{info_text}"
        "Your task is to choose the next action. Provide only the number of "
        "your next move from the list of legal actions."
    )