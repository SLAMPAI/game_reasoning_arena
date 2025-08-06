"""
llm_agent.py

Implements an agent that queries an LLM for its action.
"""

import logging
import os
import re
import ray
import random
from typing import Any, Dict, List
from ...backends import generate_response
from .base_agent import BaseAgent

MAX_TOKENS = int(os.getenv("MAX_TOKENS", 250))
# The lower the more deterministic
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))


class LLMAgent(BaseAgent):
    """
    Agent that queries a language model (LLM) to pick an action.
    """

    def __init__(self, model_name: str, game_name: str):
        """
        Args:
            model_name (str): The name of the LLM to use (from registry).
            game_name (str): The game's name for context in the prompt.
        """
        super().__init__(agent_type="llm")
        self.game_name = game_name
        self.model_name = model_name

        # If backend == vllm Loads the LLM model to GPU
        #   if model_name not in LLM_REGISTRY: #validate model is available
        #     raise ValueError(f"LLM '{model_name}' is not registered.")
        # self.llm = LLM_REGISTRY[model_name]["model_loader"]()
        # self.model_name = model_name

    def compute_action(self, observation: Dict[str, Any]) -> int:
        """
        Uses the LLM to select an action from the legal actions.

        Args:
            observation (Dict[str, Any]): The observation dictionary with:
                - legal_actions: Set of legal actions for current player.
                - state_string: The current OpenSpiel state.
                - info: Additional information to include in the prompt.

        Returns:
            int: The action chosen by the LLM.
        """
        prompt = observation.get("prompt", None)
        legal_actions = observation.get("legal_actions", [])

        # Call batch function (use Ray if initialized, otherwise direct)
        try:
            if ray.is_initialized():
                action_dict = ray.get(batch_llm_decide_moves_ray.remote(
                    {0: self.model_name},
                    {0: prompt},
                    {0: legal_actions}
                ))
            else:
                action_dict = batch_llm_decide_moves(
                    {0: self.model_name},
                    {0: prompt},
                    {0: legal_actions}
                )
        except Exception as e:
            logging.warning(
                f"Error in LLM inference (Ray: {ray.is_initialized()}): {e}"
            )
            # Fallback to direct call if Ray fails
            action_dict = batch_llm_decide_moves(
                {0: self.model_name},
                {0: prompt},
                {0: legal_actions}
            )

        chosen_action = action_dict[0]["action"]
        reasoning = action_dict[0].get("reasoning", "N/A")
        return {"action": chosen_action, "reasoning": reasoning}


def batch_llm_decide_moves(
    model_names: Dict[int, str],
    prompts: Dict[int, str],
    legal_actions_dict: Dict[int, List[int]]
) -> Dict[int, Dict[str, Any]]:
    """
    Queries LLM models to decide moves for multiple players.
    Uses the new backends system to handle both LiteLLM and vLLM.
    """
    actions_dict = {}
    for player_id, model_name in model_names.items():
        legal_actions = legal_actions_dict.get(player_id, [0])
        try:
            response_text = generate_response(
                model_name=model_name,
                prompt=prompts[player_id],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            actions_dict[player_id] = {
                'action': extract_action(response_text, legal_actions),
                'reasoning': extract_reasoning(response_text)
            }

        except Exception as e:
            error_msg = f"Error with model {model_name} for player {player_id}"
            logging.error(f"{error_msg}: {e}")
            # Fallback: return random valid action
            if legal_actions:
                fallback_action = random.choice(legal_actions)
            else:
                fallback_action = 0
            actions_dict[player_id] = {
                'action': fallback_action,
                'reasoning': f"Error occurred, using random fallback: {str(e)}"
            }

    return actions_dict


@ray.remote
def batch_llm_decide_moves_ray(
    model_names: Dict[int, str],
    prompts: Dict[int, str],
    legal_actions_dict: Dict[int, List[int]]
) -> Dict[int, Dict[str, Any]]:
    """
    Ray remote version of batch_llm_decide_moves.
    Same functionality but can be executed as a Ray task.
    """
    return batch_llm_decide_moves(model_names, prompts, legal_actions_dict)


def extract_action(response_text: str, legal_actions: List[int]) -> int:
    """
    Extracts action from LLM response with intelligent fallback to random
    valid action.
    
    Args:
        response_text: The raw text response from the LLM
        legal_actions: List of valid actions the agent can take
    
    Returns:
        int: A valid action from the legal_actions list
    """
    # Try the primary pattern first
    match = re.search(r"'action'\s*:\s*(\d+)", response_text)
    if match:
        action = int(match.group(1))
        if action in legal_actions:
            return action
    
    # Try alternative patterns for different model outputs
    match = re.search(r'"action"\s*:\s*(\d+)', response_text)
    if match:
        action = int(match.group(1))
        if action in legal_actions:
            return action
    
    # Try to find any number that might be a valid action
    matches = re.findall(r'\b(\d+)\b', response_text)
    for match in matches:
        action = int(match)
        if action in legal_actions:
            return action
    
    # If no valid action found in response, fall back to random valid action
    if legal_actions:
        fallback_action = random.choice(legal_actions)
        logging.warning(
            f"No valid action found in response: '{response_text[:100]}...' "
            f"Using random fallback: {fallback_action}"
        )
        return fallback_action
    
    # Ultimate fallback if somehow no legal actions provided
    logging.error(
        f"No legal actions provided! Response: {response_text[:100]}..."
    )
    return 0


def extract_reasoning(response_text: str) -> str:
    """Fixes the issue that the LLM outputs a string instead of a dictionary"""
    # Try the primary pattern first
    match = re.search(r"'reasoning'\s*:\s*'(.*?)'", response_text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try alternative patterns
    match = re.search(r'"reasoning"\s*:\s*"(.*?)"', response_text, re.DOTALL)
    if match:
        return match.group(1)
    
    # If no structured reasoning found, return the first part of the response
    if response_text.strip():
        # Take first 200 characters as reasoning
        reasoning = response_text.strip()
        if len(reasoning) > 200:
            return reasoning[:200] + "..."
        return reasoning
    
    return "No reasoning provided"
