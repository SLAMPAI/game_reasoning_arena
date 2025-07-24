"""
llm_agent.py

Implements an agent that queries an LLM for its action.
"""

import logging
import os
import re
import ray
from typing import Any, Dict
from ...backends import LLM_REGISTRY, generate_response
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

        # Call batch function (use Ray if initialized, otherwise direct)
        if ray.is_initialized():
            action_dict = ray.get(batch_llm_decide_moves_ray.remote(
                {0: self.model_name},
                {0: prompt}
            ))
        else:
            action_dict = batch_llm_decide_moves(
                {0: self.model_name},
                {0: prompt}
            )

        chosen_action = action_dict[0]["action"]
        reasoning = action_dict[0].get("reasoning", "N/A")
        return {"action": chosen_action, "reasoning": reasoning}


def batch_llm_decide_moves(
    model_names: Dict[int, str],
    prompts: Dict[int, str]
) -> Dict[int, Dict[str, Any]]:
    """
    Queries LLM models to decide moves for multiple players.
    Uses the new backends system to handle both LiteLLM and vLLM.
    """
    actions_dict = {}
    for player_id, model_name in model_names.items():
        try:
            response_text = generate_response(
                model_name=model_name,
                prompt=prompts[player_id],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            actions_dict[player_id] = {
                'action': extract_action(response_text),
                'reasoning': extract_reasoning(response_text)
            }

        except Exception as e:
            error_msg = f"Error with model {model_name} for player {player_id}"
            logging.error(f"{error_msg}: {e}")
            # Fallback: return first available action
            actions_dict[player_id] = {
                'action': 0,  # Default fallback action
                'reasoning': f"Error occurred: {str(e)}"
            }

    return actions_dict


# Ray-decorated version for distributed processing
@ray.remote
def batch_llm_decide_moves_ray(
    model_names: Dict[int, str],
    prompts: Dict[int, str]
) -> Dict[int, Dict[str, Any]]:
    """
    Ray remote version of batch_llm_decide_moves.
    Same functionality but can be executed as a Ray task.
    """
    return batch_llm_decide_moves(model_names, prompts)


def extract_action(response_text: str) -> int:
    """Fixes the issue that the LLM outputs a string instead of a dictionary"""
    match = re.search(r"'action'\s*:\s*(\d+)", response_text)
    return int(match.group(1))


def extract_reasoning(response_text: str) -> str:
    """Fixes the issue that the LLM outputs a string instead of a dictionary"""
    match = re.search(r"'reasoning'\s*:\s*'(.*?)'", response_text, re.DOTALL)
    return match.group(1) if match else "None"


def batch_llm_decide_moves_vllm(
    model_names: Dict[int, str],
    prompts: Dict[int, str]
) -> Dict[int, int]:
    """
    Original vLLM batch inference function.
    Queries vLLM in batch mode to decide moves for multiple players.
    """
    # Load all models in use
    llm_instances = {
        player_id: LLM_REGISTRY[model_name]["model_loader"]()
        for player_id, model_name in model_names.items()
    }
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    # Run batch inference for each LLM model separately
    actions_dict = {}
    for player_id, llm in llm_instances.items():
        response = llm.generate([prompts[player_id]], sampling_params)[0]
        response_text = response.outputs[0].text
        actions_dict[player_id] = {
            'action': extract_action(response_text),
            'reasoning': extract_reasoning(response_text)
        }
    return actions_dict
