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


class LLMEndpointError(Exception):
    """
    Custom exception raised when LLM endpoint fails.
    This allows the simulation to handle LLM errors differently
    from other issues.
    """
    def __init__(self, message: str, original_error: Exception,
                 model_name: str):
        super().__init__(message)
        self.original_error = original_error
        self.model_name = model_name


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
        except LLMEndpointError:
            # Re-raise LLM endpoint errors immediately - don't mask them
            raise
        except Exception as e:
            logging.warning(
                "Error in Ray execution (Ray: %s): %s", ray.is_initialized(), e
            )
            # Fallback to direct call if Ray fails (but not for LLM errors)
            action_dict = batch_llm_decide_moves(
                {0: self.model_name},
                {0: prompt},
                {0: legal_actions}
            )

        chosen_action = action_dict[0]["action"]
        reasoning = action_dict[0].get("reasoning", "N/A")
        return {
            "action": chosen_action,
            "reasoning": reasoning
        }


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
            logging.error("%s: %s", error_msg, e)
            # Instead of fallback, raise LLMEndpointError to terminate game
            # when an LLM endpoint fails.
            raise LLMEndpointError(
                f"LLM endpoint failed for {model_name}: {str(e)}",
                e,
                model_name
            ) from e

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
            "No valid action found in response: '%s...' "
            "Using random fallback: %s",
            response_text[:100], fallback_action
        )
        return fallback_action

    # Ultimate fallback if somehow no legal actions provided
    logging.error(
        "No legal actions provided! Response: %s...", response_text[:100]
    )
    return 0


def extract_reasoning(response_text: str) -> str:
    """
    Robustly extracts reasoning from LLM responses that may be in various
    formats. Handles JSON structures, plain text, and malformed responses.
    """
    if not response_text or not response_text.strip():
        return "No reasoning provided"

    response_text = response_text.strip()

    # Try to parse as JSON first (most structured format)
    try:
        import json
        # Handle cases where response might be wrapped in extra quotes
        # or have JSON-like structure
        cleaned_response = response_text
        if (cleaned_response.startswith('{"') or
                cleaned_response.startswith("{'{")):
            parsed = json.loads(cleaned_response)
            if 'reasoning' in parsed:
                return str(parsed['reasoning']).strip()
    except (json.JSONDecodeError, TypeError):
        pass

    # Try regex patterns for JSON-like structures with single quotes
    patterns = [
        # Standard JSON with double quotes
        r'"reasoning"\s*:\s*"(.*?)"(?:\s*[,}])',
        # JSON with single quotes
        r"'reasoning'\s*:\s*'(.*?)'(?:\s*[,}])",
        # Mixed quotes
        r'"reasoning"\s*:\s*\'(.*?)\'(?:\s*[,}])',
        r"'reasoning'\s*:\s*\"(.*?)\"(?:\s*[,}])",
        # With potential line breaks and more content after
        r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        r"'reasoning'\s*:\s*'([^']*(?:\\.[^']*)*)'",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.MULTILINE)
        if match:
            reasoning = match.group(1)
            # Clean up escaped characters
            reasoning = reasoning.replace('\\"', '"').replace("\\'", "'")
            reasoning = reasoning.replace('\\n', ' ').replace('\\t', ' ')
            # Remove excessive whitespace
            reasoning = ' '.join(reasoning.split())
            return reasoning.strip()

    # Check if the response starts with a JSON-like structure but is malformed
    if (response_text.startswith('{') or
            response_text.startswith('{\n') or
            "'reasoning':" in response_text or
            '"reasoning":' in response_text):

        # Try to extract reasoning from malformed JSON
        for line in response_text.split('\n'):
            if 'reasoning' in line and ':' in line:
                # Extract everything after the colon
                after_colon = line.split(':', 1)[1].strip()
                # Remove quotes and clean up
                after_colon = after_colon.strip('\'"')
                if after_colon and after_colon not in ['{', '}', ',']:
                    # Remove trailing punctuation that might be JSON syntax
                    after_colon = re.sub(r'[,}"\']$', '', after_colon)
                    return after_colon.strip()

    # If it's plain text reasoning (no JSON structure detected)
    # Return the full text without arbitrary truncation
    return response_text
