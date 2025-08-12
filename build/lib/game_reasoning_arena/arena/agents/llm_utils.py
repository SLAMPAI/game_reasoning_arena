# utils/llm_utils.py
"""Utility functions for Large Language Model (LLM) integration.

Provides helper functions to generate prompts and interact with LLMs
for decision-making in game simulations.
"""

from ...backends import initialize_llm_registry

initialize_llm_registry()


def format_prompt(input_text: str, request_explanation=True) -> str:
    """Formats the input prompt for reasoning-first output.

    Args:
        input_text (str): The game prompt.
        request_explanation (bool): Whether to request the model's reasoning.

    Returns:
        str: The correctly formatted prompt for the model.
    """

    # All models use simple prompt formatting since backend-specific
    # formatting (like chat templates) is handled by the backends themselves
    if request_explanation:
        input_text += (
            "\n\nFirst, think through the game strategy and explain "
            "your reasoning."
            "\nOnly after that, decide on the best action to take."
        )

    # Add JSON instruction
    json_instruction = (
        "\n\nReply only in the following JSON format:\n"
        "{\n  'reasoning': <str>,\n  'action': <int>\n}"
    )
    input_text += json_instruction

    return input_text
