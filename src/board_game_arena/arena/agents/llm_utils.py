# utils/llm_utils.py
"""Utility functions for Large Language Model (LLM) integration.

Provides helper functions to generate prompts and interact with LLMs
for decision-making in game simulations.
"""

import os
import json
from typing import List, Optional
from transformers import AutoTokenizer, pipeline

from backends import initialize_llm_registry

initialize_llm_registry()

#TODO: add model_name on the arguments so we can try different models
def format_prompt(input_text: str, request_explanation=True) -> str:
    """Formats the input prompt for reasoning-first output, using Hugging Face's
    chat template function.

    Args:
        input_text (str): The game prompt.
        request_explanation (bool): Whether to request the model's reasoning.

    Returns:
        str: The correctly formatted prompt for the model.
    """

    # Note: All models now use simple prompt formatting since backend is determined by model prefix
    # For LiteLLM models, return the prompt with JSON formatting but no chat template
    if request_explanation:
        input_text += (
            "\n\nFirst, think through the game strategy and explain your reasoning."
            "\nOnly after that, decide on the best action to take."
        )

    # Add JSON instruction
    json_instruction = (
        "\n\nReply only in the following JSON format:\n"
        "{\n  'reasoning': <str>,\n  'action': <int>\n}"
    )
    input_text += json_instruction
    return input_text

    # Legacy vLLM path with tokenizer (only used if not using LiteLLM)
    # TODO: construir esto con el model name nomas y el model path del OS
    # model_path = "/p/data1/mmlaion/marianna/models/google/codegemma-7b-it"
    # model_path = "/p/data1/mmlaion/marianna/models/mistralai/Mistral-7B-v0.1"
    model_path = "/p/data1/mmlaion/marianna/models/Qwen/Qwen2-7B-Instruct"

    print("'llm_utils.py': Using hardcoded Qwen2-7B-Instruct for tokenizer!")
    # TODO: add in the log which model we are loading

    # If explanation is requested, add a message.
    if request_explanation:
       input_text += (
            "\n\nFirst, think through the game strategy and explain your reasoning."
            "\nOnly after that, decide on the best action to take."
        )

    # Modify prompt to request structured JSON output
    json_instruction = (
        "\n\nReply only in the following JSON format:\n"
        "{\n  'reasoning': <str>,\n  'action': <int>\n}"
    )

    input_text += json_instruction  # Append JSON request
    messages = [{"role": "user", "content": input_text}]

    # Format using apply_chat_template
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Patch a simple chat template if missing
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|user|>\\n' + message['content'] + '<|end|>' if message['role'] == 'user' "
            "else '<|assistant|>\\n' + message['content'] + '<|end|>' }}"
            "{% endfor %}"
        )

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return formatted_prompt  # This is passed to vLLM
