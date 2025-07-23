'''
agent_registry.py

Centralized agent registry for dynamic agent loading.
Agent 'types' are classes
that receive an observation (board state) and return an action (move).
'''

from .random_agent import RandomAgent
from .human_agent import HumanAgent
from .llm_agent import LLMAgent


# Register all available agent types
AGENT_REGISTRY = {
    "llm": LLMAgent,
    "random": RandomAgent,
    "human": HumanAgent,
}
