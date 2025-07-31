Agents
======

Board Game Arena provides various types of agents for game playing and experimentation.

Agent Types
-----------

LLM Agent
~~~~~~~~~

Uses Large Language Models to make game decisions through natural language reasoning.

.. code-block:: python

   from board_game_arena.arena.agents.llm_agent import LLMAgent

   agent = LLMAgent(
       name="GPT_Player",
       model="gpt-3.5-turbo",
       temperature=0.7
   )

**Features:**
* Natural language game understanding
* Reasoning about game state
* Configurable temperature for exploration
* Support for multiple LLM providers

Random Agent
~~~~~~~~~~~~

Makes random legal moves - useful as a baseline opponent.

.. code-block:: python

   from board_game_arena.arena.agents.random_agent import RandomAgent

   agent = RandomAgent(name="Random_Player")

**Features:**
* Fast execution
* Perfect baseline for comparison
* No training required

Human Agent
~~~~~~~~~~~

Allows human players to participate in games through a user interface.

.. code-block:: python

   from board_game_arena.arena.agents.human_agent import HumanAgent

   agent = HumanAgent(name="Human_Player")

**Features:**
* Interactive gameplay
* Useful for testing and demonstration
* Can be combined with web interface

Agent Configuration
-------------------

Agents are typically configured through YAML files:

.. code-block:: yaml

   agents:
     - type: "llm"
       name: "Player1"
       model: "gpt-4"
       temperature: 0.5
       system_prompt: "You are an expert board game player."

     - type: "random"
       name: "Player2"

Agent Interface
---------------

All agents implement the base agent interface:

.. code-block:: python

   class BaseAgent:
       def get_action(self, state, legal_actions):
           """Return chosen action given game state"""
           pass

       def reset(self):
           """Reset agent state for new game"""
           pass

Creating Custom Agents
----------------------

You can create custom agents by inheriting from ``BaseAgent``:

.. code-block:: python

   from board_game_arena.arena.agents.base_agent import BaseAgent

   class MyAgent(BaseAgent):
       def __init__(self, name="MyAgent"):
           super().__init__(name)
           # Initialize your agent

       def get_action(self, state, legal_actions):
           # Your decision logic here
           return chosen_action

       def reset(self):
           # Reset any internal state
           pass

For more details, see the :doc:`api_reference` and :doc:`contributing` sections.
