API Reference
=============

This section contains the complete API reference for Board Game Arena.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   api/arena
   api/agents
   api/envs
   api/backends

Arena
-----

.. automodule:: board_game_arena.arena
   :members:
   :undoc-members:
   :show-inheritance:

Agents
------

Base Agent
~~~~~~~~~~

.. automodule:: board_game_arena.arena.agents.base_agent
   :members:
   :undoc-members:
   :show-inheritance:

LLM Agent
~~~~~~~~~

.. automodule:: board_game_arena.arena.agents.llm_agent
   :members:
   :undoc-members:
   :show-inheritance:

Random Agent
~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.agents.random_agent
   :members:
   :undoc-members:
   :show-inheritance:

Human Agent
~~~~~~~~~~~

.. automodule:: board_game_arena.arena.agents.human_agent
   :members:
   :undoc-members:
   :show-inheritance:

Environments
------------

Base Environment
~~~~~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.envs.env_initializer
   :members:
   :undoc-members:
   :show-inheritance:

Connect Four
~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.envs.connect_four_env
   :members:
   :undoc-members:
   :show-inheritance:

Kuhn Poker
~~~~~~~~~~

.. automodule:: board_game_arena.arena.envs.kuhn_poker_env
   :members:
   :undoc-members:
   :show-inheritance:

Backends
--------

Base Backend
~~~~~~~~~~~~

.. automodule:: board_game_arena.backends.base_backend
   :members:
   :undoc-members:
   :show-inheritance:

LiteLLM Backend
~~~~~~~~~~~~~~~

.. automodule:: board_game_arena.backends.litellm_backend
   :members:
   :undoc-members:
   :show-inheritance:

vLLM Backend
~~~~~~~~~~~~

.. automodule:: board_game_arena.backends.vllm_backend
   :members:
   :undoc-members:
   :show-inheritance:
