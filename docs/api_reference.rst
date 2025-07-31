API Reference
=============

This section contains the complete API reference for Board Game Arena.

.. note::
   This documentation is automatically generated from the source code docstrings.

   The main entry points for using Board Game Arena are:

   * :class:`board_game_arena.arena.agents.base_agent.BaseAgent` - Base class for all agents
   * :class:`board_game_arena.backends.base_backend.BaseLLMBackend` - Base class for LLM backends
   * :func:`scripts.runner` - Main simulation runner
   * :func:`scripts.simulate` - Core simulation logic

Arena Module
------------

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

Environment Initializer
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.envs.env_initializer
   :members:
   :undoc-members:
   :show-inheritance:

OpenSpiel Environment (Base)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.envs.open_spiel_env
   :members:
   :undoc-members:
   :show-inheritance:

Tic-Tac-Toe
~~~~~~~~~~~

.. automodule:: board_game_arena.arena.envs.tic_tac_toe_env
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

Matrix Games
~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.envs.matrix_game_env
   :members:
   :undoc-members:
   :show-inheritance:

Hex
~~~

.. automodule:: board_game_arena.arena.envs.hex_env
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

LLM Registry
~~~~~~~~~~~~

.. automodule:: board_game_arena.backends.llm_registry
   :members:
   :undoc-members:
   :show-inheritance:

Backend Configuration
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: board_game_arena.backends.backend_config
   :members:
   :undoc-members:
   :show-inheritance:

Games Module
------------

Game Registry
~~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.games.registry
   :members:
   :undoc-members:
   :show-inheritance:

Game Loaders
~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.games.loaders
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

Config Parser
~~~~~~~~~~~~~

.. automodule:: board_game_arena.configs.config_parser
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

Logging Utilities
~~~~~~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.utils.loggers
   :members:
   :undoc-members:
   :show-inheritance:

Seeding Utilities
~~~~~~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.utils.seeding
   :members:
   :undoc-members:
   :show-inheritance:

Game Inspection
~~~~~~~~~~~~~~~

.. automodule:: board_game_arena.arena.utils.inspect_game
   :members:
   :undoc-members:
   :show-inheritance:

Scripts
-------

.. note::
   The main execution scripts are located in the ``scripts/`` directory and are not part of the package modules.
   They can be run directly from the command line.

   * ``scripts/runner.py`` - Main entry point for running simulations
   * ``scripts/simulate.py`` - Core simulation logic
   * ``scripts/train.py`` - Training utilities
   * ``scripts/evaluate.py`` - Evaluation tools

Analysis Modules
----------------

.. note::
   Analysis modules are located in the ``analysis/`` directory and provide post-processing capabilities.
   Some modules may require additional dependencies like ``seaborn`` and ``matplotlib``.

Post-Game Processing
~~~~~~~~~~~~~~~~~~~~

.. automodule:: post_game_processing
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   The ``reasoning_analysis`` module contains advanced NLP analysis capabilities but requires additional
   dependencies. Key functions include:

   * ``LLMReasoningAnalyzer`` - Main analysis class
   * ``categorize_reasoning`` - Categorizes reasoning patterns
   * ``generate_wordcloud`` - Creates visualization of reasoning patterns
