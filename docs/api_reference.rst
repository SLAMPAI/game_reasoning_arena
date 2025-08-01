API Reference
=============

This section contains the complete API reference for Board Game Arena.


   The main entry points for using Board Game Arena are:

   * :class:`board_game_arena.arena.agents.base_agent.BaseAgent` - Base class for all agents
   * :class:`board_game_arena.backends.base_backend.BaseLLMBackend` - Base class for LLM backends
   * :func:`scripts.runner` - Main simulation runner
   * :func:`scripts.simulate` - Core simulation logic


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
