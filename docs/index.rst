Welcome to Game Reasoning Arena's documentation!
============================================

Game Reasoning Arena is a research platform for training and evaluating AI agents in games using Large Language Models and reinforcement learning techniques.

Project Overview
----------------

Game Reasoning Arena provides a comprehensive framework for:

* **Multi-Agent Testing**: Compare LLMs vs Random, LLM vs LLM, and Self-play scenarios
* **Multiple Game Types**: Strategy games, poker variants, cooperation games, and zero-sum games
* **Flexible Backends**: Support for API-based (LiteLLM) and local (vLLM) inference
* **Cross-Provider Compatibility**: Mix different LLM providers within the same game
* **Extensible Architecture**: Easy to add new games, agents, and analysis tools

Quick Start
-----------

The framework is designed around a few key concepts:

* **Environments**: Game simulations built on OpenSpiel
* **Agents**: AI players including LLMs, random agents, and human players
* **Backends**: Flexible LLM inference systems (local and API-based)
* **Analysis Tools**: Post-game reasoning analysis and visualization

See the :doc:`installation` guide to get started, or explore the :doc:`api_reference` for detailed project structure information.

Available Games
---------------

* **tic_tac_toe** - Classic 3×3 grid strategy game
* **connect_four** - Drop pieces to connect four in a row
* **kuhn_poker** - Simple poker variant with hidden information
* **prisoners_dilemma** - Cooperation vs defection scenarios
* **matching_pennies** - Zero-sum matching game
* **matrix_rps** - Rock-paper-scissors in matrix form

.. toctree::
   :caption: Getting Started
   :maxdepth: 2

   installation
   quickstart

.. toctree::
   :caption: Core Framework
   :maxdepth: 2

   game_loop
   backends
   prompting
   api_reference
   code_flow
   games
   agents

.. toctree::
   :caption: Analysis & Evaluation
   :maxdepth: 2

   analysis
   reasoning_traces
   experiments

.. toctree::
   :caption: Examples & Tutorials
   :maxdepth: 2

   examples
   tutorials

.. toctree::
   :caption: Developer Guide
   :maxdepth: 2

   contributing
   extending

.. toctree::
   :caption: Extra Information
   :maxdepth: 2

.. license  # Commented out for blind conference submission
