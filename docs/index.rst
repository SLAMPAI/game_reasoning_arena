Welcome to Board Game Arena's documentation!
============================================

Board Game Arena is a research platform for training and evaluating AI agents in board games using Large Language Models and reinforcement learning techniques.

## Project Overview

Board Game Arena provides a comprehensive framework for:

* **Multi-Agent Testing**: Compare LLMs vs Random, LLM vs LLM, and Self-play scenarios
* **Multiple Game Types**: Strategy games, poker variants, cooperation games, and zero-sum games
* **Flexible Backends**: Support for API-based (LiteLLM) and local (vLLM) inference
* **Cross-Provider Compatibility**: Mix different LLM providers within the same game
* **Extensible Architecture**: Easy to add new games, agents, and analysis tools

## Project Structure

The codebase is organized as follows:

.. code-block:: text

   board_game_arena/
   ├── src/board_game_arena/      # Core framework
   │   ├── backends/              # LLM backend management
   │   │   ├── llm_registry.py    # Registry for available models
   │   │   ├── litellm_backend.py # API-based model interface
   │   │   ├── vllm_backend.py    # Local model interface
   │   │   └── backend_config.py  # Backend configuration
   │   ├── arena/                 # Game simulation framework
   │   │   ├── games/             # Game registration system
   │   │   │   ├── registry.py    # Auto-discovery registry
   │   │   │   └── loaders.py     # Game loader implementations
   │   │   ├── envs/              # Game environments
   │   │   │   ├── tic_tac_toe_env.py
   │   │   │   ├── connect_four_env.py
   │   │   │   ├── kuhn_poker_env.py
   │   │   │   └── ...            # Additional game environments
   │   │   ├── agents/            # Agent implementations
   │   │   │   ├── llm_agent.py   # LLM-powered agent
   │   │   │   ├── random_agent.py # Random baseline agent
   │   │   │   ├── human_agent.py # Human player interface
   │   │   │   └── base_agent.py  # Agent base class
   │   │   └── utils/             # Utilities & helpers
   │   │       ├── loggers.py     # Game logging utilities
   │   │       ├── seeding.py     # Reproducibility tools
   │   │       └── cleanup.py     # Resource management
   │   └── configs/               # Configuration files (YAML)
   │       ├── litellm_models.yaml    # API model definitions
   │       ├── vllm_models.yaml       # Local model definitions
   │       ├── ray_config.yaml        # Distributed computing
   │       ├── example_config.yaml    # Basic configuration
   │       └── multi_game_base.yaml   # Multi-game template
   ├── scripts/                   # Execution scripts
   │   ├── runner.py              # Main entry point
   │   ├── simulate.py            # Core simulation logic
   │   ├── train.py               # Training utilities
   │   └── evaluate.py            # Evaluation tools
   ├── analysis/                  # Post-processing & analysis
   │   ├── reasoning_analysis.py  # LLM reasoning categorization
   │   └── post_game_processing.py # Game outcome analysis
   ├── tests/                     # Unit & integration tests
   ├── docs/                      # Documentation source
   ├── results/                   # Experiment output (CSV, JSON)
   ├── plots/                     # Generated visualizations
   ├── slurm_jobs/                # SLURM cluster scripts
   ├── environment.yaml           # Conda dependencies
   ├── pyproject.toml             # Package configuration
   └── .env                       # API keys (create manually)

## Available Games

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
   api_reference
   games
   agents

.. toctree::
   :caption: Analysis & Evaluation
   :maxdepth: 2

   analysis
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

   changelog
   license
