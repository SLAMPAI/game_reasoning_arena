API Reference
=============

This section contains the complete API reference for Board Game Arena.

   The main entry points for using Board Game Arena are:

   * :class:`board_game_arena.arena.agents.base_agent.BaseAgent` - Base class for all agents
   * :class:`board_game_arena.backends.base_backend.BaseLLMBackend` - Base class for LLM backends
   * :func:`scripts.runner` - Main simulation runner
   * :func:`scripts.simulate` - Core simulation logic

Project Structure
-----------------

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

Core Directory Overview
~~~~~~~~~~~~~~~~~~~~~~~

**src/board_game_arena/**: Main package containing all framework components

* **backends/**: LLM inference backends (LiteLLM, vLLM)
* **arena/**: Game simulation framework with agents, environments, games, and utilities
* **configs/**: YAML configuration files for models, games, and distributed computing

**scripts/**: Command-line tools and execution scripts

* **runner.py**: Main entry point for running experiments
* **simulate.py**: Core game simulation logic
* **train.py**: Training utilities and workflows
* **evaluate.py**: Evaluation and benchmarking tools

**analysis/**: Post-processing and analysis tools

* **reasoning_analysis.py**: LLM reasoning pattern analysis
* **post_game_processing.py**: Game outcome processing and statistics

**Supporting Directories**:

* **tests/**: Unit and integration tests
* **docs/**: Documentation source files (Sphinx)
* **results/**: Experiment outputs (CSV, JSON)
* **plots/**: Generated visualizations and figures
* **slurm_jobs/**: SLURM cluster job scripts


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
