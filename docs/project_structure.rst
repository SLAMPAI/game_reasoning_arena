Project Structure
=================

The Game Reasoning Arena codebase is organized as follows:

Directory Layout
----------------

.. code-block:: text

   game_reasoning_arena/
   ├── src/game_reasoning_arena/      # Core framework
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

Core Components
---------------

Source Code (``src/game_reasoning_arena/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Backends** (``backends/``)
  Core LLM integration layer supporting multiple inference backends:

  * ``llm_registry.py`` - Central registry for available models and providers
  * ``litellm_backend.py`` - API-based model interface (OpenAI, Anthropic, etc.)
  * ``vllm_backend.py`` - Local model interface for self-hosted models
  * ``backend_config.py`` - Configuration management for different backends

**Arena** (``arena/``)
  Game simulation and agent management framework:

  * ``games/`` - Game registration and auto-discovery system
  * ``envs/`` - OpenSpiel environment wrappers with Gymnasium-like interfaces
  * ``agents/`` - Agent implementations (LLM, Random, Human, Base classes)
  * ``utils/`` - Shared utilities (logging, seeding, cleanup, plotting)

**Configurations** (``configs/``)
  YAML-based configuration system:

  * ``litellm_models.yaml`` - API model definitions and parameters
  * ``vllm_models.yaml`` - Local model configurations
  * ``ray_config.yaml`` - Distributed computing setup
  * ``example_config.yaml`` - Basic usage examples
  * ``multi_game_base.yaml`` - Multi-game experiment templates

Execution Scripts (``scripts/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Main Entry Points**
  * ``runner.py`` - Primary simulation orchestrator with Ray integration
  * ``simulate.py`` - Core game simulation logic and agent coordination
  * ``train.py`` - Training utilities for RL agents
  * ``evaluate.py`` - Evaluation and benchmarking tools

Analysis & Results (``analysis/``, ``results/``, ``plots/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Post-Processing**
  * ``reasoning_analysis.py`` - LLM reasoning pattern categorization
  * ``post_game_processing.py`` - Game outcome analysis and statistics
  * ``results/`` - Experiment outputs (CSV, JSON format)
  * ``plots/`` - Generated visualizations and charts

Development & Deployment
~~~~~~~~~~~~~~~~~~~~~~~~

**Testing** (``tests/``)
  * Unit tests for core components
  * Integration tests for game environments
  * Configuration validation tests

**Documentation** (``docs/``)
  * Sphinx-based documentation source
  * API reference generation
  * User guides and tutorials

**Cluster Computing** (``slurm_jobs/``)
  * SLURM job submission scripts
  * Distributed experiment configurations
  * Resource management templates

**Environment Management**
  * ``environment.yaml`` - Conda environment specification
  * ``pyproject.toml`` - Python package configuration
  * ``.env`` - API keys and sensitive configuration (user-created)

Design Principles
-----------------

Modular Architecture
~~~~~~~~~~~~~~~~~~~~

The codebase follows a **modular design** where each component has clear responsibilities:

* **Separation of concerns** between game logic, agent behavior, and infrastructure
* **Plugin-style architecture** for easy addition of new games and agents
* **Configuration-driven** behavior to minimize code changes for experiments

Extensibility
~~~~~~~~~~~~~

* **Game environments** can be added by implementing the ``OpenSpielEnv`` interface
* **Agent types** extend the ``BaseAgent`` base class
* **LLM backends** implement the ``BaseLLMBackend`` interface
* **Analysis modules** can be added to the ``analysis/`` directory

Reproducibility
~~~~~~~~~~~~~~~

* **Deterministic seeding** across all random components
* **Comprehensive logging** of all agent decisions and game states
* **Version-controlled configurations** for experiment reproducibility
* **Standardized output formats** for analysis and comparison

Scalability
~~~~~~~~~~~

* **Ray integration** for distributed computing
* **SLURM support** for cluster environments
* **Batch processing** capabilities for large-scale experiments
* **Memory-efficient** game state management

See Also
--------

* :doc:`installation` - Setting up the development environment
* :doc:`game_loop` - Understanding the simulation architecture
* :doc:`api_reference` - Complete API documentation
* :doc:`contributing` - Guidelines for extending the framework
