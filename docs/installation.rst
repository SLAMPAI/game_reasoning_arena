Installation
============

This guide covers how to install and set up Board Game Arena for your research and development needs.

Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

* **Operating System**: Linux (Ubuntu 22.04+), macOS (10.15+), or Windows (WSL2)
* **Python**: 3.8 or later (3.9+ recommended)
* **Memory**: 4GB RAM minimum (8GB+ recommended for multi-agent experiments)
* **Storage**: 2GB free space

Python Requirements
~~~~~~~~~~~~~~~~~~~

* Python 3.8+
* pip (latest version)
* virtualenv or conda (recommended)

Quick Installation
------------------

Conda Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have conda installed, this is the fastest way to get started:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/lcipolina/board_game_arena.git
   cd board_game_arena

   # Create and activate conda environment
   conda env create -f environment.yaml
   conda activate board_game_arena

   # Install the package in development mode
   pip install -e .

Pip Installation
~~~~~~~~~~~~~~~~

If you prefer pip or don't have conda:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/lcipolina/board_game_arena.git
   cd board_game_arena

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -e .

Development Installation
------------------------

For Contributors and Developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you plan to contribute or modify the code:

.. code-block:: bash

   # Clone with development tools
   git clone https://github.com/lcipolina/board_game_arena.git
   cd board_game_arena

   # Set up conda environment with development dependencies
   conda env create -f environment.yaml
   conda activate board_game_arena

   # Install in editable mode with development extras
   pip install -e ".[dev]"

   # Install pre-commit hooks (optional)
   pre-commit install

Running Tests
~~~~~~~~~~~~~

Verify your installation by running the test suite:

.. code-block:: bash

   # Run basic tests
   python -m pytest tests/

   # Run specific game tests
   python scripts/runner.py --config configs/test_all_games.py

Configuration Setup
-------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Create a `.env` file in the project root for API keys:

.. code-block:: bash

   # Create .env file for API keys
   touch .env

Add your API keys to the `.env` file:

.. code-block:: text

   # OpenAI API Key (for LiteLLM backend)
   OPENAI_API_KEY=your_openai_api_key_here

   # Anthropic API Key (optional)
   ANTHROPIC_API_KEY=your_anthropic_api_key_here

   # Other provider keys as needed
   COHERE_API_KEY=your_cohere_key_here

**Note**: Never commit the `.env` file to version control. It's already included in `.gitignore`.

Backend Configuration
~~~~~~~~~~~~~~~~~~~~~

Choose your preferred LLM backend:

**Option 1: API-based models (LiteLLM)**

.. code-block:: bash

   # No additional setup needed, just add API keys to .env
   # Supports OpenAI, Anthropic, Cohere, and 100+ other providers

**Option 2: Local models (vLLM)**

.. code-block:: bash

   # Install vLLM for local model inference
   pip install vllm

   # Download a model (example with Hugging Face)
   python -c "
   from transformers import AutoTokenizer, AutoModelForCausalLM
   model_name = 'microsoft/DialoGPT-medium'
   AutoTokenizer.from_pretrained(model_name)
   AutoModelForCausalLM.from_pretrained(model_name)
   "

GPU Support (Optional)
~~~~~~~~~~~~~~~~~~~~~~

For accelerated local model inference:

.. code-block:: bash

   # Install PyTorch with CUDA support
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   # Or with pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Cluster Setup (Optional)
------------------------

SLURM Integration
~~~~~~~~~~~~~~~~~

For running experiments on SLURM clusters:

.. code-block:: bash

   # Install additional dependencies
   pip install ray[default]

   # Configure Ray for SLURM
   # Edit configs/ray_config.yaml as needed

   # Submit jobs using provided scripts
   sbatch slurm_jobs/run_simulation.sh

Ray Distributed Computing
~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-node experiments:

.. code-block:: bash

   # Install Ray
   pip install ray[default]

   # Start Ray cluster (head node)
   ray start --head --port=6379

   # Connect worker nodes
   ray start --address=<head_node_ip>:6379

Verification
------------

Test Your Installation
~~~~~~~~~~~~~~~~~~~~~~~

Run a quick test to verify everything is working:

.. code-block:: bash

   # Test basic functionality
   python scripts/runner.py --config configs/example_config.yaml --debug

   # Test specific games
   python scripts/runner.py --config configs/kuhn_poker_llm_vs_llm.yaml --num_games 5

   # Test with different backends
   python scripts/runner.py --config configs/example_config.yaml --backend litellm

Expected Output
~~~~~~~~~~~~~~~

You should see output similar to:

.. code-block:: text

   2024-08-01 10:30:15 [INFO] Initializing Board Game Arena...
   2024-08-01 10:30:15 [INFO] Loading configuration: configs/example_config.yaml
   2024-08-01 10:30:16 [INFO] Backend: litellm initialized successfully
   2024-08-01 10:30:16 [INFO] Game: tic_tac_toe loaded
   2024-08-01 10:30:16 [INFO] Agents: ['llm_agent', 'random_agent'] ready
   2024-08-01 10:30:16 [INFO] Starting simulation with 10 games...

Run Test Suite
~~~~~~~~~~~~~~

For a comprehensive verification:

.. code-block:: bash

   # Install test dependencies
   pip install pytest

   # Run full test suite
   pytest tests/

   # Run with verbose output
   pytest tests/ -v

   # Run specific test files
   pytest tests/test_all_games_config.yaml

Quick Interactive Test
~~~~~~~~~~~~~~~~~~~~~~

Test the installation by running a simple game:

.. code-block:: bash

   # Run a quick tic-tac-toe game with random agents
   python scripts/runner.py --config configs/example_config.yaml --override \
     env_configs.0.game_name=tic_tac_toe \
     agents.player_0.type=random \
     agents.player_1.type=random \
     num_episodes=1

Expected output should show game progress and results.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'board_game_arena'**

.. code-block:: bash

   # Make sure you installed in development mode
   pip install -e .

**OpenSpiel not found**

.. code-block:: bash

   # Install OpenSpiel via pip
   pip install open_spiel

**API Key Issues**

.. code-block:: bash

   # Check your .env file exists and has the right format
   cat .env
   # Ensure no extra spaces around the = sign

**Ray Connection Issues**

.. code-block:: bash

   # Check Ray status
   ray status

   # Restart Ray if needed
   ray stop
   ray start --head

**Memory Issues**

.. code-block:: bash

   # Reduce batch size or number of parallel games
   python scripts/runner.py --config configs/example_config.yaml --num_games 1 --batch_size 1

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/lcipolina/board_game_arena/issues>`_
2. Review the troubleshooting section in our documentation
3. Join our community discussions
4. Contact the development team

For installation-specific issues, please include:

* Your operating system and version
* Python version
* Full error message
* Steps to reproduce the issue

Next Steps
----------

Once installation is complete:

1. Read the :doc:`quickstart` guide for your first experiment
2. Explore the :doc:`examples` for common use cases
3. Check out :doc:`games` to see available game environments
4. Learn about :doc:`agents` for different AI agent types
