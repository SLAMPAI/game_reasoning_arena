Installation
============

This guide covers how to install and set up Game Reasoning Arena for your research and development needs.

Quick Installation
------------------

Conda Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have conda installed, this is the fastest way to get started:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/lcipolina/game_reasoning_arena.git
   cd game_reasoning_arena

   # Create and activate conda environment
   conda env create -f environment.yaml
   conda activate game_reasoning_arena

   # Install the package in development mode
   pip install -e .

Pip Installation
~~~~~~~~~~~~~~~~

If you prefer pip or don't have conda:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/lcipolina/game_reasoning_arena.git
   cd game_reasoning_arena

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
   git clone https://github.com/SLAMPAI/game_reasoning_arena.git
   cd game_reasoning_arena

   # Set up conda environment with development dependencies
   conda env create -f environment.yaml
   conda activate game_reasoning_arena

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

For accelerated local model inference. Install Pytorch with CUDA support


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
   python scripts/runner.py --config src/game_reasoning_arena/configs/example_config.yaml --log_level DEBUG

   # Test specific games
   python scripts/runner.py --config src/game_reasoning_arena/configs/kuhn_poker_llm_vs_llm.yaml --num_games 5

   # Test with different backends
   python scripts/runner.py --config src/game_reasoning_arena/configs/example_config.yaml --backend litellm

Expected Output
~~~~~~~~~~~~~~~

You should see output similar to:

.. code-block:: text

   Running simulation...
   Initializing LLM registry with automatic backend detection
   [DEBUG] OpenSpielEnv created with game_name: tic_tac_toe
   game terminated
   Running post-game processing...
   Starting post-game processing...
   Merged logs saved as CSV to results/merged_logs_YYYYMMDD_HHMMSS.csv
   Game Outcomes Summary:
   terminated    XXXX
   truncated      XXX
   Name: status, dtype: int64
   Simulation completed.



Quick Interactive Test
~~~~~~~~~~~~~~~~~~~~~~

Test the installation by running a simple game:

.. code-block:: bash

   # Run a quick tic-tac-toe game with random agents
   python scripts/runner.py --config src/game_reasoning_arena/configs/example_config.yaml --override \
     env_configs.0.game_name=tic_tac_toe \
     agents.player_0.type=random \
     agents.player_1.type=random \
     num_episodes=1

Expected output should show game progress and results.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'game_reasoning_arena'**

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


Next Steps
----------

Once installation is complete:

1. Read the :doc:`quickstart` guide for your first experiment
2. Explore the :doc:`examples` for common use cases
3. Check out :doc:`games` to see available game environments
4. Learn about :doc:`agents` for different AI agent types
