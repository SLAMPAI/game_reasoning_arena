Installation
============

The easiest way to get started with Board Game Arena is using conda:

.. code-block:: bash

   git clone https://github.com/lcipolina/board_game_arena.git
   cd board_game_arena
   conda env create -f environment.yaml
   conda activate board_game_arena
   pip install -e .


## Installation from Source

### Step-by-Step Instructions

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/lcipolina/board_game_arena.git
      cd board_game_arena

2. **Set up Python environment:**

   **Using conda (recommended):**

   .. code-block:: bash

      conda env create -f environment.yaml
      conda activate board_game_arena

   **Using virtualenv:**

   .. code-block:: bash

      python3 -m venv venv
      source venv/bin/activate

3. **Install dependencies:**

   .. code-block:: bash

      pip install -e .

4. **Verify installation:**

   .. code-block:: bash

      python -c "import board_game_arena; print('Installation successful!')"

## Configuration

### Environment Variables

Set up your LLM API keys (optional, only needed for LLM agents):

.. code-block:: bash

   # For OpenAI models
   export OPENAI_API_KEY="your-api-key-here"

   # For Anthropic models
   export ANTHROPIC_API_KEY="your-api-key-here"

   # Add to your ~/.bashrc or ~/.zshrc to persist

### GPU Support (Optional)

For GPU acceleration with certain backends:

.. code-block:: bash

   # Install PyTorch with CUDA support
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   # Or with pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Testing Your Installation

### Basic Functionality Test

.. code-block:: bash

   # Test with the provided example configuration
   python scripts/runner.py --config src/board_game_arena/configs/example_config.yaml

### Run Test Suite

.. code-block:: bash

   # Install test dependencies
   pip install pytest

   # Run tests
   pytest tests/

### Quick Interactive Test

Test the installation by running a simple game:

.. code-block:: bash

   # Run a quick tic-tac-toe game with random agents
   python scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
     env_configs.0.game_name=tic_tac_toe \
     agents.player_0.type=random \
     agents.player_1.type=random \
     num_episodes=1

Expected output should show game progress and results.
