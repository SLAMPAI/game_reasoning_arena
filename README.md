# Project: OpenSpiel LLM Arena

## Quick Start

Want to jump right in? Here are the fastest ways to get started:

```bash
# 1. Install dependencies
pip install -r environment.yaml

# Run a quick random vs random test
python3 scripts/runner.py --config test_config.json

# Try an LLM vs random game (requires API key)
python3 scripts/runner.py --config test_config.json --override \
  agents.player_0.type=llm \
  agents.player_0.model=groq/llama3-8b-8192

# Override to llm_vs_llm with different models
python3 scripts/runner.py --config test_config.json --override \
  mode=llm_vs_llm \
  agents.player_0.type=llm \
  agents.player_0.model=groq/gemma-7b-it \
  agents.player_1.type=llm \
  agents.player_1.model=groq/llama3-70b-8192


# 4. Play different games
python3 scripts/runner.py --config test_config.json --override env_config.game_name=connect_four
python3 scripts/runner.py --config test_config.json --override env_config.game_name=kuhn_poker


# Fast model vs High-quality model
  python3 scripts/runner.py --config test_config.json --override \
  mode=llm_vs_llm \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_groq/gemma-7b-it \
  agents.player_1.type=llm \
  agents.player_1.model=litellm_groq/llama3-70b-8192

# Groq vs Together AI models
  python3 scripts/runner.py --config test_config.json --override \
  mode=llm_vs_llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  agents.player_1.type=llm \
  agents.player_1.model=litellm_together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

# Different backends: LiteLLM vs vLLM
  python3 scripts/runner.py --config test_config.json --override \
  mode=llm_vs_llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  agents.player_1.type=llm \
  agents.player_1.model=vllm_Qwen2-7B-Instruct

# Different Groq models
  python3 scripts/runner.py --config test_config.json --override \
  mode=llm_vs_llm \
  agents.player_0.model=litellm_groq/gemma-7b-it \
  agents.player_1.type=llm \
  agents.player_1.model=litellm_groq/mixtral-8x7b-32768

# Test with Connect Four (longer game)
  python3 scripts/runner.py --config test_config.json --override \
  env_config.game_name=connect_four \
  mode=llm_vs_llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  agents.player_1.type=llm \
  agents.player_1.model=litellm_groq/llama3-70b-8192 \
  num_episodes=3


**Model Naming Convention**

Models are now identified with backend prefixes for clarity:
- **LiteLLM models**: `litellm_<model_name>` (e.g., `litellm_groq/llama3-8b-8192`)
- **vLLM models**: `vllm_<model_name>` (e.g., `vllm_Qwen2-7B-Instruct`)

This allows **mixing multiple inference providers in the same simulation**. For example, you can have one LLM using Groq's API and another using local vLLM inference:

```json
{
  "agents": {
    "player_0": {
      "type": "llm",
      "model": "litellm_groq/llama3-8b-8192"
    },
    "player_1": {
      "type": "llm",
      "model": "vllm_Qwen2-7B-Instruct"
    }
  }
}
```

**Backend Configuration**

The system automatically loads models from both configuration files:
- Models from `src/configs/litellm_models.json` (API-based inference)
- Models from `src/configs/vllm_models.json` (local GPU inference)

```

**API keys** Create a `.env` file with your keys, for example:

GROQ_API_KEY=your_groq_key_here
TOGETHER_API_KEY=your_together_key_here


**Important:** If a model is not listed in `litellm_models.json`, it will not be available for use in the simulation. Always keep this file up to date with the models you want to support. If you encounter errors about missing models, check this file first.

## 0. Project Goal
The goal of this project is to evaluate the decision-making capabilities of Large Language Models (LLMs) by engaging them in simple games implemented using Google's OpenSpiel framework. The LLMs can play against:
1. A random bot.
2. Another LLM.
3. Themselves (self-play).

This project explores how LLMs interpret game states, make strategic decisions, and adapt their behavior through natural language prompts.

---

## 1. How to Run

### Prerequisites
1. **Python Environment**:
   - Python 3.7 or higher.
   - Install the required dependencies:
     ```bash
     pip install -r environment.yaml
     ```

2. **Install OpenSpiel Framework**:
   - Clone and set up OpenSpiel from its official repository:
     ```bash
     git clone https://github.com/deepmind/open_spiel.git
     cd open_spiel
     ./install.sh
     ```

3. **Project Setup**:
   - Clone this repository:
     ```bash
     git clone <repository-url>
     cd <repository-folder>
     # Installs using pyproject.toml
     conda run -n llm pip install -e .
     ```

---

### Running the Simulator

1. **Basic Usage Examples**:

   ```bash
   # Quick test with random vs random agents (default: tic_tac_toe)
   python3 scripts/runner.py --config test_config.json

   # Run 10 episodes of Connect Four
   python3 scripts/runner.py --config test_config.json --override env_config.game_name=connect_four num_episodes=10

   # Test different games
   python3 scripts/runner.py --config test_config.json --override env_config.game_name=kuhn_poker
   python3 scripts/runner.py --config test_config.json --override env_config.game_name=prisoners_dilemma

   # LLM vs Random using Groq
   python3 scripts/runner.py --config test_config.json --override \
     agents.player_0.type=llm \
     agents.player_0.model=groq/llama3-8b-8192 \
     mode=llm_vs_random

   # LLM vs LLM (self-play)
   python3 scripts/runner.py --config test_config.json --override \
     agents.player_0.type=llm \
     agents.player_0.model=groq/llama3-8b-8192 \
     agents.player_1.type=llm \
     agents.player_1.model=groq/llama3-8b-8192 \
     mode=llm_vs_llm

   # Different LLM providers
   python3 scripts/runner.py --config test_config.json --override \
     agents.player_0.type=llm \
     agents.player_0.model=together/meta-llama/Llama-2-7b-chat-hf \
     agents.player_1.type=random

   # Test system components
   python3 -c "from arena.games.registry import registry; print('✓ Available games:', list(registry._registry.keys()))"
   ```

2. **Configuration Examples**:

   Create custom JSON configuration files for different scenarios:

   ```json
   // Simple random vs random
   {
     "env_config": {"game_name": "tic_tac_toe"},
     "num_episodes": 5,
     "seed": 42,
     "agents": {
       "player_0": {"type": "random"},
       "player_1": {"type": "random"}
     }
   }

   // LLM vs Random
   {
     "env_config": {"game_name": "connect_four"},
     "num_episodes": 3,
     "seed": 123,
     "agents": {
       "player_0": {
         "type": "llm",
         "model": "litellm_groq/llama3-8b-8192"
       },
       "player_1": {"type": "random"}
     },
     "llm_backend": {
       "max_tokens": 250,
       "temperature": 0.1
     }
   }

   // LLM vs LLM (different models)
   {
     "env_config": {"game_name": "kuhn_poker"},
     "num_episodes": 10,
     "agents": {
       "player_0": {
         "type": "llm",
         "model": "groq/llama3-8b-8192"
       },
       "player_1": {
         "type": "llm",
         "model": "together/meta-llama/Llama-2-7b-chat-hf"
       }
     }
   }
   ```

3. **Command-line Options**:
   - `--config`: Specify a JSON configuration file path.
                 Example: `python3 scripts/runner.py --config test_config.json`
   - `--override`: Allows modification of specific configuration values.
                   Examples:
                   ```bash
                   # Change game and episodes
                   python3 scripts/runner.py --config test_config.json --override env_config.game_name=connect_four num_episodes=5

                   # Change agents
                   python3 scripts/runner.py --config test_config.json --override agents.player_0.type=llm agents.player_0.model=groq/llama3-8b-8192

                   # Multiple overrides
                   python3 scripts/runner.py --config test_config.json --override \
                     env_config.game_name=kuhn_poker \
                     agents.player_0.type=llm \
                     agents.player_1.type=llm \
                     num_episodes=10
                   ```

4. Available games:
   - `tic_tac_toe`: Classic Tic-Tac-Toe
   - `connect_four`: Connect Four
   - `kuhn_poker`: Kuhn Poker
   - `prisoners_dilemma`: Iterated Prisoner's Dilemma
   - `matrix_pd`: Matrix Prisoner's Dilemma
   - `matching_pennies`: Matching Pennies (3P)
   - `matrix_rps`: Matrix Rock-Paper-Scissors

5. **Game-Specific Examples**:

   ```bash
   # Tic-Tac-Toe: Quick strategy game
   python3 scripts/runner.py --config test_config.json --override \
     env_config.game_name=tic_tac_toe \
     agents.player_0.type=llm \
     agents.player_0.model=groq/llama3-8b-8192 \
     num_episodes=5

   # Connect Four: Longer strategic game
   python3 scripts/runner.py --config test_config.json --override \
     env_config.game_name=connect_four \
     agents.player_0.type=llm \
     agents.player_0.model=together/meta-llama/Llama-2-7b-chat-hf \
     num_episodes=3

   # Kuhn Poker: Game with hidden information
   python3 scripts/runner.py --config test_config.json --override \
     env_config.game_name=kuhn_poker \
     agents.player_0.type=llm \
     agents.player_1.type=llm \
     num_episodes=10

   # Prisoner's Dilemma: Multi-round cooperation game
   python3 scripts/runner.py --config test_config.json --override \
     env_config.game_name=prisoners_dilemma \
     agents.player_0.type=llm \
     agents.player_1.type=random \
     num_episodes=1
   ```
---

## 2. Directory Structure

### Core Packages
- **`src/backends/`**: LLM backend management (LiteLLM, vLLM)
  - `llm_registry.py`: Central registry for model management
  - `litellm_backend.py`: LiteLLM integration
  - `vllm_backend.py`: vLLM integration (legacy)
  - `config.py`: Backend configuration management
- **`src/arena/games/`**: Game registration and discovery system
  - `registry.py`: Auto-discovery game registration
  - `loaders.py`: Game loader implementations with decorators
- **`src/arena/envs/`**: Environment simulator logic for each game
- **`src/arena/agents/`**: Agent implementations (Random, LLM, Human)
- **`src/arena/utils/`**: Shared utility functions (logging, plotting, etc.)
- **`src/configs/`**: Configuration files and parsing

### Configuration
- **`test_config.json`**: Simple test configuration for quick testing
- **`src/configs/example_config.json`**: Full example configuration
- **`src/configs/litellm.json`**: LiteLLM model configurations
- **`.env`**: Environment variables for API keys

### Results and Analysis
- **`results/`**: Stores CSV and JSON files with simulation results
- **`analysis/`**: Post-game processing and analysis scripts
- **`plots/`**: Generated visualizations and charts

### Scripts
- **`scripts/runner.py`**: Main entry point for running simulations
- **`scripts/simulate.py`**: Core simulation logic

### Tests
- **`tests/`**: Unit tests for utilities and simulators

---

## 3. Adding a New Game
To add a new game to the OpenSpiel LLM Arena, follow these steps:

### Step 1: Implement the Game Environment
1. Create a new environment file in **`src/arena/envs/`** folder.
   - The environment should inherit from `OpenSpielEnv`.
   - Example: `my_new_game_env.py`

### Step 2: Register the Game Loader
2. Add a new game loader in **`src/arena/games/loaders.py`** using the decorator pattern:
   ```python
   @registry.register(
       name="my_new_game",
       module_path="arena.games.loaders",
       class_name="MyNewGameLoader",
       environment_path="arena.envs.my_new_game_env.MyNewGameEnv",
       display_name="My New Game"
   )
   class MyNewGameLoader(GameLoader):
       @staticmethod
       def load():
           return pyspiel.load_game("my_new_game")
   ```

### Step 3: Test the Game
3. The game will be automatically discovered by the registry system. Test it:
   ```bash
   # Verify the game is registered
   python3 -c "from arena.games.registry import registry; print('Available games:', list(registry._registry.keys()))"

   # Test with your new game
   python3 scripts/runner.py --config test_config.json --override env_config.game_name=my_new_game
   ```

**That's it!** The new auto-discovery system will automatically detect and register your game without needing to modify any central registry files.
---

## 4. Adding a New Agent

### Step 1: Implement the Agent Class
1. Create a new file in **`src/arena/agents/`**, e.g., `rl_agent.py`
2. Ensure it inherits from `BaseAgent`
3. Implement the required methods:
   ```python
   from .base_agent import BaseAgent

   class RLAgent(BaseAgent):
       def __init__(self, model=None):
           super().__init__(model)

       def compute_action(self, observation, legal_actions):
           # Your RL logic here
           return selected_action
   ```

### Step 2: Register the Agent
1. Modify **`src/arena/agents/agent_registry.py`**:
   ```python
   from .rl_agent import RLAgent

   def register_agent(agent_type: str, agent_class):
       # Registration logic

   # Register your agent
   register_agent("rl", RLAgent)
   ```

### Step 3: Use the Agent in Configuration
1. Update your config file to use the new agent:
   ```json
   {
     "agents": {
       "player_0": {
         "type": "rl",
         "model": "my_trained_rl_model"
       },
       "player_1": {
         "type": "random"
       }
     }
   }
   ```

### Step 4: Test the Agent
```bash
python3 scripts/runner.py --config test_config.json --override agents.player_0.type=rl agents.player_0.model=my_model
```

---

## 5. LLM Backend Configuration

### Supported Backends
- **LiteLLM**: Unified interface for multiple LLM providers (OpenAI, Anthropic, Groq, etc.)
- **vLLM**: Local model hosting (legacy support)
- **Hybrid**: Both backends available

### Environment Setup
1. Create a `.env` file in the project root:
   ```bash
   # API Keys for LiteLLM providers
   GROQ_API_KEY=your_groq_key_here
   TOGETHER_API_KEY=your_together_key_here
   FIREWORKS_API_KEY=your_fireworks_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```

2. Configure models in `src/configs/litellm_models.json` and `src/configs/vllm_models.json`:

   **LiteLLM models (`src/configs/litellm_models.json`):**
   ```json
   [
     "litellm_groq/llama3-8b-8192",
     "litellm_groq/mixtral-8x7b-32768",
     "litellm_together_ai/meta-llama/Llama-2-7b-chat-hf"
   ]
   ```

   **vLLM models (`src/configs/vllm_models.json`):**
   ```json
   {
     "models": [
       {
         "name": "vllm_Qwen2-7B-Instruct",
         "model_path": "/path/to/models/Qwen/Qwen2-7B-Instruct",
         "tokenizer_path": "/path/to/models/Qwen/Qwen2-7B-Instruct",
         "description": "Qwen2 7B Instruct model for local inference"
       }
     ]
   }
   ```

### Backend Selection
Backends are now automatically selected based on model name prefixes:
- Models starting with `litellm_` use the LiteLLM backend
- Models starting with `vllm_` use the vLLM backend

Example configuration:
```json
{
  "llm_backend": {
    "max_tokens": 250,
    "temperature": 0.1,
    "default_model": "litellm_groq/llama3-8b-8192"
  }
}
```

---

## 6. Contribution Guidelines

### Steps to Contribute:
1. Fork this repository.
2. Create a feature branch.
3. Follow the directory structure and coding style outlined in this README.
4. Add appropriate unit tests for your contribution.
5. Submit a pull request with a detailed explanation of your changes.

---

## 7. Example Output

### Game: Tic-Tac-Toe
```
Current state of Tic-Tac-Toe:
x.o
...
...
LLM chooses action: 4
...
Final state of Tic-Tac-Toe:
x.o
..x
.o.
Scores: {'LLM_1': 1.0, 'Random_Bot': -1.0}
```

### Game: Rock-Paper-Scissors
```
Final state of Rock-Paper-Scissors:
Terminal? true
History: 0, 1
Returns: -1,1
Scores: {'google/flan-t5-small': -1.0, 'gpt2': 1.0}
Results saved to results/rock_paper_scissors_results.json
```

### Game: Connect Four
```
Current state of Connect Four:
.....
.....
...o.
..x..
.....
.....
...
Final state of Connect Four:
x wins!
Scores: {'google/flan-t5-small': 1.0, 'Random_Bot': -1.0}
```
