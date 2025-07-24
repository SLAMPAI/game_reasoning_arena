# Board Game Arena

A framework for evaluating Large Language Models (LLMs) through strategic game-playing using Google's OpenSpiel. Test LLM decision-making capabilities in games like Tic-Tac-Toe, Connect Four, Poker, and more.


### Key Features
- **Multi-Agent Testing**: LLMs vs Random, LLM vs LLM, Self-play
- **Multiple Game Types**: Strategy, poker, cooperation, zero-sum games
- **Flexible Backends**: Support for API-based (LiteLLM) and local (vLLM) inference
- **Cross-Provider**: Mix different LLM providers in the same game
- **Extensible**: Easy to add new games and agents

___

## Installation


### Prerequisites
- **OpenSpiel Framework** (see [detailed setup](#openspiel-setup) below)
- **API Keys** for LLM providers (liteLLM and VLLM supported)

### Setup
```bash
# Clone the repository
git clone https://github.com/SLAMPAI/board_game_arena.git
cd board_game_arena

# Install dependencies
conda env create -f environment.yaml

# Install the package in development mode
conda activate board_game_arena
pip install -e .

# Create a .env file for the environment variables
touch .env
```

#### API Keys Setup
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_key_here
TOGETHER_API_KEY=your_together_key_here
FIREWORKS_API_KEY=your_fireworks_key_here
OPENAI_API_KEY=your_openai_key_here
```

#### OpenSpiel Setup
Install OpenSpiel framework:
```bash
git clone https://github.com/deepmind/open_spiel.git
cd open_spiel
./install.sh
cd ..
```

### Test the Installation
```bash
# Run a quick random vs random test
python3 scripts/runner.py --config test_config.json
```

___
### Game Examples
```bash
# Different games
python3 scripts/runner.py --config test_config.json --override env_config.game_name=connect_four
python3 scripts/runner.py --config test_config.json --override env_config.game_name=kuhn_poker

# LLM vs Random (requires API key)
python3 scripts/runner.py --config test_config.json --override \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192

# LLM vs LLM
python3 scripts/runner.py --config test_config.json --override \
  mode=llm_vs_llm \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_groq/gemma-7b-it \
  agents.player_1.type=llm \
  agents.player_1.model=litellm_groq/llama3-70b-8192

# Mixed backends (liteLLM vs vLLM)
python3 scripts/runner.py --config test_config.json --override \
  mode=llm_vs_llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  agents.player_1.model=vllm_Qwen2-7B-Instruct

# Multi-episode tournament
python3 scripts/runner.py --config test_config.json --override \
  env_config.game_name=connect_four \
  mode=llm_vs_llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  agents.player_1.model=litellm_groq/llama3-70b-8192 \
  num_episodes=10
```

___
## Configuration


### Model Naming Convention
Models use backend prefixes for clarity:
- **LiteLLM models**: `litellm_<provider>/<model>` (e.g., `litellm_groq/llama3-8b-8192`)
- **vLLM models**: `vllm_<model>` (e.g., `vllm_Qwen2-7B-Instruct`)

### Backend Configuration

The system loads models from configuration files:
- `src/configs/litellm_models.json` - API-based models
- `src/configs/vllm_models.json` - Local GPU models

**Important**: Models must be listed in these files to be available for use.

___
### Available Games
- `tic_tac_toe` - Classic 3x3 grid game
- `connect_four` - Drop pieces to connect four
- `kuhn_poker` - Simple poker with hidden information
- `prisoners_dilemma` - Cooperation vs defection
- `matrix_pd` - Matrix form prisoner's dilemma
- `matching_pennies` - Zero-sum matching game
- `matrix_rps` - Rock-paper-scissors matrix game

---

## Usage Guide

### Command-Line Interface

**Basic Syntax:**
```bash
python3 scripts/runner.py --config <config_file> [--override key=value ...]
```

**Common Commands:**
```bash

# Verify available games
python3 -c "from board_game_arena.arena.games.registry import registry; print('Available games:', list(registry._registry.keys()))"
```

### Configuration Files

Create custom JSON configuration files for different scenarios:

**Simple Random vs Random:**
```json
{
  "env_config": {"game_name": "tic_tac_toe"},
  "num_episodes": 5,
  "seed": 42,
  "agents": {
    "player_0": {"type": "random"},
    "player_1": {"type": "random"}
  }
}
```

**LLM vs Random:**
```json
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
```

**LLM vs LLM (Cross-Provider):**
```json
{
  "env_config": {"game_name": "kuhn_poker"},
  "num_episodes": 10,
  "agents": {
    "player_0": {
      "type": "llm",
      "model": "litellm_groq/llama3-8b-8192"
    },
    "player_1": {
      "type": "llm",
      "model": "litellm_together_ai/meta-llama/Llama-2-7b-chat-hf"
    }
  }
}
```

### Game-Specific Examples
```bash
# Tic-Tac-Toe: Quick strategy game
python3 scripts/runner.py --config test_config.json --override \
  env_config.game_name=tic_tac_toe \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  num_episodes=5

# Connect Four: Longer strategic game
python3 scripts/runner.py --config test_config.json --override \
  env_config.game_name=connect_four \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_together_ai/meta-llama/Llama-2-7b-chat-hf \
  num_episodes=3

# Kuhn Poker: Game with hidden information
python3 scripts/runner.py --config test_config.json --override \
  env_config.game_name=kuhn_poker \
  agents.player_0.type=llm \
  agents.player_1.type=llm \
  num_episodes=10

# Prisoner's Dilemma: Multi-round cooperation
python3 scripts/runner.py --config test_config.json --override \
  env_config.game_name=prisoners_dilemma \
  agents.player_0.type=llm \
  agents.player_1.type=random \
  num_episodes=1
```

---

## Project Structure

```
board_game_arena/
├── src/
│   ├── backends/          # LLM backend management
│   │   ├── llm_registry.py
│   │   ├── litellm_backend.py
│   │   ├── vllm_backend.py
│   │   └── config.py
│   ├── arena/
│   │   ├── games/         # Game registration system
│   │   │   ├── registry.py
│   │   │   └── loaders.py
│   │   ├── envs/          # Game environments
│   │   ├── agents/        # Agent implementations
│   │   └── utils/         # Utilities & helpers
│   └── configs/           # Configuration files
│       ├── litellm_models.json
│       ├── vllm_models.json
│       └── example_config.json
├── scripts/
│   ├── runner.py          # Main entry point
│   └── simulate.py        # Core simulation logic
├── tests/                 # Unit tests
├── results/               # Output data (CSV, JSON)
├── analysis/              # Post-processing scripts
├── plots/                 # Generated visualizations
├── test_config.json       # Quick test config
├── environment.yaml       # Dependencies
├── pyproject.toml         # Package configuration
└── .env                   # API keys (create manually)

```

---

## Development Guide

### Adding a New Game

The system uses auto-discovery for game registration, making it easy to add new games:

**Step 1: Create Game Environment**
```python
# src/arena/envs/my_new_game_env.py
from .base_env import OpenSpielEnv

class MyNewGameEnv(OpenSpielEnv):
    def __init__(self, game_config):
        super().__init__(game_config)
        # Your game-specific initialization
```

**Step 2: Register Game Loader**
```python
# Add to src/arena/games/loaders.py
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

**Step 3: Test**
```bash
# Verify registration
python3 -c "from board_game_arena.arena.games.registry import registry; print(list(registry._registry.keys()))"

# Test the game
python3 scripts/runner.py --config test_config.json --override env_config.game_name=my_new_game
```

### Adding a New Agent

**Step 1: Implement Agent Class**
```python
# src/arena/agents/my_agent.py
from .base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, model=None, **kwargs):
        super().__init__(model)
        # Your initialization logic

    def compute_action(self, observation, legal_actions):
        # Your decision logic here
        return selected_action
```

**Step 2: Register Agent**
```python
# Update src/arena/agents/agent_registry.py
from .my_agent import MyAgent

# Add to registration
register_agent("my_agent", MyAgent)
```

**Step 3: Use in Configuration**
```json
{
  "agents": {
    "player_0": {
      "type": "my_agent",
      "model": "optional_model_parameter"
    }
  }
}
```

**Step 4: Test**
```bash
python3 scripts/runner.py --config test_config.json --override \
  agents.player_0.type=my_agent \
  agents.player_0.model=my_model
```


---

## Example Output

### Tic-Tac-Toe Game
```
Current state of Tic-Tac-Toe:
x.o
...
...
LLM (llama3-8b) chooses action: 4
...
Final state of Tic-Tac-Toe:
x.o
..x
.o.
Winner: Player 0 (LLM)
Scores: {'LLM_llama3-8b': 1.0, 'Random_Bot': -1.0}
```

### Connect Four Game
```
Current state of Connect Four:
......
......
......
...o..
..xo..
.xxo..

LLM (groq/llama3-70b) chooses action: 3
...
Final state of Connect Four:
......
......
..x...
..xo..
..xo..
.xxo..
Winner: Player 0 (LLM)
Game completed in 12 moves
```

### Tournament Results
```
Tournament Results (10 episodes):
├── connect_four_groq_llama3-8b_vs_groq_llama3-70b
│   ├── Player 0 wins: 3/10 (30%)
│   ├── Player 1 wins: 6/10 (60%)
│   └── Draws: 1/10 (10%)
└── Results saved to: results/tournament_2025-07-23_14-30-15.json
```


---

## Resources

- **OpenSpiel Documentation**: [https://github.com/deepmind/open_spiel](https://github.com/deepmind/open_spiel)
- **LiteLLM Documentation**: [https://litellm.ai](https://litellm.ai)
- **Issues & Bug Reports**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and community interaction

---

*Built for AI research and strategic game analysis*

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

---
---

## Contributing

We welcome contributions! Here's how to get started:

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Follow the directory structure and coding style outlined in this README.
4. Add appropriate unit tests for your contribution.
5. Submit a pull request with a detailed explanation of your changes.



### Code Guidelines
- Follow PEP 8 for Python code style
- Add docstrings to new functions and classes
- Include type hints where appropriate
- Write unit tests for new features
- Update documentation as needed

### Areas for Contribution
- **New Games**: Add support for additional OpenSpiel games
- **New Agents**: Implement RL agents, tree search agents, etc.
- **Analysis Tools**: Visualization and statistical analysis
- **Backend Support**: Additional LLM providers or local models
- **Performance**: Optimization and caching improvements

___
## Citation
If you found this work useful, please consider citing:

```
@article{cipolina-kun2025board_game_arena,
    title={},
    author={},
    year={2025},
    journal={arXiv},
    url={https://arxiv.org/abs/2}
}
```

## License

This code is made available under a [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license, as found in the [LICENSE](LICENSE) file. Some portions of the project are subject to separate license terms outlined in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
