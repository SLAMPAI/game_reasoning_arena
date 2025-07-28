# Board Game Arena

A framework for evaluating Large Language Models (LLMs) through strategic game-playing using Google's OpenSpiel game library. Allows to test LLM decision-making capabilities in games like Tic-Tac-Toe, Connect Four, Poker, and more.

### Key Features
- **Multi-Agent Testing**: LLMs vs Random, LLM vs LLM, Self-play
- **Multiple Game Types**: Strategy, poker, cooperation, zero-sum games
- **Flexible Backends**: Support for API-based (LiteLLM) and local (vLLM) inference
- **Cross-Provider**: Mix different LLM providers in the same game
- **Extensible**: Easy to add new games and agents

___
### Available Games
- `tic_tac_toe` - Classic 3x3 grid game
- `connect_four` - Drop pieces to connect four
- `kuhn_poker` - Simple poker with hidden information
- `prisoners_dilemma` - Cooperation vs defection (matrix form)
- `matrix_pd` - Matrix form prisoner's dilemma
- `matching_pennies` - Zero-sum matching game
- `matrix_rps` - Rock-paper-scissors matrix game

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
# Run a quick test
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml
```

___
### Game Examples
```bash
# Different games
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override env_config.game_name=connect_four
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override env_config.game_name=kuhn_poker

# LLM vs Random in a multi-episode tournament
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  num_episodes=10

# LLM vs LLM with mixed backends (liteLLM vs vLLM)
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
mode=llm_vs_llm \
agents.player_0.model=litellm_groq/llama3-8b-8192 \
agents.player_1.model=vllm_Qwen2-7B-Instruct
```

#### Game-Specific Examples
```bash
# Tic-Tac-Toe: Quick strategy game
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
  env_config.game_name=tic_tac_toe \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  num_episodes=5

# Connect Four: Longer strategic game
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
  env_config.game_name=connect_four \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_together_ai/meta-llama/Llama-2-7b-chat-hf \
  num_episodes=3

# Kuhn Poker: Game with hidden information
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
  env_config.game_name=kuhn_poker \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  agents.player_1.type=llm \
  agents.player_1.model=litellm_groq/llama3-8b-8192 \
  num_episodes=10

# Tic-Tac-Toe LLM vs Random: Classic strategy game
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
  env_config.game_name=tic_tac_toe \
  agents.player_0.type=llm \
  agents.player_1.type=random \
  num_episodes=5
```
___

## Configuration

### Model Naming Convention
Models use backend prefixes:
- **LiteLLM models**: `litellm_<provider>/<model>` (e.g., `litellm_groq/llama3-8b-8192`)
- **vLLM models**: `vllm_<model>` (e.g., `vllm_Qwen2-7B-Instruct`)

### Backend Configuration

The system loads models from configuration files:
- `src/configs/litellm_models.yaml` - API-based models
- `src/configs/vllm_models.yaml` - Local GPU models

**Important**: Models must be listed in these files to be available for use.



## Ray Integration for Parallel Execution

The Board Game Arena supports **Ray** for distributed and parallel execution, allowing you to:

- **Run multiple games in parallel** across different cores/machines
- **Parallelize episodes within games** for faster data collection
- **Distribute LLM inference** for batch processing
- **Scale experiments** on SLURM clusters or multi-GPU setups

## Configuration System


**Option 1: Combined Configuration File (YAML)**
```yaml
# Combined config with all settings in one file
env_config:
  game_name: tic_tac_toe
num_episodes: 5
agents:
  player_0:
    type: llm
    model: litellm_groq/llama3-8b-8192
  player_1:
    type: random
use_ray: true
parallel_episodes: true
ray_config:
  num_cpus: 8
  include_dashboard: false
```

**Option 2: Separate Ray Configuration (Recommended)**

```bash
# Use any existing config + separate Ray settings
python3 scripts/runner.py \
  --base-config src/board_game_arena/configs/multi_game_base.yaml \
  --ray-config src/board_game_arena/configs/ray_config.yaml \
  --override num_episodes=10 \
  --override agents.player_0.model=litellm_groq/llama3-70b-8192
```

**Option 3: Command-Line Override**
```bash
# Enable Ray with any existing configuration
  python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml \
 --override use_ray=true parallel_episodes=true
```

### Ray Configuration Options

The `ray_config.yaml` file contains only Ray-specific settings:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_ray` | Enable/disable Ray | `false` |
| `parallel_episodes` | Parallelize episodes within games | `false` |
| `ray_config.num_cpus` | Number of CPUs for Ray | Auto-detect |
| `ray_config.num_gpus` | Number of GPUs for Ray | Auto-detect |
| `ray_config.include_dashboard` | Enable Ray dashboard | `false` |
| `ray_config.dashboard_port` | Dashboard port | `8265` |
| `ray_config.object_store_memory` | Object store memory limit | Auto |



**Configuration Merging Order:**
The system merges configurations in this order (later overrides earlier):
1. Default configuration
2. Base config (`--base-config`)
3. Main config (`--config`)
4. Ray config (`--ray-config`)
5. CLI overrides (`--override`)



### SLURM Integration

For cluster environments, Ray automatically detects SLURM allocation:

```bash
# SLURM job with Ray
sbatch --nodes=2 --cpus-per-task=48 --gres=gpu:4 slurm_jobs/run_simulation.sh
```

The SLURM script (`slurm_jobs/run_simulation.sh`) handles:
- Multi-node Ray cluster setup
- Head node and worker initialization
- GPU allocation across nodes
- Environment variable configuration


**Debug Commands:**
```bash
# Check Ray status
ray status

# Monitor Ray dashboard (if enabled)
# Navigate to: http://localhost:8265

```

---

## Usage Guide

### Command-Line Interface

```bash
# Basic syntax
python3 scripts/runner.py --config <config_file> [--override key=value ...]

```

**Common Commands:**
```bash

# Verify available games
python3 -c "from src.board_game_arena.arena.games.registry import registry; print('Available games:', list(registry._registry.keys()))"
```

### Configuration Files

Create custom YAML configuration files for different scenarios:

**Simple Random vs Random (YAML):**
```yaml
env_config:
  game_name: tic_tac_toe
num_episodes: 5
seed: 42
agents:
  player_0:
    type: random
  player_1:
    type: random
```

___

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
│   └── configs/           # Configuration files (YAML)
│       ├── litellm_models.yaml
│       ├── vllm_models.yaml
│       ├── ray_config.yaml
│       └── example_config.yaml
├── scripts/
│   ├── runner.py          # Main entry point
│   └── simulate.py        # Core simulation logic
├── tests/                 # Unit tests
├── results/               # Output data (CSV, JSON)
├── analysis/              # Post-processing scripts
├── plots/                 # Generated visualizations
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
python3 -c "from src.board_game_arena.arena.games.registry import registry; print(list(registry._registry.keys()))"

# Test the game
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override env_config.game_name=my_new_game
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
```yaml
agents:
  player_0:
    type: my_agent
    model: optional_model_parameter
```

**Step 4: Test**
```bash
python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
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
