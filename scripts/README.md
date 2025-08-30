# Scripts Folder

This directory contains various top-level scripts for running game experiments and evaluations.

## Contents

### Main Execution Scripts

1. **runner.py**
   - Primary script for running game experiments with LLM agents
   - Supports single models, multiple games, and Ray parallelization
   - Usage example:
     ```bash
     python3 scripts/runner.py --config src/game_reasoning_arena/configs/human_vs_random_config.yaml
     ```

2. **run_ray_multi_model.py**
   - Ray-accelerated multi-model experiment runner
   - Maximum parallelization: Models + Games + Episodes simultaneously
   - Best for large-scale multi-model evaluations
   - Usage example:
     ```bash
     python3 scripts/run_ray_multi_model.py --config src/game_reasoning_arena/configs/ray_multi_model.yaml --override use_ray=true
     ```

3. **run_multi_model_games.py**
   - Sequential multi-model experiment runner
   - Conservative resource usage, good for testing setups
   - Usage example:
     ```bash
     python3 scripts/run_multi_model_games.py --config src/game_reasoning_arena/configs/multi_game_multi_model.yaml
     ```



5. **configs.py** (Legacy)
   - Contains predefined config dictionaries (deprecated in favor of YAML configs)
   - Example: `default_simulation_config()` sets up tic-tac-toe with 5 rounds, a seed of 42, and 2 players.

6. **train.py** (Legacy)
   - Stub for training an RL agent.

7. **evaluate.py** (Legacy)
   - For systematically evaluating a trained agent across multiple episodes, seeds, or opponent types.

8. **run_experiment.py** (Legacy)
   - Orchestrates multi-run experiments, hyperparameter sweeps, or repeated simulations with different seeds.

## Typical Usage

### Quick Start - Choose Your Script

**For single model experiments:**
```bash
python3 scripts/runner.py --config src/game_reasoning_arena/configs/human_vs_random_config.yaml
```

**For multi-model experiments (maximum speed):**
```bash
python3 scripts/run_ray_multi_model.py \
  --config src/game_reasoning_arena/configs/ray_multi_model.yaml \
  --override use_ray=true
```

**For multi-model experiments (conservative):**
```bash
python3 scripts/run_multi_model_games.py \
  --config src/game_reasoning_arena/configs/multi_game_multi_model.yaml
```

### Performance Guide

| Script | Parallelization | Best For | Expected Speedup |
|--------|----------------|----------|------------------|
| `runner.py` | Episodes | Single model testing | ~N_episodes |
| `runner.py` (Ray) | Games + Episodes | Single model, multiple games | ~N_games × N_episodes |
| `run_ray_multi_model.py` | Models + Games + Episodes | Production multi-model experiments | ~N_models × N_games × N_episodes |
| `run_multi_model_games.py` | Episodes only | Testing multi-model setups | ~N_episodes |

### Configuration Files

All scripts use YAML configuration files located in `src/game_reasoning_arena/configs/`:

- `human_vs_random_config.yaml` - Basic single game setup
- `ray_multi_model.yaml` - Optimized for Ray multi-model experiments
- `multi_game_multi_model.yaml` - Multi-model, multi-game configuration

## Legacy Usage

- To play or test the environment with a mix of agents using legacy scripts:
  ```bash
  python simulate.py \
      --game tic_tac_toe \
      --rounds 3 \
      --player-types human random_bot
