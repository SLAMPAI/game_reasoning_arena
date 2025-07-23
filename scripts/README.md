# Scripts Folder

This directory contains various top-level scripts for running, training, and evaluating games.

## Contents

1. **simulate.py**
   - Allows one to play one or more matches of an OpenSpiel game with various agent types (human, random_bot, llm, or trained agents).
   - Uses a config dict to specify the environment, agent types, number of rounds, and so on.
   - The `config` object is loaded from `configs.py` or can be replaced by a custom config in code.
   - Usage example:
     ```bash
     python simulate.py --game tic_tac_toe --rounds 5 --player-types human random_bot
     ```

2. **configs.py**
- Contains predefined config dictionaries that specify the scenario. For instance,
`default_simulation_config()` sets up tic-tac-toe with 5 rounds, a seed of 42, and 2 players.
- One can create more advanced configs (like `advanced_config()`) for different games or scenarios.


2. **train.py**
   - Stub for training an RL agent.

3. **evaluate.py**
   - For systematically evaluating a trained agent across multiple episodes, seeds, or opponent types. Produces final metrics and logs.

4. **run_experiment.py**
   - Orchestrates multi-run experiments, hyperparameter sweeps, or repeated simulations with different seeds.

## Typical Usage

- To play or test the environment with a mix of agents, run:
  ```bash
  python simulate.py \
      --game tic_tac_toe \
      --rounds 3 \
      --player-types human random_bot
