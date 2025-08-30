Experiments
===========

This section covers how to design and run experiments with Game Reasoning Arena, including distributed execution capabilities.

Ray Integration for Parallel Execution
---------------------------------------

Game Reasoning Arena supports **Ray** for distributed and parallel execution, allowing you to:

- **Run multiple games in parallel** across different cores/machines
- **Parallelize episodes within games** for faster data collection
- **Distribute LLM inference** for batch processing
- **Scale experiments** on SLURM clusters or multi-GPU setups

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

**Option 1: Combined Configuration File (YAML)**

.. code-block:: yaml

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

**Option 2: Separate Ray Configuration (Recommended)**

.. code-block:: bash

   # Use any existing config + separate Ray settings
   python3 scripts/runner.py \
     --base-config src/game_reasoning_arena/configs/multi_game_base.yaml \
     --ray-config src/game_reasoning_arena/configs/ray_config.yaml \
     --override num_episodes=10 \
     --override agents.player_0.model=litellm_groq/llama3-70b-8192

**Option 3: Command-Line Override**

.. code-block:: bash

   # Enable Ray with any existing configuration
   python3 scripts/runner.py --config src/game_reasoning_arena/configs/example_config.yaml \
     --override use_ray=true parallel_episodes=true

**Option 4: Maximum Parallelization (Multi-Model Ray)**

.. code-block:: bash

   # Run multiple models in parallel with full Ray integration
   # Parallelizes: Models + Games + Episodes simultaneously
   python3 scripts/run_ray_multi_model.py \
     --config src/game_reasoning_arena/configs/ray_multi_model.yaml \
     --override use_ray=true

Ray Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ray_config.yaml`` file contains Ray-specific settings:

.. list-table:: Ray Configuration Options
   :widths: 25 50 25
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - ``use_ray``
     - Enable/disable Ray
     - ``false``
   * - ``parallel_episodes``
     - Parallelize episodes within games
     - ``false``
   * - ``ray_config.num_cpus``
     - Number of CPUs for Ray
     - Auto-detect
   * - ``ray_config.num_gpus``
     - Number of GPUs for Ray
     - Auto-detect
   * - ``ray_config.include_dashboard``
     - Enable Ray dashboard
     - ``false``
   * - ``ray_config.dashboard_port``
     - Dashboard port
     - ``8265``
   * - ``ray_config.object_store_memory``
     - Object store memory limit
     - Auto

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Execution Modes Performance
   :widths: 30 25 25 20
   :header-rows: 1

   * - Execution Mode
     - Parallelization Level
     - Best For
     - Expected Speedup
   * - ``scripts/runner.py`` (standard)
     - Episodes only
     - Single model, single game
     - ~N_episodes
   * - ``scripts/runner.py`` (Ray enabled)
     - Games + Episodes
     - Single model, multiple games
     - ~N_games × N_episodes
   * - ``scripts/run_ray_multi_model.py``
     - Models + Games + Episodes
     - Multiple models, multiple games
     - ~N_models × N_games × N_episodes

**Recommendation**: Use ``run_ray_multi_model.py`` for multi-model experiments to achieve maximum speedup.

Configuration Merging Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system merges configurations in this order (later overrides earlier):

1. Default configuration
2. Base config (``--base-config``)
3. Main config (``--config``)
4. Ray config (``--ray-config``)
5. CLI overrides (``--override``)

SLURM Integration
~~~~~~~~~~~~~~~~~

For cluster environments, Ray automatically detects SLURM allocation:

.. code-block:: bash

   # SLURM job with Ray
   sbatch --nodes=2 --cpus-per-task=48 --gres=gpu:4 slurm_jobs/run_simulation.sh

The SLURM script (``slurm_jobs/run_simulation.sh``) handles:

- Multi-node Ray cluster setup
- Head node and worker initialization
- GPU allocation across nodes
- Environment variable configuration

Debug Commands
~~~~~~~~~~~~~~

.. code-block:: bash

   # Check Ray status
   ray status

   # Monitor Ray dashboard (if enabled)
   # Navigate to: http://localhost:8265

Experiment Design
-----------------

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~~

Use YAML configuration files to define experiments:

.. code-block:: yaml

   experiment:
     name: "llm_comparison_study"
     description: "Compare different LLM models on strategic games"

   games:
     - name: "connect_four"
       num_episodes: 100
     - name: "kuhn_poker"
       num_episodes: 200

   agents:
     - type: "llm"
       model: "gpt-4"
       name: "GPT4_Player"
     - type: "llm"
       model: "claude-3-sonnet"
       name: "Claude_Player"

Running Experiments
-------------------

Single Experiments
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/simulate.py --config experiments/my_experiment.yaml

Batch Experiments
~~~~~~~~~~~~~~~~~

For large-scale studies:

.. code-block:: bash

   # Using SLURM for cluster computing
   sbatch slurm_jobs/run_simulation.sh

   # Or parallel execution
   python scripts/runner.py --parallel --jobs 8


Distributed Computing
~~~~~~~~~~~~~~~~~~~~~

Use Ray for distributed execution:

.. code-block:: yaml

   execution:
     backend: "ray"
     num_workers: 8
     resources_per_worker:
       cpu: 2
       memory: "4GB"

Statistical Analysis
--------------------

Significance Testing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from game_reasoning_arena.analysis import statistical_tests

   # Compare win rates between agents
   p_value = statistical_tests.binomial_test(
       wins_a=75, games_a=100,
       wins_b=65, games_b=100
   )
