Experiments
===========

This section covers how to design and run experiments with Board Game Arena.

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

   from board_game_arena.analysis import statistical_tests

   # Compare win rates between agents
   p_value = statistical_tests.binomial_test(
       wins_a=75, games_a=100,
       wins_b=65, games_b=100
   )
