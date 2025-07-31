Quickstart
==========

This guide will help you run your first experiment with Board Game Arena.

Basic Usage
-----------

1. **Configure your experiment**

   Create a configuration file or use one of the provided examples:

   .. code-block:: bash

      cp src/board_game_arena/configs/example_config.yaml my_experiment.yaml

2. **Run a simple simulation**

   .. code-block:: bash

      python scripts/simulate.py --config my_experiment.yaml

3. **Analyze results**

   Results will be saved in the `run_logs/` directory. You can use the analysis tools:

   .. code-block:: bash

      python analysis/reasoning_analysis.py

Example: Human vs Random Agent
-------------------------------

Run a quick test with a human player against a random agent:

.. code-block:: bash

   python scripts/simulate.py --config src/board_game_arena/configs/human_vs_random_config.yaml

Example: LLM vs LLM
-------------------

To run an experiment with two LLM agents:

.. code-block:: bash

   python scripts/simulate.py --config src/board_game_arena/configs/kuhn_poker_llm_vs_llm.yaml

Configuration Files
-------------------

Configuration files define:

* **Game settings**: Which game to play, number of rounds
* **Agent configuration**: Types of agents and their parameters
* **Backend settings**: LLM providers and model configurations
* **Logging**: Output directories and analysis options

Example configuration:

.. code-block:: yaml

   game:
     name: "connect_four"
     num_episodes: 10

   agents:
     - type: "llm"
       model: "gpt-3.5-turbo"
       name: "Player1"
     - type: "random"
       name: "Player2"

   backend:
     provider: "litellm"

Available Games
---------------

* Connect Four
* Tic-Tac-Toe
* Kuhn Poker
* Chess (basic support)
* Hex

Next Steps
----------

* Explore the :doc:`api_reference` for detailed API documentation
* Check out :doc:`examples` for more complex use cases
* Learn about :doc:`contributing` to the project
