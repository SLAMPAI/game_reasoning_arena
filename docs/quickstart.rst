Quick Start
===========

Get up and running with Board Game Arena in minutes.

Installation
------------

.. code-block:: bash

   git clone https://github.com/lcipolina/board_game_arena.git
   cd board_game_arena
   conda env create -f environment.yaml
   conda activate board_game_arena
   pip install -e .

Your First Game
---------------

Run a simple game using the command-line interface:

.. code-block:: bash

   # Run a Tic-Tac-Toe game with random agents
   python scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
     env_configs.0.game_name=tic_tac_toe \
     agents.player_0.type=random \
     agents.player_1.type=random \
     num_episodes=1

.. code-block:: bash

   # Run a Connect Four game
   python scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
     env_configs.0.game_name=connect_four \
     agents.player_0.type=random \
     agents.player_1.type=random \
     num_episodes=1

LLM vs Random Agent
-------------------

Try an LLM agent against a random player:

.. code-block:: bash

   python scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
     env_configs.0.game_name=kuhn_poker \
     agents.player_0.type=llm \
     agents.player_0.model=litellm_groq/llama3-8b-8192 \
     agents.player_1.type=random \
     num_episodes=5

What's Next?
------------

* Learn about :doc:`games` supported by the platform
* Explore different :doc:`agents` types
* Check out detailed :doc:`examples`
* Read the full :doc:`api_reference`
