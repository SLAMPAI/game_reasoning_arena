Board Game Arena
================

Welcome to Board Game Arena - a research platform for training and evaluating AI agents in board games.

Installation
------------

.. code-block:: bash

   git clone https://github.com/lcipolina/board_game_arena.git
   cd board_game_arena
   conda env create -f environment.yaml
   conda activate board_game_arena
   pip install -e .

Quick Start
-----------

.. code-block:: python

   from board_game_arena.arena.envs.env_initializer import EnvInitializer
   from board_game_arena.arena.agents.random_agent import RandomAgent
   
   # Create a Connect Four environment
   env = EnvInitializer.create_env("connect_four")
   
   # Create random agents
   agent1 = RandomAgent(name="Player1")
   agent2 = RandomAgent(name="Player2")
   
   # Run a game
   result = env.simulate_game([agent1, agent2])
   print(f"Winner: {result['winner']}")

Features
--------

* Multi-game support (Connect Four, Tic-Tac-Toe, Kuhn Poker, etc.)
* LLM agent integration via LiteLLM and vLLM
* Flexible agent framework
* Comprehensive analysis tools
* Ray-based distributed computing

Repository
----------

* GitHub: https://github.com/lcipolina/board_game_arena
