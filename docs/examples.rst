Examples
========

This section provides detailed examples of using Board Game Arena for various scenarios.

Basic Examples
--------------

Simple Game Simulation
~~~~~~~~~~~~~~~~~~~~~~~

Here's a basic example of running a Connect Four game:

.. code-block:: python

   from game_reasoning_arena.arena.envs.env_initializer import EnvInitializer
   from game_reasoning_arena.arena.agents.random_agent import RandomAgent
   from game_reasoning_arena.arena.agents.llm_agent import LLMAgent

   # Initialize environment
   env = EnvInitializer.create_env("connect_four")

   # Create agents
   agent1 = LLMAgent(name="LLM_Player", model="gpt-3.5-turbo")
   agent2 = RandomAgent(name="Random_Player")

   # Run simulation
   result = env.simulate_game([agent1, agent2])
   print(f"Winner: {result['winner']}")

Advanced Examples
-----------------

Multi-Game Tournament
~~~~~~~~~~~~~~~~~~~~~~

Running a tournament across multiple games:

.. code-block:: python

   import yaml
   from game_reasoning_arena.scripts.simulate import run_simulation

   # Load configuration
   with open('multi_game_config.yaml', 'r') as f:
       config = yaml.safe_load(f)

   # Run tournament
   results = run_simulation(config)

Custom Agent Development
~~~~~~~~~~~~~~~~~~~~~~~~

Creating a custom agent:

.. code-block:: python

   from game_reasoning_arena.arena.agents.base_agent import BaseAgent
   import random

   class MyCustomAgent(BaseAgent):
       def __init__(self, name="CustomAgent"):
           super().__init__(name)

       def get_action(self, state, legal_actions):
           # Custom logic here
           if len(legal_actions) > 0:
               return random.choice(legal_actions)
           return None

       def reset(self):
           pass

Research Examples
-----------------

Analyzing Agent Behavior
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from game_reasoning_arena.analysis.reasoning_analysis import analyze_reasoning

   # Analyze game logs
   results = analyze_reasoning("run_logs/experiment_results.json")

   # Generate visualizations
   results.plot_reasoning_patterns()
   results.save_analysis_report("analysis_report.html")

Batch Experiments
~~~~~~~~~~~~~~~~~

Running large-scale experiments:

.. code-block:: bash

   # Using SLURM for distributed execution
   sbatch slurm_jobs/run_simulation.sh

Configuration Examples
----------------------

LLM vs LLM Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   game:
     name: "kuhn_poker"
     num_episodes: 100
     max_turns: 50

   agents:
     - type: "llm"
       name: "Player1"
       model: "gpt-4"
       temperature: 0.7
     - type: "llm"
       name: "Player2"
       model: "claude-3-sonnet"
       temperature: 0.5

   backend:
     provider: "litellm"
     api_key: "${OPENAI_API_KEY}"

   logging:
     save_reasoning: true
     output_dir: "experiments/llm_vs_llm"

Hybrid Agent Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   game:
     name: "connect_four"
     num_episodes: 50

   agents:
     - type: "llm"
       name: "LLM_Player"
       model: "gpt-3.5-turbo"
     - type: "human"
       name: "Human_Player"

   interface:
     mode: "gradio"
     port: 7860
