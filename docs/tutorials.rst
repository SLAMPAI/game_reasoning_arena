Tutorials
=========

Step-by-step tutorials for common use cases.

Tutorial 1: Your First Experiment
----------------------------------

Learn the basics by running a simple Connect Four experiment.

**Goal**: Compare an LLM agent against a random agent

**Step 1**: Set up the environment

.. code-block:: bash

   conda activate game_reasoning_arena

**Step 2**: Create a configuration file

.. code-block:: yaml

   # tutorial_1.yaml
   env_config:
     game_name: "connect_four"

   num_episodes: 10

   agents:
     player_0:
       type: "llm"
       model: "litellm_openai/gpt-3.5-turbo"
     player_1:
       type: "random"

   log_level: "INFO"

**Step 3**: Run the experiment

.. code-block:: bash

   python scripts/runner.py --config src/game_reasoning_arena/configs/tutorial_1.yaml

**Step 4**: Analyze results

.. code-block:: bash

   cd analysis/
   python reasoning_analysis.py

Tutorial 2: Multi-Game Analysis
--------------------------------

Compare agent performance across different games.

**Goal**: See how the same agents perform on different game types

**Configuration**:

.. code-block:: yaml

   # tutorial_2.yaml
   env_configs:
     - game_name: "connect_four"
     - game_name: "tic_tac_toe"
     - game_name: "kuhn_poker"

   num_episodes: 20

   agents:
     player_0:
       type: "llm"
       model: "litellm_openai/gpt-3.5-turbo"
     player_1:
       type: "random"

Tutorial 3: Custom Agent Development
------------------------------------

Create your own agent type.

**Goal**: Implement a simple heuristic-based agent

**Step 1**: Create the agent class

.. code-block:: python

   # custom_agents/heuristic_agent.py
   from src.game_reasoning_arena.arena.agents.base_agent import BaseAgent

   class HeuristicAgent(BaseAgent):
       def __init__(self, name="HeuristicAgent"):
           super().__init__(name)

       def get_action(self, state, legal_actions):
           # Simple heuristic: prefer center moves
           if hasattr(state, 'board') and legal_actions:
               center_col = len(state.board[0]) // 2
               if center_col in legal_actions:
                   return center_col
           return legal_actions[0] if legal_actions else None

       def reset(self):
           pass

**Step 2**: Register the agent

.. code-block:: python

   # Add to agent registry
   from src.game_reasoning_arena.arena.agents.agent_registry import register_agent
   register_agent("heuristic", HeuristicAgent)

**Step 3**: Use in configuration

.. code-block:: yaml

   agents:
     player_0:
       type: "heuristic"
     player_1:
       type: "random"

Tutorial 4: Large-Scale Experiments
-----------------------------------

Run experiments with many games and statistical analysis.

**Goal**: Get statistically significant results

**Configuration for large experiment**:

.. code-block:: yaml

   # large_experiment.yaml
   env_config:
     game_name: "connect_four"

   num_episodes: 200

   agents:
     player_0:
       type: "llm"
       model: "litellm_openai/gpt-4"
     player_1:
       type: "llm"
       model: "litellm_openai/gpt-3.5-turbo"

   llm_backend:
     temperature: 0.3

   use_ray: true
   parallel_episodes: true

**Analysis**:

.. code-block:: python

   from analysis import reasoning_analysis

   # Load and analyze results
   results = reasoning_analysis.load_results_from_directory("results/")

   # Generate statistical summaries
   summary = reasoning_analysis.generate_summary_statistics(results)

Tutorial 5: Distributed Computing
----------------------------------

Scale up using Ray for parallel execution.

**Setup Ray cluster**:

.. code-block:: bash

   ray start --head --port=6379

**Configuration**:

.. code-block:: yaml

   env_config:
     game_name: "connect_four"

   num_episodes: 1000

   agents:
     player_0:
       type: "llm"
       model: "litellm_groq/llama3-8b-8192"
     player_1:
       type: "random"

   use_ray: true
   parallel_episodes: true
   ray_config:
     num_cpus: 4

**Monitor progress**:

.. code-block:: bash

   # Check Ray status
   ray status

Next Steps
----------

* Explore :doc:`reasoning_traces` for in-depth LLM decision analysis
* Check the :doc:`api_reference` for advanced features
* Browse :doc:`examples` for more complex scenarios
* Read :doc:`contributing` to add your own features
* Join the community discussions
