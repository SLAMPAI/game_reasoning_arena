Tutorials
=========

Step-by-step tutorials for common use cases.

Tutorial 1: Your First Experiment
----------------------------------

Learn the basics by running a simple Connect Four experiment.

**Goal**: Compare an LLM agent against a random agent

**Step 1**: Set up the environment

.. code-block:: bash

   conda activate board_game_arena

**Step 2**: Create a configuration file

.. code-block:: yaml

   # tutorial_1.yaml
   game:
     name: "connect_four"
     num_episodes: 10

   agents:
     - type: "llm"
       model: "gpt-3.5-turbo"
       name: "GPT_Player"
     - type: "random"
       name: "Random_Player"

   logging:
     save_reasoning: true
     output_dir: "tutorial_results"

**Step 3**: Run the experiment

.. code-block:: bash

   python scripts/simulate.py --config tutorial_1.yaml

**Step 4**: Analyze results

.. code-block:: bash

   python analysis/reasoning_analysis.py --input tutorial_results/

Tutorial 2: Multi-Game Analysis
--------------------------------

Compare agent performance across different games.

**Goal**: See how the same agents perform on different game types

**Configuration**:

.. code-block:: yaml

   # tutorial_2.yaml
   games:
     - name: "connect_four"
       num_episodes: 20
     - name: "tic_tac_toe"
       num_episodes: 20
     - name: "kuhn_poker"
       num_episodes: 50

   agents:
     - type: "llm"
       model: "gpt-3.5-turbo"
       name: "GPT_Player"
     - type: "random"
       name: "Random_Player"

Tutorial 3: Custom Agent Development
------------------------------------

Create your own agent type.

**Goal**: Implement a simple heuristic-based agent

**Step 1**: Create the agent class

.. code-block:: python

   # custom_agents/heuristic_agent.py
   from board_game_arena.arena.agents.base_agent import BaseAgent

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
   from board_game_arena.arena.agents.agent_registry import register_agent
   register_agent("heuristic", HeuristicAgent)

**Step 3**: Use in configuration

.. code-block:: yaml

   agents:
     - type: "heuristic"
       name: "Heuristic_Player"

Tutorial 4: Large-Scale Experiments
-----------------------------------

Run experiments with many games and statistical analysis.

**Goal**: Get statistically significant results

**Configuration for large experiment**:

.. code-block:: yaml

   # large_experiment.yaml
   experiment:
     name: "statistical_study"
     replications: 5  # Run entire experiment 5 times

   game:
     name: "connect_four"
     num_episodes: 200  # 200 games per replication

   agents:
     - type: "llm"
       model: "gpt-4"
       temperature: 0.3
     - type: "llm"
       model: "gpt-3.5-turbo"
       temperature: 0.3

**Analysis**:

.. code-block:: python

   from board_game_arena.analysis import statistical_analysis

   results = statistical_analysis.load_experiment("statistical_study")

   # Calculate confidence intervals
   ci = results.confidence_interval(metric="win_rate", confidence=0.95)

   # Test for significant differences
   p_value = results.significance_test("gpt-4", "gpt-3.5-turbo")

Tutorial 5: Distributed Computing
----------------------------------

Scale up using Ray for parallel execution.

**Setup Ray cluster**:

.. code-block:: bash

   ray start --head --port=6379

**Configuration**:

.. code-block:: yaml

   execution:
     backend: "ray"
     num_workers: 4

   game:
     name: "connect_four"
     num_episodes: 1000  # Will be distributed across workers

**Monitor progress**:

.. code-block:: bash

   ray dashboard  # Open Ray dashboard in browser

Next Steps
----------

* Explore the :doc:`api_reference` for advanced features
* Check out :doc:`examples` for more complex scenarios
* Read :doc:`contributing` to add your own features
* Join the community discussions
