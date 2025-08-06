Extending Board Game Arena
=========================

This guide covers how to extend Board Game Arena with new features.

Adding New Games
----------------

To add support for a new game:

1. **Create the game environment**

   Create a new file in ``src/game_reasoning_arena/arena/envs/``:

   .. code-block:: python

      # src/game_reasoning_arena/arena/envs/my_game_env.py
      from .base_env import BaseEnv

      class MyGameEnv(BaseEnv):
          def __init__(self, config):
              super().__init__(config)
              # Initialize game-specific state

          def reset(self):
              # Reset game to initial state
              pass

          def step(self, action):
              # Execute action and return new state
              pass

          def get_legal_actions(self, player):
              # Return list of legal actions for player
              pass

          def is_terminal(self):
              # Check if game is finished
              pass

          def get_winner(self):
              # Return winner if game is terminal
              pass

2. **Register the game**

   Add your game to the environment initializer:

   .. code-block:: python

      # In src/game_reasoning_arena/arena/envs/env_initializer.py
      from .my_game_env import MyGameEnv

      GAME_REGISTRY = {
          # ... existing games
          "my_game": MyGameEnv,
      }

3. **Add configuration support**

   Create a configuration template in ``src/game_reasoning_arena/configs/``.

4. **Add tests**

   Create tests in ``tests/`` to verify your game works correctly.

5. **Customize prompts (for LLM games)**

   If your game will be played by LLM agents, you'll want to customize the prompting system to provide game-specific context. See the :doc:`prompting` guide for detailed instructions on creating effective prompts for your game.

Adding New Agent Types
----------------------

To create a new agent type:

1. **Implement the agent class**

   .. code-block:: python

      # src/game_reasoning_arena/arena/agents/my_agent.py
      from .base_agent import BaseAgent

      class MyAgent(BaseAgent):
          def __init__(self, name, **kwargs):
              super().__init__(name)
              # Initialize agent-specific parameters

          def get_action(self, state, legal_actions):
              # Your decision logic here
              return chosen_action

          def reset(self):
              # Reset agent state for new game
              pass

2. **Register the agent**

   Add to the agent registry:

   .. code-block:: python

      # In src/game_reasoning_arena/arena/agents/agent_registry.py
      from .my_agent import MyAgent

      AGENT_REGISTRY = {
          # ... existing agents
          "my_agent": MyAgent,
      }

Adding New Backends
-------------------

To add support for a new LLM backend:

1. **Implement backend class**

   .. code-block:: python

      # src/game_reasoning_arena/backends/my_backend.py
      from .base_backend import BaseBackend

      class MyBackend(BaseBackend):
          def __init__(self, config):
              super().__init__(config)
              # Initialize backend-specific setup

          def generate(self, prompt, **kwargs):
              # Call your LLM service
              pass

2. **Register the backend**

   Add to backend registry and configuration files.

Adding New Analysis Tools
-------------------------

To add new analysis capabilities:

1. **Create analysis module**

   .. code-block:: python

      # analysis/my_analysis.py
      def analyze_my_metric(game_logs):
          # Your analysis logic
          pass

2. **Add visualization support**

   Integrate with existing plotting infrastructure.

3. **Update analysis pipeline**

   Add to the main analysis workflow.

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

- Follow PEP 8 style guidelines
- Use type hints where possible
- Add comprehensive docstrings
- Include unit tests for new features

Testing
~~~~~~~

- Write tests for all new functionality
- Ensure tests pass before submitting
- Include integration tests for complex features

Documentation
~~~~~~~~~~~~~

- Update relevant documentation pages
- Add examples for new features
- Include configuration examples
- Update API reference as needed

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Profile new code for performance bottlenecks
- Consider memory usage for large-scale experiments
- Optimize critical paths in game simulation
- Test scalability with distributed execution

For more detailed guidelines, see the :doc:`contributing` section.
