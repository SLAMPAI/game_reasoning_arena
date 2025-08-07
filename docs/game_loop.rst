Game Loop & Environment Design
==============================

Game Reasoning Arena follows a **multi-agent reinforcement learning paradigm** built on top of OpenSpiel, providing a Gymnasium-like interface for game interactions. This design enables seamless integration with RL frameworks while supporting diverse agent types including LLMs, random agents, and human players.

Reinforcement Learning Paradigm
--------------------------------

The framework implements the standard **agent-environment interaction loop** from reinforcement learning:

.. code-block:: text

   ┌─────────────┐    observation    ┌─────────────┐
   │  LLM Agent  │ ◄──────────────── │ Environment │
   │             │                   │             │
   │  (Policy)   │ ────────────────► │ (OpenSpiel) │
   └─────────────┘      action       └─────────────┘
                                            │
                                            ▼
                                       reward +
                                    next observation

Key Components
~~~~~~~~~~~~~~

**Environment (OpenSpielEnv)**
  - Wraps OpenSpiel games in a Gymnasium-compatible interface
  - Manages state transitions, legal action validation, and game termination
  - Handles both turn-based and simultaneous-move games
  - Provides rich observations including game state and legal actions

**Agents (Policies)**
  - Implement the ``BaseAgent`` interface with ``compute_action()`` method
  - Receive observations and return actions (integers)
  - Support multiple agent types: LLM, Random, Human
  - Can maintain internal state across game steps

**Observations**
  - Dictionary format containing game state information
  - Includes legal actions, state strings, and formatted prompts
  - Tailored per agent with player-specific information

**Actions**
  - Integer values representing legal moves in the game
  - Validated against OpenSpiel's legal action set
  - Support both single-agent (turn-based) and multi-agent (simultaneous) scenarios

**Rewards**
  - Dictionary mapping player IDs to reward values
  - Computed using OpenSpiel's built-in reward functions
  - Available at each step (sparse) or episode termination (dense)

Gymnasium Compatibility
------------------------

Game Reasoning Arena closely follows the **Gymnasium API standard**:

Environment Interface
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Standard Gymnasium pattern
   observation, info = env.reset(seed=42)

   while not terminated and not truncated:
       action_dict = {current_player: agent.compute_action(observation)}
       observation, reward, terminated, truncated, info = env.step(action_dict)

Key Similarities
~~~~~~~~~~~~~~~~

==================== ===================== ================================
**Gymnasium**        **Game Reasoning Arena**  **Description**
==================== ===================== ================================
``env.reset()``      ``env.reset()``       Initialize episode, return observation
``env.step(action)`` ``env.step(actions)`` Apply action(s), return transition
``env.render()``     ``env.render()``      Display current game state
``env.seed()``       ``env.set_seed()``    Set random seed for reproducibility
Observation space    Observation dict      Structured state information
Action space         Legal actions list    Valid moves for current state
Reward signal        Reward dictionary     Per-player reward values
==================== ===================== ================================

Multi-Agent Extensions
~~~~~~~~~~~~~~~~~~~~~~~

Game Reasoning Arena extends Gymnasium for **multi-agent scenarios**:

.. code-block:: python

   # Turn-based games (like Chess, Tic-Tac-Toe)
   action_dict = {current_player: action}

   # Simultaneous games (like Rock-Paper-Scissors)
   action_dict = {0: action_0, 1: action_1}

RLLib Multi-Agent Compatibility
--------------------------------

The framework is designed with **RLLib multi-agent training** in mind:

Policy Mapping
~~~~~~~~~~~~~~

.. code-block:: python

   # RLLib-style policy mapping
   def policy_mapping_fn(agent_id, episode, worker, **kwargs):
       return f"policy_{agent_id}"

   # Game Reasoning Arena equivalent
   player_to_agent = {
       0: LLMAgent(model="gpt-4"),
       1: RandomAgent()
   }

Action Computation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # RLLib pattern
   actions = {agent_id: policy.compute_action(obs)
             for agent_id, obs in observations.items()}

   # Game Reasoning Arena implementation
   actions = {player: agent(observations[player])
             for player in active_players}

Episode Management
~~~~~~~~~~~~~~~~~~

The simulation loop mirrors RLLib's training workflow:

.. code-block:: python

   def simulate_episode():
       observations = env.reset()
       episode_rewards = {agent_id: 0 for agent_id in agents}

       while not done:
           # Compute actions for active agents
           actions = compute_actions(env, agents, observations)

           # Step environment
           obs, rewards, terminated, truncated, info = env.step(actions)

           # Accumulate rewards
           for agent_id, reward in rewards.items():
               episode_rewards[agent_id] += reward

           # Update state
           observations = obs
           done = terminated or truncated

       return episode_rewards

Game Loop Architecture
----------------------

Turn-Based Games
~~~~~~~~~~~~~~~~

For sequential games like Chess or Tic-Tac-Toe:

.. code-block:: python

   while not game_over:
       # 1. Get current player
       current_player = env.state.current_player()

       # 2. Generate observation
       observation = env._state_to_observation()[current_player]

       # 3. Agent selects action
       action = agents[current_player].compute_action(observation)

       # 4. Validate and apply action
       if action in observation["legal_actions"]:
           obs, rewards, terminated, truncated, info = env.step({current_player: action})
       else:
           # Handle illegal action (terminate episode)
           break

Simultaneous Games
~~~~~~~~~~~~~~~~~~

For concurrent games like Rock-Paper-Scissors:

.. code-block:: python

   while not game_over:
       # 1. All players act simultaneously
       observations = env._state_to_observation()

       # 2. Collect actions from all agents
       action_dict = {}
       for player_id, agent in agents.items():
           action_dict[player_id] = agent.compute_action(observations[player_id])

       # 3. Apply all actions together
       obs, rewards, terminated, truncated, info = env.step(action_dict)

Chance Node Handling
~~~~~~~~~~~~~~~~~~~~

OpenSpiel games often include chance events (card dealing, dice rolls):

.. code-block:: python

   def _solve_chance_nodes(self):
       """Automatically resolve probabilistic events."""
       while self.state.is_chance_node():
           outcomes, probabilities = zip(*self.state.chance_outcomes())
           action = random.choices(outcomes, probabilities)[0]
           self.state.apply_action(action)

Observation Structure
---------------------

Observations follow a **rich dictionary format** providing comprehensive game information:

.. code-block:: python

   observation = {
       "state_string": "X.O\\n.X.\\n...",  # Human-readable state
       "legal_actions": [0, 2, 5, 6, 7, 8],  # Valid move indices
       "prompt": "You are playing Tic-Tac-Toe\\n..."  # Formatted for LLMs
   }

Per-Agent Observations
~~~~~~~~~~~~~~~~~~~~~~

Each agent receives **player-specific information**:

- **Partial observability**: Hidden information (e.g., opponent cards in Poker)
- **Player perspective**: Board orientation and symbol assignment
- **Legal actions**: Only moves valid for that specific player
- **Context prompts**: Tailored natural language descriptions for LLM agents

Action Space Design
-------------------

Actions are represented as **integer indices** corresponding to OpenSpiel's action encoding:

.. code-block:: python

   # Tic-Tac-Toe: positions 0-8
   # 0 1 2
   # 3 4 5
   # 6 7 8

   # Connect Four: columns 0-6
   # Kuhn Poker: 0=Pass, 1=Bet

.. note::
   All action indices are validated against OpenSpiel's legal action constraints to ensure game rule compliance.

Action Validation
~~~~~~~~~~~~~~~~~

The framework provides **automatic legal action checking**:

.. code-block:: python

   legal_actions = env.state.legal_actions(current_player)

   if chosen_action not in legal_actions:
       # Log illegal move and terminate episode
       logger.error(f"Illegal action {chosen_action} by player {current_player}")
       env.truncated = True

Reward Structure
----------------

Rewards follow **OpenSpiel's game-theoretic conventions**:

Zero-Sum Games
~~~~~~~~~~~~~~
- Winner: +1, Loser: -1, Draw: 0
- Total rewards sum to zero across all players

Cooperative Games
~~~~~~~~~~~~~~~~~
- Shared objectives with aligned reward signals
- All players receive same reward for joint success

Reward Timing
~~~~~~~~~~~~~
.. code-block:: python

   # Sparse rewards (typical)
   rewards = {0: 0.0, 1: 0.0}  # During game
   rewards = {0: 1.0, 1: -1.0}  # At termination

   # Dense rewards (optional)
   rewards = {0: step_reward, 1: step_reward}  # Each step


See Also
--------

- :doc:`agents` - Detailed agent implementation guide
- :doc:`games` - Available game environments
- :doc:`api_reference` - Complete API documentation
- :doc:`experiments` - Advanced multi-agent training setups
