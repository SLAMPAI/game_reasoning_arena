Code Flow Analysis
==================

This section provides a comprehensive analysis of the code execution flow in Board Game Arena, helping developers understand how the framework orchestrates game simulations and agent interactions.

Overview
--------

The Board Game Arena framework follows a structured execution flow that handles multiple game types, agent configurations, and backend systems. The code flow analysis covers three main perspectives:

1. **High-level Architecture Flow** - Overall system organization
2. **Method Call Hierarchy** - Detailed function call sequences
3. **Data Flow Analysis** - Information transformation through the system

Method Call Flow for Matrix Games
----------------------------------

The following diagram illustrates the complete method call flow, with special focus on matrix games (simultaneous-move games like Rock-Paper-Scissors):

.. note::

   For interactive Mermaid diagrams, see the source files ``method_call_flow.md`` or ``flow_visualization.html`` in the project root.

**High-Level Flow Overview:**

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  runner.py      │───▶│ simulate_game   │───▶│ compute_actions │
   │  ::main()       │    │                 │    │                 │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │ parse_config    │    │ initialize_     │    │ env.step        │
   │ run_simulation  │    │ policies        │    │ apply_actions   │
   └─────────────────┘    └─────────────────┘    └─────────────────┘

**Detailed Method Call Flow:**

.. code-block:: text

   runner.py::main()
   ├── parse_config()
   ├── run_simulation()
   │   ├── initialize_ray() [if enabled]
   │   └── simulate_game()
   │       ├── initialize_llm_registry()
   │       ├── initialize_policies()
   │       │   ├── AGENT_REGISTRY[agent_type]
   │       │   ├── LLMAgent.__init__(model_name, game_name)
   │       │   └── RandomAgent.__init__(seed)
   │       ├── registry.make_env()
   │       │   ├── registry.get_game_loader()
   │       │   └── MatrixGameEnv.__init__()
   │       │       └── OpenSpielEnv.__init__()
   │       │
   │       └── Episode Loop:
   │           ├── env.reset(seed)
   │           │   ├── game.new_initial_state()
   │           │   ├── _solve_chance_nodes()
   │           │   └── _state_to_observation()
   │           │       └── _generate_prompt(agent_id)
   │           │
   │           └── while not terminated:
   │               ├── compute_actions()
   │               │   ├── is_simultaneous_node() → True [matrix games]
   │               │   ├── LLMAgent.compute_action()
   │               │   │   └── generate_response()
   │               │   └── RandomAgent.compute_action()
   │               │       └── random.choice(legal_actions)
   │               │
   │               ├── env.step(action_dict)
   │               │   ├── state.apply_actions([action_0, action_1])
   │               │   ├── _compute_reward()
   │               │   └── state.is_terminal()
   │               │
   │               └── Logging:
   │                   ├── log_llm_action()
   │                   ├── SQLiteLogger.log_move()
   │                   └── TensorBoard metrics

**Key Decision Points:**

* **Simultaneous vs Turn-based**: ``is_simultaneous_node()`` determines whether all players act (matrix games) or only current player (turn-based games)
* **LLM vs Local**: ``generate_response()`` uses Ray batch processing for remote LLMs or direct calls for local models
* **Game Termination**: ``state.is_terminal()`` controls episode completion

Initialization Chain
---------------------

The system startup follows this hierarchical initialization pattern:

.. code-block:: text

   runner.py::main()
   ├── parse_config()
   ├── run_simulation()
       ├── initialize_ray() [if enabled]
       └── simulate_game()
           ├── initialize_llm_registry()
           ├── initialize_policies()
           │   ├── registry.get_game_loader().num_players()
           │   ├── AGENT_REGISTRY[agent_type]
           │   ├── LLMAgent.__init__(model_name, game_name)
           │   └── RandomAgent.__init__(seed)
           └── registry.make_env()
               ├── registry.get_game_loader()
               ├── loader_class.load() [OpenSpiel game]
               └── MatrixGameEnv.__init__()
                   └── OpenSpielEnv.__init__()

Episode Execution Chain
------------------------

Each game episode follows this execution pattern:

.. code-block:: text

   simulate_game()
   └── for episode in range(num_episodes):
       ├── env.reset(seed)
       │   ├── game.new_initial_state()
       │   ├── _solve_chance_nodes()
       │   └── _state_to_observation()
       │       └── _generate_prompt(agent_id)
       │           ├── state.legal_actions(agent_id)
       │           └── state.action_to_string(agent_id, action)
       │
       └── while not (terminated or truncated):
           ├── compute_actions()
           │   ├── env.state.is_simultaneous_node() [True for matrix games]
           │   └── for each player:
           │       └── player_to_agent[player](observation)
           │           ├── LLMAgent.compute_action()
           │           │   ├── generate_response() or Ray call
           │           │   └── extract action from LLM response
           │           └── RandomAgent.compute_action()
           │               └── random.choice(legal_actions)
           │
           ├── env.step(action_dict)
           │   ├── state.apply_actions([action_0, action_1, ...])
           │   ├── _compute_reward()
           │   ├── state.is_terminal()
           │   └── _state_to_observation() [if not terminal]
           │
           └── Logging:
               ├── log_llm_action()
               ├── SQLiteLogger.log_move()
               └── TensorBoard metrics

Matrix Game Specific Flow
--------------------------

For matrix games like **Rock-Paper-Scissors** between an LLM and Random agent:

**1. State to Observation**
   - For Player 0 (LLM): legal_actions: [0, 1, 2] (Rock, Paper, Scissors), prompt: "You are Player 0 in matrix_rps..."
   - For Player 1 (Random): legal_actions: [0, 1, 2], prompt: "You are Player 1 in matrix_rps..."

**2. Action Computation**
   - is_simultaneous_node() → True
   - Player 0: LLMAgent.compute_action() → Send prompt to LLM → Parse response for action number → Return {"action": 1, "reasoning": "..."}
   - Player 1: RandomAgent.compute_action() → Return random.choice([0, 1, 2])

**3. Environment Step**
   - env.step({0: 1, 1: 2})  # Player 0: Paper, Player 1: Scissors
   - state.apply_actions([1, 2])
   - OpenSpiel calculates: Player 1 wins (Scissors cuts Paper)
   - rewards: {0: -1, 1: +1}
   - Check if terminal (single round) or continue

Key Class Interactions
----------------------

The main class relationships and interactions follow this hierarchy:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                     RUNNER & SIMULATION                         │
   │                                                                 │
   │  ┌─────────────────┐         ┌─────────────────────────────────┐│
   │  │    runner.py    │────────▶│        simulate.py              ││
   │  │                 │         │                                 ││
   │  │ + main()        │         │ + simulate_game()               ││
   │  │ + run_simulation│         │ + compute_actions()             ││
   │  │ + initialize_ray│         │ + log_llm_action()              ││
   │  └─────────────────┘         └─────────────────────────────────┘│
   └─────────────────────────────────────────────────────────────────┘
              │                               │
              ▼                               ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                 REGISTRY & POLICY MANAGEMENT                    │
   │                                                                 │
   │  ┌─────────────────┐         ┌─────────────────────────────────┐│
   │  │  GameRegistry   │         │      PolicyManager              ││
   │  │                 │         │                                 ││
   │  │ + register()    │         │ + initialize_policies()         ││
   │  │ + get_game_     │         │ + policy_mapping_fn()           ││
   │  │   loader()      │         │                                 ││
   │  │ + make_env()    │         │                                 ││
   │  └─────────────────┘         └─────────────────────────────────┘│
   └─────────────────────────────────────────────────────────────────┘
              │                               │
              ▼                               ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                ENVIRONMENTS & AGENTS                            │
   │                                                                 │
   │  ┌─────────────────┐         ┌─────────────────────────────────┐│
   │  │ MatrixGameEnv   │         │          AGENTS                 ││
   │  │                 │         │                                 ││
   │  │ + __init__()    │         │  ┌─────────────┐ ┌─────────────┐││
   │  │ + _state_to_    │         │  │  LLMAgent   │ │ RandomAgent │││
   │  │   observation() │◄────────┤  │             │ │             │││
   │  │ + _generate_    │         │  │+ compute_   │ │+ compute_   │││
   │  │   prompt()      │         │  │  action()   │ │  action()   │││
   │  │                 │         │  │+ __call__() │ │+ __call__() │││
   │  │      ▲          │         │  └─────────────┘ └─────────────┘││
   │  │      │          │         │                                 ││
   │  │ ┌─────────────────┐       │                                 ││
   │  │ │ OpenSpielEnv    │       │                                 ││
   │  │ │                 │       │                                 ││
   │  │ │ + reset()       │       │                                 ││
   │  │ │ + step()        │       │                                 ││
   │  │ │ + _solve_chance_│       │                                 ││
   │  │ │   nodes()       │       │                                 ││
   │  │ │ + _compute_     │       │                                 ││
   │  │ │   reward()      │       │                                 ││
   │  │ └─────────────────┘       │                                 ││
   │  └─────────────────────────────────────────────────────────────┘│
   └─────────────────────────────────────────────────────────────────┘

**Class Responsibilities:**

* **runner.py**: Main entry point, configuration parsing, simulation orchestration
* **simulate.py**: Core game loop, action coordination, logging management
* **GameRegistry**: Game discovery, environment factory, loader management
* **PolicyManager**: Agent instantiation, policy mapping, multi-agent coordination
* **MatrixGameEnv**: Game-specific observation generation, prompt formatting
* **OpenSpielEnv**: Base environment interface, state management, reward computation
* **LLMAgent**: Language model interaction, response parsing, reasoning extraction
* **RandomAgent**: Baseline random policy, action sampling

Data Flow Overview
------------------

The complete data transformation flow follows this pattern:

.. code-block:: text

   Configuration → Game Setup → Episode Loop → Action Computation → Environment Step → Logging

**Configuration Example:**

.. code-block:: json

   {
     "env_configs": [{"game_name": "matrix_rps"}],
     "agents": {
       "player_0": {"type": "llm", "model": "gpt-4"},
       "player_1": {"type": "random"}
     }
   }

**Game Setup:**

.. code-block:: python

   policies_dict = {
     "policy_0": LLMAgent("gpt-4", "matrix_rps"),
     "policy_1": RandomAgent(seed=42)
   }

   env = MatrixGameEnv(openspiel_game, "matrix_rps", ...)

**Episode Loop:**

.. code-block:: python

   observations = {
     0: {"legal_actions": [0,1,2], "prompt": "..."},
     1: {"legal_actions": [0,1,2], "prompt": "..."}
   }

**Action Computation:**

.. code-block:: python

   action_dict = {
     0: 1,  # LLM chooses Paper
     1: 2   # Random chooses Scissors
   }

**Environment Step:**

.. code-block:: python

   rewards = {0: -1, 1: +1}  # Player 1 wins
   terminated = True          # Single round game

Turn-based vs. Simultaneous Games
----------------------------------

The framework handles both game types differently:

**Matrix Games (Simultaneous)**
   - All players act at the same time
   - ``is_simultaneous_node()`` returns ``True``
   - Actions collected from all agents before environment step
   - Examples: Rock-Paper-Scissors, Prisoner's Dilemma

**Turn-based Games**
   - Only current player acts each step
   - ``state.current_player()`` identifies active player
   - Single action passed to environment step
   - Examples: Tic-Tac-Toe, Connect Four

This comprehensive flow analysis shows how matrix games are handled as simultaneous-move games where all players act at once, contrasting with turn-based games where only the current player acts each step.

.. note::

   For more detailed technical implementation, see the source files:

   - ``method_call_flow.md`` - Complete method call documentation
   - ``code_flow_analysis.md`` - Detailed architectural analysis
   - ``flow_visualization.html`` - Interactive flow visualization
