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

The following Mermaid diagram illustrates the complete method call flow, with special focus on matrix games (simultaneous-move games like Rock-Paper-Scissors):

.. raw:: html

   <div class="mermaid">
   graph TD
       %% Entry Point
       A[runner.py::main] --> B[parse_config]
       A --> C[run_simulation]

       %% Ray Initialization
       C --> D[initialize_ray]

       %% Simulation Orchestration
       C --> E[simulate_game]

       %% Core Simulation Setup
       E --> F[initialize_llm_registry]
       E --> G[initialize_policies]
       E --> H[registry.make_env]

       %% Policy Management
       G --> I[AGENT_REGISTRY lookup]
       G --> J[LLMAgent.__init__]
       G --> K[RandomAgent.__init__]
       G --> L[HumanAgent.__init__]

       %% Environment Creation
       H --> M[registry.get_game_loader]
       H --> N[MatrixGameEnv.__init__]
       N --> O[OpenSpielEnv.__init__]

       %% Episode Loop
       E --> P[env.reset]
       P --> Q[game.new_initial_state]
       P --> R[_solve_chance_nodes]
       P --> S[_state_to_observation]

       %% Matrix Game Observation
       S --> T[_generate_prompt]
       T --> U[state.legal_actions]
       T --> V[state.action_to_string]

       %% Action Computation Loop
       E --> W[compute_actions]
       W --> X{is_simultaneous_node?}

       %% Simultaneous Actions (Matrix Games)
       X -->|Yes| Y[All players act]
       Y --> Z[player_to_agent[0]]
       Y --> AA[player_to_agent[1]]

       %% Agent Decision Making
       Z --> BB[LLMAgent.compute_action]
       AA --> CC[RandomAgent.compute_action]

       %% LLM Processing
       BB --> DD[generate_response]
       DD --> EE[Ray batch processing]
       DD --> FF[Direct LLM call]

       %% Random Agent
       CC --> GG[random.choice]

       %% Environment Step
       W --> HH[env.step]
       HH --> II[state.apply_actions]
       II --> JJ[OpenSpiel game logic]

       %% Reward and State Update
       HH --> KK[_compute_reward]
       HH --> LL[state.is_terminal]
       HH --> MM[_state_to_observation]

       %% Logging
       E --> NN[log_llm_action]
       E --> OO[SQLiteLogger.log_move]
       E --> PP[TensorBoard logging]

       %% Turn-based Alternative
       X -->|No| QQ[Current player only]
       QQ --> RR[state.current_player]
       QQ --> Z

       %% Cleanup and Results
       E --> SS[Post-game processing]
       A --> TT[full_cleanup]

       %% Styling
       classDef entryPoint fill:#e1f5fe
       classDef simulation fill:#f3e5f5
       classDef environment fill:#e8f5e8
       classDef agent fill:#fff3e0
       classDef matrix fill:#ffebee

       class A,B,C entryPoint
       class E,F,G,W,HH simulation
       class H,N,O,P,S environment
       class I,J,K,L,BB,CC agent
       class T,U,V,Y,Z,AA,II matrix
   </div>

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

The following diagram shows the main class relationships and interactions:

.. raw:: html

   <div class="mermaid">
   classDiagram
       class runner {
           +main()
           +run_simulation()
           +initialize_ray()
       }

       class simulate {
           +simulate_game()
           +compute_actions()
           +log_llm_action()
       }

       class GameRegistry {
           +register()
           +get_game_loader()
           +make_env()
       }

       class PolicyManager {
           +initialize_policies()
           +policy_mapping_fn()
       }

       class MatrixGameEnv {
           +__init__()
           +_state_to_observation()
           +_generate_prompt()
           +apply_action()
       }

       class OpenSpielEnv {
           +reset()
           +step()
           +_solve_chance_nodes()
           +_compute_reward()
       }

       class LLMAgent {
           +compute_action()
           +__call__()
       }

       class RandomAgent {
           +compute_action()
           +__call__()
       }

       runner --> simulate
       simulate --> GameRegistry
       simulate --> PolicyManager
       GameRegistry --> MatrixGameEnv
       MatrixGameEnv --> OpenSpielEnv
       PolicyManager --> LLMAgent
       PolicyManager --> RandomAgent
       simulate --> LLMAgent
       simulate --> RandomAgent
   </div>

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
