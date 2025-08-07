Analysis & Evaluation
====================

Board Game Arena provides comprehensive tools for analyzing agent behavior and game outcomes.

Analysis Tools
--------------

Reasoning Traces Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Board Game Arena includes reasoning traces functionality that captures LLM decision-making processes during gameplay. This provides deep insights into how LLMs think through game strategies.

.. note::
   For a comprehensive tutorial on reasoning traces analysis, see :doc:`reasoning_traces`.

Quick Start with Reasoning Traces:

.. code-block:: bash

   # Run a game with LLM agents (traces collected automatically)
   python3 scripts/runner.py --config src/game_reasoning_arena/configs/example_config.yaml --override \
     env_config.game_name=tic_tac_toe \
     agents.player_0.type=llm \
     agents.player_0.model=litellm_groq/llama3-8b-8192 \
     num_episodes=5

   # View the collected reasoning traces
   python3 show_reasoning_traces.py

**Example Reasoning Trace Output:**

.. code-block:: text

   🧠 Reasoning Trace #1
   ----------------------------------------
   🎯 Game: tic_tac_toe
   📅 Episode: 1, Turn: 0
   🤖 Agent: litellm_groq/llama3-8b-8192
   🎲 Action Chosen: 4

   📋 Board State at Decision Time:
        ...
        ...
        ...

   🧠 Agent's Reasoning:
        I'll take the center position for strategic advantage.
        The center square gives me the most control over the
        board and creates multiple winning opportunities.

   ⏰ Timestamp: 2025-08-04 10:15:23

   🧠 Reasoning Trace #2
   ----------------------------------------
   🎯 Game: tic_tac_toe
   📅 Episode: 1, Turn: 1
   🤖 Agent: litellm_groq/llama3-8b-8192
   🎲 Action Chosen: 0

   📋 Board State at Decision Time:
        ...
        .x.
        ...

   🧠 Agent's Reasoning:
        Opponent took center, I need to take a corner to
        create diagonal threats and prevent them from
        controlling too much of the board.

   ⏰ Timestamp: 2025-08-04 10:15:24

**Key Features:**
* Automatic collection during LLM gameplay
* Board state capture at decision time
* Comprehensive reasoning categorization
* Multi-game support and analysis tools

Reasoning Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze the reasoning patterns of LLM agents using the reasoning analysis module:

.. code-block:: python

   # Import the analyzer class
   import sys
   sys.path.append('analysis/')
   from reasoning_analysis import LLMReasoningAnalyzer

   # Analyze game logs
   analyzer = LLMReasoningAnalyzer("run_logs/experiment_results.csv")

   # Categorize reasoning patterns
   analyzer.categorize_reasoning()

   # Generate comprehensive metrics and visualizations
   analyzer.compute_metrics(output_csv="metrics.csv", plot_dir="plots/")

   # Create word clouds by agent
   analyzer.plot_wordclouds_by_agent(output_dir="plots/")

   # Generate reasoning heatmaps
   analyzer.plot_heatmaps_by_agent(output_dir="plots/")

Alternatively, run as a script:

.. code-block:: bash

   cd analysis/
   python reasoning_analysis.py

**Features:**
* Categorizes reasoning types (strategic, tactical, random)
* Word cloud generation for common patterns
* Entropy analysis of decision-making
* Heatmap visualizations by agent type
* Export to various formats

Post-Game Processing
~~~~~~~~~~~~~~~~~~~~

Process and visualize game outcomes:

.. code-block:: python

   import sys
   sys.path.append('analysis/')
   from post_game_processing import PostGameProcessor

   processor = PostGameProcessor("run_logs/")
   processor.generate_win_rate_analysis()
   processor.create_heatmaps()

**Available Visualizations:**
* Win rate heatmaps by agent type
* Game length distributions
* Move frequency analysis
* Performance over time

TensorBoard Integration
~~~~~~~~~~~~~~~~~~~~~~~

Board Game Arena includes **TensorBoard integration** for real-time monitoring and visualization of agent performance metrics during experiments.

.. note::
   TensorBoard provides complementary visualization to the built-in analysis tools, focusing on real-time performance monitoring.

**What is Logged:**

* **Agent Rewards**: Final reward scores for each agent per episode
* **Performance Tracking**: Real-time visualization of win/loss patterns
* **Multi-Agent Comparison**: Side-by-side performance metrics for different agents
* **Episode-by-Episode Analysis**: Track performance evolution over multiple games

**Starting TensorBoard:**

.. code-block:: bash

   # After running experiments, launch TensorBoard
   tensorboard --logdir=runs

   # Open in browser: http://localhost:6006/

**Log Structure:**

.. code-block::

   runs/
   ├── tic_tac_toe/           # Game-specific TensorBoard logs
   │   └── events.out.tfevents.*
   ├── connect_four/
   │   └── events.out.tfevents.*
   └── kuhn_poker/
       └── events.out.tfevents.*

**Example Metrics:**

* ``Rewards/llm_litellm_groq_llama3_8b_8192``: LLM agent reward progression
* ``Rewards/random_None``: Random agent reward progression
* ``Rewards/llm_gpt_4``: GPT-4 agent reward progression

Evaluation Metrics
------------------

Agent Performance
~~~~~~~~~~~~~~~~~

* **Win Rate**: Percentage of games won
* **Average Game Length**: Typical number of moves per game
* **Decision Time**: Time taken per move
* **Reasoning Quality**: Analysis of LLM explanations

Reasoning Categories
~~~~~~~~~~~~~~~~~~~~

The analysis tool categorizes LLM reasoning into:

* **Positional**: Center control, corner/edge strategies
* **Blocking**: Preventing opponent wins
* **Opponent Modeling**: Understanding opponent strategy
* **Winning Logic**: Direct winning moves, threats
* **Heuristic**: General strategic principles
* **Rule-Based**: Following explicit strategies
* **Random/Unjustified**: Unclear or random reasoning

Comparative Analysis
~~~~~~~~~~~~~~~~~~~~

Compare different agents using the Python API:

.. code-block:: python

   # Import the analyzer class
   from analysis.reasoning_analysis import LLMReasoningAnalyzer

   # Analyze game logs
   analyzer = LLMReasoningAnalyzer("run_logs/experiment_results.csv")

   # Categorize reasoning patterns
   analyzer.categorize_reasoning()

   # Generate metrics and visualizations for comparison
   analyzer.compute_metrics(output_csv="comparison_metrics.csv", plot_dir="plots/")

**Comparison Capabilities:**
* Agent-specific reasoning pattern analysis
* Cross-game performance visualizations
* Reasoning category distributions by agent
* Word clouds showing agent-specific reasoning terms

Experiment Tracking
-------------------

All experiments are automatically logged with:

* Game configurations
* Agent parameters
* Full game transcripts
* Reasoning traces (for LLM agents)
* Performance metrics

**Actual Log Structure:**

.. code-block::

   results/
   ├── llm_<model_name>.db              # SQLite database per LLM agent
   ├── random_None.db                   # Random agent database
   ├── merged_logs_YYYYMMDD_HHMMSS.csv  # Processed data for analysis
   └── ...

   plots/                               # Generated visualizations
   ├── wordcloud_<agent>_<game>.png
   ├── pie_reasoning_type_<agent>_<game>.png
   └── heatmap_<agent>_<game>.png

   run_logs.txt                         # Raw execution logs
   run_logs_<game_name>.txt            # Game-specific logs

Generated Visualizations
------------------------

The analysis tools generate various plots and charts:

* **Reasoning Type Pie Charts**: Distribution of reasoning categories
* **Word Clouds**: Common phrases in agent reasoning
* **Heatmaps**: Performance across different game conditions
* **Entropy Plots**: Decision randomness over time
* **Win Rate Analysis**: Comparative performance metrics

Example Analysis Workflow
--------------------------

.. code-block:: python

   # Complete analysis pipeline
   import sys
   sys.path.append('analysis/')
   from reasoning_analysis import LLMReasoningAnalyzer

   # Initialize analyzer
   analyzer = LLMReasoningAnalyzer("run_logs/llm_experiments.csv")

   # Step 1: Categorize all reasoning
   analyzer.categorize_reasoning()

   # Step 2: Generate summary metrics
   game_summary = analyzer.summarize_games("results/game_summary.csv")

   # Step 3: Create all visualizations
   analyzer.compute_metrics(plot_dir="analysis_plots/")
   analyzer.plot_wordclouds_by_agent("analysis_plots/")
   analyzer.plot_entropy_trendlines("analysis_plots/")

   # Step 4: Save processed data
   analyzer.save_output("processed_analysis.csv")

For detailed analysis examples, see the :doc:`examples` section.
