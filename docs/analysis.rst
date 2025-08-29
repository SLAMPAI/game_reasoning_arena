Analysis & Evaluation
====================

Board Game Arena provides comprehensive tools for analyzing agent behavior and game outcomes. The analysis pipeline supports both automated workflows and detailed manual analysis.

Quick Start: Automated Analysis Pipeline
----------------------------------------

The easiest way to get started with analysis is using the automated pipeline:

.. code-block:: bash

   # Complete analysis with all games and models
   python3 analysis/run_full_analysis.py

   # Game-specific analysis
   python3 analysis/run_full_analysis.py --game hex

   # Model-specific analysis
   python3 analysis/run_full_analysis.py --model llama3

   # Combined filtering for targeted research
   python3 analysis/run_full_analysis.py --game hex --model llama3

   # Additional options
   python3 analysis/run_full_analysis.py --quiet --plots-dir custom_plots

**Pipeline Features:**
* 🔍 **Auto-discovery** of SQLite databases in ``results/``
* 🔄 **Automatic merging** of databases into consolidated CSV files
* 🎯 **Smart filtering** by game type and/or model
* 🧠 **Reasoning categorization** using rule-based classification
* 📊 **Comprehensive visualizations** (plots, charts, heatmaps, word clouds)
* 📁 **Organized output** in game/model-specific directories
* ⚡ **Error handling** with detailed logging

Focused Analysis
-----------------------

The analysis pipeline now supports filtering for specific games and models, enabling targeted research questions:

Game-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Focus on specific game strategies
   python3 analysis/run_full_analysis.py --game hex         


Model-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Compare model families (partial string matching)
   python3 analysis/run_full_analysis.py --model llama        # All Llama variants
   python3 analysis/run_full_analysis.py --model gpt          # All GPT models
   python3 analysis/run_full_analysis.py --model llama3-8b    # Specific model size

Combined Filtering for Research Questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Answer specific research questions
   python3 analysis/run_full_analysis.py --game hex --model llama3
   # → "How does Llama3 approach HEX connection strategies?"

   python3 analysis/run_full_analysis.py --game kuhn_poker --model gpt
   # → "How do GPT models handle hidden information in poker?"

**Output Organization:**

When filters are applied, results are organized in subdirectories:

* ``plots/game_hex/`` - HEX-specific analysis and visualizations
* ``plots/model_llama/`` - Llama model family analysis
* ``plots/game_hex_model_llama3/`` - Combined game+model filtering results

**Benefits:**
* ⚡ **Faster processing** by analyzing only relevant data
* 🎯 **Research-focused** analysis for specific hypotheses
* 💾 **Memory efficient** for large datasets
* 📊 **Cleaner visualizations** with focused data

Command-Line Options Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``run_full_analysis.py`` script supports the following options:

.. code-block:: bash

   python3 analysis/run_full_analysis.py [OPTIONS]

**Core Options:**

* ``--game GAME`` - Filter analysis for specific game (e.g., ``hex``, ``tic_tac_toe``, ``connect_four``)
* ``--model MODEL`` - Filter analysis for specific model (supports partial matching, e.g., ``llama3``, ``gpt``)
* ``--results-dir DIR`` - Directory containing SQLite database files (default: ``results``)
* ``--plots-dir DIR`` - Directory for output plots and visualizations (default: ``plots``)
* ``--quiet`` - Run in quiet mode with minimal output
* ``--skip-existing`` - Skip analysis steps if output files already exist

**Example Commands:**

.. code-block:: bash

   # Get help
   python3 analysis/run_full_analysis.py --help

   # Basic usage
   python3 analysis/run_full_analysis.py

   # Game-specific analysis
   python3 analysis/run_full_analysis.py --game hex

   # Model-specific analysis
   python3 analysis/run_full_analysis.py --model llama3

   # Combined filtering
   python3 analysis/run_full_analysis.py --game hex --model llama3

   # Custom directories with quiet mode
   python3 analysis/run_full_analysis.py --results-dir my_results --plots-dir my_plots --quiet

   # Skip existing files for faster re-runs
   python3 analysis/run_full_analysis.py --skip-existing

Detailed Analysis Tools
-----------------------

Reasoning Traces Collection & Viewing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Board Game Arena automatically captures LLM decision-making processes during gameplay, providing deep insights into strategic thinking.

.. note::
   For a comprehensive tutorial on reasoning traces analysis, see :doc:`reasoning_traces`.

**Automatic Collection:**

.. code-block:: bash

   # Run a game with LLM agents (traces collected automatically)
   python3 scripts/runner.py --config src/game_reasoning_arena/configs/example_config.yaml --override \
     env_config.game_name=tic_tac_toe \
     agents.player_0.type=llm \
     agents.player_0.model=litellm_groq/llama3-8b-8192 \
     num_episodes=5

**Viewing Traces:**

.. code-block:: bash

   # View all reasoning traces
   python3 show_reasoning_traces.py

   # Extract specific traces with filtering
   python3 analysis/extract_reasoning_traces.py --game tic_tac_toe --episode 1
   python3 analysis/extract_reasoning_traces.py --db results/llm_model.db --analyze-only

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

Analyze reasoning patterns using both automated pipeline and manual analysis:

**Automated Analysis (Recommended):**

.. code-block:: bash

   # Complete reasoning analysis
   python3 analysis/run_full_analysis.py

   # 🎯 Game-specific reasoning analysis
   python3 analysis/run_full_analysis.py --game hex
   python3 analysis/run_full_analysis.py --game tic_tac_toe

   # 🎯 Model-specific reasoning analysis
   python3 analysis/run_full_analysis.py --model llama3
   python3 analysis/run_full_analysis.py --model gpt

   # 🎯 Combined filtering for focused research
   python3 analysis/run_full_analysis.py --game hex --model llama3

**Manual Analysis (Advanced):**

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

**Manual Filtering (for custom analysis):**

.. code-block:: python

   # Load and filter data manually
   analyzer = LLMReasoningAnalyzer("merged_logs.csv")

   # Filter for specific game
   hex_data = analyzer.df[analyzer.df['game_name'] == 'hex']

   # Filter for specific model (partial matching)
   llama_data = analyzer.df[
       analyzer.df['agent_model'].str.contains('llama3', case=False, na=False)
   ]

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

Entropy Analysis
~~~~~~~~~~~~~~~~

Board Game Arena provides comprehensive **entropy analysis** to measure the diversity and predictability of agent reasoning patterns over time.

**What is Entropy?**

Shannon entropy quantifies the diversity of reasoning categories used by an agent:

.. math::

   H = -\sum_{i} p_i \log_2(p_i)

Where :math:`p_i` is the probability of reasoning category :math:`i`.

**Entropy Interpretation:**

* **High Entropy (2.5-3.0)**: Diverse reasoning, using many different strategies
* **Medium Entropy (1.5-2.5)**: Moderate diversity, some preferred strategies
* **Low Entropy (0.0-1.5)**: Focused reasoning, few dominant strategies

**Key Entropy Metrics:**

* **Reasoning Entropy**: Diversity of reasoning categories per game turn
* **Temporal Trends**: How entropy changes throughout gameplay
* **Cross-Game Comparison**: Entropy patterns across different game types
* **Agent Comparison**: Reasoning diversity between different models

**Entropy Analysis Tools:**

.. code-block:: python

   from analysis.reasoning_analysis import LLMReasoningAnalyzer

   # Initialize analyzer
   analyzer = LLMReasoningAnalyzer("run_logs/experiment_results.csv")

   # Generate entropy trendline plots
   analyzer.plot_entropy_trendlines(output_dir="plots/")

   # Plot average entropy across all games
   analyzer.plot_avg_entropy_across_games(output_dir="plots/")

   # Calculate entropy for specific game/agent combinations
   entropy_data = analyzer.calculate_entropy_by_turn(
       game_name="tic_tac_toe",
       agent_type="llm_litellm_groq_llama3_8b_8192"
   )

**Generated Entropy Visualizations:**

* ``entropy_trend_[agent]_[game].png``: Entropy evolution over game turns
* ``avg_entropy_all_games.png``: Average entropy comparison across games
* ``entropy_heatmap_[agent].png``: Entropy patterns across different conditions

**Example Entropy Interpretation:**

A decreasing entropy trend might indicate that an agent becomes more focused on specific strategies as the game progresses, while fluctuating entropy could suggest adaptive reasoning based on changing game states.

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

**Reasoning Analysis Plots:**

* **Reasoning Type Pie Charts**: Distribution of reasoning categories
* **Word Clouds**: Common phrases in agent reasoning
* **Stacked Bar Evolution**: Reasoning category transitions over game turns
* **Reasoning Heatmaps**: Performance across different game conditions

**Entropy Analysis Plots:**

* **Entropy Trendlines**: Decision diversity evolution over game turns (``entropy_trend_[agent]_[game].png``)
* **Average Entropy Comparison**: Cross-game entropy comparison (``avg_entropy_all_games.png``)
* **Entropy Heatmaps**: Reasoning diversity patterns across conditions

**Performance Analysis:**

* **Win Rate Analysis**: Comparative performance metrics
* **Evolution Plots**: Enhanced single-panel stacked bar visualizations showing reasoning transitions
* **Cross-Agent Comparisons**: Side-by-side performance and reasoning analysis

Example Analysis Workflows
--------------------------

Complete Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Option 1: Automated complete analysis
   python3 analysis/run_full_analysis.py --quiet

   # Option 2: Focused analysis for specific research question
   python3 analysis/run_full_analysis.py --game hex --model llama3 --plots-dir hex_llama_analysis

**Automated pipeline handles:**
* Database discovery and merging
* Data filtering (if specified)
* Reasoning categorization
* All visualizations generation
* Organized output structure

Game-Specific Research Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Research question: "How do different models approach HEX strategy?"

   # Step 1: Collect HEX data with multiple models
   python3 scripts/runner.py --config configs/multi_model_hex.yaml

   # Step 2: Analyze HEX-specific patterns
   python3 analysis/run_full_analysis.py --game hex

   # Results in: plots/game_hex/
   # - HEX-specific reasoning categories
   # - HEX move pattern heatmaps
   # - HEX strategy word clouds

Model Comparison Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Research question: "How does Llama3 reasoning differ from GPT models?"

   # Step 1: Analyze Llama3 family
   python3 analysis/run_full_analysis.py --model llama3 --plots-dir llama3_analysis

   # Step 2: Analyze GPT family
   python3 analysis/run_full_analysis.py --model gpt --plots-dir gpt_analysis

   # Step 3: Compare results in respective directories

Manual Advanced Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For custom research requiring manual control
   import sys
   sys.path.append('analysis/')
   from reasoning_analysis import LLMReasoningAnalyzer

   # Initialize analyzer
   analyzer = LLMReasoningAnalyzer("run_logs/llm_experiments.csv")

   # Step 1: Apply custom filtering
   hex_data = analyzer.df[analyzer.df['game_name'] == 'hex']
   llama_hex = hex_data[hex_data['agent_model'].str.contains('llama3')]

   # Step 2: Categorize filtered reasoning
   analyzer.df = llama_hex  # Apply filter
   analyzer.categorize_reasoning()

   # Step 3: Generate targeted visualizations
   analyzer.compute_metrics(plot_dir="custom_analysis/")
   analyzer.plot_wordclouds_by_agent("custom_analysis/")
   analyzer.plot_entropy_trendlines("custom_analysis/")

   # Step 4: Export results
   analyzer.save_output("llama3_hex_analysis.csv")

For detailed analysis examples, see the :doc:`examples` section.
