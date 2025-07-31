Analysis & Evaluation
====================

Board Game Arena provides comprehensive tools for analyzing agent behavior and game outcomes.

Analysis Tools
--------------

Reasoning Analysis
~~~~~~~~~~~~~~~~~~

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

Alternatively, run as a command-line script:

.. code-block:: bash

   python analysis/reasoning_analysis.py --input run_logs/ --output plots/

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

Evaluation Metrics
------------------

Agent Performance
~~~~~~~~~~~~~~~~~

* **Win Rate**: Percentage of games won
* **Average Game Length**: Typical number of moves per game
* **Decision Time**: Time taken per move
* **Reasoning Quality**: Analysis of LLM explanations

Game Complexity
~~~~~~~~~~~~~~~

* **Branching Factor**: Average number of legal moves
* **Game Tree Depth**: Typical game length
* **State Space Size**: Number of possible positions

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

Compare different agents across multiple metrics:

.. code-block:: bash

   python analysis/reasoning_analysis.py --compare --agents llm random

**Comparison Features:**
* Head-to-head win rates
* Strategy pattern differences
* Performance across different games
* Statistical significance testing

Experiment Tracking
-------------------

All experiments are automatically logged with:

* Game configurations
* Agent parameters
* Full game transcripts
* Reasoning traces (for LLM agents)
* Performance metrics

**Log Structure:**

.. code-block::

   run_logs/
   ├── experiment_YYYYMMDD_HHMMSS/
   │   ├── config.yaml
   │   ├── games_log.csv
   │   ├── agent_reasoning.json
   │   └── summary.json

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
   game_summary = analyzer.summarize_games("game_summary.csv")

   # Step 3: Create all visualizations
   analyzer.compute_metrics(plot_dir="analysis_plots/")
   analyzer.plot_wordclouds_by_agent("analysis_plots/")
   analyzer.plot_entropy_trendlines("analysis_plots/")

   # Step 4: Save processed data
   analyzer.save_output("processed_analysis.csv")

For detailed analysis examples, see the :doc:`examples` section.
