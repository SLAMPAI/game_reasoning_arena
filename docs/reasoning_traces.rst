Reasoning Traces Analysis
=========================

Board Game Arena provides powerful **reasoning traces** functionality that captures and analyzes LLM decision-making processes during gameplay. This tutorial will guide you through obtaining, viewing, and analyzing reasoning traces to gain deep insights into how LLMs think through game strategies.

What are Reasoning Traces?
--------------------------

Reasoning traces capture three key pieces of information for each LLM move:

* **Board State**: The exact game position when the decision was made
* **Agent Reasoning**: The LLM's thought process and explanation for the move
* **Action Context**: The chosen action along with metadata (timestamp, episode, turn)

This combination provides unprecedented insight into AI decision-making patterns, strategic thinking, and potential weaknesses in LLM game-playing abilities.

Key Features
~~~~~~~~~~~~

* **Automatic Collection**: No special configuration required - traces are collected automatically during LLM gameplay
* **Multi-Game Support**: Works across all supported games (Tic-Tac-Toe, Connect Four, Kuhn Poker, etc.)
* **Comprehensive Logging**: Stores complete game context in SQLite databases
* **Analysis Tools**: Built-in categorization and visualization of reasoning patterns
* **Research Ready**: Designed for academic analysis of AI decision-making

Getting Started with Reasoning Traces
-------------------------------------

Step 1: Run Games with LLM Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reasoning traces are automatically collected whenever LLM agents play games. No special configuration is needed:

.. code-block:: bash

   # Basic LLM vs Random game - traces will be automatically collected
   python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
     env_config.game_name=tic_tac_toe \
     agents.player_0.type=llm \
     agents.player_0.model=litellm_groq/llama3-8b-8192 \
     num_episodes=5

.. code-block:: bash

   # LLM vs LLM - both agents' reasoning will be captured
   python3 scripts/runner.py --config src/board_game_arena/configs/example_config.yaml --override \
     env_config.game_name=connect_four \
     agents.player_0.type=llm \
     agents.player_0.model=litellm_groq/llama3-8b-8192 \
     agents.player_1.type=llm \
     agents.player_1.model=litellm_groq/llama3-70b-8192 \
     num_episodes=3

**Results Location**: Traces are automatically stored in ``results/llm_<model_name>.db``

Step 2: View Reasoning Traces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the built-in display script to examine the collected traces:

.. code-block:: bash

   # Display all reasoning traces from recent games
   python3 show_reasoning_traces.py

This will show detailed output like:

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

Advanced Analysis
-----------------

Extracting Specific Traces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For targeted analysis, use the extraction script with filters:

.. code-block:: bash

   # Extract traces for specific games
   python3 extract_reasoning_traces.py --game tic_tac_toe --episode 1

   # Extract all traces from database and save to CSV
   python3 extract_reasoning_traces.py --output-format csv --output traces.csv

Reasoning Pattern Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate comprehensive analysis and visualizations:

.. code-block:: bash

   # Analyze reasoning patterns and generate visualizations
   python3 -c "
   from analysis.reasoning_analysis import LLMReasoningAnalyzer
   analyzer = LLMReasoningAnalyzer('results/merged_logs.csv')
   analyzer.analyze_all_games()
   "

This generates multiple outputs:

* **Word Clouds**: ``plots/wordcloud_<model>_<game>.png`` - Common reasoning terms
* **Pie Charts**: ``plots/pie_reasoning_type_<model>_<game>.png`` - Reasoning category distributions
* **Heatmaps**: ``plots/heatmap_<model>_<game>.png`` - Move position preferences

Database Queries
~~~~~~~~~~~~~~~~~

For custom analysis, access the SQLite database directly:

.. code-block:: python

   import sqlite3
   import pandas as pd

   # Connect to the reasoning traces database
   conn = sqlite3.connect('results/llm_litellm_groq_llama3_8b_8192.db')

   # Query all reasoning traces
   df = pd.read_sql_query("""
       SELECT game_name, episode, turn, action, reasoning, board_state, timestamp
       FROM moves
       WHERE reasoning IS NOT NULL
       ORDER BY timestamp
   """, conn)

   # Analyze reasoning length by game
   reasoning_stats = df.groupby('game_name')['reasoning'].apply(
       lambda x: x.str.len().describe()
   )

   conn.close()

Understanding Reasoning Categories
----------------------------------

The analysis system automatically categorizes LLM reasoning into seven types:

Positional Strategy
~~~~~~~~~~~~~~~~~~~
Focuses on board position and control:

* Center control and positioning
* Corner and edge play strategies
* Spatial advantage concepts

**Example**: *"I'll take the center position for strategic advantage"*

Blocking & Defense
~~~~~~~~~~~~~~~~~~
Preventing opponent wins and defensive moves:

* Blocking immediate threats
* Preventing opponent strategies
* Defensive positioning

**Example**: *"I need to block their winning opportunity in column 3"*

Opponent Modeling
~~~~~~~~~~~~~~~~~
Understanding and predicting opponent behavior:

* Analyzing opponent patterns
* Predicting next moves
* Counter-strategy development

**Example**: *"Based on their previous moves, they prefer corner positions"*

Winning Logic
~~~~~~~~~~~~~
Direct winning opportunities and offensive play:

* Identifying winning moves
* Creating threats and forks
* Forcing winning positions

**Example**: *"This creates a fork - I can win on my next turn"*

Heuristic Reasoning
~~~~~~~~~~~~~~~~~~~
General strategic principles and rules of thumb:

* Best practices application
* General strategy guidelines
* Experience-based decisions

**Example**: *"Opening with corner moves is generally a good strategy"*

Rule-Based Decisions
~~~~~~~~~~~~~~~~~~~~
Following explicit game rules or predetermined strategies:

* Algorithmic approaches
* Systematic decision-making
* Rule application

**Example**: *"According to basic strategy, I should prioritize the center columns"*

Random/Unjustified
~~~~~~~~~~~~~~~~~~~
Unclear, random, or poorly justified reasoning:

* Unclear explanations
* Random choices
* Weak justifications

**Example**: *"I'll just pick this move randomly"*

Research Applications
---------------------

Model Comparison Studies
~~~~~~~~~~~~~~~~~~~~~~~~

Compare reasoning patterns between different LLMs:

.. code-block:: python

   # Compare reasoning quality between models
   import sqlite3
   import pandas as pd

   models = ['llm_groq_llama3_8b', 'llm_groq_llama3_70b', 'llm_openai_gpt4']

   for model in models:
       conn = sqlite3.connect(f'results/{model}.db')
       df = pd.read_sql_query("""
           SELECT reasoning, LENGTH(reasoning) as reasoning_length
           FROM moves WHERE reasoning IS NOT NULL
       """, conn)

       print(f"{model}: Avg reasoning length = {df['reasoning_length'].mean():.1f}")
       conn.close()

Strategy Evolution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track how reasoning changes throughout games:

.. code-block:: python

   # Analyze reasoning evolution within games
   df = pd.read_sql_query("""
       SELECT episode, turn, reasoning, action
       FROM moves
       WHERE game_name = 'tic_tac_toe'
       ORDER BY episode, turn
   """, conn)

   # Group by turn number to see patterns
   turn_patterns = df.groupby('turn')['reasoning'].apply(list)

Debugging LLM Decision-Making
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identify problematic reasoning patterns:

.. code-block:: python

   # Find games where LLM lost despite good reasoning
   losing_games = pd.read_sql_query("""
       SELECT episode, reasoning, action, board_state
       FROM moves
       WHERE game_result = 'loss' AND reasoning IS NOT NULL
   """, conn)

   # Analyze what went wrong
   for idx, game in losing_games.iterrows():
       print(f"Episode {game['episode']}: {game['reasoning'][:100]}...")

Best Practices
--------------

Data Collection
~~~~~~~~~~~~~~~

* **Run Multiple Episodes**: Collect sufficient data for statistical analysis (recommended: 10+ episodes per condition)
* **Use Consistent Models**: Keep model parameters constant for fair comparisons
* **Document Experiments**: Record experimental conditions and model configurations

Analysis Workflow
~~~~~~~~~~~~~~~~~

1. **Collect Data**: Run games with LLM agents
2. **Initial Exploration**: Use ``show_reasoning_traces.py`` to understand the data
3. **Pattern Analysis**: Apply reasoning categorization and generate visualizations
4. **Custom Analysis**: Write specific queries for your research questions
5. **Validation**: Manually verify automatic categorizations for accuracy

Interpretation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Context Matters**: Consider game state when evaluating reasoning quality
* **Length ≠ Quality**: Longer reasoning isn't necessarily better reasoning
* **Model Variations**: Different models may use different reasoning styles
* **Game Complexity**: Reasoning patterns vary significantly between simple and complex games

Troubleshooting
---------------

No Reasoning Traces Found
~~~~~~~~~~~~~~~~~~~~~~~~~

If you see "❌ No reasoning traces found":

1. Ensure you're running games with LLM agents (not just random agents)
2. Check that the database file exists in the ``results/`` directory
3. Verify your model configuration is correct

Database Connection Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check available databases
   import os
   db_files = [f for f in os.listdir('results/') if f.endswith('.db')]
   print("Available databases:", db_files)

Memory Issues with Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large reasoning trace datasets:

.. code-block:: python

   # Process data in chunks
   import sqlite3
   import pandas as pd

   conn = sqlite3.connect('results/large_dataset.db')

   # Use chunking for large datasets
   for chunk in pd.read_sql_query(
       "SELECT * FROM moves WHERE reasoning IS NOT NULL",
       conn, chunksize=1000
   ):
       # Process each chunk
       process_reasoning_chunk(chunk)

Next Steps
----------

Now that you understand reasoning traces analysis, explore:

* :doc:`analysis` - Advanced analysis techniques and metrics
* :doc:`examples` - More complex experimental setups
* :doc:`api_reference` - Technical details about the logging system
* :doc:`extending` - Adding custom reasoning analysis methods

The reasoning traces feature provides a unique window into LLM decision-making processes, enabling researchers to understand not just what decisions AI systems make, but how they arrive at those decisions.
