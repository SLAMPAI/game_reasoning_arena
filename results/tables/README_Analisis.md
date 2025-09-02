# Performance Analysis Tables

## Table Generation

### How to Generate These Tables

The performance analysis tables in this directory are generated from SQLite databases using the same methodology as the main application (`app.py`). This ensures complete accuracy and consistency with the leaderboard data displayed in the Game Reasoning Arena interface.

#### Publication-Ready Tables Generation (Recommended)

**Generate Main Publication Tables:**
```bash
# From the project root directory:
cd /path/to/game_reasoning_arena
python -c "from analysis.performance_tables import generate_publication_tables; generate_publication_tables('results/tables')"
```

This command generates the main publication-ready tables:
- `overall_performance.csv/.tex` - Overall model performance across all games
- `win_rate_by_game.csv/.tex` - Win rates by model and game (pivot table)

#### Update All Pivot Tables

**Update Additional Analysis Tables:**
```bash
# From the project root directory:
python -c "
from analysis.performance_tables import PerformanceTableGenerator, display_game_name
import pandas as pd
from pathlib import Path

generator = PerformanceTableGenerator(use_databases=True)
performance = generator.compute_win_rates(by_game=True)

# Update win_rate_pivot_table.csv
win_rate_pivot = performance.pivot_table(index='agent_name', columns='game_name', values='win_rate_vs_random', fill_value=0)
win_rate_pivot.columns = [display_game_name(col) for col in win_rate_pivot.columns]
win_rate_pivot['Overall'] = win_rate_pivot.mean(axis=1)
win_rate_pivot = win_rate_pivot.sort_values('Overall', ascending=False).reset_index()
win_rate_pivot.to_csv('results/tables/win_rate_pivot_table.csv', index=False)

# Update games_played_pivot_table.csv
games_played_pivot = performance.pivot_table(index='agent_name', columns='game_name', values='games_played', fill_value=0)
games_played_pivot.columns = [display_game_name(col) for col in games_played_pivot.columns]
games_played_pivot['Overall'] = games_played_pivot.sum(axis=1)
games_played_pivot = games_played_pivot.sort_values('Overall', ascending=False).reset_index()
games_played_pivot.to_csv('results/tables/games_played_pivot_table.csv', index=False)

# Update reward_pivot_table.csv
reward_pivot = performance.pivot_table(index='agent_name', columns='game_name', values='avg_reward', fill_value=0)
reward_pivot.columns = [display_game_name(col) for col in reward_pivot.columns]
reward_pivot['Overall'] = reward_pivot.mean(axis=1)
reward_pivot = reward_pivot.sort_values('Overall', ascending=False).reset_index()
reward_pivot.to_csv('results/tables/reward_pivot_table.csv', index=False)

print('✅ All pivot tables updated with SQLite data')
"
```

#### Source Scripts

- **Primary Script**: `analysis/performance_tables.py` - Contains the `PerformanceTableGenerator` class with SQLite database integration
- **Data Source**: SQLite databases in `results/*.db` (same as used by `app.py`)
- **Model Name Cleaning**: Uses `ui/utils.py` clean_model_name function for consistency

#### Generated Files

The performance table generator creates the following files in `results/tables/`:

**Main Publication Tables:**
- `overall_performance.csv/.tex` - Overall performance with organization grouping
- `win_rate_by_game.csv/.tex` - Win rates by model and game (publication format)

**Analysis Pivot Tables:**
- `win_rate_pivot_table.csv` - Win rates in matrix format (models × games)
- `reward_pivot_table.csv` - Average rewards in matrix format
- `games_played_pivot_table.csv` - Number of games played per model-game combination

**Additional Files:**
- `reasoning_evolution_patterns.json` - Temporal reasoning pattern analysis
- `performance_tables_summary.json` - Generation metadata and top performers

## Overview

This directory contains comprehensive performance tables and analytical data generated from the Game Reasoning Arena experiments. The tables provide quantitative assessments of large language model (LLM) agents across multiple strategic game domains, focusing on both performance outcomes and reasoning characteristics.

## Performance Evaluation Tables

### Overall Performance Table (`overall_performance.csv/.tex`)

The overall performance table presents aggregate performance metrics across all game domains for each evaluated model. This publication-ready table uses SQLite database data (same as `app.py`) and features organization-based grouping (OpenAI, Meta (Llama), Google, Qwen, etc.) for clear presentation.

**Key Metrics:**
- **Win Rate (%)**: Percentage of games won against random opponents (primary metric)
- **Avg Reward**: Mean reward achieved across all games with standard deviation
- **Organization Grouping**: Models grouped by company for publication clarity

The table focuses on **win rate vs random opponents** as the primary performance metric, following the established methodology in `app.py`. This metric provides a standardized baseline for comparing strategic reasoning capabilities across different model architectures and training approaches.

**Available Formats:**
- CSV: `overall_performance.csv` (for analysis)
- LaTeX: `overall_performance.tex` (for publications)

### Win Rate by Game Table (`win_rate_by_game.csv/.tex`)

This publication-ready pivot table shows detailed win rates for each model across all seven strategic games. The matrix format enables easy comparison of model performance across different strategic contexts.


**Key Features:**
- Models sorted by overall performance (average across games)
- Win rates calculated against random opponents for consistency
- Reveals domain-specific strengths and weaknesses
- Identifies games that consistently challenge certain model types

**Available Formats:**
- CSV: `win_rate_by_game.csv` (for analysis)
- LaTeX: `win_rate_by_game.tex` (for publications)

### Win Rate Pivot Table (`win_rate_pivot_table.csv`)

Extended analysis version of the win rate by game table with additional statistical information. This table restructures performance data into a matrix format where models constitute rows and games constitute columns, with win rates and overall averages populating the intersections.


### Reward Pivot Table (`reward_pivot_table.csv`)

The reward pivot table presents average rewards in a matrix structure, focusing on continuous reward signals rather than binary win/loss outcomes. This perspective captures the quality of strategic decision-making beyond simple success or failure.

**Key Insights:**
- **Strategic Efficiency**: How well models optimize their strategic objectives
- **Decision Quality**: Margin of victory and quality of play analysis
- **Game-Specific Performance**: Rewards vary significantly across game types
- **Model Comparison**: Direct comparison of strategic optimization capabilities



### Games Played Pivot Table (`games_played_pivot_table.csv`)

This metadata table documents the experimental design by recording the number of games conducted for each model-game combination. Essential for interpreting statistical significance and ensuring comparative analyses account for sample size differences.

**Importance:**
- **Statistical Validity**: Enables proper confidence interval calculations
- **Experimental Coverage**: Documents completeness of data collection
- **Sample Size Analysis**: Identifies where additional data may be needed
- **Meta-Analysis Support**: Provides weighting information for aggregate studies

- **Data Distribution:**
- Most comprehensive coverage: Llama-3-8b-8192 (166 total games)
- Balanced across game types with strategic sampling
- Sufficient sample sizes for statistical significance testing


## Data Methodology and Accuracy

### SQLite Database Integration

All performance tables now use **SQLite database methodology identical to `app.py`** for guaranteed accuracy and consistency. This represents a significant improvement over previous CSV-based approaches.

**Key Improvements:**
- **Complete Data Access**: Uses all game history from SQLite databases, not limited CSV subsets
- **Identical Calculations**: Win rates and metrics match exactly with the main application leaderboard
- **Real-time Consistency**: Tables reflect the same data users see in the web interface
- **No Data Loss**: Includes all completed games and proper final result determination

### Performance Metrics Definition

**Primary Metric - Win Rate vs Random:**
- Calculated as: `(wins_vs_random / total_vs_random) * 100`
- Uses games where `opponent = 'random_None'`
- Focuses on strategic competence against baseline random strategy
- Provides standardized comparison across all models

**Average Reward Calculation:**
- Mean reward value across all completed games for each model
- Includes both positive and negative rewards to capture full performance spectrum
- Not normalized to preserve game-specific reward structures
- Standard deviation calculated to show consistency of performance

**Games Played Counting:**
- Counts completed game instances from `game_results` table
- Each unique `(game_name, episode)` combination represents one completed game
- Excludes incomplete or crashed games for accuracy
- Provides proper denominators for statistical calculations


## Reasoning Analysis Tables

### Agent Metrics Summary (`agent_metrics_summary.csv`)

The agent metrics summary table provides detailed analysis of reasoning characteristics exhibited during gameplay. This table moves beyond performance outcomes to examine the cognitive processes underlying strategic behavior, offering insights into how different models approach strategic reasoning tasks.

The metrics include total moves made, average reasoning length, percentage of opponent mentions in reasoning traces, reasoning diversity scores, and reasoning entropy measures. Average reasoning length quantifies the verbosity and depth of model reasoning, while opponent mentions indicate the degree to which models explicitly consider adversarial dynamics. Reasoning diversity measures the variety of strategic concepts and approaches employed, calculated through lexical and semantic diversity metrics applied to reasoning traces.

Reasoning entropy quantifies the unpredictability and complexity of reasoning patterns, computed through information-theoretic measures applied to reasoning trace sequences. Higher entropy values indicate more varied and complex reasoning approaches, while lower values suggest more stereotyped or formulaic strategic thinking. These measures collectively provide a multidimensional characterization of strategic reasoning styles and capabilities.

### Reasoning Evolution Patterns (`reasoning_evolution_patterns.json`)

The reasoning evolution patterns file contains structured analysis of how reasoning characteristics change over the course of gameplay. This temporal analysis reveals learning and adaptation patterns, strategic flexibility, and the dynamic nature of strategic reasoning in LLM agents.

The analysis tracks reasoning complexity, strategic focus, and decision confidence across game turns, providing insights into how models adapt their reasoning approaches as games progress. Early-game reasoning patterns often focus on establishing strategic positions and understanding game mechanics, while late-game patterns typically emphasize tactical optimization and endgame planning. This temporal dimension is crucial for understanding the sophistication of strategic reasoning and identifying models capable of dynamic strategy adaptation.

The evolution patterns also capture shifts in reasoning diversity and entropy over time, revealing whether models maintain consistent reasoning approaches or adapt their cognitive strategies based on game state and opponent behavior. These patterns provide evidence for or against sophisticated strategic meta-cognition in LLM agents.

## Statistical Methodology

### Reward Normalization

All reward values in performance tables are normalized to ensure fair comparison across games with different reward structures. This normalization prevents games with larger reward ranges from disproportionately influencing aggregate performance metrics.

#### Min-Max Normalization Methodology

This analysis employs **min-max normalization** (also called range normalization), a linear transformation that maps values from their original range to a standardized [-1, +1] scale. This approach was specifically chosen over alternative normalization methods for several critical reasons related to the nature of game-theoretic performance evaluation.

**Mathematical Formulation:**

The normalization formula applied is:

```
normalized = 2 × (reward - min_reward) / (max_reward - min_reward) - 1
```

**Understanding the Mathematical Transformation:**

The formula combines two transformations to convert from the original reward range to [-1, +1]:

1. **Basic Min-Max Normalization** (maps to [0,1]):
   ```
   basic_normalized = (reward - min_reward) / (max_reward - min_reward)
   ```

2. **Range Transformation** (maps from [0,1] to [-1,+1]):
   ```
   final_normalized = 2 × basic_normalized - 1
   ```

**Step-by-Step Breakdown:**

Let's trace through the transformation with examples:

- **Worst performance** (`reward = min_reward`):
  - Basic: `(min_reward - min_reward) / (max_reward - min_reward) = 0`
  - Final: `2 × 0 - 1 = -1`

- **Middle performance** (`reward = (min_reward + max_reward) / 2`):
  - Basic: `0.5`
  - Final: `2 × 0.5 - 1 = 0`

- **Best performance** (`reward = max_reward`):
  - Basic: `(max_reward - min_reward) / (max_reward - min_reward) = 1`
  - Final: `2 × 1 - 1 = +1`

**Why [-1, +1] Instead of [0, 1]?**

The [-1, +1] range is preferred because:

1. **Intuitive interpretation**: -1 = worst, 0 = neutral/average, +1 = best
2. **Symmetric around zero**: Makes it easier to identify positive vs negative performance
3. **Common in game theory**: Many games naturally use [-1, +1] (loss/draw/win)
4. **Statistical properties**: Zero-centered data often works better with analytical methods

Where:
- `reward` is the original reward value for a specific game outcome
- `min_reward` is the minimum reward observed in the normalization scope
- `max_reward` is the maximum reward observed in the normalization scope
- The factor of 2 stretches the [0,1] range to [0,2], then subtracting 1 shifts it to [-1,+1]

**Normalization Scope:**

Two normalization approaches are employed depending on the analysis context:

1. **Per-game normalization**: Used when calculating performance metrics within individual game domains, where rewards are normalized within each game separately to a [-1, +1] scale. In this approach, -1 represents the worst observed outcome within that specific game and +1 represents the best observed outcome.

2. **Cross-game normalization**: Applied when computing overall performance metrics that aggregate across multiple games, where all rewards from all games are normalized together to the same [-1, +1] scale.

### Statistical Analysis

All performance tables include confidence intervals calculated using appropriate statistical methods for the underlying data distributions. Win rates employ binomial confidence intervals, while reward measures use t-distribution-based intervals when sample sizes permit, falling back to bootstrap methods for small samples. Standard deviations are reported alongside means to provide measures of performance consistency and reliability.

**Implementation Status:**
Reward normalization using the min-max methodology described above is **fully implemented** in the current analysis pipeline. All reward values in the performance tables reflect normalized scores in the [-1, +1] range.

**Current Implementation Note:**
The reward standard deviations shown in the tables are currently calculated as `avg_reward * 0.5` as a placeholder. This is not a true statistical calculation and should be replaced with actual standard deviation computation from individual game rewards.

The aggregation methods used in overall performance tables weight individual game contributions equally unless otherwise specified, ensuring that performance in complex, longer games does not disproportionately influence overall assessments. Where multiple games of the same type were played, individual game results are treated as independent observations for statistical purposes.

## Data Integration and Formats

Tables are provided in multiple formats to support different analytical workflows. CSV formats enable integration with statistical software and database systems, while LaTeX formats support direct inclusion in academic publications. JSON files provide structured data for programmatic analysis and visualization tools.

The performance tables summary JSON file contains metadata about table generation, including timestamps, data sources, and aggregation parameters. This metadata ensures reproducibility and enables tracking of data lineage through the analytical pipeline. All tables maintain consistent naming conventions and identifier schemes to support automated analysis and cross-referencing between different analytical perspectives.
