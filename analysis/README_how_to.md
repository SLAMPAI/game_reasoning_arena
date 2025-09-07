# Game Reasoning Arena - Analysis Module

This directory contains tools for analyzing LLM reasoning patterns and game performance data collected from the Game Reasoning Arena experiments.

## � Quick Start

Use these automated solutions:

### Option 1: Simple One-Command Analysis
```bash
# From the project root directory:
./run_analysis.sh
```

### Option 2: Python Quick Analysis
```bash
# From the project root directory:
python3 analysis/quick_analysis.py
```

### Option 3: Full Pipeline with Options
```bash
# From the project root directory:
PYTHONPATH=. python3 analysis/run_full_analysis.py --help

# Examples:
python3 analysis/run_full_analysis.py                    # Default settings
python3 analysis/run_full_analysis.py --quiet            # Less verbose
python3 analysis/run_full_analysis.py --plots-dir custom_plots  # Custom output

# Game-specific and Model-specific Analysis
python3 analysis/run_full_analysis.py --game hex         # Analyze only HEX games
python3 analysis/run_full_analysis.py --model llama3     # Analyze only Llama3 models
python3 analysis/run_full_analysis.py --game hex --model llama3  # Combined filtering
python3 analysis/run_full_analysis.py --game tic_tac_toe --quiet # Quiet HEX analysis
```

These automated solutions will:
1. 🔍 **Auto-discover** all SQLite databases in `results/`
2. 🔄 **Merge databases** into consolidated CSV files
3. 🎯 **Apply filters** (optional) for specific games or models
4. 🧠 **Analyze reasoning patterns** using rule-based categorization
5. 📊 **Generate visualizations** (plots, charts, heatmaps, word clouds)
6. 📋 **Create summary reports** with pipeline statistics
7. ⚡ **Handle errors gracefully** with detailed logging

**Output**: All results saved to `plots/` directory + detailed logs

---

## 🎯 Game-Specific and Model-Specific Analysis

The analysis pipeline now supports filtering for specific games and models, allowing you to focus your analysis on particular scenarios.

### Filter by Game
```bash
# Analyze only HEX games
python3 analysis/run_full_analysis.py --game hex

# Analyze only Tic-Tac-Toe games
python3 analysis/run_full_analysis.py --game tic_tac_toe

# Analyze only Connect Four games
python3 analysis/run_full_analysis.py --game connect_four
```

### Filter by Model
```bash
# Analyze only Llama models (partial matching)
python3 analysis/run_full_analysis.py --model llama

# Analyze only GPT models
python3 analysis/run_full_analysis.py --model gpt

# Analyze specific model variant
python3 analysis/run_full_analysis.py --model llama3-8b
```

### Combined Filtering
```bash
# Analyze HEX games played by Llama models only
python3 analysis/run_full_analysis.py --game hex --model llama

# Analyze Tic-Tac-Toe games with GPT models in quiet mode
python3 analysis/run_full_analysis.py --game tic_tac_toe --model gpt --quiet
```

### Output Organization
When filters are applied, plots are automatically organized in subdirectories:
- `plots/game_hex/` - HEX-specific analysis
- `plots/model_llama/` - Llama model-specific analysis
- `plots/game_hex_model_llama/` - Combined filtering results

This makes it easy to focus on specific research questions without processing all data.

---

## �📁 Directory Contents

### Core Analysis Scripts

#### `reasoning_analysis.py` - Main Analysis Engine
The primary module for reasoning pattern analysis and categorization.

**Key Features:**
- **Reasoning Categorization**: Automatically classifies LLM reasoning into 7 types:
  - `Positional`: Center control, corner play, spatial strategies
  - `Blocking`: Defensive moves, preventing opponent wins
  - `Opponent Modeling`: Predicting opponent behavior
  - `Winning Logic`: Identifying winning opportunities, creating threats
  - `Heuristic`: General strategic principles
  - `Rule-Based`: Following explicit game rules
  - `Random/Unjustified`: Unclear or random reasoning

- **Visualization Generation**: Creates multiple plot types:
  - Word clouds of reasoning patterns
  - Pie charts showing reasoning category distributions
  - Heatmaps of move position preferences
  - Statistical summaries of decision-making behavior

**Usage:**
```python
from reasoning_analysis import LLMReasoningAnalyzer

# Initialize with CSV file from post-game processing
analyzer = LLMReasoningAnalyzer('results/merged_logs_YYYYMMDD_HHMMSS.csv')

# Categorize reasoning patterns
analyzer.categorize_reasoning()

# Generate visualizations
analyzer.compute_metrics(plot_dir='plots')
analyzer.plot_heatmaps_by_agent(output_dir='plots')
analyzer.plot_wordclouds_by_agent(output_dir='plots')
```

**Dependencies:** pandas, matplotlib, seaborn, wordcloud, transformers, numpy

---

#### `extract_reasoning_traces.py` - Data Extraction Tool (Standalone)
Comprehensive command-line tool for extracting and viewing reasoning traces from SQLite databases. This tool runs independently and is not part of the automated pipeline, allowing for ad-hoc detailed trace inspection.

**Key Features:**
- **Database Discovery**: Automatically finds available database files
- **Flexible Filtering**: Extract by game type, episode, or custom criteria
- **Multiple Output Formats**: Text display, CSV export, JSON export
- **Pattern Analysis**: Built-in statistics and reasoning pattern detection

**Usage:**
```bash
# List available databases (run without --db argument)
python extract_reasoning_traces.py

# Extract all traces from a specific database
python extract_reasoning_traces.py --db results/llm_litellm_groq_llama3_8b_8192.db

# Filter by game and episode
python extract_reasoning_traces.py --db results/llm_model.db --game tic_tac_toe --episode 1

# Export to CSV for further analysis
python extract_reasoning_traces.py --db results/llm_model.db --export-csv traces.csv

# Export formatted traces to text file (perfect for academic papers)
python extract_reasoning_traces.py --db results/llm_model.db --game tic_tac_toe --export-txt detailed_report.txt

# View analysis only (no detailed traces)
python extract_reasoning_traces.py --db results/llm_model.db --analyze-only
```

**Dependencies:** sqlite3, pandas, pathlib, argparse

---

#### `generate_reasoning_plots.py` - Reasoning Analysis Visualization
Generates comprehensive reasoning analysis plots including model comparisons, game-specific analysis, and evolution plots.

**Key Features:**
- **Model Name Cleaning**: Standardizes model names for clear visualization
- **Multiple Plot Types**: Bar charts, pie charts, stacked charts, and evolution plots
- **Game-Specific Analysis**: Individual plots per game per model
- **Aggregated Views**: Cross-game reasoning pattern analysis

**Usage:**
```python
from generate_reasoning_plots import plot_reasoning_bar_chart, clean_model_name

# Generate reasoning distribution chart for a model
reasoning_percentages = {'Positional': 30.0, 'Blocking': 25.0, 'Winning Logic': 20.0}
model_name = clean_model_name('llm_litellm_groq_llama3_8b_8192')
plot_reasoning_bar_chart(reasoning_percentages, model_name, 'output_chart.png')
```

**Command-line usage:**
```bash
python analysis/generate_reasoning_plots.py
```

**Dependencies:** matplotlib, pandas, pathlib

---

#### `post_game_processing.py` - Data Aggregation and Processing
Merges individual agent SQLite databases into consolidated CSV files for analysis.

**Key Features:**
- **Database Merging**: Combines all agent-specific SQLite logs
- **Data Validation**: Ensures data consistency and completeness
- **Summary Statistics**: Computes game-level and episode-level metrics
- **Timestamped Output**: Generates uniquely named merged files

**Usage:**
```python
from post_game_processing import merge_sqlite_logs, compute_summary_statistics

# Merge all SQLite logs in results directory
merged_df = merge_sqlite_logs('results/')

# Compute summary statistics
summary_stats = compute_summary_statistics(merged_df)

# Results saved to: results/merged_logs_YYYYMMDD_HHMMSS.csv
```

**Dependencies:** sqlite3, pandas, pathlib, datetime

---

---

## 🔄 Automated Analysis Workflow (RECOMMENDED)

The automated pipeline replaces the manual multi-step process below. Instead of running each script individually, you can now:

### Single Command Analysis
```bash
./run_analysis.sh                    # Interactive analysis with progress tracking
./run_analysis.sh --quiet            # Quiet mode
./run_analysis.sh --full             # Full analysis with all options

# Note: For game/model filtering, use the Python pipeline:
python3 analysis/run_full_analysis.py --game hex --model llama
```

### Python Pipeline
```bash
python3 analysis/run_full_analysis.py [options]

# Filtering options:
python3 analysis/run_full_analysis.py --game hex      # HEX games only
python3 analysis/run_full_analysis.py --model llama   # Llama models only
python3 analysis/run_full_analysis.py --game hex --model llama  # Combined
```

**What the automated pipeline does:**
1. **Database Discovery & Merging**: Automatically finds and merges all `.db` files in `results/`
2. **Reasoning Analysis**: Categorizes reasoning patterns using rule-based classification
3. **Visualization Generation**: Creates comprehensive plots, charts, heatmaps, and word clouds
4. **Error Handling**: Continues execution even if individual steps fail
5. **Progress Tracking**: Provides detailed logging and progress updates
6. **Summary Reporting**: Generates JSON reports with pipeline statistics

**Note:** For detailed reasoning trace extraction, use the standalone tool:
```bash
python analysis/extract_reasoning_traces.py --db results/your_database.db
```

**Generated Output:**
- `plots/*.png` - All visualization files
- `results/merged_logs_*.csv` - Consolidated data files
- `results/tables/*.csv` - Performance tables and metrics

All execution details are displayed in the console output.

---

## 🔧 Manual Analysis (Legacy Workflow)

> **Note**: The manual workflow below is still available but **not recommended**.
> Use the automated pipeline above for much better experience.

## 🔄 Typical Analysis Workflow

### 1. Data Collection
Run games with LLM agents - reasoning traces are automatically collected:
```bash
python scripts/runner.py --config configs/example_config.yaml --override \
  env_config.game_name=tic_tac_toe \
  agents.player_0.type=llm \
  agents.player_0.model=litellm_groq/llama3-8b-8192 \
  num_episodes=10
```

### 2. Data Processing
Merge individual databases into analysis-ready format:
```python
python -c "
from analysis.post_game_processing import merge_sqlite_logs
merge_sqlite_logs('results/')
"
```

### 3. Quick Data Exploration
View reasoning traces to understand the data:
```bash
python analysis/extract_reasoning_traces.py --db results/llm_model.db --analyze-only
```

### 4. Comprehensive Analysis
Generate full reasoning analysis and visualizations:
```python
python -c "
from analysis.reasoning_analysis import LLMReasoningAnalyzer
analyzer = LLMReasoningAnalyzer('results/merged_logs_latest.csv')
analyzer.categorize_reasoning()
analyzer.compute_metrics(plot_dir='plots')
analyzer.plot_heatmaps_by_agent(output_dir='plots')
analyzer.plot_wordclouds_by_agent(output_dir='plots')
"
```

### 5. Model Comparison
Compare reasoning patterns across different models:
```python
from analysis.generate_reasoning_plots import ReasoningPlotGenerator
plotter = ReasoningPlotGenerator('results/merged_logs_latest.csv')
plotter.generate_model_plots('plots/')
```

## 📊 Generated Outputs

### Database Files (`results/*.db`)
- Individual agent reasoning traces
- SQLite format for efficient querying
- Contains: game_name, episode, turn, action, reasoning, board_state, timestamp

### Merged CSV Files (`results/merged_logs_*.csv`)
- Consolidated data from all agents
- Ready for statistical analysis
- Timestamped for version control

### Visualization Files (`plots/`)
- `wordcloud_<model>_<game>.png` - Common reasoning terms
- `pie_reasoning_type_<model>_<game>.png` - Reasoning category distributions
- `heatmap_<model>_<game>.png` - Move position preferences
- `reasoning_bar_chart_<model>.png` - Model-specific reasoning breakdowns
- `entropy_by_turn_all_agents_<game>.png` - Reasoning diversity over time
- `reasoning_evolution_<model>_<game>.png` - How reasoning patterns change during games

### ⚠️ Important Note: Short Game Limitations

**Short games** (like Kuhn Poker, Matching Pennies, and Prisoner's Dilemma) may have limited or empty entropy/evolution visualizations due to:

- **Few turns per game**: Games lasting only 1-2 turns provide insufficient data for meaningful entropy calculations
- **Limited reasoning diversity**: With only 1-2 reasoning entries per agent per turn, entropy values are often zero
- **No evolution patterns**: Reasoning evolution requires multiple turns to show meaningful progression
- **Sparse data**: Individual agents may have too few data points for statistical analysis

**Recommendation**: Focus on **longer games** (Tic-Tac-Toe, Connect Four) for entropy and evolution analysis. Short games are better suited for reasoning category distribution analysis (pie charts, bar charts).

## 🧪 Research Applications

### Model Comparison Studies
- Compare reasoning sophistication across different LLMs
- Identify model-specific strategic preferences
- Evaluate reasoning consistency within models

### Game Strategy Analysis
- Understand how LLMs approach different game types
- Identify common strategic patterns and misconceptions
- Analyze adaptation to different opponents

### Reasoning Quality Assessment
- Categorize and quantify reasoning types
- Identify gaps in strategic thinking
- Evaluate decision-making consistency

### Performance Correlation
- Link reasoning quality to game outcomes
- Identify which reasoning types lead to better performance
- Study the relationship between reasoning length and quality

## 🔧 Configuration and Customization

### Adding New Reasoning Categories
Modify `REASONING_RULES` in `reasoning_analysis.py`:
```python
REASONING_RULES = {
    "Custom_Category": [
        re.compile(r"\bcustom_pattern\b"),
        re.compile(r"\banother_pattern\b")
    ],
    # ... existing categories
}
```

### Custom Visualization Themes
Modify plotting functions in `generate_reasoning_plots.py` for custom styling, colors, and layouts.

### Database Schema Extensions
The SQLite schema can be extended by modifying the logging functions in the main arena codebase.

## 📚 Dependencies

Core requirements for the analysis module:

```bash
pip install pandas matplotlib seaborn wordcloud transformers numpy
```

## 🐛 Troubleshooting


### Empty or Missing Entropy/Evolution Plots
- **Short games** (Kuhn Poker, Matching Pennies, Prisoner's Dilemma) naturally produce sparse entropy data
- Games with only 1-2 turns cannot show meaningful reasoning evolution
- Consider focusing analysis on longer games (Tic-Tac-Toe, Connect Four) for temporal analysis
- Use reasoning category distribution plots (pie/bar charts) for short games instead

### Memory Issues with Large Datasets
- Process data in chunks using pandas `chunksize` parameter
- Filter data by game type or time period before analysis
- Use `--game` and `--model` filters to analyze specific subsets
- Use SQLite queries to pre-filter before loading into memory

### Focused Analysis
- Use `--game hex` to analyze only HEX games for faster processing
- Use `--model llama` to compare only Llama model variants
- Combine filters: `--game hex --model llama` for targeted research questions

---
