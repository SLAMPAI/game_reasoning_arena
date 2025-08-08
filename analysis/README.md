# Game Reasoning Arena - Analysis Module

This directory contains tools for analyzing LLM reasoning patterns and game performance data collected from the Game Reasoning Arena experiments.

## 📁 Directory Contents

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

#### `extract_reasoning_traces.py` - Data Extraction Tool
Comprehensive command-line tool for extracting and viewing reasoning traces from SQLite databases.

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
```
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
wordcloud>=1.8.0
transformers>=4.0.0
numpy>=1.20.0
sqlite3 (built-in)
pathlib (built-in)
```

Install with:
```bash
pip install pandas matplotlib seaborn wordcloud transformers numpy
```

## 🐛 Troubleshooting

### No Data Found
- Ensure games were run with LLM agents (not just random agents)
- Check that database files exist in `results/` directory
- Verify model configuration was correct during game runs

### Memory Issues with Large Datasets
- Process data in chunks using pandas `chunksize` parameter
- Filter data by game type or time period before analysis
- Use SQLite queries to pre-filter before loading into memory

### Visualization Problems
- Ensure output directories exist before running plotting functions
- Check file permissions for output directories
- Verify all required fonts are available for matplotlib

---

*This analysis module is part of the Game Reasoning Arena project for studying LLM decision-making in strategic games.*
