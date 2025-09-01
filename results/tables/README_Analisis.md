# Performance Analysis Tables

## Overview

This directory contains comprehensive performance tables and analytical data generated from the Game Reasoning Arena experiments. The tables provide quantitative assessments of large language model (LLM) agents across multiple strategic game domains, focusing on both performance outcomes and reasoning characteristics.

## Performance Evaluation Tables

### Overall Performance Table (`overall_performance_table.csv`)

The overall performance table presents aggregate performance metrics across all game domains for each evaluated model. This table serves as the primary comparative assessment tool, providing a holistic view of model capabilities independent of specific game mechanics. The analysis aggregates performance across diverse strategic contexts to identify models with robust general strategic reasoning abilities.

The table includes total games played, games won, win rates with statistical confidence intervals, and average reward scores. Win rates are calculated as the proportion of games won relative to total games played, with standard deviations computed to assess performance consistency. Average rewards represent the mean utility achieved across all games, with rewards normalized to a standard [-1, +1] scale to ensure fair comparison across games with different reward structures. This normalization transforms rewards within each game to a common scale where -1 represents the worst possible outcome and +1 represents the best, enabling meaningful aggregation across diverse game types.

### Per-Game Performance Table (`per_game_performance_table.csv`)

The per-game performance table provides detailed performance breakdowns for each model within individual game domains. This granular analysis enables identification of domain-specific strengths and weaknesses, revealing how different architectural approaches and training methodologies affect performance across varying strategic contexts.

Each entry represents a model's performance within a specific game environment, including the number of games played, win rate with confidence intervals, and average reward achieved. This decomposition is essential for understanding the generalizability of strategic reasoning capabilities, as it reveals whether superior overall performance stems from consistent competence across domains or exceptional performance in specific strategic contexts. The table supports analysis of game complexity effects on model performance and identification of strategic reasoning patterns that transfer across domains.

### Win Rate Pivot Table (`win_rate_pivot_table.csv`)

The win rate pivot table restructures performance data into a matrix format where models constitute rows and games constitute columns, with win rates populating the intersections. This tabular arrangement facilitates comparative analysis across both dimensions simultaneously, enabling researchers to identify performance patterns and clusters in the data.

This format is particularly valuable for identifying models with similar performance profiles and games that consistently challenge or favor certain types of reasoning approaches. The matrix structure supports clustering analysis and correlation studies, helping to reveal underlying relationships between model architectures and strategic task characteristics. Statistical analysis of this matrix can reveal whether certain game types systematically favor particular modeling approaches.

### Reward Pivot Table (`reward_pivot_table.csv`)

The reward pivot table presents average rewards in a similar matrix structure to the win rate table, but focuses on the continuous reward signals rather than binary win/loss outcomes. This perspective is crucial because reward values capture the quality of strategic decision-making beyond simple success or failure, providing insight into how effectively models optimize their strategic objectives.

Reward-based analysis is particularly important in games with continuous or multi-level scoring systems, where the margin of victory or quality of play matters as much as the final outcome. This table enables analysis of strategic efficiency and decision quality, supporting investigations into whether models that achieve high win rates do so through optimal play or through exploiting specific strategic vulnerabilities in opponents.

### Games Played Pivot Table (`games_played_pivot_table.csv`)

The games played pivot table documents the experimental design by recording the number of games conducted for each model-game combination. This metadata is essential for interpreting the statistical significance of performance measures and ensuring that comparative analyses account for differences in sample sizes across experimental conditions.

Uneven sample sizes can significantly impact the reliability of performance comparisons, particularly when calculating confidence intervals and conducting statistical tests. This table enables researchers to weight their analyses appropriately and identify cases where additional data collection may be necessary to achieve statistically robust conclusions. The documentation of experimental coverage also supports meta-analyses and systematic reviews of strategic reasoning research.

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

Where:
- `reward` is the original reward value for a specific game outcome
- `min_reward` is the minimum reward observed in the normalization scope
- `max_reward` is the maximum reward observed in the normalization scope
- The factor of 2 and subtraction of 1 transforms the [0,1] range to [-1,+1]

**Normalization Scope:**

Two normalization approaches are employed depending on the analysis context:

1. **Per-game normalization**: Used when calculating performance metrics within individual game domains, where rewards are normalized within each game separately to a [-1, +1] scale. In this approach, -1 represents the worst observed outcome within that specific game and +1 represents the best observed outcome.

2. **Cross-game normalization**: Applied when computing overall performance metrics that aggregate across multiple games, where all rewards from all games are normalized together to the same [-1, +1] scale.

**Why Min-Max Normalization Over Z-Score (Mean/Standard Deviation) Normalization:**

Min-max normalization was selected over z-score normalization (which uses mean and standard deviation) for several methodological and theoretical reasons:

1. **Bounded Output Range**: Min-max normalization guarantees that all normalized values fall within the exact [-1, +1] range, regardless of the original distribution shape. Z-score normalization can produce values outside any predetermined range, particularly for distributions with outliers or non-normal characteristics common in game outcomes.

2. **Interpretability**: In the min-max approach, -1 always represents "worst possible performance" and +1 always represents "best possible performance" within the normalization scope. This provides intuitive interpretation where 0 represents median performance. Z-score normalization centers around the mean, which may not correspond to meaningful performance benchmarks in strategic games.

3. **Robustness to Distribution Shape**: Game reward distributions are often non-normal, featuring discrete outcomes, multimodal patterns, or strategic equilibria that create unusual distribution shapes. Min-max normalization is distribution-agnostic and preserves the relative ordering of all values. Z-score normalization assumes underlying normal distributions and can distort relative performance relationships when this assumption is violated.

4. **Handling of Ties and Discrete Outcomes**: Many games produce discrete reward structures (e.g., win=+1, draw=0, loss=-1). Min-max normalization preserves these discrete relationships proportionally, while z-score normalization can artificially inflate the importance of small performance differences in games with limited outcome variety.

5. **Cross-Game Comparability**: When aggregating across games with fundamentally different reward scales (e.g., a simple game with rewards {-1, 0, +1} versus a complex game with rewards {-100, -50, 0, +50, +100}), min-max normalization ensures that both games contribute equally to overall performance assessment. Z-score normalization would weight games differently based on their variance, potentially overemphasizing performance differences in high-variance games.

6. **Preservation of Performance Hierarchies**: Min-max normalization maintains the exact relative ordering of all agents within each game or across games. If Agent A outperforms Agent B in the original reward space, this relationship is preserved in the normalized space. Z-score normalization can potentially alter relative rankings when distributions have different shapes or when outliers are present.

**Practical Example:**

Consider two games in the evaluation:
- **Simple Game**: Rewards {-1, 0, +1} with agents achieving mean rewards of [-0.5, 0.2, 0.8]
- **Complex Game**: Rewards {-5, -2, 0, +3, +5} with agents achieving mean rewards of [-2.5, 1.0, 4.0]

**Min-Max Normalization:**
- Simple Game: [-0.5, 0.2, 0.8] → [0, 0.7, 1.0] → [-1, 0.4, 1.0]
- Complex Game: [-2.5, 1.0, 4.0] → [0.25, 0.75, 1.0] → [-0.5, 0.5, 1.0]

**Z-Score Normalization Problems:**
- Different games would have different variance scales
- Performance differences would be weighted by variance, not actual strategic superiority
- Cross-game aggregation would not be meaningful without additional weighting schemes

This transformation ensures that games with different reward structures contribute equally to performance assessments while preserving the essential performance relationships within each strategic domain.

### Statistical Analysis

All performance tables include confidence intervals calculated using appropriate statistical methods for the underlying data distributions. Win rates employ binomial confidence intervals, while reward measures use t-distribution-based intervals when sample sizes permit, falling back to bootstrap methods for small samples. Standard deviations are reported alongside means to provide measures of performance consistency and reliability.

The aggregation methods used in overall performance tables weight individual game contributions equally unless otherwise specified, ensuring that performance in complex, longer games does not disproportionately influence overall assessments. Where multiple games of the same type were played, individual game results are treated as independent observations for statistical purposes.

## Data Integration and Formats

Tables are provided in multiple formats to support different analytical workflows. CSV formats enable integration with statistical software and database systems, while LaTeX formats support direct inclusion in academic publications. JSON files provide structured data for programmatic analysis and visualization tools.

The performance tables summary JSON file contains metadata about table generation, including timestamps, data sources, and aggregation parameters. This metadata ensures reproducibility and enables tracking of data lineage through the analytical pipeline. All tables maintain consistent naming conventions and identifier schemes to support automated analysis and cross-referencing between different analytical perspectives.
