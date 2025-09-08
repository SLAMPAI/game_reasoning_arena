# Reasoning Pattern Color and Pattern Reference

This file documents the visual coding scheme used in all reasoning analysis plots to ensure consistent interpretation across all visualizations.

## Color and Pattern Mapping

### Spatial/Board Game Categories
These categories apply primarily to games like Tic-Tac-Toe, Connect Four, Hex, etc.

| Category | Color | Description |
|----------|-------|-------------|
| **Positional** | Blue (#1f77b4) | Strategic placement based on spatial considerations |
| **Blocking** | Orange (#ff7f0e) | Defensive reasoning to prevent opponent success |
| **Opponent Modeling** | Green (#2ca02c) | Reasoning about opponent's intentions and strategies |
| **Winning Logic** | Red (#d62728) | Direct reasoning about achieving victory conditions |
| **Heuristic** | Brown (#8c564b) | General strategic principles without specific justification |
| **Rule-Based** | Light Pink (#ff9896) | Reasoning based on explicit strategies or formal principles |
| **Random/Unjustified** | Light Purple (#c5b0d5) | Moves made without clear strategic reasoning |

### Strategic/Matrix Game Categories
These categories apply primarily to games like Prisoner's Dilemma, Rock Paper Scissors, etc.

| Category | Color | Description |
|----------|-------|-------------|
| **Strategic Reasoning** | **Bright Magenta (#e377c2)** | **Formal game-theoretic analysis and optimization** |
| **Cooperative Strategy** | Olive (#bcbd22) | Seeking mutual benefit and long-term collaboration |
| **Competitive Strategy** | Red-Orange (#ff4500) | Focusing on individual advantage over opponents |
| **Risk Management** | Purple (#9467bd) | Minimizing worst-case scenarios and managing uncertainty |

### Special Categories

| Category | Color | Description |
|----------|-------|-------------|
| **Uncategorized** | **Gray (#7f7f7f)** | **Reasoning that doesn't match any specific pattern** |

## Visual Distinction Features

### High-Contrast Color Design
The color scheme is designed for maximum visual distinction between all reasoning categories:

- **Uncategorized vs Strategic Reasoning**: Gray (#7f7f7f) vs Bright Magenta (#e377c2)
- **Competitive vs Cooperative Strategy**: Red-Orange (#ff4500) vs Olive (#bcbd22)
- **Blocking vs Positional**: Orange (#ff7f0e) vs Blue (#1f77b4)

### Consistent Visual Coding
- **All chart types** use the same color scheme for consistency
- **Black edge outlines** on bars for additional clarity and definition
- **High contrast** colors chosen for accessibility and distinction

### Accessibility Features

1. **Color Blindness Support**:
   - High contrast between similar categories
   - Different hue families used for category groups
   - Primary distinctions avoid problematic color combinations

2. **Print-Friendly Design**:
   - Colors remain distinct in grayscale conversion
   - Strong contrast ratios for readability
   - Clean black outlines maintain definition

3. **Consistent Usage**:
   - Same colors used across all plot types (bar charts, pie charts, stacked charts)
   - Uniform application ensures easy cross-reference between visualizations

## Usage in Research

### Interpreting Plots
- **Bar Charts**: Distinct colors with black edge outlines for clarity
- **Pie Charts**: Color-coded segments for category identification
- **Stacked Charts**: Consistent colors for cross-chart comparison
- **Evolution Plots**: Color trends show reasoning changes over game turns

### Key Research Insights
- **Strategic Reasoning** (bright magenta) vs **Uncategorized** (gray) are highly distinct
- Strategic game categories use warmer colors (reds, oranges, purples)
- Spatial game categories use cooler colors (blues, greens)
- Consistent color application enables easy cross-reference between chart types

## Implementation Notes

- Colors defined in `analysis/reasoning_analysis.py` in `REASONING_COLOR_MAP`
- Function `get_reasoning_colors()` provides consistent color mapping
- All plotting functions use the same color retrieval system
- Black edge outlines (`edgecolor='black', linewidth=0.5`) add definition

This unified color scheme resolves the previous issue where "Strategic Reasoning" and "Uncategorized" had similar gray colors, and ensures **complete consistency** across all chart types in your research visualizations.
