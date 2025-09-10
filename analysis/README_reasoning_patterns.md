# Reasoning Pattern Classification System

This document explains the rationale behind each reasoning pattern category used in the Game Reasoning Arena analysis pipeline. The classification system is designed to identify and categorize different types of strategic thinking patterns that emerge from LLM agents when playing various games.

## Overview

The reasoning classification system uses regex pattern matching to identify key phrases and concepts in the reasoning traces provided by LLM agents. Each reasoning trace is scored against multiple categories, and the category with the highest score is assigned to that reasoning instance.

## Classification Categories

### Spatial/Board Game Categories

These categories are primarily designed for spatial games like Tic-Tac-Toe, Connect Four, Hex, etc.

#### **Positional**
*Strategic placement based on spatial considerations*

**Patterns:** "center column", "center square", "center position", "center cell", "corner", "edge", "position"

**Rationale:** In many board games, certain positions hold inherent strategic value. Center positions often provide maximum flexibility and control, while corners and edges may offer defensive advantages or specific strategic opportunities. This category captures reasoning that explicitly considers the spatial properties of move choices.

**Example:** "I'll place my piece in the center column because it gives me the most options for future moves."

#### **Blocking**
*Defensive reasoning focused on preventing opponent success*

**Patterns:** "block", "blocking", "prevent", "stop opponent", "avoid opponent", "counter", "defense", "defensive"

**Rationale:** Defensive play is a fundamental aspect of competitive games. This category identifies reasoning that prioritizes preventing the opponent from achieving their goals rather than directly pursuing one's own winning strategy. It represents a reactive strategic approach.

**Example:** "I need to block their potential winning move by placing my piece here."

#### **Opponent Modeling**
*Reasoning about opponent's intentions, strategies, or likely moves*

**Patterns:** "opponent", "they are trying", "their strategy", "their move", "other player", "expected behavior", "predict", "anticipate", "other player's choice"

**Rationale:** Advanced strategic thinking often involves modeling the opponent's mental state and predicting their behavior. This category captures reasoning that explicitly considers what the opponent might do, their strategic intentions, or attempts to predict their next moves.

**Example:** "Based on their previous moves, I think they're trying to control the diagonal, so I should anticipate their next move."

#### **Winning Logic**
*Direct reasoning about achieving victory conditions*

**Patterns:** "win", "winning move", "connect", "fork", "threat", "chance of winning", "victory", "win condition"

**Rationale:** This category identifies proactive strategic reasoning focused on achieving the game's victory conditions. It represents forward-thinking that directly aims at winning rather than just playing defensively or following general heuristics.

**Example:** "This move creates a fork, giving me two ways to win on my next turn."

#### **Heuristic**
*General strategic principles without specific game-theoretic justification*

**Patterns:** "best move", "most likely", "advantageous", "better chance", "optimal", "effective"

**Rationale:** Players often use general strategic intuitions or rules of thumb that aren't rigorously justified but seem reasonable. This category captures reasoning based on general strategic principles or intuitive judgments about move quality.

**Example:** "This seems like the most advantageous position based on general strategy principles."

#### **Rule-Based**
*Reasoning based on explicit strategies, theories, or formal principles*

**Patterns:** "according to", "rule", "strategy", "rational choice", "game theory", "according to theory", "standard strategy"

**Rationale:** Some reasoning explicitly references established strategies, theoretical principles, or formal rules. This category identifies reasoning that grounds move choices in established strategic doctrine or theoretical frameworks.

**Example:** "According to standard opening theory, controlling the center is the optimal strategy."

#### **Random/Unjustified**
*Moves made without clear strategic reasoning*

**Patterns:** "random", "guess", "no particular reason", "arbitrarily"

**Rationale:** Not all moves are made with clear strategic intent. This category identifies instances where players explicitly acknowledge making arbitrary choices or admit to lacking a clear strategic rationale.

**Example:** "I'll just pick this move randomly since I can't see a clear advantage."

### Strategic/Matrix Game Categories

These categories are designed for matrix games and strategic scenarios like Prisoner's Dilemma, matching games, etc.

#### **Strategic Reasoning**
*Formal game-theoretic analysis and optimization*

**Patterns:** "dominant strategy", "payoff", "maximize", "optimal choice", "nash equilibrium", "equilibrium", "game theory", "strategic", "rational", "analysis", "calculate"

**Rationale:** This category captures reasoning that employs formal game-theoretic concepts and analytical thinking. It identifies instances where players engage in systematic analysis of payoffs, equilibria, or strategic dominance. This represents the most sophisticated level of strategic thinking.

**Example:** "Cooperation is the dominant strategy because it maximizes the expected payoff regardless of what the opponent chooses."

#### **Risk Management**
*Minimizing worst-case scenarios and managing uncertainty*

**Patterns:** "worst case", "minimize risk", "safest choice", "avoid", "protect against", "safe", "minimize", "risk", "cautious", "conservative"

**Rationale:** In situations with uncertainty or potential negative outcomes, players often adopt risk-averse strategies. This category identifies reasoning that prioritizes security and worst-case scenario avoidance over potential gains. It represents a conservative approach to strategic decision-making.

**Example:** "I'll choose the safest option to protect against the worst-case scenario, even if it means giving up potential higher rewards."

#### **Cooperative Strategy**
*Seeking mutual benefit and long-term collaboration*

**Patterns:** "mutual benefit", "both benefit", "cooperate", "cooperation", "establish trust", "long term", "long-term", "trust", "mutually beneficial", "work together", "shared benefit"

**Rationale:** In repeated interactions or situations where mutual benefit is possible, players may adopt cooperative strategies. This category identifies reasoning that prioritizes collective outcomes, trust-building, and long-term relationship considerations over immediate individual gains.

**Example:** "If we both cooperate, we'll both benefit more in the long run than if we try to exploit each other."

#### **Competitive Strategy**
*Focusing on individual advantage and outperforming opponents*

**Patterns:** "advantage", "exploit", "outperform", "beat opponent", "individual gain", "personal gain", "self-interest", "defect", "competitive", "maximize my", "highest.*reward"

**Rationale:** In competitive scenarios, players may prioritize individual success over collective outcomes. This category identifies reasoning that focuses on gaining advantages over opponents, maximizing personal rewards, or exploiting opportunities for individual benefit.

**Example:** "I'll defect because it gives me the highest individual reward, regardless of what happens to my opponent."



## Implementation Details

### Scoring System
Each reasoning trace is scored against all categories using the following method:
- Each pattern match contributes 1 point to the category score
- The category with the highest total score is assigned to the reasoning trace
- In case of ties, the first category (in definition order) is selected
- If no patterns match, the reasoning is classified as "Uncategorized"

### Pattern Matching
- All patterns use case-insensitive regex matching
- Word boundaries (`\b`) are used to ensure precise matching
- Patterns are designed to capture both exact phrases and conceptual variations



## Usage

The classification system is implemented in `reasoning_analysis.py` and used by:
- `run_full_analysis.py` - Main analysis pipeline
- `post_game_processing.py` - Individual game analysis
- Various plotting and statistical analysis scripts

To modify the classification rules, edit the `REASONING_RULES` dictionary in `reasoning_analysis.py`.
