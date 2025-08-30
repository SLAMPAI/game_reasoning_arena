#!/usr/bin/env python3
"""
Performance Tables Module

This module generates comprehensive performance tables showing win rates,
standard deviations, and other key metrics for LLM agents across games.
Similar to the language-only performance tables in research papers.

Reward Normalization:
All reward values are normalized to ensure fair comparison across games with
different reward structures. Two normalization modes are supported:

1. Per-game normalization: Rewards within each game are normalized to [-1, +1]
   where -1 represents the worst outcome observed in that game and +1 the best.
   This is used for per-game performance analysis.

2. Cross-game normalization: All rewards across all games are normalized to
   the same [-1, +1] scale for overall performance comparison.

Formula: normalized = 2 * (reward - min_reward) / (max_reward - min_reward) - 1

This ensures that games with different reward ranges (e.g., +5/-5 vs +1/-1)
contribute equally to aggregate performance metrics.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import json
from utils import display_game_name


def clean_model_name(model_name: str) -> str:
    """
    Clean up long model names to display only the essential model name.
    This is a copy of the function from ui/utils.py for use in analysis.
    """
    # Handle NaN/None values
    if pd.isna(model_name) or not model_name or model_name == "Unknown":
        return "Unknown"

    # Convert to string if it's not already
    model_name = str(model_name)

    # Handle special cases first
    if model_name == "None" or model_name.lower() == "random":
        return "Random Bot"

    # Handle random_None specifically
    if model_name == "random_None":
        return "Random Bot"

    # Remove leading "llm_" prefix if present (common in database)
    if model_name.startswith("llm_"):
        model_name = model_name[4:]

    # Remove leading "human_" prefix if present (filtered out already)
    if model_name.startswith("human_"):
        model_name = model_name[6:]

    # GPT models - keep the GPT part
    if "gpt" in model_name.lower():
        # Extract GPT model variants
        if "gpt_3.5" in model_name.lower() or "gpt-3.5" in model_name.lower():
            return "GPT-3.5-turbo"
        elif "gpt_4" in model_name.lower() or "gpt-4" in model_name.lower():
            if "turbo" in model_name.lower():
                return "GPT-4-turbo"
            elif "mini" in model_name.lower():
                return "GPT-4-mini"
            else:
                return "GPT-4"
        elif "gpt_5" in model_name.lower() or "gpt-5" in model_name.lower():
            if "mini" in model_name.lower():
                return "GPT-5-mini"
            else:
                return "GPT-5"
        elif "gpt2" in model_name.lower() or "gpt-2" in model_name.lower():
            return "GPT-2"
        elif "distilgpt2" in model_name.lower():
            return "DistilGPT-2"
        elif "gpt-neo" in model_name.lower():
            return "GPT-Neo-125M"

    # For litellm models, extract everything after the last slash
    if "litellm_" in model_name and "/" in model_name:
        # Split by "/" and take the last part
        model_part = model_name.split("/")[-1]
        # Clean up underscores and make it more readable
        cleaned = model_part.replace("_", "-")
        return cleaned

    # For vllm models, extract the model name part
    if model_name.startswith("vllm_"):
        # Remove vllm_ prefix
        model_part = model_name[5:]
        # Clean up underscores
        cleaned = model_part.replace("_", "-")
        return cleaned

    # For litellm models without slashes (from database storage)
    if model_name.startswith("litellm_"):
        parts = model_name.split("_")

        # Handle Fireworks AI pattern:
        # litellm_fireworks_ai_accounts_fireworks_models_*
        if (
            "fireworks" in model_name
            and "accounts" in model_name
            and "models" in model_name
        ):
            try:
                models_idx = parts.index("models")
                model_parts = parts[models_idx + 1:]
                return "-".join(model_parts)
            except (ValueError, IndexError):
                pass

        # Handle GROQ pattern: litellm_groq_*
        if "groq" in model_name:
            try:
                groq_idx = parts.index("groq")
                model_parts = parts[groq_idx + 1:]
                return "-".join(model_parts)
            except (ValueError, IndexError):
                pass

        # Handle Together AI pattern: litellm_together_ai_*
        if "together" in model_name and "ai" in model_name:
            try:
                ai_idx = parts.index("ai")
                model_parts = parts[ai_idx + 1:]
                # Clean up common prefixes
                if model_parts and model_parts[0] == "meta":
                    model_parts = model_parts[1:]
                if model_parts and model_parts[0] == "llama":
                    model_parts = model_parts[1:]
                return "-".join(model_parts)
            except (ValueError, IndexError):
                pass

        # General fallback: take parts after litellm
        return "-".join(parts[1:])

    # For standard model names, clean up underscores
    return model_name.replace("_", "-")


class PerformanceTableGenerator:
    """Generates performance tables from game data."""

    def __init__(self, csv_path: str):
        """
        Initialize with merged CSV data.

        Args:
            csv_path: Path to the merged CSV file containing game data
        """
        self.df = pd.read_csv(csv_path)
        self.df['clean_model_name'] = self.df['agent_model'].apply(
            clean_model_name)

        # Ensure we have the required columns
        required_cols = ['game_name', 'agent_name', 'episode',
                         'reward', 'status']
        missing_cols = [col for col in required_cols
                        if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def compute_win_rates(self, by_game: bool = True) -> pd.DataFrame:
        """
        Compute win rates for each agent, optionally broken down by game.

        This method calculates performance metrics including win rates and
        normalized average rewards. Rewards are normalized to ensure fair
        comparison across games with different reward structures.

        Normalization Methodology:
        - Per-game normalization (by_game=True): Rewards are normalized within
          each game separately to a [-1, +1] scale, where -1 represents the
          worst observed outcome in that game and +1 represents the best.
        - Cross-game normalization (by_game=False): Rewards are normalized
          across all games to the same [-1, +1] scale for overall comparison.
        - Formula: normalized = 2 * (reward - min_reward) /
                   (max_reward - min_reward) - 1

        This ensures that a game with +5/-5 rewards and a game with +1/-1
        rewards contribute equally to overall performance assessments.

        Args:
            by_game: If True, compute win rates per game with per-game
                    normalization. If False, compute overall metrics with
                    cross-game normalization.

        Returns:
            DataFrame with agent performance metrics including:
            - Games played, games won, win rates with confidence intervals
            - Normalized average rewards with standard deviations
        """
        # Filter to only completed games
        completed_games = self.df[self.df['status'].isin(['terminated'])]

        # Group by agent and optionally by game
        if by_game:
            group_cols = ['game_name', 'clean_model_name', 'episode']
        else:
            group_cols = ['clean_model_name', 'episode']

        # Get one row per episode (game) to calculate episode-level outcomes
        episode_data = completed_games.groupby(group_cols).agg({
            'reward': 'mean',  # Average reward per episode
        }).reset_index()

        # Normalize rewards to -1/+1 scale per game to enable fair comparison
        if by_game:
            # Normalize within each game separately
            for game in episode_data['game_name'].unique():
                game_mask = episode_data['game_name'] == game
                game_rewards = episode_data.loc[game_mask, 'reward']

                # Skip normalization if all rewards are the same
                if game_rewards.nunique() <= 1:
                    continue

                # Normalize to [-1, 1] range
                min_reward = game_rewards.min()
                max_reward = game_rewards.max()
                if max_reward != min_reward:
                    normalized = (2 * (game_rewards - min_reward) /
                                  (max_reward - min_reward) - 1)
                    episode_data.loc[game_mask, 'reward'] = normalized
        else:
            # Normalize across all games for overall comparison
            all_rewards = episode_data['reward']
            if all_rewards.nunique() > 1:
                min_reward = all_rewards.min()
                max_reward = all_rewards.max()
                if max_reward != min_reward:
                    normalized_rewards = (2 * (all_rewards - min_reward) /
                                         (max_reward - min_reward) - 1)
                    episode_data['reward'] = normalized_rewards

        # Calculate win rate (assuming reward > 0 means win)
        episode_data['won'] = episode_data['reward'] > 0

        if by_game:
            # Group by game and model to calculate statistics
            performance = episode_data.groupby(
                ['game_name', 'clean_model_name']).agg({
                    'won': ['count', 'sum', 'mean', 'std'],
                    'reward': ['mean', 'std']
                }).reset_index()
        else:
            # Group by model only
            performance = episode_data.groupby(['clean_model_name']).agg({
                'won': ['count', 'sum', 'mean', 'std'],
                'reward': ['mean', 'std']
            }).reset_index()

        # Flatten column names
        performance.columns = [
            '_'.join(col).strip('_') if col[1] else col[0]
            for col in performance.columns
        ]

        # Rename columns for clarity
        rename_map = {
            'won_count': 'games_played',
            'won_sum': 'games_won',
            'won_mean': 'win_rate',
            'won_std': 'win_rate_std',
            'reward_mean': 'avg_reward',
            'reward_std': 'reward_std'
        }
        performance = performance.rename(columns=rename_map)

        # Convert win rate to percentage
        performance['win_rate'] = performance['win_rate'] * 100
        performance['win_rate_std'] = performance['win_rate_std'] * 100

        # Fill NaN standard deviations with 0 (happens when only 1 game played)
        performance['win_rate_std'] = performance['win_rate_std'].fillna(0)
        performance['reward_std'] = performance['reward_std'].fillna(0)

        return performance

    def generate_overall_performance_table(self,
                                         output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate an overall performance table across all games.

        Args:
            output_path: If provided, save the table to this path

        Returns:
            DataFrame with overall performance metrics
        """
        performance = self.compute_win_rates(by_game=False)

        # Sort by win rate (descending)
        performance = performance.sort_values('win_rate', ascending=False)

        # Format the table for display
        display_table = performance.copy()
        display_table['Win Rate (%)'] = display_table.apply(
            lambda row: f"{row['win_rate']:.2f} ± {row['win_rate_std']:.2f}",
            axis=1
        )
        display_table['Avg Reward'] = display_table.apply(
            lambda row: f"{row['avg_reward']:.3f} ± {row['reward_std']:.3f}",
            axis=1
        )

        # Select and rename columns for final table
        final_table = display_table[[
            'clean_model_name', 'games_played', 'games_won',
            'Win Rate (%)', 'Avg Reward'
        ]].copy()
        final_table = final_table.rename(columns={
            'clean_model_name': 'Model',
            'games_played': 'Games Played',
            'games_won': 'Games Won'
        })

        if output_path:
            final_table.to_csv(output_path, index=False)
            print(f"Overall performance table saved to: {output_path}")

        return final_table

    def generate_per_game_performance_table(self,
                                          output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a performance table broken down by game.

        Args:
            output_path: If provided, save the table to this path

        Returns:
            DataFrame with per-game performance metrics
        """
        performance = self.compute_win_rates(by_game=True)

        # Sort by game name and then by win rate
        performance = performance.sort_values(['game_name', 'win_rate'],
                                            ascending=[True, False])

        # Format the table for display
        display_table = performance.copy()
        display_table['game_name'] = display_table['game_name'].apply(display_game_name)
        display_table['Win Rate (%)'] = display_table.apply(
            lambda row: f"{row['win_rate']:.2f} ± {row['win_rate_std']:.2f}",
            axis=1
        )
        display_table['Avg Reward'] = display_table.apply(
            lambda row: f"{row['avg_reward']:.3f} ± {row['reward_std']:.3f}",
            axis=1
        )

        # Select and rename columns for final table
        final_table = display_table[[
            'game_name', 'clean_model_name', 'games_played', 'games_won',
            'Win Rate (%)', 'Avg Reward'
        ]].copy()
        final_table = final_table.rename(columns={
            'game_name': 'Game',
            'clean_model_name': 'Model',
            'games_played': 'Games Played',
            'games_won': 'Games Won'
        })

        if output_path:
            final_table.to_csv(output_path, index=False)
            print(f"Per-game performance table saved to: {output_path}")

        return final_table

    def generate_pivot_performance_table(self,
                                       metric: str = 'win_rate',
                                       output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a pivot table with models as rows and games as columns.

        Args:
            metric: Metric to display ('win_rate', 'avg_reward', 'games_played')
            output_path: If provided, save the table to this path

        Returns:
            DataFrame pivot table
        """
        performance = self.compute_win_rates(by_game=True)

        if metric not in performance.columns:
            raise ValueError(f"Metric '{metric}' not found in performance data")

        # Create pivot table
        pivot_table = performance.pivot_table(
            index='clean_model_name',
            columns='game_name',
            values=metric,
            fill_value=0
        )

        # Clean up game names for display
        pivot_table.columns = [display_game_name(col) for col in pivot_table.columns]

        # Sort by overall performance (row means)
        pivot_table['Overall'] = pivot_table.mean(axis=1)
        pivot_table = pivot_table.sort_values('Overall', ascending=False)

        # Format values based on metric type
        if metric == 'win_rate':
            # Format as percentage with 1 decimal place
            pivot_table = pivot_table.round(1)
        elif metric == 'avg_reward':
            # Format with 3 decimal places
            pivot_table = pivot_table.round(3)
        else:
            # Integer metrics (games_played, etc.)
            pivot_table = pivot_table.round(0).astype(int)

        if output_path:
            pivot_table.to_csv(output_path)
            print(f"Pivot performance table saved to: {output_path}")

        return pivot_table

    def generate_latex_table(self,
                           table_type: str = 'overall',
                           output_path: Optional[str] = None) -> str:
        """
        Generate LaTeX table code for publication.

        Args:
            table_type: Type of table ('overall', 'per_game', 'pivot')
            output_path: If provided, save LaTeX code to this file

        Returns:
            LaTeX table code as string
        """
        if table_type == 'overall':
            df = self.generate_overall_performance_table()
            caption = "Overall Model Performance Across All Games"

            latex_code = "\\begin{table}[htbp]\n"
            latex_code += "\\centering\n"
            latex_code += f"\\caption{{{caption}}}\n"
            latex_code += "\\begin{tabular}{lcccc}\n"
            latex_code += "\\toprule\n"
            latex_code += "Model & Games Played & Games Won & Win Rate (\\%) & Avg Reward \\\\\n"
            latex_code += "\\midrule\n"

            for _, row in df.iterrows():
                latex_code += f"{row['Model']} & {row['Games Played']} & {row['Games Won']} & {row['Win Rate (%)']} & {row['Avg Reward']} \\\\\n"

            latex_code += "\\bottomrule\n"
            latex_code += "\\end{tabular}\n"
            latex_code += "\\end{table}\n"

        elif table_type == 'pivot':
            pivot_df = self.generate_pivot_performance_table(metric='win_rate')
            caption = "Win Rate (\\%) by Model and Game"

            games = [col for col in pivot_df.columns if col != 'Overall']
            col_spec = 'l' + 'c' * len(games) + 'c'  # Left-align model names, center-align games

            latex_code = "\\begin{table}[htbp]\n"
            latex_code += "\\centering\n"
            latex_code += f"\\caption{{{caption}}}\n"
            latex_code += f"\\begin{{tabular}}{{{col_spec}}}\n"
            latex_code += "\\toprule\n"

            # Header row
            header = "Model & " + " & ".join(games) + " & Overall \\\\\n"
            latex_code += header
            latex_code += "\\midrule\n"

            # Data rows
            for model, row in pivot_df.iterrows():
                values = [f"{row[game]:.1f}" for game in games]
                overall = f"{row['Overall']:.1f}"
                latex_code += f"{model} & " + " & ".join(values) + f" & {overall} \\\\\n"

            latex_code += "\\bottomrule\n"
            latex_code += "\\end{tabular}\n"
            latex_code += "\\end{table}\n"

        else:
            raise ValueError(f"Unsupported table type: {table_type}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_code)
            print(f"LaTeX table saved to: {output_path}")

        return latex_code

    def generate_all_performance_tables(self, results_dir: str = "results"):
        """
        Generate all performance tables and save them to results/tables directory.

        Args:
            results_dir: Base results directory where tables subdirectory will be created
        """
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)

        # Create dedicated tables subdirectory under results
        tables_path = results_path / "tables"
        tables_path.mkdir(exist_ok=True)

        print("Generating performance tables...")

        # Overall performance table
        overall_table = self.generate_overall_performance_table(
            tables_path / "overall_performance_table.csv"
        )

        # Per-game performance table
        per_game_table = self.generate_per_game_performance_table(
            tables_path / "per_game_performance_table.csv"
        )

        # Pivot tables
        win_rate_pivot = self.generate_pivot_performance_table(
            metric='win_rate',
            output_path=tables_path / "win_rate_pivot_table.csv"
        )

        reward_pivot = self.generate_pivot_performance_table(
            metric='avg_reward',
            output_path=tables_path / "reward_pivot_table.csv"
        )

        games_pivot = self.generate_pivot_performance_table(
            metric='games_played',
            output_path=tables_path / "games_played_pivot_table.csv"
        )

        # LaTeX tables
        self.generate_latex_table(
            table_type='overall',
            output_path=tables_path / "overall_performance_table.tex"
        )

        self.generate_latex_table(
            table_type='pivot',
            output_path=tables_path / "win_rate_pivot_table.tex"
        )

        # Generate summary report
        summary = {
            "total_models": len(overall_table),
            "total_games": len(self.df['game_name'].unique()),
            "top_performer": overall_table.iloc[0]['Model'],
            "top_win_rate": overall_table.iloc[0]['Win Rate (%)'],
            "tables_generated": [
                "tables/overall_performance_table.csv",
                "tables/per_game_performance_table.csv",
                "tables/win_rate_pivot_table.csv",
                "tables/reward_pivot_table.csv",
                "tables/games_played_pivot_table.csv",
                "tables/overall_performance_table.tex",
                "tables/win_rate_pivot_table.tex"
            ]
        }

        with open(tables_path / "performance_tables_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Generated {len(summary['tables_generated'])} "
              "performance tables")
        print(f"📊 Top performer: {summary['top_performer']} "
              f"({summary['top_win_rate']})")
        print(f"📁 All tables saved to: {tables_path}")

        return summary


def main():
    """Main function for testing the performance table generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate performance tables from game data")
    parser.add_argument("--csv", required=True, help="Path to merged CSV file")
    parser.add_argument("--results-dir", default="results",
                        help="Results directory for tables")

    args = parser.parse_args()

    # Generate all performance tables
    generator = PerformanceTableGenerator(args.csv)
    summary = generator.generate_all_performance_tables(args.results_dir)

    # Display overall performance table
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE TABLE")
    print("="*80)
    overall_table = generator.generate_overall_performance_table()
    print(overall_table.to_string(index=False))


if __name__ == "__main__":
    main()
