#!/usr/bin/env python3
"""
Generate Reasoning Analysis Plots

This script generates comprehensive reasoning analysis visualizations for LLM
game-playing data, including bar charts, pie charts, stacked charts, and
evolution plots showing how reasoning patterns change over time.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add analysis directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reasoning_analysis import LLMReasoningAnalyzer  # noqa: E402


def plot_reasoning_bar_chart(
    reasoning_percentages: Dict[str, float],
    model_name: str,
    output_path: str = "reasoning_bar_chart.png",
    games_list: Optional[List[str]] = None
) -> None:
    """
    Plots a horizontal bar chart for reasoning type percentages.

    Args:
        reasoning_percentages: Dictionary mapping reasoning types to
            percentages.
        model_name: Name of the model for the title.
        output_path: File path to save the figure.
        games_list: Optional list of games included in the analysis.
    """
    df = pd.DataFrame(
        list(reasoning_percentages.items()),
        columns=["reasoning_type", "percentage"]
    )
    df = df.sort_values("percentage", ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df["reasoning_type"], df["percentage"])

    # Generate title
    title = f"Reasoning Type Distribution - {model_name}"
    if games_list:
        title += f" ({', '.join(games_list)})"

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Percentage (%)", fontsize=14)
    plt.ylabel("Reasoning Type", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add percentage labels on bars
    for bar, percentage in zip(bars, df["percentage"]):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{percentage:.1f}%', ha='left', va='center', fontsize=11)


def plot_reasoning_pie_chart(
    reasoning_percentages: Dict[str, float],
    model_name: str,
    output_path: str = "reasoning_pie_chart.png"
) -> None:
    """
    Plots a pie chart showing the reasoning type distribution.

    Args:
        reasoning_percentages: Dictionary mapping reasoning types to
            percentages.
        model_name: Name of the model for the title.
        output_path: File path to save the figure.
    """
    series = pd.Series(reasoning_percentages)

    plt.figure(figsize=(8, 8))
    _, _, autotexts = plt.pie(
        series.values,
        labels=series.index,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    plt.title(f"Reasoning Type Distribution - {model_name}",
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_stacked_bar_chart(
    game_reasoning_percentages: Dict[str, Dict[str, float]],
    model_name: str,
    output_path: str = "reasoning_stacked_bar.png"
) -> None:
    """
    Plots a stacked bar chart showing reasoning type percentages per game.

    Args:
        game_reasoning_percentages: Nested dictionary mapping
            game_name -> reasoning_type -> percentage.
        model_name: Name of the model for the title.
        output_path: File path to save the figure.
    """
    df = pd.DataFrame(game_reasoning_percentages).T.fillna(0)

    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', stacked=True, colormap="tab20c")

    plt.ylabel("Percentage (%)", fontsize=14)
    plt.xlabel("Game", fontsize=14)
    plt.title(f"Reasoning Types per Game - {model_name}",
              fontsize=16, fontweight='bold')
    plt.legend(title="Reasoning Type", bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=12, title_fontsize=13)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Add percentage labels on stacked bars
    for i, game in enumerate(df.index):
        cumulative = 0
        for reasoning_type in df.columns:
            percentage = df.loc[game, reasoning_type]
            if percentage > 5:  # Only show label if segment is large enough
                plt.text(i, cumulative + percentage/2, f'{percentage:.0f}%',
                         ha='center', va='center', fontweight='bold',
                         fontsize=10)
            cumulative += percentage

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reasoning_evolution_over_turns(
    reasoning_per_turn: Dict[int, Dict[str, int]],
    model_name: str,
    game_name: str,
    output_path: str = "reasoning_evolution.png"
) -> None:
    """
    Plots how reasoning type distribution changes over turns for a
    model-game pair using a stacked area chart.

    Args:
        reasoning_per_turn: Dict mapping turn_number -> reasoning_type ->
            count.
        model_name: Name of the model for the title.
        game_name: Name of the game for the title.
        output_path: File path to save the figure.
    """
    # Convert nested dict into DataFrame
    df = pd.DataFrame.from_dict(reasoning_per_turn, orient="index").fillna(0)
    df = df.sort_index()

    # Normalize to proportions per turn
    df_prop = df.div(df.sum(axis=1), axis=0).fillna(0)

    # If we have sparse data, skip this plot
    if len(df_prop) < 2:
        print(f"Skipping evolution plot for {model_name}-{game_name}: "
              f"insufficient turn data")
        return

    # Create single figure with stacked bar chart
    plt.figure(figsize=(12, 8))

    # Create stacked bar chart where each bar represents a turn
    # and each segment represents a reasoning category
    bottom = np.zeros(len(df_prop))
    colors = plt.cm.Set3(np.linspace(0, 1, len(df_prop.columns)))

    for i, reasoning_type in enumerate(df_prop.columns):
        plt.bar(df_prop.index, df_prop[reasoning_type],
                bottom=bottom, label=reasoning_type,
                color=colors[i], alpha=0.8, edgecolor='black',
                linewidth=0.5)
        bottom += df_prop[reasoning_type]

    plt.title(f"{model_name} - {game_name}: Reasoning Category Evolution",
              fontsize=16, fontweight='bold')
    plt.xlabel("Game Turn", fontsize=14)
    plt.ylabel("Proportion of Reasoning Types", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5),
               title="Reasoning Type", fontsize=12, title_fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add sample size annotations above each bar
    for turn in df_prop.index:
        total_count = df.loc[turn].sum()
        plt.annotate(f'n={int(total_count)}',
                     xy=(turn, 1.02),
                     ha='center', va='bottom',
                     fontsize=11, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reasoning_evolution_heatmap(
    reasoning_per_turn: Dict[int, Dict[str, int]],
    model_name: str,
    game_name: str,
    output_path: str = "reasoning_evolution_heatmap.png"
) -> None:
    """
    Creates a heatmap showing reasoning evolution over turns.
    Better for sparse data or when you want to see patterns clearly.
    """
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(reasoning_per_turn, orient="index").fillna(0)
    df = df.sort_index()

    # If insufficient data, skip
    if len(df) < 2 or df.empty:
        print(f"Skipping heatmap for {model_name}-{game_name}: "
              f"insufficient data")
        return

    # Normalize to proportions
    df_prop = df.div(df.sum(axis=1), axis=0).fillna(0)

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_prop.T, annot=True, cmap='RdYlBu_r',
                cbar_kws={'label': 'Proportion'}, fmt='.2f',
                annot_kws={'fontsize': 12})

    plt.title(f"{model_name} - {game_name}: Reasoning Evolution Heatmap",
              fontsize=16, fontweight='bold')
    plt.xlabel("Game Turn", fontsize=14)
    plt.ylabel("Reasoning Type", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add sample size annotations
    for i, turn in enumerate(df.index):
        total = df.loc[turn].sum()
        plt.text(i + 0.5, len(df.columns) + 0.1, f'n={int(total)}',
                 ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


class ReasoningPlotGenerator:
    """Generates comprehensive reasoning analysis plots for LLM game data."""

    def __init__(self, csv_path: str):
        """Initialize with the reasoning analysis data."""
        self.analyzer = LLMReasoningAnalyzer(csv_path)
        self.analyzer.categorize_reasoning()
        self.df = self.analyzer.df

    def extract_all_evolution_data(self):
        """
        Extract evolution data for all model-game pairs.

        Returns:
            Dict[model][game][turn] -> reasoning_type -> count
        """
        non_random_df = self.df[
            ~self.df['agent_name'].str.startswith("random")
        ]

        all_data = {}

        for agent_name in non_random_df['agent_name'].unique():
            agent_df = non_random_df[non_random_df['agent_name'] == agent_name]
            model_name = agent_name
            all_data[model_name] = {}

            for game in agent_df['game_name'].unique():
                game_df = agent_df[agent_df['game_name'] == game]

                # Only include if there are multiple turns
                if len(game_df['turn'].unique()) > 1:
                    reasoning_per_turn = {}
                    for turn in sorted(game_df['turn'].unique()):
                        turn_df = game_df[game_df['turn'] == turn]
                        turn_counts = turn_df['reasoning_type'].value_counts()
                        reasoning_per_turn[turn] = turn_counts.to_dict()

                    if reasoning_per_turn:  # Only add if we have turn data
                        all_data[model_name][game] = reasoning_per_turn

        return all_data

    def plot_all_reasoning_evolutions(self, output_dir: str = "plots"):
        """
        Plots reasoning evolution over time for all model-game pairs.

        Args:
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        all_data = self.extract_all_evolution_data()
        generated_files = []

        for model_name, model_games in all_data.items():
            for game_name, reasoning_per_turn in model_games.items():
                # Clean the model and game names for filename
                clean_model = model_name.replace(" ", "_").replace(".", "_")
                clean_game = game_name.replace(" ", "_").replace(".", "_")
                filename = f"evolution_{clean_model}_{clean_game}.png".lower()
                output_path = Path(output_dir) / filename

                plot_reasoning_evolution_over_turns(
                    reasoning_per_turn=reasoning_per_turn,
                    model_name=model_name,
                    game_name=game_name,
                    output_path=str(output_path)
                )
                generated_files.append(str(output_path))

        return generated_files

    def plot_all_reasoning_evolutions_enhanced(self,
                                               output_dir: str = "plots"):
        """
        Plots both standard evolution plots and heatmaps for all
        model-game pairs.

        Args:
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        all_data = self.extract_all_evolution_data()
        generated_files = []

        for model_name, model_games in all_data.items():
            for game_name, reasoning_per_turn in model_games.items():
                # Clean the model and game names for filename
                clean_model = model_name.replace(" ", "_").replace(".", "_")
                clean_game = game_name.replace(" ", "_").replace(".", "_")

                # Generate standard evolution plot
                filename = f"evolution_{clean_model}_{clean_game}.png".lower()
                output_path = Path(output_dir) / filename

                plot_reasoning_evolution_over_turns(
                    reasoning_per_turn=reasoning_per_turn,
                    model_name=model_name,
                    game_name=game_name,
                    output_path=str(output_path)
                )
                generated_files.append(str(output_path))

                # Generate heatmap version
                heatmap_filename = (f"evolution_heatmap_{clean_model}_"
                                    f"{clean_game}.png").lower()
                heatmap_path = Path(output_dir) / heatmap_filename

                plot_reasoning_evolution_heatmap(
                    reasoning_per_turn=reasoning_per_turn,
                    model_name=model_name,
                    game_name=game_name,
                    output_path=str(heatmap_path)
                )
                generated_files.append(str(heatmap_path))

        return generated_files

    def analyze_reasoning_evolution_patterns(self) -> Dict:
        """
        Analyze and summarize reasoning evolution patterns across models.
        Returns insights about how reasoning changes over turns.
        """
        all_data = self.extract_all_evolution_data()
        analysis = {
            'models': {},
            'summary': {
                'total_models': len(all_data),
                'evolution_patterns': []
            }
        }

        for model_name, games in all_data.items():
            model_analysis = {'games': {}}

            for game_name, turn_data in games.items():
                turns = sorted(turn_data.keys())

                # Analyze transition patterns
                dominant_categories = {}

                for turn in turns:
                    reasoning_counts = turn_data[turn]
                    total = sum(reasoning_counts.values())

                    # Find dominant category for this turn
                    if reasoning_counts:
                        dominant = max(reasoning_counts.items(),
                                       key=lambda x: x[1])
                        dominant_categories[turn] = {
                            'category': dominant[0],
                            'proportion': dominant[1] / total,
                            'count': dominant[1],
                            'total': total
                        }

                # Detect evolution patterns
                pattern_description = self._describe_evolution_pattern(
                    dominant_categories)

                model_analysis['games'][game_name] = {
                    'turns_analyzed': turns,
                    'dominant_by_turn': dominant_categories,
                    'pattern': pattern_description
                }

            analysis['models'][model_name] = model_analysis

        return analysis

    def _describe_evolution_pattern(self, dominant_categories: Dict) -> str:
        """Helper method to describe evolution patterns in natural language."""
        turns = sorted(dominant_categories.keys())
        categories = [dominant_categories[turn]['category'] for turn in turns]

        if len(set(categories)) == 1:
            return f"Consistent {categories[0]} throughout"
        elif len(categories) >= 3:
            # Look for progression patterns
            if categories[0] != categories[-1]:
                return (f"Evolution: {categories[0]} → ... → {categories[-1]} "
                        f"(progresses through {len(set(categories))} "
                        f"categories)")
            else:
                return f"Mixed pattern with {len(set(categories))} categories"
        else:
            return f"Transition: {' → '.join(categories)}"

    def generate_model_plots(self, output_dir: str = "plots"):
        """Generate all plot types for each model."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Filter out random agents
        non_random_df = self.df[
            ~self.df['agent_name'].str.startswith("random")
        ]

        if non_random_df.empty:
            print("No non-random agent data found.")
            return

        generated_files = []

        # Process each model
        for agent_name in non_random_df['agent_name'].unique():
            agent_df = non_random_df[non_random_df['agent_name'] == agent_name]
            model_name = agent_name
            games = agent_df['game_name'].unique().tolist()

            # Calculate reasoning percentages
            reasoning_counts = agent_df['reasoning_type'].value_counts()
            total_moves = reasoning_counts.sum()
            reasoning_percentages = (
                reasoning_counts / total_moves * 100
            ).to_dict()

            # 1. Aggregated horizontal bar chart (across all games)
            bar_output = Path(output_dir) / f"reasoning_bar_{agent_name}.png"
            plot_reasoning_bar_chart(
                reasoning_percentages,
                model_name,
                str(bar_output),
                games_list=games
            )
            generated_files.append(str(bar_output))

            # 2. Pie chart
            pie_output = Path(output_dir) / f"reasoning_pie_{agent_name}.png"
            plot_reasoning_pie_chart(
                reasoning_percentages,
                model_name,
                str(pie_output)
            )
            generated_files.append(str(pie_output))

            # 3. Stacked bar chart (if multiple games)
            games = agent_df['game_name'].unique()
            if len(games) > 1:
                # Calculate game-reasoning percentages
                game_reasoning_data = {}
                for game in games:
                    game_df = agent_df[agent_df['game_name'] == game]
                    game_counts = game_df['reasoning_type'].value_counts()
                    game_total = game_counts.sum()
                    game_percentages = (
                        game_counts / game_total * 100
                    ).to_dict()
                    game_reasoning_data[game] = game_percentages

                stacked_output = Path(output_dir) / (
                    f"reasoning_stacked_{agent_name}.png"
                )
                plot_stacked_bar_chart(
                    game_reasoning_data,
                    model_name,
                    str(stacked_output)
                )
                generated_files.append(str(stacked_output))

            # 4. Individual bar charts per game
            for game in games:
                game_df = agent_df[agent_df['game_name'] == game]
                game_counts = game_df['reasoning_type'].value_counts()
                game_total_moves = game_counts.sum()
                game_reasoning_percentages = (
                    game_counts / game_total_moves * 100
                ).to_dict()

                game_bar_output = (
                    Path(output_dir) / f"reasoning_bar_{agent_name}_{game}.png"
                )
                plot_reasoning_bar_chart(
                    game_reasoning_percentages,
                    model_name,
                    str(game_bar_output),
                    games_list=[game]  # Single game
                )
                generated_files.append(str(game_bar_output))

            # 5. Evolution plots for each game (if multiple turns exist)
            for game in games:
                game_df = agent_df[agent_df['game_name'] == game]
                if len(game_df['turn'].unique()) > 1:
                    # Build reasoning_per_turn dict
                    reasoning_per_turn = {}
                    for turn in sorted(game_df['turn'].unique()):
                        turn_df = game_df[game_df['turn'] == turn]
                        turn_counts = turn_df['reasoning_type'].value_counts()
                        reasoning_per_turn[turn] = turn_counts.to_dict()

                    evolution_output = Path(output_dir) / (
                        f"reasoning_evolution_{agent_name}_{game}.png"
                    )
                    plot_reasoning_evolution_over_turns(
                        reasoning_per_turn,
                        model_name,
                        game,
                        str(evolution_output)
                    )
                    generated_files.append(str(evolution_output))

            print(f"Generated plots for {model_name}:")
            print(f"  - Bar chart: {bar_output.name}")
            print(f"  - Pie chart: {pie_output.name}")
            if len(games) > 1:
                print(f"  - Stacked bar: {stacked_output.name}")

            # Count evolution plots
            evolution_count = 0
            for game in games:
                game_turns = agent_df[agent_df['game_name'] == game]['turn']
                if len(game_turns.unique()) > 1:
                    evolution_count += 1

            if evolution_count > 0:
                print(f"  - Evolution plots: {evolution_count} games")
            print()

        return generated_files

    def print_data_summary(self):
        """Print a summary of the data being used."""
        non_random_df = self.df[
            ~self.df['agent_name'].str.startswith("random")
        ]

        print("=== DATA SUMMARY ===")
        print(f"Total moves: {len(non_random_df)}")
        print(f"Models: {len(non_random_df['agent_name'].unique())}")
        print(f"Games: {len(non_random_df['game_name'].unique())}")
        print()

        print("Models and their move counts:")
        for agent_name in non_random_df['agent_name'].unique():
            agent_df = non_random_df[non_random_df['agent_name'] == agent_name]
            model_name = agent_name
            print(f"  {model_name}: {len(agent_df)} moves")
        print()

        print("Overall reasoning distribution:")
        reasoning_counts = non_random_df['reasoning_type'].value_counts()
        total = reasoning_counts.sum()
        for reasoning_type, count in reasoning_counts.items():
            percentage = count / total * 100
            print(f"  {reasoning_type}: {count} ({percentage:.1f}%)")
        print()


def main():
    """Main function to generate model-specific plots."""
    print("Generating model-specific reasoning plots...")

    # Find the latest merged log file
    latest_csv = LLMReasoningAnalyzer.find_latest_log("results")
    if not latest_csv:
        print("No merged log files found in results/ directory.")
        print("Please run post_game_processing.py first.")
        return

    print(f"Using data from: {latest_csv}")

    # Initialize the plotter
    plotter = ReasoningPlotGenerator(str(latest_csv))

    # Print data summary
    plotter.print_data_summary()

    # Generate plots
    output_dir = "plots"
    generated_files = plotter.generate_model_plots(output_dir)

    # Also generate all evolution plots in one batch
    print("\nGenerating batch evolution plots...")
    evolution_files = plotter.plot_all_reasoning_evolutions(output_dir)
    generated_files.extend(evolution_files)

    print(f"Generated {len(generated_files)} plot files in {output_dir}/")
    individual_count = len(generated_files) - len(evolution_files)
    print(f"  - Individual model plots: {individual_count}")
    print(f"  - Evolution plots: {len(evolution_files)}")
    print("\nPlot types generated:")
    print("  - Horizontal bar charts: Show reasoning distribution percentages")
    print("  - Pie charts: Visual reasoning type proportions")
    print("  - Stacked bar charts: Reasoning percentages across games")
    print("  - Evolution plots: Reasoning changes over game turns")


if __name__ == "__main__":
    main()
