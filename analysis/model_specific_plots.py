#!/usr/bin/env python3
"""
Model-Specific Reasoning Plots

This script generates reasoning analysis plots adapted for our game reasoning
data, with one plot per model and using percentages instead of raw counts.
"""

import sys
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

# Add analysis directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reasoning_analysis import LLMReasoningAnalyzer  # noqa: E402


def clean_model_name(agent_name: str) -> str:
    """Extract a clean model name from the full agent name."""
    if 'llama3_70b' in agent_name:
        return 'Llama3 70B'
    elif 'llama3_8b' in agent_name:
        return 'Llama3 8B'
    elif 'Meta_Llama_3.1_70B' in agent_name:
        return 'Llama3.1 70B'
    elif 'Meta_Llama_3.1_8B' in agent_name:
        return 'Llama3.1 8B'
    else:
        # Fallback: use last part of agent name
        return agent_name.split('_')[-1]


def plot_reasoning_bar_chart(
    reasoning_percentages: Dict[str, float],
    model_name: str,
    output_path: str = "reasoning_bar_chart.png"
) -> None:
    """
    Plots a horizontal bar chart for reasoning type percentages.

    Args:
        reasoning_percentages: Dictionary mapping reasoning types to
            percentages.
        model_name: Name of the model for the title.
        output_path: File path to save the figure.
    """
    df = pd.DataFrame(
        list(reasoning_percentages.items()),
        columns=["reasoning_type", "percentage"]
    )
    df = df.sort_values("percentage", ascending=True)

    plt.figure(figsize=(8, 6))
    bars = plt.barh(df["reasoning_type"], df["percentage"], color="skyblue")

    # Add percentage labels on bars
    for bar, percentage in zip(bars, df["percentage"]):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{percentage:.1f}%', ha='left', va='center',
                 fontweight='bold')

    plt.title(f"Reasoning Type Distribution - {model_name}")
    plt.xlabel("Percentage (%)")
    plt.ylabel("Reasoning Type")
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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
        startangle=90
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    plt.title(f"Reasoning Type Distribution - {model_name}", fontsize=14)
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

    plt.ylabel("Percentage (%)")
    plt.xlabel("Game")
    plt.title(f"Reasoning Types per Game - {model_name}")
    plt.legend(title="Reasoning Type", bbox_to_anchor=(1.05, 1),
               loc='upper left')
    plt.xticks(rotation=45)

    # Add percentage labels on stacked bars
    for i, game in enumerate(df.index):
        cumulative = 0
        for reasoning_type in df.columns:
            percentage = df.loc[game, reasoning_type]
            if percentage > 5:  # Only show label if segment is large enough
                plt.text(i, cumulative + percentage/2, f'{percentage:.0f}%',
                         ha='center', va='center', fontweight='bold',
                         fontsize=8)
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
    model-game pair.

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

    # Plot
    plt.figure(figsize=(10, 6))
    for reasoning_type in df_prop.columns:
        plt.plot(
            df_prop.index,
            df_prop[reasoning_type],
            label=reasoning_type,
            linewidth=2,
            marker='o'
        )

    plt.title(f"{model_name} - {game_name}: Reasoning Evolution Over Turns")
    plt.xlabel("Turn")
    plt.ylabel("Proportion of Reasoning")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1),
               title="Reasoning Type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


class ModelSpecificPlotter:
    """Generates model-specific reasoning plots."""

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
            model_name = clean_model_name(agent_name)
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
                filename = f"evolution_{model_name}_{game_name}.png"
                filename = filename.replace(" ", "_").replace(".", "_").lower()
                output_path = Path(output_dir) / filename

                plot_reasoning_evolution_over_turns(
                    reasoning_per_turn=reasoning_per_turn,
                    model_name=model_name,
                    game_name=game_name,
                    output_path=str(output_path)
                )
                generated_files.append(str(output_path))

        return generated_files

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
            model_name = clean_model_name(agent_name)

            # Calculate reasoning percentages
            reasoning_counts = agent_df['reasoning_type'].value_counts()
            total_moves = reasoning_counts.sum()
            reasoning_percentages = (
                reasoning_counts / total_moves * 100
            ).to_dict()

            # 1. Horizontal bar chart
            bar_output = Path(output_dir) / f"reasoning_bar_{agent_name}.png"
            plot_reasoning_bar_chart(
                reasoning_percentages,
                model_name,
                str(bar_output)
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

            # 4. Evolution plots for each game (if multiple turns exist)
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
            model_name = clean_model_name(agent_name)
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
    plotter = ModelSpecificPlotter(str(latest_csv))

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
    print("\nAll plots use percentages instead of raw counts and include")
    print("model names in titles for easy identification.")


if __name__ == "__main__":
    main()
