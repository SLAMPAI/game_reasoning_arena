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

from utils import display_game_name
from reasoning_analysis import (LLMReasoningAnalyzer,
                                get_reasoning_colors)  # noqa: E402

# Add analysis directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def clean_model_name(model_name: str) -> str:
    """
    Clean up long model names to display only the essential model name.

    This function handles various model naming patterns
    from different providers:
    - LiteLLM models with provider prefixes
    - vLLM models with prefixes
    - Models with slash-separated paths
    - GPT model variants

    Args:
        model_name: Full model name from database
                   (e.g., "llm_litellm_together_ai_meta_llama_Meta_Llama_3.1...")

    Returns:
        Cleaned model name (e.g., "Meta-Llama-3.1-8B-Instruct-Turbo")
    """
    if not model_name or model_name == "Unknown":
        return model_name

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
    # These correspond to the slash-separated patterns in the YAML
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
            except ValueError:
                pass

        # Handle Together AI pattern: litellm_together_ai_meta_llama_*
        if (
            "together" in model_name
            and "meta" in model_name
            and "llama" in model_name
        ):
            try:
                # Find "meta" and "llama" -
                # the model name starts after "meta_llama_"
                for i, part in enumerate(parts):
                    if (
                        part == "meta"
                        and i + 1 < len(parts)
                        and parts[i + 1] == "llama"
                    ):
                        # Model name starts after "meta_llama_"
                        model_parts = parts[i + 2:]
                        return "-".join(model_parts)
            except Exception:
                pass

        # Handle Groq pattern: litellm_groq_*
        # These are simpler patterns
        if parts[1] == "groq" and len(parts) >= 3:
            model_parts = parts[2:]  # Everything after "litellm_groq_"
            cleaned = "-".join(model_parts)
            # Special handling for common models
            if "llama3" in cleaned.lower():
                cleaned = cleaned.replace("llama3", "Llama-3")
            elif "qwen" in cleaned.lower():
                cleaned = cleaned.replace("qwen", "Qwen")
            elif "gemma" in cleaned.lower():
                cleaned = cleaned.replace("gemma", "Gemma")
            return cleaned

        # For other patterns, skip first two parts (litellm_provider_)
        if len(parts) >= 3:
            model_parts = parts[2:]  # Everything after provider
            cleaned = "-".join(model_parts)
            return cleaned

    # For models with slashes but not litellm (like direct model paths)
    if "/" in model_name:
        return model_name.split("/")[-1].replace("_", "-")

    # Default: just replace underscores with dashes
    return model_name.replace("_", "-")


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
    colors = get_reasoning_colors(df["reasoning_type"])
    bars = plt.barh(df["reasoning_type"], df["percentage"], color=colors)

    # Generate title
    clean_name = clean_model_name(model_name)
    title = f"Reasoning Type Distribution - {clean_name}"
    if games_list:
        pretty_games = [display_game_name(g) for g in games_list]
        title += f" ({', '.join(pretty_games)})"

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Percentage (%)", fontsize=14)
    plt.ylabel("Reasoning Type", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add percentage labels on bars
    for bar, percentage in zip(bars, df["percentage"]):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{percentage:.1f}%', ha='left', va='center', fontsize=11)

    # Save and close
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
    colors = get_reasoning_colors(series.index)

    plt.figure(figsize=(8, 8))
    _, _, autotexts = plt.pie(
        series.values,
        labels=series.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12}
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    plt.title(f"Reasoning Type Distribution - {clean_model_name(model_name)}",
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
    colors = get_reasoning_colors(df.columns)

    plt.figure(figsize=(12, 6))
    ax = df.plot(kind='bar', stacked=True, color=colors)

    plt.ylabel("Percentage (%)", fontsize=14)
    plt.xlabel("Game", fontsize=14)
    plt.title(f"Reasoning Types per Game - {clean_model_name(model_name)}",
              fontsize=16, fontweight='bold')
    plt.legend(title="Reasoning Type",
               bbox_to_anchor=(1.02, 0.5),
               loc='center left', fontsize=12, title_fontsize=13,
               borderaxespad=0.0, frameon=False)
    # Use friendly game names on x-axis
    pretty_index = [display_game_name(g) for g in df.index]
    ax.set_xticklabels(pretty_index, rotation=45, fontsize=12)
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
    # Reserve a consistent right margin for the legend (after tight_layout)
    plt.subplots_adjust(right=0.82)
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
    colors = get_reasoning_colors(df_prop.columns)

    for i, reasoning_type in enumerate(df_prop.columns):
        plt.bar(df_prop.index, df_prop[reasoning_type],
                bottom=bottom, label=reasoning_type,
                color=colors[i], alpha=0.8, edgecolor='black',
                linewidth=0.5)
        bottom += df_prop[reasoning_type]

    plt.title(
        f"{clean_model_name(model_name)} - {display_game_name(game_name)}: "
        f"Reasoning Category Evolution",
        fontsize=16, fontweight='bold'
    )
    plt.xlabel("Game Turn", fontsize=14)
    plt.ylabel("Proportion of Reasoning Types", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
               title="Reasoning Type", fontsize=12, title_fontsize=13,
               borderaxespad=0.0, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add sample size annotations above each bar
    for turn in df_prop.index:
        total_count = df.loc[turn].sum()
        plt.annotate(f'n={int(total_count)}',
                     xy=(turn, 1.02),
                     ha='center', va='bottom',
                     fontsize=11, alpha=0.7)

    plt.tight_layout()
    # Reserve a consistent right margin for the legend (after tight_layout)
    plt.subplots_adjust(right=0.82)
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

    plt.title(
        f"{clean_model_name(model_name)} - {display_game_name(game_name)}: "
        f"Reasoning Evolution Heatmap",
        fontsize=16, fontweight='bold'
    )
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

                # HEATMAP GENERATION DISABLED
                # # Generate heatmap version
                # heatmap_filename = (f"evolution_heatmap_{clean_model}_"
                #                     f"{clean_game}.png").lower()
                # heatmap_path = Path(output_dir) / heatmap_filename

                # plot_reasoning_evolution_heatmap(
                #     reasoning_per_turn=reasoning_per_turn,
                #     model_name=model_name,
                #     game_name=game_name,
                #     output_path=str(heatmap_path)
                # )
                # generated_files.append(str(heatmap_path))

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
                        dominant_categories[int(turn)] = {
                            'category': dominant[0],
                            'proportion': float(dominant[1] / total),
                            'count': int(dominant[1]),
                            'total': int(total)
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
