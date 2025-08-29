"""
Reasoning Analysis Module

This module analyzes LLM reasoning patterns in game play data.
It categorizes reasoning types, generates visualizations, and creates
word clouds and statistical summaries of agent behavior patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from wordcloud import WordCloud  # DISABLED - not using wordclouds
import numpy as np
import re
import os
from typing import Optional
from transformers import pipeline
from pathlib import Path
from utils import display_game_name


def clean_model_name(model_name: str) -> str:
    """
    Clean up long model names to display only the essential model name.
    This is a copy of the function from ui/utils.py for use in analysis.
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


REASONING_RULES = {
    "Positional": [
        re.compile(r"\bcenter column\b"),
        re.compile(r"\bcenter square\b"),
        re.compile(r"\bcorner\b"),
        re.compile(r"\bedge\b")
    ],
    "Blocking": [
        re.compile(r"\bblock\b"),
        re.compile(r"\bblocking\b"),
        re.compile(r"\bprevent\b"),
        re.compile(r"\bstop opponent\b"),
        re.compile(r"\bavoid opponent\b"),
        re.compile(r"\bcounter\b")
    ],
    "Opponent Modeling": [
        re.compile(r"\bopponent\b"),
        re.compile(r"\bthey are trying\b"),
        re.compile(r"\btheir strategy\b"),
        re.compile(r"\btheir move\b")
    ],
    "Winning Logic": [
        re.compile(r"\bwin\b"),
        re.compile(r"\bwinning move\b"),
        re.compile(r"\bconnect\b"),
        re.compile(r"\bfork\b"),
        re.compile(r"\bthreat\b"),
        re.compile(r"\bchance of winning\b")
    ],
    "Heuristic": [
        re.compile(r"\bbest move\b"),
        re.compile(r"\bmost likely\b"),
        re.compile(r"\badvantageous\b"),
        re.compile(r"\bbetter chance\b")
    ],
    "Rule-Based": [
        re.compile(r"\baccording to\b"),
        re.compile(r"\brule\b"),
        re.compile(r"\bstrategy\b")
    ],
    "Random/Unjustified": [
        re.compile(r"\brandom\b"),
        re.compile(r"\bguess\b")
    ]
}

# Consistent color mapping for all reasoning types
REASONING_COLOR_MAP = {
    "Positional": "#1f77b4",          # Blue
    "Blocking": "#ff7f0e",            # Orange
    "Opponent Modeling": "#2ca02c",   # Green
    "Winning Logic": "#d62728",       # Red
    "Heuristic": "#9467bd",           # Purple
    "Rule-Based": "#8c564b",          # Brown
    "Random/Unjustified": "#e377c2",  # Pink
    "Uncategorized": "#7f7f7f"        # Gray
}


def get_reasoning_colors(reasoning_types):
    """Get consistent colors for a list of reasoning types.

    Args:
        reasoning_types: List or iterable of reasoning type names

    Returns:
        List of hex color codes matching the reasoning types
    """
    return [REASONING_COLOR_MAP.get(rtype, "#999999")
            for rtype in reasoning_types]


LLM_PROMPT_TEMPLATE = (
    "You are a reasoning classifier. Your job is to categorize a move "
    "explanation into one of the following types:\n"
    "- Positional\n- Blocking\n- Opponent Modeling\n- Winning Logic\n"
    "- Heuristic\n- Rule-Based\n- Random/Unjustified\n\n"
    "Examples:\n"
    "REASONING: I placed in the center square to prevent the opponent "
    "from winning.\nCATEGORY: Blocking\n\n"
    "REASONING: The center square gives me the best control.\n"
    "CATEGORY: Positional\n\n"
    "Now classify this:\n"
    "REASONING: {reasoning}\nCATEGORY:"
)


class LLMReasoningAnalyzer:
    def __init__(self, csv_path: str):
        """Initialize the analyzer with a path to the LLM game log CSV.

        Args:
            csv_path: Path to the reasoning CSV file.
        """
        self.df = pd.read_csv(csv_path)
        self._preprocess()
        self.llm_pipe = pipeline(
            "text2text-generation", model="google/flan-t5-small"
        )

    @staticmethod
    def find_latest_log(folder: str) -> str:
        """Find the most recent log file in the given folder.

        Args:
            folder: Directory where the merged_logs_*.csv files are stored.

        Returns:
            Path to the most recent CSV file.
        """
        files = list(Path(folder).glob("merged_logs_*.csv"))
        if not files:
            return None
        files.sort(key=lambda f: f.name.split("_")[2], reverse=True)
        return files[0]

    def _preprocess(self) -> None:
        """Prepare the DataFrame by filling NaNs and stripping whitespace."""
        self.df['reasoning'] = self.df['reasoning'].fillna("").astype(str)
        self.df['reasoning'] = self.df['reasoning'].str.strip()

    def categorize_reasoning(self) -> None:
        """Assign a reasoning category to each reasoning entry using scoring
        and precompiled regexes.

        This version scores each reasoning type by the number of matching
        patterns and assigns the category with the highest score.
        """
        def classify(reasoning: str, agent: str) -> str:
            if not reasoning or agent.startswith("random"):
                return "Uncategorized"

            text = reasoning.lower()
            scores = {}

            for label, patterns in REASONING_RULES.items():
                match_count = sum(
                    1 for pattern in patterns if pattern.search(text)
                )
                if match_count > 0:
                    scores[label] = match_count

            return (
                max(scores.items(), key=lambda x: x[1])[0]
                if scores else "Uncategorized"
            )

        self.df['reasoning_type'] = self.df.apply(
            lambda row: classify(row['reasoning'], row['agent_name']), axis=1
        )

    def categorize_with_llm(self, max_samples: Optional[int] = None) -> None:
        """Use a Hugging Face model to categorize reasoning types.

        Args:
            max_samples: Optional limit for debugging or testing with a subset.
        """
        def classify_llm(reasoning: str) -> str:
            prompt = LLM_PROMPT_TEMPLATE.format(reasoning=reasoning)
            response = self.llm_pipe(prompt, max_new_tokens=10)[0][
                'generated_text']
            return response.strip().split("\n")[0].replace(
                "CATEGORY:", "").strip()

        df_to_classify = self.df
        if max_samples is not None and max_samples > 0:
            df_to_classify = df_to_classify.sample(
                n=min(max_samples, len(df_to_classify)), random_state=42
            )
        # Default to Uncategorized for all, then fill for sampled rows
        self.df['reasoning_type_llm'] = "Uncategorized"
        classified = df_to_classify.apply(
            lambda row: classify_llm(row['reasoning'])
            if row['reasoning'] and not row['agent_name'].startswith("random")
            else "Uncategorized",
            axis=1
        )
        # Assign back to the original indices
        self.df.loc[classified.index, 'reasoning_type_llm'] = classified

    def summarize_reasoning(self) -> None:
        """Generate a short summary for each reasoning entry.

        Simple heuristic; later we can replace this with LLM compression.
        """
        def summarize(reasoning: str) -> str:
            if "." in reasoning:
                first = reasoning.split(".")[0]
            else:
                first = reasoning
            return " ".join(first.strip().split()[:10])

        self.df['summary'] = self.df['reasoning'].apply(summarize)

    def summarize_games(
        self, output_csv: str = "results/game_summary.csv"
    ) -> pd.DataFrame:
        """Summarize the reasoning data by game and agent."""
        # Ensure the results directory exists
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

        summary = self.df.groupby(["game_name", "agent_name"]).agg(
            episodes=('episode', 'nunique'),
            turns=('turn', 'count')
        ).reset_index()
        summary.to_csv(output_csv, index=False)
        return summary

    def _calculate_agent_metrics(self, group_df: pd.DataFrame) -> dict:
        """Calculate metrics for a single agent-game group.

        Args:
            group_df: DataFrame subset for one agent-game combination

        Returns:
            Dictionary containing calculated metrics
        """
        total = len(group_df)
        opponent_mentions = group_df['reasoning'].str.lower().str.contains(
            "opponent"
        ).sum()
        reasoning_len_avg = group_df['reasoning'].apply(
            lambda r: len(r.split())
        ).mean()
        unique_types = group_df['reasoning_type'].nunique()
        type_counts = group_df['reasoning_type'].value_counts(
            normalize=True
        ).to_dict()
        entropy = -sum(
            p * np.log2(p) for p in type_counts.values() if p > 0
        )

        return {
            "total_moves": total,
            "avg_reasoning_length": reasoning_len_avg,
            "%_opponent_mentions": opponent_mentions / total,
            "reasoning_diversity": unique_types,
            "reasoning_entropy": entropy
        }

    def _create_pie_chart(self, group_df: pd.DataFrame, agent: str,
                          game: str, plot_dir: str) -> None:
        """Create a pie chart for reasoning type distribution.

        Args:
            group_df: DataFrame subset for one agent-game combination
            agent: Agent name
            game: Game name
            plot_dir: Directory to save the plot
        """
        type_dist = group_df['reasoning_type'].value_counts()
        colors = get_reasoning_colors(type_dist.index)
        plt.figure()
        type_dist.plot.pie(autopct='%1.1f%%', colors=colors)
        plt.title(
            f"Reasoning Type Distribution - {clean_model_name(agent)}\n"
            f"(Game: {display_game_name(game)})"
        )
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(
            Path(plot_dir) / f"pie_reasoning_type_{agent}_{game}.png"
        )
        plt.close()

    def _create_agent_aggregate_heatmap(self, df_agent: pd.DataFrame,
                                        agent: str, plot_dir: str) -> None:
        """Create aggregate heatmap for one agent across all games.

        Args:
            df_agent: DataFrame subset for one agent across all games
            agent: Agent name
            plot_dir: Directory to save the plot
        """
        pivot = df_agent.pivot_table(
            index="turn", columns="reasoning_type", values="agent_name",
            aggfunc="count", fill_value=0
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True)
        games = df_agent['game_name'].unique()
        plt.title(
            f"Reasoning Type by Turn - {clean_model_name(agent)}\n"
            f"Games:\n{', '.join(display_game_name(g) for g in games)}"
        )
        plt.ylabel("Turn")
        plt.xlabel("Reasoning Type")
        plt.tight_layout()
        out_path = os.path.join(plot_dir, f"heatmap_{agent}_all_games.png")
        plt.savefig(out_path)
        plt.close()

    def compute_agent_metrics(
        self, output_csv: str = "results/agent_metrics_summary.csv"
    ) -> pd.DataFrame:
        """Compute metrics for each agent and game combination.

        Args:
            output_csv: Path to save the metrics CSV

        Returns:
            DataFrame containing computed metrics
        """
        rows = []
        for (game, agent), group_df in self.df.groupby(
            ["game_name", "agent_name"]
        ):
            if agent.startswith("random"):
                continue

            metrics = self._calculate_agent_metrics(group_df)
            metrics.update({
                "agent_name": agent,
                "game_name": game
            })
            rows.append(metrics)

        df_metrics = pd.DataFrame(rows)

        # Save to CSV if path provided
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df_metrics.to_csv(output_csv, index=False)

        return df_metrics

    def plot_reasoning_type_pie_charts(self, plot_dir: str = "plots") -> None:
        """Create pie charts showing reasoning type distribution for
        each agent-game pair.

        Args:
            plot_dir: Directory to save the plots
        """
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        for (game, agent), group_df in self.df.groupby(
            ["game_name", "agent_name"]
        ):
            if agent.startswith("random"):
                continue
            self._create_pie_chart(group_df, agent, game, plot_dir)

    def plot_agent_aggregate_heatmaps(self, plot_dir: str = "plots") -> None:
        """Create aggregate heatmaps for each agent across all their games.

        Args:
            plot_dir: Directory to save the plots
        """
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        for agent, df_agent in self.df.groupby("agent_name"):
            if agent.startswith("random"):
                continue
            self._create_agent_aggregate_heatmap(df_agent, agent, plot_dir)

    def compute_metrics(
        self, output_csv: str = "results/agent_metrics_summary.csv",
        plot_dir: str = "plots"
    ) -> None:
        """Compute metrics and generate visualizations for each agent and game.

        This is the main interface that combines metric computation with
        visualization generation for backward compatibility.

        Args:
            output_csv: Path to save the metrics CSV
            plot_dir: Directory to save the plots
        """
        # Compute and save metrics
        self.compute_agent_metrics(output_csv)

        # Generate visualizations
        self.plot_reasoning_type_pie_charts(plot_dir)
        self.plot_agent_aggregate_heatmaps(plot_dir)

    def plot_heatmaps_by_agent(self, output_dir: str = "plots") -> None:
        """Plot per-agent heatmaps and one aggregated heatmap across
        all games.

        Individual heatmaps are saved per agent-game pair. This also
        includes a general heatmap per agent showing all turns merged
        across all games. Useful for seeing broad reasoning type patterns.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for (agent, game), df_agent in self.df.groupby(
            ["agent_name", "game_name"]
        ):
            if agent.startswith("random"):
                continue
            pivot = df_agent.pivot_table(
                index="turn", columns="reasoning_type",
                values="agent_name", aggfunc="count", fill_value=0
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, cmap="YlGnBu", annot=True)
            plt.title(
                f"Reasoning Type by Turn - {agent} "
                f"(Game: {display_game_name(game)})"
            )
            plt.ylabel("Turn")
            plt.xlabel("Reasoning Type")
            plt.tight_layout()
            out_path = os.path.join(
                output_dir, f"heatmap_{agent}_{game}.png"
            )
            plt.savefig(out_path)
            plt.close()

    def plot_wordclouds_by_agent(self, output_dir: str = "plots") -> None:
        """Plot per-agent word clouds and one aggregated word cloud across
        all games.

        Word clouds are created per agent-game pair and also aggregated per
        agent over all games. The full version helps summarize LLM behavior
        globally.

        Note: WordCloud import is disabled at module level, so this function
        will fail if called. Use this only when WordCloud dependency is
        available.
        """
        from wordcloud import WordCloud  # Import only when needed

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for (agent, game), agent_df in self.df.groupby(
                ["agent_name", "game_name"]
        ):
            if agent.startswith("random"):
                continue
            text = " ".join(agent_df['reasoning'].tolist())
            wc = WordCloud(
                width=800, height=400, background_color='white'
            ).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            title = (
                f"Reasoning Word Cloud - {clean_model_name(agent)} "
                f"(Game: {display_game_name(game)})"
            )
            plt.title(title)
            plt.tight_layout()
            out_path = os.path.join(
                output_dir, f"wordcloud_{agent}_{game}.png"
            )
            plt.savefig(out_path)
            plt.close()

    def plot_entropy_trendlines(self, output_dir: str = "plots") -> None:
        """Plot entropy over turns for each agent-game pair.

        This shows how each LLM agent's reasoning diversity evolves
        throughout the game, based on Shannon entropy of reasoning types.
        Higher entropy means more varied reasoning types.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for (agent, game), df_group in self.df.groupby(
                ["agent_name", "game_name"]
        ):
            if agent.startswith("random"):
                continue
            entropy_by_turn = (
                df_group.groupby("turn")["reasoning_type"]
                .apply(lambda s: -sum(
                    s.value_counts(normalize=True).apply(
                        lambda p: p * np.log2(p)
                    )
                ))
            )
            plt.figure()
            entropy_by_turn.plot(marker='o')
            plt.title(
                f"Reasoning Entropy by Turn - {agent} "
                f"(Game: {display_game_name(game)})"
            )
            plt.xlabel("Turn")
            plt.ylabel("Entropy")
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(
                output_dir, f"entropy_trend_{agent}_{game}.png"
            )
            plt.savefig(out_path)
            plt.close()

    def plot_entropy_by_turn_across_agents(self,
                                           output_dir: str = "plots"
                                           ) -> None:
        """Plot entropy over turns across all agents per game.

        This compares how different LLM agents behave during the same game,
        highlighting agents that adapt their reasoning more flexibly.
        Useful to detect which models generalize or explore more.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for game, df_game in self.df.groupby("game_name"):
            plt.figure(figsize=(10, 6))
            for agent, df_agent in df_game.groupby("agent_name"):
                if agent.startswith("random"):
                    continue
                entropy_by_turn = (
                    df_agent.groupby("turn")["reasoning_type"]
                    .apply(lambda s: -sum(
                        s.value_counts(normalize=True).apply(
                            lambda p: p * np.log2(p)
                        )
                    ))
                )
                entropy_by_turn.plot(label=clean_model_name(agent))
            plt.title(
                f"Entropy by Turn Across Agents - {display_game_name(game)}"
            )
            plt.xlabel("Turn")
            plt.ylabel("Entropy")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
            plt.grid(True)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            out_path = os.path.join(
                output_dir, f"entropy_by_turn_all_agents_{game}.png"
            )
            plt.savefig(out_path)
            plt.close()

    def plot_avg_entropy_across_games(self, output_dir: str = "plots") -> None:
        """Plot average entropy over time across all games and agents.

        This reveals the general trend of reasoning diversity (entropy) per
        turn for all agents collectively, helping to understand how LLM
        reasoning evolves globally across gameplay.
        """

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        df_all = self.df[~self.df['agent_name'].str.startswith("random")]
        avg_entropy = (
            df_all.groupby("turn")["reasoning_type"]
            .apply(lambda s: -sum(
                s.value_counts(normalize=True).apply(lambda p: p * np.log2(p))
            ))
        )
        avg_entropy.plot(marker='o')
        plt.title("Average Reasoning Entropy Across All Games and Agents")
        plt.xlabel("Turn")
        plt.ylabel("Entropy")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "avg_entropy_all_games.png")
        plt.savefig(out_path)
        plt.close()

    def save_output(self, path: str) -> None:
        self.df.to_csv(path, index=False)


if __name__ == "__main__":
    # Use scripts/results directory for the current workspace
    project_root = Path(__file__).resolve().parent.parent
    scripts_results = str(project_root / "scripts" / "results")
    latest_csv = LLMReasoningAnalyzer.find_latest_log(scripts_results)
    analyzer = LLMReasoningAnalyzer(latest_csv)

    # Choose one of the methods below to analyze the reasoning data
    analyzer.categorize_reasoning()
    # analyzer.categorize_with_llm(max_samples=50)
    # Remove limit for full analysis

    analyzer.compute_metrics(plot_dir="plots")
    # analyzer.plot_heatmaps_by_agent(output_dir="plots")
    #   HEATMAP DISABLED
    # analyzer.plot_wordclouds_by_agent(output_dir="plots")
    #   WORDCLOUD DISABLED
    # analyzer.plot_entropy_trendlines()
    #   Individual entropy trends per agent-game pair
    analyzer.plot_entropy_by_turn_across_agents()
    analyzer.plot_avg_entropy_across_games(output_dir="plots")

    # Save augmented output to scripts/results directory
    output_path = (
        Path(__file__).resolve().parent.parent / 'scripts' / 'results' /
        'augmented_reasoning_output.csv'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer.save_output(str(output_path))
    print("Analysis completed successfully!.")
