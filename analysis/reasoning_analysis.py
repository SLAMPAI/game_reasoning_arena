"""
Reasoning Analysis Module

This module analyzes LLM reasoning patterns in game play data.
It categorizes reasoning types, generates visualizations, and creates
word clouds and statistical summaries of agent behavior patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import re
import os
from typing import List, Optional
import glob
from transformers import pipeline
from pathlib import Path


REASONING_RULES = {
    "Positional": [re.compile(r"\bcenter column\b"), re.compile(r"\bcenter square\b"), re.compile(r"\bcorner\b"), re.compile(r"\bedge\b")],
    "Blocking": [re.compile(r"\bblock\b"), re.compile(r"\bblocking\b"), re.compile(r"\bprevent\b"),
                 re.compile(r"\bstop opponent\b"), re.compile(r"\bavoid opponent\b"), re.compile(r"\bcounter\b")],
    "Opponent Modeling": [re.compile(r"\bopponent\b"), re.compile(r"\bthey are trying\b"),
                          re.compile(r"\btheir strategy\b"), re.compile(r"\btheir move\b")],
    "Winning Logic": [re.compile(r"\bwin\b"), re.compile(r"\bwinning move\b"), re.compile(r"\bconnect\b"),
                      re.compile(r"\bfork\b"), re.compile(r"\bthreat\b"), re.compile(r"\bchance of winning\b")],
    "Heuristic": [re.compile(r"\bbest move\b"), re.compile(r"\bmost likely\b"), re.compile(r"\badvantageous\b"),
                  re.compile(r"\bbetter chance\b")],
    "Rule-Based": [re.compile(r"\baccording to\b"), re.compile(r"\brule\b"), re.compile(r"\bstrategy\b")],
    "Random/Unjustified": [re.compile(r"\brandom\b"), re.compile(r"\bguess\b")]
}

LLM_PROMPT_TEMPLATE = (
    "You are a reasoning classifier. Your job is to categorize a move explanation into one of the following types:\n"
    "- Positional\n- Blocking\n- Opponent Modeling\n- Winning Logic\n- Heuristic\n- Rule-Based\n- Random/Unjustified\n\n"
    "Examples:\n"
    "REASONING: I placed in the center square to prevent the opponent from winning.\nCATEGORY: Blocking\n\n"
    "REASONING: The center square gives me the best control.\nCATEGORY: Positional\n\n"
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
        self.llm_pipe = pipeline("text2text-generation", model="google/flan-t5-small")

    @staticmethod
    def find_latest_log(folder: str) -> str:
        """Find the most recent log file in the given folder.

        Args:
            folder: Directory where the merged_logs_*.csv files are stored.

        Returns:
            Path to the most recent CSV file.
        """
        files = glob.glob(os.path.join(folder, "merged_logs_*.csv"))
        if not files:
            raise FileNotFoundError("No log files found in folder")
        files.sort(key=lambda f: os.path.basename(f).split("_")[2], reverse=True)
        return files[0]

    def _preprocess(self) -> None:
        """Prepare the DataFrame by filling NaNs and stripping whitespace."""
        self.df['reasoning'] = self.df['reasoning'].fillna("").astype(str)
        self.df['reasoning'] = self.df['reasoning'].str.strip()

    def categorize_reasoning(self) -> None:
        """Assign a reasoning category to each reasoning entry using scoring and precompiled regexes.

        This version scores each reasoning type by the number of matching
        patterns and assigns the category with the highest score.
        """
        def classify(reasoning: str, agent: str) -> str:
            if not reasoning or agent.startswith("random"):
                return "Uncategorized"

            text = reasoning.lower()
            scores = {}

            for label, patterns in REASONING_RULES.items():
                match_count = sum(1 for pattern in patterns if pattern.search(text))
                if match_count > 0:
                    scores[label] = match_count

            return max(scores.items(), key=lambda x: x[1])[0] if scores else "Uncategorized"

        self.df['reasoning_type'] = self.df.apply(
            lambda row: classify(row['reasoning'], row['agent_name']), axis=1
        )

    def categorize_with_llm(self, max_samples: Optional[int] = None) -> None:
            """Use a Hugging Face model to categorize reasoning types using natural language.

            Args:
                max_samples: Optional limit for debugging or testing with a subset.
            """
            if max_samples:
                df_subset = self.df.head(max_samples).copy()
            else:
                df_subset = self.df.copy()

            def classify_llm(reasoning: str) -> str:
                prompt = LLM_PROMPT_TEMPLATE.format(reasoning=reasoning)
                response = self.llm_pipe(prompt, max_new_tokens=10)[0]['generated_text']
                return response.strip().split("\n")[0].replace("CATEGORY:", "").strip()

            self.df['reasoning_type_llm'] = self.df.apply(
                lambda row: classify_llm(row['reasoning']) if row['reasoning'] and not row['agent_name'].startswith("random") else "Uncategorized",
                axis=1
            )

    def summarize_reasoning(self) -> None:
        """Generate a short summary for each reasoning entry (simple heuristic).

        OBS: Later we can replace this with LLM compression.
        """
        def summarize(reasoning: str) -> str:
            if "." in reasoning:
                first = reasoning.split(".")[0]
            else:
                first = reasoning
            return " ".join(first.strip().split()[:10])

        self.df['summary'] = self.df['reasoning'].apply(summarize)

    def summarize_games(self, output_csv: str = "game_summary.csv") -> pd.DataFrame:
        """Summarize the reasoning data by game and agent."""
        summary = self.df.groupby(["game_name", "agent_name"]).agg(
            episodes=('episode', 'nunique'),
            turns=('turn', 'count')
        ).reset_index()
        summary.to_csv(output_csv, index=False)
        return summary

    def compute_metrics(self, output_csv: str = "agent_metrics_summary.csv", plot_dir: str = "plots") -> None:
        """Compute metrics for each agent and game."""

        os.makedirs(plot_dir, exist_ok=True)
        game_summary = self.summarize_games()
        rows = []
        for (game, agent), group_df in self.df.groupby(["game_name", "agent_name"]):
            if agent.startswith("random"):
                continue
            total = len(group_df)
            opponent_mentions = group_df['reasoning'].str.lower().str.contains("opponent").sum()
            reasoning_len_avg = group_df['reasoning'].apply(lambda r: len(r.split())).mean()
            unique_types = group_df['reasoning_type'].nunique()
            type_counts = group_df['reasoning_type'].value_counts(normalize=True).to_dict()
            entropy = -sum(p * np.log2(p) for p in type_counts.values() if p > 0)

            rows.append({
                "agent_name": agent,
                "game_name": game,
                "total_moves": total,
                "avg_reasoning_length": reasoning_len_avg,
                "%_opponent_mentions": opponent_mentions / total,
                "reasoning_diversity": unique_types,
                "reasoning_entropy": entropy
            })

            # Pie chart for this (agent, game)
            type_dist = group_df['reasoning_type'].value_counts()
            plt.figure()
            type_dist.plot.pie(autopct='%1.1f%%')
            plt.title(f"Reasoning Type Distribution - {agent}\n(Game: {game})")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"pie_reasoning_type_{agent}_{game}.png"))
            plt.close()

    # Aggregate heatmap for each agent across all games
        for agent, df_agent in self.df.groupby("agent_name"):
            if agent.startswith("random"):
                continue
            pivot = df_agent.pivot_table(
                index="turn", columns="reasoning_type", values="agent_name",
                aggfunc="count", fill_value=0
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, cmap="YlGnBu", annot=True)
            games = df_agent['game_name'].unique()
            plt.title(f"Reasoning Type by Turn - {agent}\n Games:\n{', '.join(games)}")
            plt.ylabel("Turn")
            plt.xlabel("Reasoning Type")
            plt.tight_layout()
            out_path = os.path.join(plot_dir, f"heatmap_{agent}_all_games.png")
            plt.savefig(out_path)
            plt.close()

        # Aggregate by agent across all games
        for agent, agent_df in self.df.groupby("agent_name"):
            if agent.startswith("random"):
                continue
            text = " ".join(agent_df['reasoning'].tolist())
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            games = agent_df['game_name'].unique()
            game_list = ", ".join(games)
            title = (
                f"Reasoning Word Cloud - {agent}\n"
                f"Games:{game_list}"
            )
            plt.title(title)
            plt.tight_layout()
            out_path = os.path.join(plot_dir, f"wordcloud_{agent}_all_games.png")
            plt.savefig(out_path)
            plt.close()

      #  pd.DataFrame(rows).to_csv(output_csv, index=False) # Uncomment to save the metrics to a CSV

    def plot_heatmaps_by_agent(self, output_dir: str = "plots") -> None:
        """Plot per-agent heatmaps and one aggregated heatmap across all games.

        Individual heatmaps are saved per agent-game pair. This also includes
        a general heatmap per agent showing all turns merged across all games.
        Useful for seeing broad reasoning type patterns.
        """
        os.makedirs(output_dir, exist_ok=True)
        for (agent, game), df_agent in self.df.groupby(["agent_name", "game_name"]):
            if agent.startswith("random"):
                continue
            pivot = df_agent.pivot_table(
                index="turn", columns="reasoning_type", values="agent_name",
                aggfunc="count", fill_value=0
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, cmap="YlGnBu", annot=True)
            plt.title(f"Reasoning Type by Turn - {agent} (Game: {game})")
            plt.ylabel("Turn")
            plt.xlabel("Reasoning Type")
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"heatmap_{agent}_{game}.png")
            plt.savefig(out_path)
            plt.close()

    def plot_wordclouds_by_agent(self, output_dir: str = "plots") -> None:
        """Plot per-agent word clouds and one aggregated word cloud across all games.

        Word clouds are created per agent-game pair and also aggregated per agent
        over all games. The full version helps summarize LLM behavior globally.
        """

        os.makedirs(output_dir, exist_ok=True)
        for (agent, game), agent_df in self.df.groupby(["agent_name", "game_name"]):
            if agent.startswith("random"):
                continue
            text = " ".join(agent_df['reasoning'].tolist())
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            title = f"Reasoning Word Cloud - {agent} (Game: {game})"
            plt.title(title)
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"wordcloud_{agent}_{game}.png")
            plt.savefig(out_path)
            plt.close()

    def plot_entropy_trendlines(self, output_dir: str = "plots") -> None:
        """Plot entropy over turns for each agent-game pair.

        This shows how each LLM agent's reasoning diversity evolves
        throughout the game, based on Shannon entropy of reasoning types.
        Higher entropy means more varied reasoning types.
        """
        os.makedirs(output_dir, exist_ok=True)
        for (agent, game), df_group in self.df.groupby(["agent_name", "game_name"]):
            if agent.startswith("random"):
                continue
            entropy_by_turn = (
                df_group.groupby("turn")["reasoning_type"]
                .apply(lambda s: -sum((v := s.value_counts(normalize=True)).apply(lambda p: p * np.log2(p))))
            )
            plt.figure()
            entropy_by_turn.plot(marker='o')
            plt.title(f"Reasoning Entropy by Turn - {agent} (Game: {game})")
            plt.xlabel("Turn")
            plt.ylabel("Entropy")
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"entropy_trend_{agent}_{game}.png")
            plt.savefig(out_path)
            plt.close()

    def plot_entropy_by_turn_across_agents(self, output_dir: str = "plots") -> None:
        """Plot entropy over turns across all agents per game.

        This compares how different LLM agents behave during the same game,
        highlighting agents that adapt their reasoning more flexibly.
        Useful to detect which models generalize or explore more.
        """
        os.makedirs(output_dir, exist_ok=True)
        for game, df_game in self.df.groupby("game_name"):
            plt.figure()
            for agent, df_agent in df_game.groupby("agent_name"):
                if agent.startswith("random"):
                    continue
                entropy_by_turn = (
                    df_agent.groupby("turn")["reasoning_type"]
                    .apply(lambda s: -sum((v := s.value_counts(normalize=True)).apply(lambda p: p * np.log2(p))))
                )
                entropy_by_turn.plot(label=agent)
            plt.title(f"Entropy by Turn Across Agents - {game}")
            plt.xlabel("Turn")
            plt.ylabel("Entropy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"entropy_by_turn_all_agents_{game}.png")
            plt.savefig(out_path)
            plt.close()

    def plot_avg_entropy_across_games(self, output_dir: str = "plots") -> None:
        """Plot average reasoning entropy over time across all games and agents.
        This reveals the general trend of reasoning diversity (entropy) per turn
        for all agents collectively, helping to understand how LLM reasoning
        evolves globally across gameplay.
        """

        os.makedirs(output_dir, exist_ok=True)
        plt.figure()
        df_all = self.df[~self.df['agent_name'].str.startswith("random")]
        avg_entropy = (
            df_all.groupby("turn")["reasoning_type"]
            .apply(lambda s: -sum((v := s.value_counts(normalize=True)).apply(lambda p: p * np.log2(p))))
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
    output_dir = '/p/project/ccstdl/cipolina-kun1/open_spiel_arena/results'
    latest_csv = LLMReasoningAnalyzer.find_latest_log("results")
    analyzer = LLMReasoningAnalyzer(latest_csv)

    # Choose one of the methods below to analyze the reasoning data
    analyzer.categorize_reasoning()
    #analyzer.categorize_with_llm(max_samples=50)  # or remove limit for full analysis

    analyzer.compute_metrics()
    analyzer.plot_heatmaps_by_agent()
    analyzer.plot_wordclouds_by_agent()
   # analyzer.plot_entropy_trendlines()
   # analyzer.plot_entropy_by_turn_across_agents()
    analyzer.plot_avg_entropy_across_games()
   # output_path = Path(__file__).resolve().parent.parent.parent / 'results'
    output_path = '/p/project/ccstdl/cipolina-kun1/open_spiel_arena/results/' #TODO: change this!
    #output_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer.save_output(output_path + 'augmented_reasoning_output.csv')
    print("Analysis completed successfully!.")
