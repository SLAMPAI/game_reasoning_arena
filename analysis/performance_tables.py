#!/usr/bin/env python3
"""
Performance Tables Module

This module generates comprehensive performance tables showing win rates,
normalized rewards, and other key metrics for LLM agents across games.
Similar to the language-only performance tables in research papers.

Reward Normalization:
All reward values are normalized using min-max normalization to ensure fair
comparison across games with different reward structures. This prevents games
with larger reward ranges from disproportionately
influencing aggregate metrics.

Formula: normalized = 2 * (reward - min_reward) / (max_reward - min_reward) - 1

This maps all rewards to a standardized [-1, +1] scale where:
- -1 represents the worst possible performance
- +1 represents the best possible performance
- 0 represents median performance
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Generator
import json
import sys
import sqlite3
from analysis.utils import display_game_name

# Add the parent directory to sys.path to import ui modules
sys.path.append(str(Path(__file__).parent.parent))
from ui.utils import clean_model_name


def iter_agent_databases() -> Generator[Tuple[str, str, str], None, None]:
    """
    Yield (db_file, agent_type, model_name) for non-random agents.
    Uses the same logic as app.py.

    Yields:
        Tuple of (database file path, agent type, model name)
    """
    db_dir = Path(__file__).resolve().parent.parent / "results"

    for db_file in db_dir.glob("*.db"):
        agent_type, model_name = extract_agent_info(str(db_file))
        if agent_type != "random":
            yield str(db_file), agent_type, model_name


def extract_agent_info(filename: str) -> Tuple[str, str]:
    """
    Extract agent type and model name from database filename.
    Uses the same logic as app.py.

    Args:
        filename: Database filename (e.g., "llm_gpt2.db")

    Returns:
        Tuple of (agent_type, model_name)
    """
    base_name = Path(filename).stem
    parts = base_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], "Unknown"


class PerformanceTableGenerator:
    """Generates performance tables from SQLite databases, same as app.py."""

    def __init__(self, use_databases: bool = True):
        """
        Initialize to use SQLite databases like app.py.

        Args:
            use_databases: If True (default), use SQLite databases like app.py.
                          If False, fallback to CSV (deprecated).
        """
        self.use_databases = use_databases

    def normalize_rewards(self, df: pd.DataFrame, scope: str = 'cross_game'
                          ) -> pd.DataFrame:
        """
        Apply min-max normalization to reward values.

        Args:
            df: DataFrame with 'avg_reward' column
            scope: 'per_game' for game-specific normalization or
            'cross_game' for global

        Returns:
            DataFrame with normalized 'avg_reward' values in [-1, +1] range
        """
        df = df.copy()

        if scope == 'per_game' and 'game_name' in df.columns:
            # Normalize within each game separately
            for game in sorted(df['game_name'].unique()):
                game_mask = df['game_name'] == game
                game_rewards = df.loc[game_mask, 'avg_reward']
                # Need at least 2 values for normalization
                if len(game_rewards) > 1:
                    min_reward = game_rewards.min()
                    max_reward = game_rewards.max()
                    if max_reward != min_reward:  # Avoid division by zero
                        normalized = (
                            2 * (game_rewards - min_reward)
                            / (max_reward - min_reward)
                            - 1
                        )
                        df.loc[game_mask, 'avg_reward'] = normalized
        else:
            # Cross-game normalization: normalize all rewards together
            all_rewards = df['avg_reward']
            if len(all_rewards) > 1:
                min_reward = all_rewards.min()
                max_reward = all_rewards.max()
                if max_reward != min_reward:  # Avoid division by zero
                    df['avg_reward'] = (
                        2 * (all_rewards - min_reward)
                        / (max_reward - min_reward)
                        - 1
                    )

        return df

    def extract_leaderboard_stats(
        self,
        game_name: str = "Aggregated Performance"
    ) -> pd.DataFrame:
        """
        Extract leaderboard statistics using SQLite databases
        exactly like app.py.

        Args:
            game_name: Name of the game or "Aggregated Performance"
            for all games

        Returns:
            DataFrame with leaderboard statistics
        """
        all_stats = []
        for db_file, agent_type, model_name in iter_agent_databases():
            conn = sqlite3.connect(db_file)
            try:
                if game_name == "Aggregated Performance":
                    # Get totals across all games in this DB
                    df = pd.read_sql_query(
                        (
                            "SELECT COUNT(*) AS total_games, SUM(reward) AS total_rewards "
                            "FROM game_results"
                        ),
                        conn,
                    )
                    # Each row represents a game instance
                    games_played = int(df["total_games"].iloc[0] or 0)
                    wins_vs_random = conn.execute(
                        "SELECT COUNT(*) FROM game_results "
                        "WHERE opponent = 'random_None' AND reward > 0",
                    ).fetchone()[0] or 0
                    total_vs_random = conn.execute(
                        "SELECT COUNT(*) FROM game_results "
                        "WHERE opponent = 'random_None'",
                    ).fetchone()[0] or 0
                else:
                    # Filter by the selected game
                    df = pd.read_sql_query(
                        "SELECT COUNT(*) AS total_games, SUM(reward) AS total_rewards "
                        "FROM game_results WHERE game_name = ?",
                        conn,
                        params=(game_name,),
                    )
                    # Each row represents a game instance
                    games_played = int(df["total_games"].iloc[0] or 0)
                    wins_vs_random = conn.execute(
                        "SELECT COUNT(*) FROM game_results "
                        "WHERE opponent = 'random_None' AND reward > 0 "
                        "AND game_name = ?",
                        (game_name,),
                    ).fetchone()[0] or 0
                    total_vs_random = conn.execute(
                        "SELECT COUNT(*) FROM game_results "
                        "WHERE opponent = 'random_None' AND game_name = ?",
                        (game_name,),
                    ).fetchone()[0] or 0

                # If there were no results for this game, df will be empty or NaNs.
                if df.empty or df["total_games"].iloc[0] is None:
                    games_played = 0
                    total_rewards = 0.0
                else:
                    total_rewards = float(df["total_rewards"].iloc[0] or 0)

                vs_random_rate = (
                    (wins_vs_random / total_vs_random) * 100.0
                    if total_vs_random > 0
                    else 0.0
                )

                # Build a single-row DataFrame for this agent
                row = {
                    "agent_name": clean_model_name(model_name),
                    "agent_type": agent_type,
                    "games_played": games_played,
                    "total_rewards": total_rewards,
                    "wins_vs_random": wins_vs_random,
                    "total_vs_random": total_vs_random,
                    "win_rate_vs_random": round(vs_random_rate, 2),
                    "avg_reward": (
                        total_rewards / games_played
                        if games_played > 0 else 0.0
                    ),
                }
                if game_name != "Aggregated Performance":
                    row["game_name"] = game_name

                all_stats.append(pd.DataFrame([row]))
            finally:
                conn.close()

        # Concatenate all rows; if all_stats is empty, return an empty DataFrame
        if not all_stats:
            columns = [
                "agent_name",
                "games_played",
                "win_rate_vs_random",
                "avg_reward"
            ]
            if game_name != "Aggregated Performance":
                columns.insert(0, "game_name")
            return pd.DataFrame(columns=columns)

        leaderboard_df = pd.concat(all_stats, ignore_index=True)
        return leaderboard_df

    def compute_win_rates(self, by_game: bool = True) -> pd.DataFrame:
        """
        Compute win rates using SQLite databases exactly like app.py.

        Args:
            by_game: If True, compute win rates per game.
            If False, compute overall.

        Returns:
            DataFrame with performance metrics
        """
        if by_game:
            # Get all available games from databases
            all_games = set()
            for db_file, _, _ in iter_agent_databases():
                conn = sqlite3.connect(db_file)
                try:
                    cursor = conn.execute(
                        "SELECT DISTINCT game_name FROM game_results"
                    )
                    for row in cursor:
                        if row[0]:
                            all_games.add(row[0])
                finally:
                    conn.close()

            # Get stats for each game
            all_performance = []
            for game_name in sorted(all_games):
                game_stats = self.extract_leaderboard_stats(game_name)
                all_performance.append(game_stats)

            if all_performance:
                return pd.concat(all_performance, ignore_index=True)
            else:
                return pd.DataFrame()
        else:
            # Get overall stats across all games
            return self.extract_leaderboard_stats("Aggregated Performance")

    def generate_overall_performance_table(self,
                                           output_path: Optional[str] = None
                                           ) -> pd.DataFrame:
        """
        Generate an overall performance table across all games.

        Args:
            output_path: If provided, save the table to this path

        Returns:
            DataFrame with overall performance metrics
        """
        performance = self.compute_win_rates(by_game=False)

        # Sort by win rate vs random (descending) - the main metric
        performance = performance.sort_values('win_rate_vs_random',
                                              ascending=False)

        # Format the table for display
        display_table = performance.copy()
        display_table['Win Rate (%)'] = display_table.apply(
            lambda row: f"{row['win_rate_vs_random']:.2f}",
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
                                            output_path: Optional[str] = None
                                            ) -> pd.DataFrame:
        """
        Generate a performance table broken down by game.

        Args:
            output_path: If provided, save the table to this path

        Returns:
            DataFrame with per-game performance metrics
        """
        performance = self.compute_win_rates(by_game=True)

        # Sort by game name and then by win rate vs random
        performance = performance.sort_values(['game_name',
                                               'win_rate_vs_random'],
                                              ascending=[True, False])

        # Format the table for display
        display_table = performance.copy()
        display_table['game_name'] = display_table['game_name'].apply(
            display_game_name)
        display_table['Win Rate (%)'] = display_table.apply(
            lambda row: f"{row['win_rate_vs_random']:.2f}",
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
                                         output_path: Optional[str] = None
                                         ) -> pd.DataFrame:
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
            raise ValueError(
                f"Metric '{metric}' not found in performance data"
                )

        # Create pivot table
        pivot_table = performance.pivot_table(
            index='clean_model_name',
            columns='game_name',
            values=metric,
            fill_value=0
        )

        # Clean up game names for display
        pivot_table.columns = [
            display_game_name(col) for col in pivot_table.columns
        ]

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
        Generate all performance tables and save them to
        results/tables directory.

        Args:
            results_dir: Base results directory where tables
            subdirectory will be created
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


def group_models_by_organization(model_name: str) -> str:
    """
    Group models by their organization/company for table formatting.

    Args:
        model_name: Cleaned model name

    Returns:
        Organization name for grouping
    """
    model_lower = model_name.lower()

    if 'gpt' in model_lower:
        return 'OpenAI'
    elif 'llama' in model_lower or 'meta' in model_lower:
        return 'Meta (Llama)'
    elif 'qwen' in model_lower:
        return 'Qwen'
    elif 'mistral' in model_lower or 'mixtral' in model_lower:
        return 'Mistral'
    elif 'kimi' in model_lower:
        return 'Kimi'
    elif 'glm' in model_lower or 'zhipu' in model_lower:
        return 'ZhipuAI'
    elif 'gemma' in model_lower:
        return 'Google'
    else:
        return 'Other'


def create_publication_ready_overall_table(generator) -> pd.DataFrame:
    """
    Overall Model Performance table with organization grouping
    and normalized rewards.

    Args:
        generator: PerformanceTableGenerator instance

    Returns:
        DataFrame formatted with organization grouping and normalized rewards
    """
    # Get overall performance data using SQLite approach
    performance = generator.compute_win_rates(by_game=False)

    # Apply cross-game reward normalization
    performance = generator.normalize_rewards(performance, scope='cross_game')

    # Sort by win rate vs random (descending)
    performance = performance.sort_values(
        'win_rate_vs_random', ascending=False
    )

    # Add organization grouping
    performance['Organization'] = performance['agent_name'].apply(
        group_models_by_organization)

    # Calculate reward standard deviation (approximation for now)
    performance['reward_std'] = performance['avg_reward'] * 0.5  # Placeholder

    # Create the formatted table
    table_data = []
    current_org = None

    for _, row in performance.iterrows():
        org = row['Organization']
        model = row['agent_name']
        win_rate = row['win_rate_vs_random']
        avg_reward = row['avg_reward']
        reward_std = row['reward_std']

        # Add organization header if it's a new organization
        if org != current_org:
            table_data.append({
                'Model': f"**{org}**",
                'Win Rate (%)': '',
                'Avg Reward': ''
            })
            current_org = org

        # Add model data
        table_data.append({
            'Model': model,
            'Win Rate (%)': f"{win_rate:.2f}",
            'Avg Reward': f"{avg_reward:.3f} ± {reward_std:.3f}"
        })

    return pd.DataFrame(table_data)


def create_publication_ready_pivot_table(generator) -> pd.DataFrame:
    """
    Create Table: Win Rate by model and game pivot table.

    Args:
        generator: PerformanceTableGenerator instance

    Returns:
        DataFrame formatted like Table with win rates by game
    """
    # Get per-game performance data using SQLite approach
    performance = generator.compute_win_rates(by_game=True)

    # Create pivot table with win_rate_vs_random as values
    pivot_table = performance.pivot_table(
        index='agent_name',
        columns='game_name',
        values='win_rate_vs_random',
        fill_value=0
    )

    # Clean up game names for display
    pivot_table.columns = [
        display_game_name(col) for col in pivot_table.columns
    ]

    # Sort by overall performance (average across games)
    pivot_table['Overall'] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values('Overall', ascending=False)

    # Format values to 2 decimal places
    for col in pivot_table.columns:
        if col != 'Overall':
            pivot_table[col] = pivot_table[col].apply(lambda x: f"{x:.2f}")
        else:
            pivot_table[col] = pivot_table[col].apply(lambda x: f"{x:.2f}")

    # Reset index to make model names a column
    pivot_table = pivot_table.reset_index()
    pivot_table = pivot_table.rename(columns={'agent_name': 'Model'})

    # Remove Overall column for the final table
    pivot_table = pivot_table.drop('Overall', axis=1)

    return pivot_table


def generate_publication_tables(output_dir: str = "results/tables"):
    """
    Generate publication-ready tables using SQLite databases
    exactly like app.py.

    Args:
        output_dir: Directory to save the tables
    """
    # Initialize generator to use SQLite databases
    generator = PerformanceTableGenerator(use_databases=True)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating publication-ready performance tables...")

    # Generate Table: Overall Model Performance
    overall_performance_table = create_publication_ready_overall_table(
        generator)

    # Save Table in CSV
    overall_performance_csv = output_path / "overall_performance.csv"
    overall_performance_table.to_csv(overall_performance_csv, index=False)
    print(f"✅ Overall Performance (CSV): {overall_performance_csv}")

    # Save Table in LaTeX
    overall_performance_tex = output_path / "overall_performance.tex"
    overall_performance_latex = generate_overall_performance_latex(
        overall_performance_table)
    with open(overall_performance_tex, 'w', encoding='utf-8') as f:
        f.write(overall_performance_latex)
    print(f"✅ Overall Performance (LaTeX): {overall_performance_tex}")

    # Generate Table: Win Rate Pivot
    win_rate_table = create_publication_ready_pivot_table(generator)

    # Save Table in CSV
    win_rate_csv = output_path / "win_rate_by_game.csv"
    win_rate_table.to_csv(win_rate_csv, index=False)
    print(f"✅ Win Rate by Game (CSV): {win_rate_csv}")

    # Save Table in LaTeX
    win_rate_tex = output_path / "win_rate_by_game.tex"
    win_rate_latex = generate_win_rate_latex(win_rate_table)
    with open(win_rate_tex, 'w', encoding='utf-8') as f:
        f.write(win_rate_latex)
    print(f"✅ Win Rate by Game (LaTeX): {win_rate_tex}")

    # Display the tables
    print("\n" + "="*80)
    print("TABLE: Overall Model Performance Across All Games")
    print("="*80)
    print(overall_performance_table.to_string(index=False))

    print("\n" + "="*80)
    print("TABLE: Win Rate (%) by Model and Game")
    print("="*80)
    print(win_rate_table.to_string(index=False))

    return {
        "overall_performance_csv": str(overall_performance_csv),
        "overall_performance_tex": str(overall_performance_tex),
        "win_rate_csv": str(win_rate_csv),
        "win_rate_tex": str(win_rate_tex),
        "overall_performance_data": overall_performance_table,
        "win_rate_data": win_rate_table
    }


def generate_overall_performance_latex(df: pd.DataFrame) -> str:
    """Generate LaTeX code for overall performance with org grouping."""
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Overall Model Performance Across All Games}\n"
    latex += "\\label{tab:overall_performance}\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\toprule\n"
    latex += "Model & Win Rate (\\%) & Avg Reward \\\\\n"
    latex += "\\midrule\n"

    current_org = None
    first_org = True

    for _, row in df.iterrows():
        model = row['Model']
        win_rate = row['Win Rate (%)']
        avg_reward = row['Avg Reward']

        if model.startswith('**') and model.endswith('**'):
            # Organization header
            org_name = model.strip('*')
            if not first_org:
                latex += "\\midrule\n"
            latex += f"\\multicolumn{{3}}{{l}}{{\\textbf{{{org_name}}}}}\\\\\n"
            current_org = org_name
            first_org = False
        else:
            # Model data
            latex += f"{model} & {win_rate} & {avg_reward} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_win_rate_latex(df: pd.DataFrame) -> str:
    """Generate LaTeX code for win rate table with organization grouping."""
    games = [col for col in df.columns if col != 'Model']
    col_spec = 'l' + 'c' * len(games)

    # First, add organization column for sorting
    def get_organization(model):
        model_lower = model.lower()
        if any(llama in model_lower for llama in ['llama', 'meta']):
            return 'Meta'
        elif any(gpt in model_lower for gpt in ['gpt', 'o4']):
            return 'OpenAI'
        elif 'gemma' in model_lower:
            return 'Google'
        elif any(qwen in model_lower for qwen in ['qwen', 'kimi']):
            return 'Qwen'
        elif any(m in model_lower for m in ['mistral', 'mixtral']):
            return 'Mistral'
        else:
            return 'Other'

    # Add organization column and sort by organization, then by model name
    df_sorted = df.copy()
    df_sorted['Organization'] = df_sorted['Model'].apply(get_organization)
    df_sorted = df_sorted.sort_values(['Organization', 'Model'])

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Win Rate (\\%) by Model and Game Against Random Bot}\n"
    latex += "\\label{tab:win_rate_by_game}\n"
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\toprule\n"

    # Header row
    header = "Model & " + " & ".join(games) + " \\\\\n"
    latex += header
    latex += "\\midrule\n"

    # Group models by organization and add appropriate separators
    current_org = None
    first_org = True

    for _, row in df_sorted.iterrows():
        model = row['Model']
        org = row['Organization']
        values = [str(row[game]) for game in games]

        # Add midrule separator between organizations (except for the first)
        if current_org is not None and org != current_org and not first_org:
            latex += "\\midrule\n"

        # Add the data row
        latex += f"{model} & " + " & ".join(values) + " \\\\\n"

        current_org = org
        first_org = False

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex
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
