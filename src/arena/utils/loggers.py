'''
SQLiteLogger class for logging agent decisions and game results into a SQLite database.
This class provides methods to log moves, rewards, and game results, as well as retrieve this data for analysis.
'''

import sqlite3
import os
from datetime import datetime
import pandas as pd

class SQLiteLogger:
    """Handles logging of agent decisions into a structured SQLite database."""

    def __init__(self, agent_type: str, model_name: str):
        """
        Initializes the logger with a unique database file for each agent type and model.

        Args:
            agent_type (str): The type of agent (e.g., "llm", "random", "human").
            model_name (str): The model name (for LLMs), or "None" for random/human.
        """
        # Sanitize model name for use in filename
        sanitized_model_name = model_name.replace('-', '_').replace('/', '_').replace('\\', '_')
        self.db_path = f"results/{agent_type}_{sanitized_model_name}.db"
        self.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        self._create_database()

    def _create_database(self):
        """
        Creates necessary tables if they do not exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create 'moves' table (logs every move taken in a game)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                turn INTEGER,
                action INTEGER,
                reasoning TEXT,
                opponent TEXT,
                generation_time REAL,
                agent_type TEXT,
                agent_model TEXT,
                timestamp TEXT,
                run_id TEXT,
                seed INTEGER
            )
        """)

        # Create 'rewards' table (stores final rewards for each episode)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                reward REAL,
                timestamp TEXT,
                run_id TEXT
            )
        """)

        # Create 'illegal_moves' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS illegal_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_name TEXT,
            episode INTEGER,
            turn INTEGER,
            agent_id INTEGER,
            illegal_action INTEGER,
            reason TEXT,
            board_state TEXT,
            timestamp TEXT,
            run_id TEXT
        )
    """)

        # Create 'game_results' table (stores final results of games played)
        # TODO: add opponents to this table !
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                status TEXT,
                reward REAL,
                timestamp TEXT,
                run_id TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log_game_result(self, game_name: str, episode: int, status: str, reward: float):
        """
        Logs the final result of a game into table 'game_results'.

        Args:
            game_name (str): Name of the game played.
            episode (int): Episode number.
            status (str): 'terminated' or 'truncated'.
            reward (float): Final reward the agent received.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO game_results (game_name, episode, status, reward, timestamp, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            game_name, episode, status, reward, datetime.now().isoformat(),self.run_id
        ))

        conn.commit()
        conn.close()

    def log_move(self, game_name: str, episode: int, turn: int, action: int,
                 reasoning: str, opponent: str, generation_time: float,
                 agent_type: str, agent_model: str, seed: int):
        """
        Logs an agent's move into the table 'moves'

        Args:
            game_name (str): Name of the game.
            episode (int): Episode number.
            turn (int): Turn number in the episode.
            action (int): The action taken by the agent.
            reasoning (str): The agent's reasoning for the action.
            opponent (str): The opponent's identifier.
            generation_time (float): Time taken to generate the move.
            agent_type (str): Type of agent (e.g., "llm", "random", "human").
            agent_model (str): Specific model name for LLMs.
            seed (int): Random seed used for the episode.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO moves (game_name, episode, turn, action, reasoning,
                            opponent, generation_time, agent_type, agent_model, timestamp, run_id,seed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
        """, (
            game_name, episode, turn, action, reasoning,
            opponent, generation_time, agent_type, agent_model,
            datetime.now().isoformat(), self.run_id, seed
        ))

        conn.commit()
        conn.close()

    def log_rewards(self, game_name: str, episode: int, reward: float):
        """
        Logs the final reward for an agent after an episode.

        Args:
            game_name (str): Name of the game.
            episode (int): Episode number.
            reward (float): Final reward received.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO rewards (game_name, episode, reward, timestamp, run_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            game_name, episode, reward, datetime.now().isoformat(), self.run_id
        ))

        conn.commit()
        conn.close()

    def log_illegal_move(self, game_name: str, episode: int, turn: int, agent_id: int, illegal_action: int, reason: str,board_state: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO illegal_moves (
                game_name, episode, turn, agent_id, illegal_action,
                reason, board_state, timestamp, run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_name, episode, turn, agent_id, illegal_action,
            reason, board_state, datetime.now().isoformat(), self.run_id
        ))
        conn.commit()
        conn.close()

    def get_agent_moves(self):
        """
        Retrieves all actions taken by the agent.

        Returns:
            List of tuples containing game_name, episode, turn, action,
            reasoning, opponent, generation_time, and timestamp.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT game_name, episode, turn, action, reasoning, opponent, generation_time, timestamp, run_id
            FROM moves
            ORDER BY game_name, episode, turn
        """)

        moves = cursor.fetchall()
        conn.close()
        return moves

    def get_agent_rewards(self):
        """
        Retrieves all rewards logged for the agent.

        Returns:
            List of tuples containing game_name, episode, reward, and timestamp.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT game_name, episode, reward, timestamp, run_id
            FROM rewards
            ORDER BY game_name, episode
        """)

        rewards = cursor.fetchall()
        conn.close()
        return rewards

    def get_game_results(self, game_name: str):
        """
        Retrieves all game results for a specific game.

        Args:
            game_name (str): Name of the game to fetch results for.

        Returns:
            List of tuples containing game_name, episode, status, reward, and timestamp.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT game_name, episode, status, reward, timestamp, run_id
            FROM game_results
            WHERE game_name = ?
            ORDER BY episode
        """, (game_name,))

        game_results = cursor.fetchall()
        conn.close()
        return game_results

    def get_agent_results(self):
        """
        Fetches all game results for the agent's database.

        Returns:
            Pandas DataFrame containing game_name, episode, status, reward, and timestamp.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT game_name, episode, status, reward, timestamp, run_id
            FROM game_results
            ORDER BY game_name, episode
        """, conn)
        conn.close()
        return df

def get_run_ids(self) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT run_id FROM moves ORDER BY run_id")
        run_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return run_ids

def get_moves_by_run(self, run_id: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT *
            FROM moves
            WHERE run_id = ?
            ORDER BY game_name, episode, turn
        """, conn, params=(run_id,))
        conn.close()
        return df

def get_game_results_by_run(self, run_id: str) -> pd.DataFrame:
    """
    Retrieves all game results for a specific run ID.

    Args:
        run_id (str): Unique run identifier.

    Returns:
        pd.DataFrame: Game results for that run.
    """
    conn = sqlite3.connect(self.db_path)
    df = pd.read_sql_query("""
        SELECT *
        FROM game_results
        WHERE run_id = ?
        ORDER BY game_name, episode
    """, conn, params=(run_id,))
    conn.close()
    return df

def list_all_runs(self) -> pd.DataFrame:
    """
    Lists all runs stored in the DB with metadata.

    Returns:
        pd.DataFrame: One row per run_id with timestamp and game count.
    """
    conn = sqlite3.connect(self.db_path)
    df = pd.read_sql_query("""
        SELECT run_id, MIN(timestamp) AS start_time, COUNT(DISTINCT episode) AS games_played
        FROM game_results
        GROUP BY run_id
        ORDER BY start_time
    """, conn)
    conn.close()
    return df
