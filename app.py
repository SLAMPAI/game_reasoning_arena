# # !/usr/bin/env python3
"""
Board Game Arena - HuggingFace Gradio App

A Gradio interface for playing board games with LLM agents and analyzing
their performance. Optimized for deployment on HuggingFace Spaces.

Features:
- Play games against LLM agents (free HuggingFace models supported)
- View leaderboards and performance metrics
- Analyze LLM reasoning and decision-making
- Interactive visualizations and charts
"""

import sqlite3
import pandas as pd
import gradio as gr
from pathlib import Path
from typing import List
import importlib.util
from transformers import pipeline

# Add the src directory to Python path
import sys
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Try importing from the board game arena package
# Prioritize src.board_game_arena path for HuggingFace Spaces deployment

from src.board_game_arena.arena.games.registry import registry as games_registry


# ============================================================================
# INITIALIZATION - Try to import game modules if available
# ============================================================================

# Default game types based on database entries
GAMES_REGISTRY = {
    "tic_tac_toe": "Tic Tac Toe",
    "kuhn_poker": "Kuhn Poker",
    "connect_four": "Connect Four"
}

# Local LLM models using transformers pipeline
LOCAL_LLM_MODELS = {
    "gpt2": {
        "model_id": "gpt2",
        "prompt_template": """You are playing a game. Current state: {game_state}
You are player {player}. Valid moves: {valid_moves}

Think step by step:
1. Analyze the current game state
2. Consider your options
3. Choose the best move

Reasoning: [Explain your thinking here]
Move: [Your chosen move number]""",
        "max_new_tokens": 50
    },
    "google/flan-t5-small": {
        "model_id": "google/flan-t5-small",
        "prompt_template": """Game state: {game_state}
Player: {player}
Valid moves: {valid_moves}

Explain your reasoning and choose a move:
Reasoning: [Why this move is good]
Move: [Your move number]""",
        "max_new_tokens": 30
    }
}


print("Loading local LLM models...")

# Initialize pipelines without device_map for better compatibility
llms = {}
for model_name in LOCAL_LLM_MODELS.keys():
    try:
        print(f"Loading {model_name}...")

        # Use the correct pipeline type for each model
        if "flan-t5" in model_name.lower():
            # T5 models use text2text-generation pipeline
            llms[model_name] = pipeline("text2text-generation", model=model_name)
        else:
            # GPT-2 and similar models use text-generation pipeline
            llms[model_name] = pipeline("text-generation", model=model_name)

        print(f"✓ Successfully loaded {model_name}")
    except Exception as e:
        print(f"✗ Failed to load {model_name}: {e}")

print(f"✓ Loaded {len(llms)} local LLM models")

# Try to import board game arena modules
GAME_REGISTRY_AVAILABLE = False
ENV_INITIALIZER_AVAILABLE = False

def is_package_installed(package_name):
    """Check if a Python package is installed."""
    try:
        importlib.util.find_spec(package_name)
        return True
    except ImportError:
        return False


# Update registries if successful
GAMES_REGISTRY = {name: cls for name, cls in games_registry._registry.items()}
GAME_REGISTRY_AVAILABLE = True
ENV_INITIALIZER_AVAILABLE = True
print("✅ Successfully imported full board game arena - GAMES ARE PLAYABLE!")

print(f"Environment initializer available: {ENV_INITIALIZER_AVAILABLE}")

# Directory to store SQLite results
db_dir = Path("results")


# ============================================================================
# HUGGINGFACE API HELPER FUNCTIONS
# ============================================================================

def create_local_llm_player(model_id):
    """Create a player class that uses local transformers pipeline."""
    try:
        from board_game_arena.arena.player import Player
    except ImportError:
        try:
            from src.board_game_arena.arena.player import Player
        except ImportError:
            # Define a minimal Player base class if we can't import
            class Player:
                def __init__(self, name="Player"):
                    self.name = name

                def make_move(self, game_state):
                    raise NotImplementedError("Base player can't make moves")

    class LocalLLMPlayer(Player):
        def __init__(self, name=None):
            name = name or f"Local-{model_id}"
            super().__init__(name=name)
            self.model_id = model_id
            self.model_info = LOCAL_LLM_MODELS[model_id]
            self.pipeline = llms.get(model_id)

        def make_move(self, game_state):
            if not self.pipeline:
                # Fallback to random if model not loaded
                return game_state.legal_actions()[0]

            # Create a prompt for the model based on the game state
            prompt = self.model_info["prompt_template"].format(
                game_state=str(game_state),
                valid_moves=", ".join(map(str, game_state.legal_actions())),
                player=game_state.current_player(),
            )

            try:
                # Generate response using local pipeline
                max_tokens = self.model_info.get("max_new_tokens", 10)

                # Handle potential tokenizer issues
                try:
                    pad_token_id = self.pipeline.tokenizer.eos_token_id
                except AttributeError:
                    pad_token_id = None

                # Check if this is a text2text-generation pipeline (like T5)
                is_text2text = hasattr(self.pipeline, 'task') and self.pipeline.task == 'text2text-generation'

                if is_text2text:
                    # For T5/Flan-T5 models, use text2text-generation
                    response = self.pipeline(
                        prompt,
                        max_length=len(prompt.split()) + max_tokens,
                        do_sample=True,
                        temperature=0.3
                    )
                else:
                    # For GPT-2 and similar models, use text-generation
                    if pad_token_id is not None:
                        response = self.pipeline(
                            prompt,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=0.3,
                            pad_token_id=pad_token_id
                        )
                    else:
                        response = self.pipeline(
                            prompt,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=0.3
                        )

                # Extract text from response
                if isinstance(response, list) and len(response) > 0:
                    generated_text = response[0].get('generated_text', '')
                    # Remove the original prompt from the response
                    response_text = generated_text.replace(prompt, '').strip()
                else:
                    response_text = str(response)

                # Parse reasoning and move from response
                reasoning = ""
                move = None

                # Try to extract reasoning and move
                lines = response_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("Reasoning:"):
                        reasoning = line.replace("Reasoning:", "").strip()
                    elif line.startswith("Move:"):
                        move_text = line.replace("Move:", "").strip()
                        # Extract number from move text
                        import re
                        move_match = re.search(r'\d+', move_text)
                        if move_match:
                            try:
                                move = int(move_match.group())
                            except ValueError:
                                pass

                # Store reasoning for display (this is a hack for demo purposes)
                if hasattr(self, '_last_reasoning'):
                    self._last_reasoning = reasoning
                else:
                    self._last_reasoning = reasoning

                # Validate move is legal
                valid_moves = game_state.legal_actions()
                if move is not None and move in valid_moves:
                    # Store both action and reasoning for full simulation
                    self._last_action = move
                    self._last_reasoning = reasoning
                    return move

                # Fallback: try to find any valid move number in the response
                for valid_move in valid_moves:
                    if str(valid_move) in response_text:
                        self._last_action = valid_move
                        self._last_reasoning = reasoning
                        return valid_move

                fallback_move = valid_moves[0]
                self._last_action = fallback_move
                if reasoning:
                    self._last_reasoning = reasoning
                else:
                    self._last_reasoning = "Using fallback move"
                return fallback_move

            except Exception as e:
                print(f"Error generating move: {e}")
                fallback_move = game_state.legal_actions()[0]
                self._last_action = fallback_move
                self._last_reasoning = f"Error occurred: {str(e)}"
                return fallback_move

        def compute_action(self, observation):
            """Compute action for full game simulation."""
            action = self.make_move(observation)
            reasoning = getattr(self, '_last_reasoning',
                                'No reasoning available')
            return {
                "action": action,
                "reasoning": reasoning
            }

    return LocalLLMPlayer


def query_local_llm(model_id, prompt, max_length=10):
    """Query a local LLM using transformers pipeline."""
    if model_id not in llms:
        return {"response": f"Model {model_id} not available", "reasoning": ""}

    try:
        pipeline_obj = llms[model_id]

        # Handle potential tokenizer issues
        try:
            pad_token_id = pipeline_obj.tokenizer.eos_token_id
        except AttributeError:
            pad_token_id = None

        # Check if this is a text2text-generation pipeline (like T5)
        is_text2text = hasattr(pipeline_obj, 'task') and pipeline_obj.task == 'text2text-generation'

        if is_text2text:
            # For T5/Flan-T5 models, use text2text-generation
            response = pipeline_obj(
                prompt,
                max_length=len(prompt.split()) + max_length,
                do_sample=True,
                temperature=0.3
            )
        else:
            # For GPT-2 and similar models, use text-generation
            if pad_token_id is not None:
                response = pipeline_obj(
                    prompt,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=pad_token_id
                )
            else:
                response = pipeline_obj(
                    prompt,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.3
                )

        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get('generated_text', '')
            # Remove the original prompt from the response
            response_text = generated_text.replace(prompt, '').strip()
        else:
            response_text = str(response)

        # Parse reasoning and move from response
        reasoning = ""
        move = ""

        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif line.startswith("Move:"):
                move = line.replace("Move:", "").strip()

        return {
            "response": response_text,
            "reasoning": reasoning or "No reasoning provided",
            "move": move or "No move specified"
        }

    except Exception as e:
        return {"response": f"Error querying model: {str(e)}", "reasoning": ""}
# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================


def find_or_download_db():
    """Check if SQLite .db files exist; if not, attempt to download from
    cloud storage."""
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)

    db_files = list(db_dir.glob("*.db"))

    # Ensure the random bot database exists
    random_db_path = db_dir / "random_None.db"
    if not random_db_path.exists():
        try:
            # Create an empty SQLite database if it doesn't exist
            conn = sqlite3.connect(str(random_db_path))
            conn.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY,
                    game_name TEXT,
                    player1 TEXT,
                    player2 TEXT,
                    winner INTEGER,
                    timestamp TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            raise FileNotFoundError(
                "Please upload results for the random agent in a file named "
                "'random_None.db'.")

    db_files = list(db_dir.glob("*.db"))
    return [str(f) for f in db_files]


def extract_agent_info(filename: str):
    """Extract agent type and model name from the filename."""
    base_name = Path(filename).stem
    parts = base_name.split("_", 1)
    if len(parts) == 2:
        agent_type, model_name = parts
    else:
        agent_type, model_name = parts[0], "Unknown"
    return agent_type, model_name


def get_available_games(include_aggregated=True) -> List[str]:
    """Extracts all unique game names from all SQLite databases.
    Includes 'Aggregated Performance' only when required."""
    db_files = find_or_download_db()
    game_names = set()

    # First, add games from database files
    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        try:
            query = "SELECT DISTINCT game_name FROM moves"
            df = pd.read_sql_query(query, conn)
            game_names.update(df["game_name"].tolist())
        except Exception:
            pass  # Ignore errors if table doesn't exist
        finally:
            conn.close()

    # Always include the default games from GAMES_REGISTRY
    game_names.update(GAMES_REGISTRY.keys())

    game_list = sorted(game_names) if game_names else ["No Games Found"]
    if include_aggregated:
        # Ensure 'Aggregated Performance' is always first
        game_list.insert(0, "Aggregated Performance")
    return game_list


def extract_illegal_moves_summary() -> pd.DataFrame:
    """Extracts the number of illegal moves made by each LLM agent.

    Returns:
        pd.DataFrame: DataFrame with columns [agent_name, illegal_moves].
    """
    db_files = find_or_download_db()
    summary = []
    for db_file in db_files:
        agent_type, model_name = extract_agent_info(db_file)
        if agent_type == "random":
            continue  # Skip the random agent from this analysis
        conn = sqlite3.connect(db_file)
        try:
            # Count number of illegal moves from the illegal_moves table
            query = "SELECT COUNT(*) AS illegal_moves FROM illegal_moves"
            df = pd.read_sql_query(query, conn)
            count = int(df["illegal_moves"].iloc[0]) if not df.empty else 0
        except Exception:
            count = 0  # If the table does not exist or error occurs
        summary.append({"agent_name": model_name, "illegal_moves": count})
        conn.close()
    return pd.DataFrame(summary)


# ============================================================================
# HELPER FUNCTIONS FOR PLAYER CONFIGURATION
# ============================================================================

def setup_player_config(player_type: str, player_model: str,
                       player_id: str) -> dict:
    """Configure a single player for the game environment."""
    if player_type == "random_bot":
        return {"type": "random"}
    elif player_type.startswith("llm_"):
        # Extract model ID from the player type (format: llm_model_id)
        model_id = player_type[4:]  # Remove 'llm_' prefix
        if model_id in LOCAL_LLM_MODELS:
            return {
                "type": "custom",
                "name": f"Local-{model_id}",
                "player_class": create_local_llm_player(model_id)
            }
    elif player_type == "llm" and player_model in LOCAL_LLM_MODELS:
        return {
            "type": "custom",
            "name": f"Local-{player_model}",
            "player_class": create_local_llm_player(player_model)
        }

    return {"type": "random"}  # Fallback


def run_full_game_simulation(game_name: str, config: dict) -> str:
    """Run a complete game simulation using the board game arena."""
    try:
        # Try multiple import paths
        try:
            from src.board_game_arena.arena.games.registry import registry
            from src.board_game_arena.arena.agents.policy_manager import initialize_policies
            from src.board_game_arena.arena.utils.seeding import set_seed
        except ImportError:
            raise ImportError("Board game arena modules not available")

        # Set seed for reproducibility
        set_seed(config["seed"])

        # Create environment
        env_config_full = {
            "env_config": config["env_configs"][0],
            **config
        }
        env = registry.make_env(game_name, env_config_full)

        # Initialize agent policies
        policies_dict = initialize_policies(config, game_name, config["seed"])
        player_to_agent = {0: policies_dict["policy_0"], 1: policies_dict["policy_1"]}

        game_states = []

        # Run episodes
        for episode in range(config["num_episodes"]):
            game_states.append(f"\n🎯 Episode {episode + 1}")
            game_states.append("=" * 30)

            observation_dict, _ = env.reset(seed=config["seed"] + episode)
            episode_rewards = {0: 0, 1: 0}
            terminated = truncated = False
            step_count = 0

            while not (terminated or truncated):
                step_count += 1
                game_states.append(f"\n📋 Step {step_count}")

                # Show board state
                board = env.render_board(0)
                game_states.append("Current board:")
                game_states.append(board)

                # Determine active players
                if env.state.is_simultaneous_node():
                    active_players = list(player_to_agent.keys())
                else:
                    current_player = env.state.current_player()
                    active_players = [current_player]
                    game_states.append(f"Player {current_player}'s turn")

                # Get actions
                action_dict = {}
                for player_id in active_players:
                    agent = player_to_agent[player_id]
                    observation = observation_dict[player_id]
                    action_result = agent.compute_action(observation)

                    if isinstance(action_result, dict):
                        action = action_result.get("action")
                        reasoning = action_result.get("reasoning", "No reasoning provided")
                    else:
                        action = action_result
                        reasoning = "Random choice"

                    action_dict[player_id] = action
                    game_states.append(f"  Player {player_id} chooses action {action}")

                    if reasoning:
                        reasoning_preview = reasoning[:100] + ("..." if len(reasoning) > 100 else "")
                        game_states.append(f"  Reasoning: {reasoning_preview}")

                # Take step
                observation_dict, rewards, terminated, truncated, info = env.step(action_dict)
                for player_id, reward in rewards.items():
                    episode_rewards[player_id] += reward

            # Episode results
            game_states.append(f"\n🏁 Episode {episode + 1} Complete!")
            game_states.append("Final board:")
            game_states.append(env.render_board(0))

            if episode_rewards[0] > episode_rewards[1]:
                winner = "Player 0"
            elif episode_rewards[1] > episode_rewards[0]:
                winner = "Player 1"
            else:
                winner = "Draw"

            game_states.append(f"🏆 Winner: {winner}")
            game_states.append(f"📊 Scores: Player 0={episode_rewards[0]}, Player 1={episode_rewards[1]}")

        return "\n".join(game_states)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during game simulation: {str(e)}"


# ============================================================================
# MAIN GAME FUNCTION (CLEANED UP)
# ============================================================================

def play_game(game_name: str, player1_type: str, player2_type: str,
              player1_model: str = None, player2_model: str = None,
              rounds: int = 1) -> str:
    """Play the selected game with specified players.

    Args:
        game_name: Name of the game to play
        player1_type: Type of player 1 (human, random_bot, llm_*)
        player2_type: Type of player 2 (human, random_bot, llm_*)
        player1_model: LLM model for player 1 (if applicable)
        player2_model: LLM model for player 2 (if applicable)
        rounds: Number of rounds to play

    Returns:
        str: Game log/output to display
    """
    if game_name == "No Games Found":
        return "No games available. Please add game databases."

    # Try full game simulation if environment is available
    if ENV_INITIALIZER_AVAILABLE:
        try:
            # Configure players
            agents = {
                "player_0": setup_player_config(player1_type, player1_model,
                                               "player_0"),
                "player_1": setup_player_config(player2_type, player2_model,
                                               "player_1")
            }

            # Create game configuration
            config = {
                "env_configs": [{"game_name": game_name,
                               "max_game_rounds": None}],
                "num_episodes": int(rounds),
                "seed": 42,
                "use_ray": False,
                "mode": f"{player1_type}_vs_{player2_type}",
                "agents": agents,
                "log_level": "INFO"
            }

            return run_full_game_simulation(game_name, config)

        except Exception as e:
            return f"Error during game simulation: {str(e)}"
    else:
        return ("Board game arena modules not available. "
                "Please ensure the full repository is properly set up.")


def extract_leaderboard_stats(game_name: str) -> pd.DataFrame:
    """Extract and aggregate leaderboard stats from all SQLite databases."""
    db_files = find_or_download_db()
    all_stats = []

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        agent_type, model_name = extract_agent_info(db_file)

        # Skip random agent rows
        if agent_type == "random":
            conn.close()
            continue

        if game_name == "Aggregated Performance":
            query = "SELECT COUNT(DISTINCT episode) AS games_played, " \
                    "SUM(reward) AS total_rewards " \
                    "FROM game_results"
            df = pd.read_sql_query(query, conn)

            # Use avg_generation_time from a specific game (e.g., Kuhn Poker)
            game_query = ("SELECT AVG(generation_time) FROM moves "
                          "WHERE game_name = 'kuhn_poker'")
            avg_gen_time = conn.execute(game_query).fetchone()[0] or 0
        else:
            query = "SELECT COUNT(DISTINCT episode) AS games_played, " \
                    "SUM(reward) AS total_rewards " \
                    "FROM game_results WHERE game_name = ?"
            df = pd.read_sql_query(query, conn, params=(game_name,))

            # Fetch average generation time from moves table
            gen_time_query = ("SELECT AVG(generation_time) FROM moves "
                              "WHERE game_name = ?")
            result = conn.execute(gen_time_query, (game_name,)).fetchone()
            avg_gen_time = result[0] or 0

        # Keep division by 2 for total rewards
        df["total_rewards"] = df["total_rewards"].fillna(0).astype(float) / 2

        # Ensure avg_gen_time has decimals
        avg_gen_time = round(avg_gen_time, 3)

        # Calculate win rate against random bot using opponent column
        vs_random_query = """
            SELECT COUNT(*) FROM game_results
            WHERE opponent = 'random_None' AND reward > 0
        """
        total_vs_random_query = """
            SELECT COUNT(*) FROM game_results
            WHERE opponent = 'random_None'
        """
        wins_vs_random = conn.execute(vs_random_query).fetchone()[0] or 0
        result = conn.execute(total_vs_random_query).fetchone()
        total_vs_random = result[0] or 0
        if total_vs_random > 0:
            vs_random_rate = (wins_vs_random / total_vs_random * 100)
        else:
            vs_random_rate = 0

        # Ensure agent_name is the first column
        df.insert(0, "agent_name", model_name)
        # Ensure agent_type is second column
        df.insert(1, "agent_type", agent_type)
        df["avg_generation_time (sec)"] = avg_gen_time
        df["win vs_random (%)"] = round(vs_random_rate, 2)

        all_stats.append(df)
        conn.close()

    if all_stats:
        leaderboard_df = pd.concat(all_stats, ignore_index=True)
    else:
        leaderboard_df = pd.DataFrame()

    if leaderboard_df.empty:
        columns = ["agent_name", "agent_type", "# games", "total rewards",
                   "avg_generation_time (sec)", "win-rate",
                   "win vs_random (%)"]
        leaderboard_df = pd.DataFrame(columns=columns)

    return leaderboard_df


# ============================================================================
# GRADIO INTERFACE SETUP
# ============================================================================

def create_player_config():
    """Create player configuration for the interface."""
    # Get available games and models
    available_games = get_available_games(include_aggregated=False)

    db_files = find_or_download_db()
    database_models = []
    for db_file in db_files:
        if "random" not in db_file.lower():
            agent_type, model_name = extract_agent_info(db_file)
            database_models.append(model_name)

    # Setup player types
    base_player_types = ["random_bot"]

    # Add local LLM models as player types
    llm_player_types = [f"llm_{key}" for key in LOCAL_LLM_MODELS.keys()]

    # Display names for dropdowns
    player_type_display = {
        "random_bot": "Random Bot",
    }

    # Add display names for local LLM models
    for key in LOCAL_LLM_MODELS.keys():
        model_name = key.split('/')[-1] if '/' in key else key
        player_type_display[f"llm_{key}"] = f"Local LLM: {model_name}"

    # Configure available options
    allowed_player_types = base_player_types + llm_player_types
    available_models = list(LOCAL_LLM_MODELS.keys()) + database_models
    model_info = "Local transformer models are available as player types."

    return {
        'available_games': available_games,
        'allowed_player_types': allowed_player_types,
        'player_type_display': player_type_display,
        'available_models': available_models,
        'model_info': model_info
    }


def create_game_arena_tab():
    """Create the Game Arena tab interface."""
    config = create_player_config()

    gr.Markdown("# LLM Game Arena\n"
                "Play games against LLMs or watch LLMs play against each other!")

    # Model availability info
    gr.Markdown(f"""
    > **🤖 Available AI Players**: {config['model_info']}
    >
    > Local transformer models (GPT-2, Flan-T5) run directly in this space
    > using HuggingFace transformers library. No API tokens required!
    """)

    # Game selection and rounds
    with gr.Row():
        game_dropdown = gr.Dropdown(
            choices=config['available_games'],
            label="Select a Game",
            value=(config['available_games'][0] if config['available_games']
                   else "No Games Found")
        )

        rounds_slider = gr.Slider(
            minimum=1, maximum=10, value=1, step=1,
            label="Number of Rounds"
        )

    # Player configuration
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Player 1")
            p1_choices = [(key, config['player_type_display'][key])
                         for key in config['allowed_player_types']]
            player1_type = gr.Dropdown(
                choices=p1_choices,
                label="Player 1 Type",
                value=p1_choices[0][0] if p1_choices else "random_bot"
            )
            player1_model = gr.Dropdown(
                choices=config['available_models'],
                label="Player 1 Model (if LLM API)",
                visible=False
            )

        with gr.Column():
            gr.Markdown("### Player 2")
            p2_choices = [(key, config['player_type_display'][key])
                         for key in config['allowed_player_types']]
            player2_type = gr.Dropdown(
                choices=p2_choices,
                label="Player 2 Type",
                value=p2_choices[0][0] if p2_choices else "random_bot"
            )
            player2_model = gr.Dropdown(
                choices=config['available_models'],
                label="Player 2 Model (if LLM API)",
                visible=False
            )

    # Game controls and output
    play_button = gr.Button("Start Game", variant="primary")
    game_output = gr.Textbox(label="Game Log", lines=20)

    # Event handlers
    def update_model_visibility(player_type):
        return gr.update(visible=player_type == "llm")

    player1_type.change(update_model_visibility, inputs=player1_type, outputs=player1_model)
    player2_type.change(update_model_visibility, inputs=player2_type, outputs=player2_model)

    play_button.click(
        play_game,
        inputs=[game_dropdown, player1_type, player2_type,
               player1_model, player2_model, rounds_slider],
        outputs=[game_output]
    )


# ============================================================================
# MAIN GRADIO INTERFACE
# ============================================================================

with gr.Blocks() as interface:
    # Game Arena Tab
    with gr.Tab("Game Arena"):
        create_game_arena_tab()

    # Tab for leaderboard and performance tracking
    with gr.Tab("Leaderboard"):
        gr.Markdown("# LLM Model Leaderboard\n"
                    "Track performance across different games!")
        # Dropdown to select a game, including 'Aggregated Performance'
        leaderboard_game_dropdown = gr.Dropdown(
            choices=get_available_games(),
            label="Select Game",
            value="Aggregated Performance"
        )
        # Table to display leaderboard statistics
        leaderboard_table = gr.Dataframe(
            value=extract_leaderboard_stats("Aggregated Performance"),
            headers=["agent_name", "agent_type", "# games", "total rewards",
                     "avg_generation_time (sec)", "win-rate",
                     "win vs_random (%)"],
            every=5
        )
        # Update the leaderboard when a new game is selected
        leaderboard_game_dropdown.change(
            fn=extract_leaderboard_stats,
            inputs=[leaderboard_game_dropdown],
            outputs=[leaderboard_table]
        )

    # Tab for visual insights and performance metrics
    with gr.Tab("Metrics Dashboard"):
        gr.Markdown("# 📊 Metrics Dashboard\n"
                    "Visual summaries of LLM performance across games.")

        # Extract data for visualizations
        metrics_df = extract_leaderboard_stats("Aggregated Performance")

        with gr.Row():
            gr.BarPlot(
                value=metrics_df,
                x="agent_name",
                y="win vs_random (%)",
                title="Win Rate vs Random Bot",
                x_label="LLM Model",
                y_label="Win Rate (%)"
            )

        with gr.Row():
            gr.BarPlot(
                value=metrics_df,
                x="agent_name",
                y="avg_generation_time (sec)",
                title="Average Generation Time",
                x_label="LLM Model",
                y_label="Time (sec)"
            )

        with gr.Row():
            gr.Dataframe(value=metrics_df, label="Performance Summary")

    # Tab for LLM reasoning and illegal move analysis
    with gr.Tab("Analysis of LLM Reasoning"):
        gr.Markdown("# 🧠 Analysis of LLM Reasoning\n"
                    "Insights into move legality and decision behavior.")

        # Load illegal move stats using global function
        illegal_df = extract_illegal_moves_summary()

        with gr.Row():
            gr.BarPlot(
                value=illegal_df,
                x="agent_name",
                y="illegal_moves",
                title="Illegal Moves by Model",
                x_label="LLM Model",
                y_label="# of Illegal Moves"
            )

        with gr.Row():
            gr.Dataframe(value=illegal_df, label="Illegal Move Summary")

    # Add an "About" tab with information about the project
    with gr.Tab("About"):
        gr.Markdown("""
        # About Board Game Arena

        This application provides analysis and visualization of LLM performance
        playing various board games. It includes:

        - **Game Arena**: Play games against LLMs or watch LLMs compete
        - **Leaderboard**: View performance statistics across different games
        - **Metrics Dashboard**: Visual insights into LLM performance
        - **Reasoning Analysis**: Analyze LLM decision-making and errors

        ## Data Sources

        Game results are stored in SQLite databases in the `results` directory.
        Each database contains game logs, moves, and performance metrics for a
        specific LLM agent.

        ## Repository

        This project is part of research on LLM reasoning and decision-making
        strategic game environments.
        """)

    # Launch the Gradio interface with parameters for HuggingFace Spaces
    interface.launch(
        share=False,  # Do not create a public link
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=None,  # Let Gradio find an available port
        show_api=False,  # Hide API docs
        favicon_path=None,  # Use default favicon
        auth=None,  # No authentication required
        # Allow uploads of .db files for database analysis
        allowed_paths=["*.db"]
    )


if __name__ == "__main__":
    # This ensures the interface only launches when run directly
    pass
