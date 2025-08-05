
''' Used by the Gradio app to display game results, leaderboards,
and performance metrics for LLM agents.
'''

import sqlite3
import pandas as pd
import gradio as gr
import os
import json
import requests
from pathlib import Path
from typing import List, Dict
import importlib.util
import sys

# Try to import game-related modules with proper error handling
GAME_REGISTRY_AVAILABLE = False
AGENT_REGISTRY_AVAILABLE = False
ENV_INITIALIZER_AVAILABLE = False

# Define default player types and registries
class PlayerType:
    HUMAN = "human"
    RANDOM = "random_bot"
    LLM = "llm"


# Default game types based on database entries
GAMES_REGISTRY = {
    "tic_tac_toe": "Tic Tac Toe",
    "kuhn_poker": "Kuhn Poker",
    "connect_four": "Connect Four"
}

# Default LLM models with their HuggingFace model IDs
LLM_REGISTRY = {
    "llama3_8b": "Llama 3 8B",  # External API (costs tokens)
    "random_None": "Random Bot", # Not an LLM
    "gpt2": "GPT-2 (free)",      # Free on HuggingFace
    "bloom-560m": "BLOOM 560M (free)", # Free smaller model
    "flan-t5-small": "Flan-T5-Small (free)" # Free instruction model
}

# Models available for free inference through HuggingFace
FREE_HUGGINGFACE_MODELS = {
    "gpt2": {
        "model_id": "gpt2",
        "prompt_template": """
You are playing a game. Here is the current state:
{game_state}

You are player {player}.
Valid moves: {valid_moves}

Choose one valid move from the list above. Just respond with the move number.
""",
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 0.2,
            "do_sample": True
        }
    },
    "bloom-560m": {
        "model_id": "bigscience/bloom-560m",
        "prompt_template": """
You are playing a game. Current state:
{game_state}

You are player {player}.
Valid moves: {valid_moves}

Choose one move from the valid moves. Only respond with the move number.
""",
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 0.1
        }
    },
    "flan-t5-small": {
        "model_id": "google/flan-t5-small",
        "prompt_template": """
Game state: {game_state}
Player: {player}
Valid moves: {valid_moves}
Choose move:
""",
        "parameters": {
            "max_new_tokens": 5,
            "temperature": 0.1,
            "do_sample": False
        }
    }
}

# Function to create a player class that uses HuggingFace models
def create_huggingface_player(model_id):
    """Create a player class that uses HuggingFace Inference API.

    Args:
        model_id: The Hugging Face model ID

    Returns:
        A player class that can be instantiated for game playing
    """
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

    class HuggingFacePlayer(Player):
        def __init__(self, name=None):
            name = name or f"HF-{model_id}"
            super().__init__(name=name)
            self.model_id = model_id
            self.model_info = FREE_HUGGINGFACE_MODELS[model_id]

        def make_move(self, game_state):
            # Create a prompt for the model based on the game state
            prompt = self.model_info["prompt_template"].format(
                game_state=str(game_state),
                valid_moves=", ".join(map(str, game_state.legal_actions())),
                player=game_state.current_player(),
            )

            # Query the HuggingFace model
            response = query_huggingface_model(
                self.model_info.get("model_id", model_id),
                prompt,
                max_length=self.model_info["parameters"].get("max_new_tokens", 10)
            )

            # Parse the response to extract the move
            valid_moves = game_state.legal_actions()
            for move in valid_moves:
                if str(move) in response:
                    return move

            # Fallback: if no valid move found, return the first legal action
            print(f"No valid move found in response: {response}")
            print(f"Using fallback move: {valid_moves[0]}")
            return valid_moves[0]

    return HuggingFacePlayer


# Function to check if HuggingFace API is available
def check_huggingface_api():
    """Check if the HuggingFace Inference API is accessible.
    
    Returns:
        bool: True if API is accessible, False otherwise
    """
    try:
        # Use a simple health check endpoint
        response = requests.get(
            "https://api-inference.huggingface.co/status",
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking HuggingFace API: {e}")
        return False

# Function to query HuggingFace's inference API for free models
def query_huggingface_model(model_id, prompt, max_length=100):
    """
    Query a model through HuggingFace's inference API.

    Args:
        model_id: ID of the model on HuggingFace Hub
        prompt: Text prompt to send to the model
        max_length: Maximum response length

    Returns:
        str: Model's response text
    """
    try:
        # HuggingFace Inference API endpoint
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"

        # Headers with API token if provided
        headers = {}
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        # Parameters vary by model type
        if "t5" in model_id.lower():
            # T5 models expect direct input without specific formatting
            payload = {"inputs": prompt}
        else:
            # For generative models like GPT-2, BLOOM
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }

        # Make the request to the API
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return str(result[0])
            else:
                return str(result)
        else:
            return f"Error: {response.status_code}, {response.text}"

    except Exception as e:
        return f"Failed to query model: {str(e)}"

# Function to check if a package is installed
def is_package_installed(package_name):
    """Check if a Python package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        bool: True if installed, False otherwise
    """
    try:
        importlib.util.find_spec(package_name)
        return True
    except ImportError:
        return False

# Try to import from the proper package structure
print("\n--- Initializing Board Game Arena Modules ---")
print("Checking required packages:")
for pkg in ["numpy", "pyspiel", "openspiel"]:
    installed = is_package_installed(pkg)
    print(f"  - {pkg}: {'✓ Installed' if installed else '✗ Not found'}")

try:
    # First try importing from board_game_arena package (installed case)
    print("\nAttempting to import from board_game_arena package...")
    from board_game_arena.arena.games.registry import registry as games_registry
    from board_game_arena.arena.agents.agent_registry import registry as agent_registry
    from board_game_arena.arena.envs.env_initializer import get_environment

    # If successful, update our registry dictionaries
    GAMES_REGISTRY = {name: cls for name, cls in games_registry._registry.items()}
    GAME_REGISTRY_AVAILABLE = True
    AGENT_REGISTRY_AVAILABLE = True
    ENV_INITIALIZER_AVAILABLE = True
    print("✓ Successfully imported board_game_arena modules")
    print(f"Available games: {list(GAMES_REGISTRY.keys())}")

    # Verify get_environment works
    try:
        print("Testing environment initialization with a basic game setup...")
        test_env = get_environment(
            game_name="tic_tac_toe",
            player_configs={0: {"type": "random"}, 1: {"type": "random"}}
        )
        print("✓ Environment initialization test successful")
        
        # Verify the created environment has expected attributes
        if hasattr(test_env, 'simulate'):
            print("✓ Environment has simulate method")
        else:
            print("✗ Environment missing simulate method")
            ENV_INITIALIZER_AVAILABLE = False
            
        # Try to create a custom player to verify player creation works
        try:
            from board_game_arena.arena.player import Player
            class TestPlayer(Player):
                def __init__(self): 
                    super().__init__(name="TestPlayer")
                def make_move(self, state): 
                    return state.legal_actions()[0]
                    
            print("✓ Successfully created test player class")
        except Exception as player_error:
            print(f"✗ Failed to create player class: {player_error}")
            
    except Exception as test_error:
        print(f"✗ Environment test failed: {test_error}")
        print("  This suggests the board_game_arena package is not correctly installed")
        ENV_INITIALIZER_AVAILABLE = False

except ImportError as e:
    # Then try from src.board_game_arena (development case)
    print(f"✗ Failed to import from board_game_arena: {e}")
    print("Attempting to import from src.board_game_arena...")
    try:
        from src.board_game_arena.arena.games.registry import registry as games_registry
        from src.board_game_arena.arena.agents.agent_registry import registry as agent_registry
        from src.board_game_arena.arena.envs.env_initializer import get_environment

        # If successful, update our registry dictionaries
        GAMES_REGISTRY = {name: cls for name, cls in games_registry._registry.items()}
        GAME_REGISTRY_AVAILABLE = True
        AGENT_REGISTRY_AVAILABLE = True
        ENV_INITIALIZER_AVAILABLE = True
        print("✓ Successfully imported src.board_game_arena modules")
        print(f"Available games: {list(GAMES_REGISTRY.keys())}")

        # Verify get_environment works
        try:
            test_env = get_environment(
                game_name="tic_tac_toe",
                player_configs={0: {"type": "random"}, 1: {"type": "random"}}
            )
            print("✓ Environment initialization test successful")
        except Exception as test_error:
            print(f"✗ Environment test failed: {test_error}")
            ENV_INITIALIZER_AVAILABLE = False

    except ImportError as e2:
        print(f"✗ Failed to import from src.board_game_arena: {e2}")
        print("Using placeholder implementations")
        GAME_REGISTRY_AVAILABLE = False
        AGENT_REGISTRY_AVAILABLE = False
        ENV_INITIALIZER_AVAILABLE = False

print(f"Environment initializer available: {ENV_INITIALIZER_AVAILABLE}")
print("--- Initialization Complete ---\n")

# Directory to store SQLite results
db_dir = Path("results")


def find_or_download_db():
    """Check if SQLite .db files exist; if not, attempt to download from
    cloud storage."""
    print(f"DEBUG: Looking for DB files in {db_dir}")
    if not db_dir.exists():
        print("DEBUG: Results directory doesn't exist, creating it")
        db_dir.mkdir(parents=True, exist_ok=True)

    db_files = list(db_dir.glob("*.db"))
    print(f"DEBUG: Found {len(db_files)} DB files: {[f.name for f in db_files]}")

    # Ensure the random bot database exists
    random_db_path = db_dir / "random_None.db"
    if not random_db_path.exists():
        print(f"DEBUG: Required file {random_db_path} does not exist")
        try:
            # Create an empty SQLite database if it doesn't exist
            print("DEBUG: Attempting to create a placeholder random_None.db file")
            import sqlite3
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
            print("DEBUG: Created placeholder database")
        except Exception as e:
            print(f"DEBUG: Error creating placeholder database: {e}")
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


def play_game(game_name: str, player1_type: str, player2_type: str,
              player1_model: str = None, player2_model: str = None,
              rounds: int = 1) -> str:
    """Play the selected game with specified players.

    Args:
        game_name: Name of the game to play
        player1_type: Type of player 1 (human, random_bot, llm)
        player2_type: Type of player 2 (human, random_bot, llm)
        player1_model: LLM model for player 1 (if applicable)
        player2_model: LLM model for player 2 (if applicable)
        rounds: Number of rounds to play

    Returns:
        str: Game log/output to display
    """
    if game_name == "No Games Found":
        return "No games available. Please add game databases."

    # Track token usage to prevent excessive costs (especially on HuggingFace)
    # This counter would ideally be persisted between sessions
    token_usage_path = Path("token_usage.txt")
    max_daily_tokens = 50000  # Set a reasonable limit

    # Check if using paid LLM APIs (free HF models don't count toward token limit)
    using_paid_llm = (
        (player1_type == "llm" and player1_model and
         player1_model not in FREE_HUGGINGFACE_MODELS) or
        (player2_type == "llm" and player2_model and
         player2_model not in FREE_HUGGINGFACE_MODELS)
    )

    if using_paid_llm:
        # Simple token tracking
        try:
            # Read current token count if file exists
            if token_usage_path.exists():
                with open(token_usage_path, "r") as f:
                    try:
                        last_reset, tokens_used = f.read().strip().split(",")
                        # Reset daily if needed
                        if last_reset != str(pd.Timestamp.now().date()):
                            last_reset = str(pd.Timestamp.now().date())
                            tokens_used = 0
                        else:
                            tokens_used = int(tokens_used)
                    except (ValueError, IndexError):
                        # Invalid format, reset
                        last_reset = str(pd.Timestamp.now().date())
                        tokens_used = 0
            else:
                # Initialize token tracking
                last_reset = str(pd.Timestamp.now().date())
                tokens_used = 0

            # Estimate token usage for this game
            # Very rough estimate: ~100 tokens per move, 10 moves per round
            estimated_new_tokens = rounds * 10 * 100

            # Check if we're over the limit
            if tokens_used + estimated_new_tokens > max_daily_tokens:
                return (
                    "⚠️ Daily token limit reached. Please try again tomorrow, "
                    "use the random bot opponent, or select a free HuggingFace model."
                )

            # Otherwise, update the projected usage
            tokens_used += estimated_new_tokens
            with open(token_usage_path, "w") as f:
                f.write(f"{last_reset},{tokens_used}")

        except Exception as e:
            # Log error but continue (token tracking is optional)
            print(f"Token tracking error: {e}")
            # Don't prevent gameplay if token tracking fails

    # If we have the environment initializer available, use it
    if ENV_INITIALIZER_AVAILABLE:
        try:
            # Print debug info
            print(f"Game initialization started: {game_name}")
            print(f"Player 1: {player1_type}, Model: {player1_model}")
            print(f"Player 2: {player2_type}, Model: {player2_model}")

            # Configure agents for the game config
            agents = {}

            # Set up player 1
            if player1_type == "human":
                agents["player_0"] = {"type": "human"}
                print("Player 1 configured as human")
            elif player1_type == "random_bot":
                agents["player_0"] = {"type": "random"}
                print("Player 1 configured as random bot")
            elif player1_type.startswith("hf_"):
                # Extract model ID from the player type (format: hf_model_id)
                hf_model_id = player1_type[3:]  # Remove 'hf_' prefix
                print(f"Setting up Player 1 as HuggingFace model: {hf_model_id}")
                # Instead of custom player class, set up config correctly for initialize_policies
                agents["player_0"] = {
                    "type": "custom",
                    "name": f"HF-{hf_model_id}",
                    "player_class": create_huggingface_player(hf_model_id)
                }
            elif player1_type == "llm" and player1_model:
                if player1_model in FREE_HUGGINGFACE_MODELS:
                    # Special handling for free HuggingFace models
                    print(f"Setting up Player 1 as HuggingFace model: {player1_model}")
                    agents["player_0"] = {
                        "type": "custom",
                        "name": f"HF-{player1_model}",
                        "player_class": create_huggingface_player(player1_model)
                    }
                else:
                    # Regular LLM player using paid APIs
                    print(f"Setting up Player 1 as LLM: {player1_model}")
                    agents["player_0"] = {"type": "llm", "model": player1_model}

            # Set up player 2
            if player2_type == "human":
                agents["player_1"] = {"type": "human"}
                print("Player 2 configured as human")
            elif player2_type == "random_bot":
                agents["player_1"] = {"type": "random"}
                print("Player 2 configured as random bot")
            elif player2_type.startswith("hf_"):
                # Extract model ID from the player type (format: hf_model_id)
                hf_model_id = player2_type[3:]  # Remove 'hf_' prefix
                print(f"Setting up Player 2 as HuggingFace model: {hf_model_id}")
                agents["player_1"] = {
                    "type": "custom",
                    "name": f"HF-{hf_model_id}",
                    "player_class": create_huggingface_player(hf_model_id)
                }
            elif player2_type == "llm" and player2_model:
                if player2_model in FREE_HUGGINGFACE_MODELS:
                    # Special handling for free HuggingFace models
                    print(f"Setting up Player 2 as HuggingFace model: {player2_model}")
                    agents["player_1"] = {
                        "type": "custom",
                        "name": f"HF-{player2_model}",
                        "player_class": create_huggingface_player(player2_model)
                    }
                else:
                    # Regular LLM player using paid APIs
                    print(f"Setting up Player 2 as LLM: {player2_model}")
                    agents["player_1"] = {"type": "llm", "model": player2_model}

            # Create configuration similar to what works in simple_game_demo.py
            config = {
                "env_configs": [
                    {
                        "game_name": game_name,
                        "max_game_rounds": None
                    }
                ],
                "num_episodes": int(rounds),
                "seed": 42,
                "use_ray": False,
                "mode": f"{player1_type}_vs_{player2_type}",
                "agents": agents,
                "log_level": "INFO"
            }

            print(f"Creating game environment for: {game_name}")
            print(f"Config: {config}")
            print(f"DEBUG: Environment variables:")
            print(f"DEBUG: PATH: {os.environ.get('PATH', 'Not set')}")
            print(f"DEBUG: PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            print(f"DEBUG: Working directory: {os.getcwd()}")

            try:
                from board_game_arena.arena.games.registry import registry
                from board_game_arena.arena.agents.policy_manager import initialize_policies
                from board_game_arena.arena.utils.seeding import set_seed

                # Set seed for reproducibility
                set_seed(config["seed"])
                
                # Get the game configuration
                env_config = config["env_configs"][0]
                game_name = env_config["game_name"]
                
                # Create environment with the correct config structure
                env_config_full = {
                    "env_config": config["env_configs"][0],  # Make it available as env_config
                    **config  # Include all other config parameters
                }
                
                # Create environment
                env = registry.make_env(game_name, env_config_full)
                
                # Initialize agent policies
                policies_dict = initialize_policies(config, game_name, config["seed"])
                
                print("Game environment created successfully")
            except Exception as env_error:
                print(f"ERROR creating game environment: {env_error}")
                print(f"DEBUG: ENV_INITIALIZER_AVAILABLE = {ENV_INITIALIZER_AVAILABLE}")
                import traceback
                traceback.print_exc()
                return (f"Failed to create game environment: {str(env_error)}\n\n"
                        "Detailed error information has been printed to the console.")

            # Track game states
            game_states = []
            
            # Mapping from player IDs to agents (using policies_dict from before)
            player_to_agent = {
                0: policies_dict["policy_0"], 
                1: policies_dict["policy_1"]
            }
            
            # Run the game simulation
            try:
                print(f"Starting game simulation with {rounds} rounds")
                
                # Run episodes (same approach as simple_game_demo.py)
                for episode in range(int(rounds)):
                    game_states.append(f"\n🎯 Episode {episode + 1}")
                    game_states.append("=" * 30)
                    
                    # Reset environment
                    observation_dict, _ = env.reset(seed=config["seed"] + episode)
                    
                    # Game variables
                    episode_rewards = {0: 0, 1: 0}
                    terminated = False
                    truncated = False
                    step_count = 0
                    
                    while not (terminated or truncated):
                        step_count += 1
                        game_states.append(f"\n📋 Step {step_count}")
                        
                        # Show current board state
                        board = env.render_board(0)
                        game_states.append("Current board:")
                        game_states.append(board)
                        
                        # Determine which player(s) should act
                        if env.state.is_simultaneous_node():
                            # All players act simultaneously (not typical for tic-tac-toe)
                            active_players = list(player_to_agent.keys())
                        else:
                            # Turn-based: only current player acts
                            current_player = env.state.current_player()
                            active_players = [current_player]
                            game_states.append(f"Player {current_player}'s turn")
                        
                        # Compute actions for active players
                        action_dict = {}
                        for player_id in active_players:
                            agent = player_to_agent[player_id]
                            observation = observation_dict[player_id]
                            
                            # Get action from agent
                            action_result = agent.compute_action(observation)
                            
                            if isinstance(action_result, dict):
                                action = action_result.get("action")
                                reasoning = action_result.get("reasoning", "No reasoning provided")
                            else:
                                action = action_result
                                reasoning = "Random choice"
                                
                            action_dict[player_id] = action
                            
                            # Log the action and reasoning
                            game_states.append(f"  Player {player_id} chooses action {action}")
                            if reasoning:
                                # Show the first 100 chars of reasoning
                                reasoning_preview = reasoning[:100]
                                if len(reasoning) > 100:
                                    reasoning_preview += "..."
                                game_states.append(f"  Reasoning: {reasoning_preview}")
                        
                        # Take environment step
                        observation_dict, rewards, terminated, truncated, info = env.step(action_dict)
                        
                        # Update episode rewards
                        for player_id, reward in rewards.items():
                            episode_rewards[player_id] += reward
                    
                    # Episode finished
                    game_states.append(f"\n🏁 Episode {episode + 1} Complete!")
                    game_states.append("Final board:")
                    game_states.append(env.render_board(0))
                    
                    # Determine winner
                    if episode_rewards[0] > episode_rewards[1]:
                        winner = "Player 0"
                    elif episode_rewards[1] > episode_rewards[0]:
                        winner = "Player 1"
                    else:
                        winner = "Draw"
                    
                    game_states.append(f"🏆 Winner: {winner}")
                    game_states.append(f"📊 Scores: Player 0={episode_rewards[0]}, Player 1={episode_rewards[1]}")
                
                print("Game simulation completed")
                return "\n".join(game_states)
            except Exception as sim_error:
                print(f"ERROR during game simulation: {sim_error}")
                import traceback
                traceback.print_exc()
                return f"Error during game simulation: {str(sim_error)}"

        except Exception as e:
            # If anything fails, provide error details
            print(f"ERROR in game setup: {e}")
            import traceback
            traceback.print_exc()
            return (f"Error setting up game: {str(e)}\n\n"
                    "Make sure the board_game_arena package is properly installed.")

    # Fallback implementation when game engine isn't available
    print("DEBUG: Fallback mode activated - game engine not available")
    print("DEBUG: Checking if board_game_arena module exists:")
    try:
        import importlib
        bga_spec = importlib.util.find_spec("board_game_arena")
        print(f"DEBUG: board_game_arena module found: {bga_spec}")
        if bga_spec:
            print(f"DEBUG: Module location: {bga_spec.origin}")
            print(f"DEBUG: Submodule locations: "
                  f"{bga_spec.submodule_search_locations}")
            # Try importing some key components
            print("DEBUG: Trying to import key components:")
            try:
                print("DEBUG: Import board_game_arena.arena")
                import board_game_arena.arena
                print("DEBUG: Successfully imported arena module")
            except Exception as e:
                print(f"DEBUG: Failed to import arena: {e}")
    except Exception as import_error:
        print(f"DEBUG: Error importing board_game_arena: {import_error}")
        import traceback
        traceback.print_exc()

    # Connect to database to get game rules and possible moves
    print("DEBUG: Checking for database files in results/")
    db_files_direct = list(db_dir.glob("*.db"))
    print(f"DEBUG: Found DB files directly: {db_files_direct}")

    db_files = find_or_download_db()
    print(f"DEBUG: DB files from find_or_download_db: {db_files}")
    
    # Simple fallback mode that at least attempts to use the HuggingFace API
    # for direct HF models
    game_log = []
    game_log.append(f"Initializing {game_name} game in simplified mode...")
    
    # Determine if we have any HuggingFace models
    has_hf_model = False
    
    # Format player info and check for HuggingFace models
    player1_info = f"Player 1: {player1_type}"
    if player1_type.startswith("hf_"):
        hf_model_id = player1_type[3:]  # Remove 'hf_' prefix
        player1_info += f" ({hf_model_id})"
        has_hf_model = True
    elif player1_model:
        player1_info += f" ({player1_model})"
        if player1_model in FREE_HUGGINGFACE_MODELS:
            has_hf_model = True
    game_log.append(player1_info)
    
    player2_info = f"Player 2: {player2_type}"
    if player2_type.startswith("hf_"):
        hf_model_id = player2_type[3:]  # Remove 'hf_' prefix
        player2_info += f" ({hf_model_id})"
        has_hf_model = True
    elif player2_model:
        player2_info += f" ({player2_model})"
        if player2_model in FREE_HUGGINGFACE_MODELS:
            has_hf_model = True
    game_log.append(player2_info)
    
    game_log.append(f"Rounds: {rounds}")
    game_log.append("")
    
    # If we have a HuggingFace model, try to demonstrate it
    if has_hf_model:
        game_log.append("Making API call to HuggingFace Inference Endpoint:")
        
        # First check if HuggingFace API is accessible
        if check_huggingface_api():
            game_log.append("✓ HuggingFace API is accessible")
            
            try:
                # Determine which model to query
                model_to_query = None
                if player1_type.startswith("hf_"):
                    model_to_query = player1_type[3:]
                elif player1_model in FREE_HUGGINGFACE_MODELS:
                    model_to_query = player1_model
                elif player2_type.startswith("hf_"):
                    model_to_query = player2_type[3:]
                elif player2_model in FREE_HUGGINGFACE_MODELS:
                    model_to_query = player2_model
                    
                if model_to_query:
                    # Create a simple test prompt
                    test_prompt = f"""
You are playing a game of {game_name}.
Board state:
Empty board
Valid moves: 0, 1, 2, 3, 4, 5, 6, 7, 8
Choose one move from the valid moves. Only respond with the move number.
"""
                    game_log.append(f"Model: {model_to_query}")
                    game_log.append(f"Prompt: {test_prompt}")
                    
                    # Call the API
                    response = query_huggingface_model(
                        model_to_query,
                        test_prompt,
                        max_length=10
                    )
                    
                    game_log.append(f"Response: {response}")
                    game_log.append("")
                    game_log.append("✓ API call successful! HuggingFace model working.")
                else:
                    game_log.append("✗ Could not determine which model to query.")
                    
            except Exception as e:
                game_log.append(f"✗ Error making API call: {str(e)}")
                game_log.append("Check your internet connection and API tokens.")
        else:
            game_log.append("✗ HuggingFace API is not accessible")
            game_log.append("Please check your internet connection.")
    
    game_log.append("")
    game_log.append("Note: Running in simplified demonstration mode.")
    game_log.append("For full gameplay with the game engine, ensure the board_game_arena package")
    game_log.append("is properly installed and configured.")

    # Add database information
    game_log.append("\nAvailable database files:")
    for db in db_files:
        if game_name in db or game_name.replace("_", "") in db.lower():
            game_log.append(f" - {db} [RELEVANT]")
        else:
            game_log.append(f" - {db}")

    return "\n".join(game_log)


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


##########################################################
with gr.Blocks() as interface:
    # Tab for playing games against LLMs
    with gr.Tab("Game Arena"):
        gr.Markdown("# LLM Game Arena\n"
                    "Play games against LLMs or watch LLMs play against each other!")

        # Get available games
        available_games = get_available_games(include_aggregated=False)

        # Get available LLM models from database files
        db_files = find_or_download_db()
        database_models = []
        for db_file in db_files:
            if "random" not in db_file.lower():  # Exclude random bot
                agent_type, model_name = extract_agent_info(db_file)
                database_models.append(model_name)

        # Check if running on HuggingFace
        is_huggingface = os.environ.get("SPACE_ID") is not None

        # Create player types including specific free models
        base_player_types = ["human", "random_bot"]

        # Add free HuggingFace models as player types
        hf_model_player_types = [f"hf_{key}" for key in FREE_HUGGINGFACE_MODELS.keys()]

        # Display names for the player types in the dropdown
        player_type_display = {
            "human": "Human Player",
            "random_bot": "Random Bot",
            # Removed "llm" option as requested
        }

        # Add display names for HuggingFace models
        for key in FREE_HUGGINGFACE_MODELS.keys():
            model_name = key.split('/')[-1] if '/' in key else key
            player_type_display[f"hf_{key}"] = f"HF: {model_name} (free)"

        # Determine which player types to offer
        if is_huggingface:
            # When on HuggingFace, offer human, random_bot and free models
            allowed_player_types = base_player_types + hf_model_player_types
            available_models = list(FREE_HUGGINGFACE_MODELS.keys())
            model_info = "Free HuggingFace models are available directly as player types."
        else:
            # When running locally, offer all options (removed "llm" option)
            allowed_player_types = base_player_types + hf_model_player_types
            available_models = list(FREE_HUGGINGFACE_MODELS.keys()) + database_models
            model_info = "Free HuggingFace models are available as player types."

        # Display appropriate message about model usage
        gr.Markdown(f"""
        > **🤖 Available AI Players**: {model_info}
        >
        > Free HuggingFace models (GPT-2, BLOOM, etc.) are now available as
        > direct player options. These models run on HuggingFace's servers
        > without using your API tokens.
        """)

        with gr.Row():
            # Game selection
            game_dropdown = gr.Dropdown(
                choices=available_games,
                label="Select a Game",
                value=(available_games[0] if available_games
                       else "No Games Found")
            )

            # Number of rounds
            rounds_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                label="Number of Rounds"
            )

        with gr.Row():
            with gr.Column():
                # Player 1 configuration
                gr.Markdown("### Player 1")
                # Create choices with nice display names
                p1_choices = [(key, player_type_display[key])
                             for key in allowed_player_types]
                player1_type = gr.Dropdown(
                    choices=p1_choices,
                    label="Player 1 Type",
                    value="human"
                )
                player1_model = gr.Dropdown(
                    choices=available_models,
                    label="Player 1 Model (if LLM API)",
                    visible=False
                )

                # Show model dropdown only when generic LLM is selected
                def update_p1_model_visibility(player_type):
                    return gr.Dropdown.update(visible=player_type == "llm")

                player1_type.change(
                    update_p1_model_visibility,
                    inputs=player1_type,
                    outputs=player1_model
                )

            with gr.Column():
                # Player 2 configuration
                gr.Markdown("### Player 2")
                # Create choices with nice display names
                p2_choices = [(key, player_type_display[key])
                             for key in allowed_player_types]
                player2_type = gr.Dropdown(
                    choices=p2_choices,
                    label="Player 2 Type",
                    value="random_bot"
                )
                player2_model = gr.Dropdown(
                    choices=available_models,
                    label="Player 2 Model (if LLM API)",
                    visible=False
                )

                # Show model dropdown only when generic LLM is selected
                def update_p2_model_visibility(player_type):
                    return gr.Dropdown.update(visible=player_type == "llm")

                player2_type.change(
                    update_p2_model_visibility,
                    inputs=player2_type,
                    outputs=player2_model
                )        # Button to start the game and output area
        play_button = gr.Button("Start Game", variant="primary")
        game_output = gr.Textbox(label="Game Log", lines=20)

        # Event to start the game when the button is clicked
        play_button.click(
            play_game,
            inputs=[
                game_dropdown,
                player1_type,
                player2_type,
                player1_model,
                player2_model,
                rounds_slider
            ],
            outputs=[game_output]
        )

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
        server_port=7860,  # Default Gradio port
        show_api=False,  # Hide API docs
        favicon_path=None,  # Use default favicon
        auth=None,  # No authentication required
        # Allow uploads of .db files for database analysis
        allowed_paths=["*.db"]
    )
