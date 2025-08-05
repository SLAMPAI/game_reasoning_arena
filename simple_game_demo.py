"""
Simple Tic-Tac-Toe demo script for Board Game Arena.

This script demonstrates how to use the Board Game Arena framework
to run a simple tic-tac-toe game between two bots.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_simple_demo():
    """Run a simple tic-tac-toe demonstration using random vs. random agents."""

    # Add parent directory to path if needed
    current_dir = Path(__file__).parent.resolve()
    parent_dir = current_dir.parent

    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))

    try:
        # First, try to import required components
        print("Importing Board Game Arena components...")

        try:
            # Try standard import paths
            from board_game_arena.arena.games.registry import registry
            from board_game_arena.arena.agents.policy_manager import initialize_policies
            from board_game_arena.arena.utils.seeding import set_seed

            import_source = "standard package"
        except ImportError:
            try:
                # Try source paths
                from src.board_game_arena.arena.games.registry import registry
                from src.board_game_arena.arena.agents.policy_manager import \
                    initialize_policies
                from src.board_game_arena.arena.utils.seeding import set_seed

                import_source = "source directory"
            except ImportError:
                print("❌ Failed to import Board Game Arena components!")
                print("Make sure the package is installed correctly.")
                return

        print(f"✓ Successfully imported components from {import_source}")

        # Configuration for a simple tic-tac-toe game
        config = {
            "env_configs": [
                {
                    "game_name": "tic_tac_toe",
                    "max_game_rounds": None
                }
            ],
            "num_episodes": 2,  # Run 2 games
            "seed": 42,
            "use_ray": False,
            "mode": "random_vs_random",  # Both players are random
            "agents": {
                "player_0": {
                    "type": "random"
                },
                "player_1": {
                    "type": "random"
                }
            },
            "log_level": "INFO"
        }

        print("\n🎮 Running Simple Tic-Tac-Toe Demo")
        print("-" * 50)

        # Set seed for reproducibility
        set_seed(config["seed"])

        # Get game configuration
        env_config = config["env_configs"][0]
        game_name = env_config["game_name"]

        print(f"🎮 Starting {game_name.replace('_', ' ').title()} Demo")
        print(f"🎲 Random Bot vs Random Bot")
        print(f"📊 Episodes: {config['num_episodes']}")
        print("-" * 50)

        # Create environment
        # We need to include env_config correctly
        env_config_full = {
            "env_config": config["env_configs"][0],  # Make it available as env_config
            **config  # Include all other config parameters
        }

        # Pass both the game name and updated config
        env = registry.make_env(game_name, env_config_full)

        # Initialize agent policies
        policies_dict = initialize_policies(config, game_name, config["seed"])

        # Mapping from player IDs to agents
        player_to_agent = {
            0: policies_dict["policy_0"],
            1: policies_dict["policy_1"]
        }

        # Run episodes
        for episode in range(config["num_episodes"]):
            print(f"\n🎯 Episode {episode + 1}")
            print("=" * 30)

            # Reset environment
            observation_dict, _ = env.reset(seed=config["seed"] + episode)

            # Game variables
            episode_rewards = {0: 0, 1: 0}
            terminated = False
            truncated = False
            step_count = 0

            while not (terminated or truncated):
                step_count += 1
                print(f"\n📋 Step {step_count}")

                # Show current board state
                print("Current board:")
                print(env.render_board(0))

                # Determine which player(s) should act
                if env.state.is_simultaneous_node():
                    # All players act simultaneously (not typical for tic-tac-toe)
                    active_players = list(player_to_agent.keys())
                else:
                    # Turn-based: only current player acts
                    current_player = env.state.current_player()
                    active_players = [current_player]
                    print(f"Player {current_player}'s turn")

                # Compute actions for active players
                action_dict = {}
                for player_id in active_players:
                    agent = player_to_agent[player_id]
                    observation = observation_dict[player_id]

                    # Get action from agent
                    action = agent.compute_action(observation)
                    action_dict[player_id] = action

                    print(f"  Player {player_id} chooses action {action}")

                # Take environment step
                observation_dict, rewards, terminated, truncated, info = env.step(
                    action_dict)

                # Update episode rewards
                for player_id, reward in rewards.items():
                    episode_rewards[player_id] += reward

            # Episode finished
            print(f"\n🏁 Episode {episode + 1} Complete!")
            print("Final board:")
            print(env.render_board(0))

            # Determine winner
            if episode_rewards[0] > episode_rewards[1]:
                winner = "Player 0"
            elif episode_rewards[1] > episode_rewards[0]:
                winner = "Player 1"
            else:
                winner = "Draw"

            print(f"🏆 Winner: {winner}")
            print(f"📊 Scores: Player 0={episode_rewards[0]}, " +
                  f"Player 1={episode_rewards[1]}")

        print("\n✅ Demo completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure the Board Game Arena package is installed correctly.")
        return False


if __name__ == "__main__":
    run_simple_demo()
