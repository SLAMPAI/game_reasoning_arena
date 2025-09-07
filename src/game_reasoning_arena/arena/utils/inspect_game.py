"""
inspect_game.py

Used to retrieve the player's symbols (checkers) used in OpenSpiel games
by playing one move per player and visualizing the board.

Useful to understand how the game is represented in OpenSpiel.
and create the prompts for LLMs.
"""

import pyspiel


def inspect_game(game_name: str):
    print("=" * 60)
    print(f"  Game: {game_name}")
    print("=" * 60)

    try:
        game = pyspiel.load_game(game_name)
    except Exception as e:
        print(f" Failed to load {game_name}: {e}")
        return

    num_players = game.num_players()
    print(f" Number of players: {num_players}")

    state = game.new_initial_state()

    # Each player makes one move if possible
    for pid in range(num_players):
        if state.is_terminal():
            print(" Reached terminal state before all players could move.")
            break

        if state.current_player() != pid:
            print(
                f" It's player {state.current_player()}'s turn, not {pid}. "
                f"Skipping."
            )
            continue

        legal = state.legal_actions()
        if not legal:
            print(f"  No legal actions for player {pid}. Skipping.")
            continue

        move = legal[0]
        print(f"\n Player {pid} makes move: {move}")
        state.apply_action(move)

        print("\n Board state after move:")
        print(state.to_string())

    print("\n\n")


def main():
    symbolic_games = [
        "chess",
        "tic_tac_toe",
        "connect_four",
        "breakthrough",
        "othello",
        "hex",
    ]

    for game in symbolic_games:
        inspect_game(game)


if __name__ == "__main__":
    main()
