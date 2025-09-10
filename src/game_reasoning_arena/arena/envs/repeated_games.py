# ================================================================
# Example: How to create a repeated version of any stage game in OpenSpiel
#
# OpenSpiel provides a "repeated_game" wrapper that can take *any*
# normal-form (matrix) or stage game (e.g., Prisoner's Dilemma,
# Matching Pennies, Rock-Paper-Scissors) and convert it into an
# iterated / repeated version.
#
# Usage:
#
# import pyspiel
#
# # Load a repeated Prisoner's Dilemma with 10 repetitions
# game = pyspiel.load_game(
#     "repeated_game(prisoners_dilemma(),num_repetitions=10)"
# )
#
# Key options for the "repeated_game" wrapper:
#   - num_repetitions (int): number of times to repeat the stage game.
#   - discount (float): optional discount factor for future payoffs
#                       (default = 1.0, i.e. no discounting).
#   - enable_abbreviated_history (bool): if true, the game tracks history
#                                        in a more compact form.
#
# Example with discounting:
# game = pyspiel.load_game(
#     "repeated_game(prisoners_dilemma(),num_repetitions=20,discount=0.95)"
# )
#
# Note:
#   - This is not a parameter inside the base game (like prisoners_dilemma).
#   - "repeated_game" is its own wrapper type. You wrap the base game string.
#   - Works with any normal-form or matrix game supported in OpenSpiel.
#
# ================================================================
