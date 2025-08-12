"""
/games/loaders.py

Centralized game loader module with decorator-based registration.
Loaded by 'registry.py' according to the mapping in 'game_specs.py'.
"""

import pyspiel
from .registry import registry


class GameLoader:
    """Base class for game loaders"""


# Registering games with simplified registry format
@registry.register(
    name="prisoners_dilemma",
    module_path="game_reasoning_arena.arena.games.loaders",
    class_name="PrisonersDilemmaLoader",
    environment_path="game_reasoning_arena.arena.envs.matrix_game_env.MatrixGameEnv",
    display_name="Prisoner's Dilemma (Matrix)"
)
class PrisonersDilemmaLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matrix_pd")


@registry.register(
    name="tic_tac_toe",
    module_path="game_reasoning_arena.arena.games.loaders",
    class_name="TicTacToeLoader",
    environment_path="game_reasoning_arena.arena.envs.tic_tac_toe_env.TicTacToeEnv",
    display_name="Tic-Tac-Toe"
)
class TicTacToeLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("tic_tac_toe")


@registry.register(
    name="connect_four",
    module_path="game_reasoning_arena.arena.games.loaders",
    class_name="ConnectFourLoader",
    environment_path="game_reasoning_arena.arena.envs.connect_four_env.ConnectFourEnv",
    display_name="Connect Four"
)
class ConnectFourLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("connect_four")


@registry.register(
    name="kuhn_poker",
    module_path="game_reasoning_arena.arena.games.loaders",
    class_name="KuhnPokerLoader",
    environment_path="game_reasoning_arena.arena.envs.kuhn_poker_env.KuhnPokerEnv",
    display_name="Kuhn Poker"
)
class KuhnPokerLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("kuhn_poker")


# Prisoner's Dilemma, Matching Pennies and Rock-Paper-Scissors as matrix games

@registry.register(
    name="matrix_pd",
    module_path="game_reasoning_arena.arena.games.loaders",
    class_name="MatrixPDLoader",
    environment_path="game_reasoning_arena.arena.envs.matrix_game_env.MatrixGameEnv",
    display_name="Matrix Prisoner's Dilemma"
)
class MatrixPDLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matrix_pd")


@registry.register(
    name="matching_pennies",
    module_path="game_reasoning_arena.arena.games.loaders",
    class_name="MatchingPenniesLoader",
    environment_path="game_reasoning_arena.arena.envs.matrix_game_env.MatrixGameEnv",
    display_name="Matching Pennies (2P)"
)
class MatchingPenniesLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matrix_mp")


@registry.register(
    name="matrix_rps",
    module_path="game_reasoning_arena.arena.games.loaders",
    class_name="MatrixRPSLoader",
    environment_path="game_reasoning_arena.arena.envs.matrix_game_env.MatrixGameEnv",
    display_name="Matrix Rock-Paper-Scissors"
)
class MatrixRPSLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matrix_rps")
