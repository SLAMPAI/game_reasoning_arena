Games
=====

Game Reasoning Arena supports multiple classic games for AI agent training and evaluation.

Supported Games
---------------

Connect Four
~~~~~~~~~~~~

A classic connection game where players drop colored discs into a grid, trying to connect four in a row.

* **Players**: 2
* **State space**: Medium complexity
* **Action space**: 7 possible columns
* **Game length**: Variable (typically 10-42 moves)

.. code-block:: python

   env = EnvInitializer.create_env("connect_four")

Tic-Tac-Toe
~~~~~~~~~~~

The classic 3x3 grid game where players try to get three in a row.

* **Players**: 2
* **State space**: Small (362,880 possible states)
* **Action space**: 9 possible positions
* **Game length**: 5-9 moves

.. code-block:: python

   env = EnvInitializer.create_env("tic_tac_toe")

Kuhn Poker
~~~~~~~~~~

A simplified poker variant that's perfect for AI research.

* **Players**: 2
* **State space**: Small but with hidden information
* **Action space**: Fold, Call, or Bet
* **Game length**: 1-2 rounds

.. code-block:: python

   env = EnvInitializer.create_env("kuhn_poker")


Prisoner's Dilemma (Matrix Form)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classic two-player dilemma game in matrix form, modeling cooperation vs defection.

* **Players**: 2
* **State space**: 2x2 matrix
* **Action space**: 2 (Cooperate, Defect)
* **Game length**: 1 round

.. code-block:: python

   env = EnvInitializer.create_env("prisoners_dilemma")


Matrix Prisoner's Dilemma
~~~~~~~~~~~~~~~~~~~~~~~~~

Matrix representation of the prisoner's dilemma, useful for payoff analysis and agent strategy.

* **Players**: 2
* **State space**: 2x2 matrix
* **Action space**: 2 (Cooperate, Defect)
* **Game length**: 1 round

.. code-block:: python

   env = EnvInitializer.create_env("matrix_pd")


Matching Pennies
~~~~~~~~~~~~~~~~

Zero-sum game where each player chooses heads or tails; one wins if the choices match, the other if they differ.

* **Players**: 2
* **State space**: 2x2 matrix
* **Action space**: 2 (Heads, Tails)
* **Game length**: 1 round

.. code-block:: python

   env = EnvInitializer.create_env("matching_pennies")


Matrix Rock-Paper-Scissors
~~~~~~~~~~~~~~~~~~~~~~~~~~

A strategic variant of rock-paper-scissors with a matrix representation.

* **Players**: 2
* **State space**: 3x3 matrix
* **Action space**: 3 (Rock, Paper, Scissors)
* **Game length**: 1 round

.. code-block:: python

   env = EnvInitializer.create_env("matrix_rps")


Hex
~~~

Abstract connection game played on a hexagonal grid. Players aim to connect opposite sides of the board.

* **Players**: 2
* **State space**: Large (depends on board size)
* **Action space**: Number of empty hexes
* **Game length**: Variable (until one player connects their sides)

.. code-block:: python

   env = EnvInitializer.create_env("hex")


Chess
~~~~~

Classic 8x8 board game of strategy and tactics, featuring a variety of pieces and complex rules.

* **Players**: 2
* **State space**: Extremely large
* **Action space**: Varies by position (moves, captures, special moves)
* **Game length**: Variable (typically 20-60 moves)

.. code-block:: python

   env = EnvInitializer.create_env("chess")


Game Properties
---------------

Each game environment provides:

* **State representation**: Current game state
* **Legal actions**: Available moves for the current player
* **Game termination**: Win/loss/draw detection
* **Reward structure**: Scoring system for agent training

Adding New Games
----------------

To add support for a new game, see the :doc:`contributing` guide for details on implementing the game interface.
