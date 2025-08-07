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

Iterated Prisoners Dilemma
~~~~~

Cooperation versus competition dynamics

* **Players**: 2
* **Action space**: 2
* **Game length**: Variable (can be very long)

.. code-block:: python

   env = EnvInitializer.create_env("iterated_prisoners_dilemma")


Matrix Rock-Paper-Scissors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A strategic variant of rock-paper-scissors with a matrix representation.

* **Players**: 2
* **State space**: 3x3 matrix
* **Action space**: 3 (Rock, Paper, Scissors)
* **Game length**: 1 round

.. code-block:: python

   env = EnvInitializer.create_env("matrix_rps")


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
