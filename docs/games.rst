Games
=====

Board Game Arena supports multiple classic board games for AI agent training and evaluation.

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

Chess
~~~~~

Basic chess support for more complex strategic gameplay.

* **Players**: 2
* **State space**: Extremely large
* **Action space**: Variable (legal moves)
* **Game length**: Variable (can be very long)

.. code-block:: python

   env = EnvInitializer.create_env("chess")

Hex
~~~

A connection game played on a hexagonal grid.

* **Players**: 2
* **State space**: Large
* **Action space**: Variable board sizes
* **Game length**: Variable

.. code-block:: python

   env = EnvInitializer.create_env("hex")

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
