Contributing
============

We welcome contributions to Board Game Arena! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/board_game_arena.git
   cd board_game_arena

3. Create a development environment:

.. code-block:: bash

   conda env create -f environment.yaml
   conda activate board_game_arena
   pip install -e .

4. Install development dependencies:

.. code-block:: bash

   pip install pytest black flake8 mypy

Code Style
----------

We follow PEP 8 style guidelines. Please ensure your code is formatted with `black`:

.. code-block:: bash

   black src/ tests/

Run linting checks:

.. code-block:: bash

   flake8 src/ tests/
   mypy src/

Testing
-------

Run the test suite before submitting changes:

.. code-block:: bash

   pytest tests/

Add tests for new functionality:

.. code-block:: python

   # tests/test_new_feature.py
   import pytest
   from board_game_arena.your_module import YourClass

   def test_your_function():
       # Test implementation
       assert True

Documentation
-------------

Update documentation for new features:

1. Add docstrings to new functions and classes
2. Update relevant `.rst` files in the `docs/` directory
3. Build documentation locally to verify:

.. code-block:: bash

   cd docs/
   make html

Submitting Changes
------------------

1. Create a new branch for your feature:

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. Make your changes and commit:

.. code-block:: bash

   git add .
   git commit -m "Add your feature description"

3. Push to your fork:

.. code-block:: bash

   git push origin feature/your-feature-name

4. Create a Pull Request on GitHub

Pull Request Guidelines
-----------------------

* Include a clear description of the changes
* Reference any related issues
* Ensure all tests pass
* Update documentation as needed
* Keep changes focused and atomic

Adding New Games
----------------

To add support for a new game:

1. Create a new environment class in `src/board_game_arena/arena/envs/`
2. Inherit from the base environment class
3. Implement required methods: `reset()`, `step()`, `get_legal_actions()`
4. Add configuration support
5. Include tests and documentation

Example:

.. code-block:: python

   from .base_env import BaseEnv
   
   class NewGameEnv(BaseEnv):
       def __init__(self, config):
           super().__init__(config)
           # Game-specific initialization
       
       def reset(self):
           # Reset game state
           pass
       
       def step(self, action):
           # Execute action and return new state
           pass

Adding New Agents
-----------------

To add a new agent type:

1. Create a new agent class in `src/board_game_arena/arena/agents/`
2. Inherit from `BaseAgent`
3. Implement `get_action()` and `reset()` methods
4. Add to agent registry if needed

Bug Reports
-----------

When reporting bugs, please include:

* Python version and operating system
* Full error traceback
* Minimal example to reproduce the issue
* Expected vs actual behavior

Feature Requests
----------------

For feature requests:

* Describe the use case and motivation
* Provide examples of how the feature would be used
* Consider implementation complexity

Community
---------

* Join our discussions on GitHub Issues
* Follow our coding standards and be respectful
* Help review other contributors' pull requests

Thank you for contributing to Board Game Arena!
