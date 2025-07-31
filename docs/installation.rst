Installation
============

This guide will help you set up Board Game Arena on your system.

Prerequisites
-------------

* Python 3.10 or higher
* Conda (recommended) or pip
* Git

Installing with Conda (Recommended)
------------------------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/lcipolina/board_game_arena.git
   cd board_game_arena

2. Create and activate the conda environment:

.. code-block:: bash

   conda env create -f environment.yaml
   conda activate board_game_arena

3. Install the package in development mode:

.. code-block:: bash

   pip install -e .

Installing with pip
-------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/lcipolina/board_game_arena.git
   cd board_game_arena

2. Create a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Verification
------------

To verify your installation, run:

.. code-block:: bash

   python -c "import board_game_arena; print('Installation successful!')"

Next Steps
----------

After installation, check out the :doc:`quickstart` guide to run your first experiment.
