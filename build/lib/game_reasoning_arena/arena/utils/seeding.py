"""
Random Seeding Utilities

Provides utilities for setting random seeds across different libraries
to ensure reproducible results in experiments.
"""

import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Sets the global seed for reproducibility across all used random number generators.

    Args:
        seed (int): The seed value to be used for all random operations.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set PyTorch seed if used
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
