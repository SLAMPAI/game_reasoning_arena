"""
Random Seeding Utilities

Provides utilities for setting random seeds across different libraries
to ensure reproducible results in experiments.
"""

import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None

def set_seed(seed: int) -> None:
    """Sets the global seed for reproducibility across all used random number generators.

    Args:
        seed (int): The seed value to be used for all random operations.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    # Set PyTorch seed if used
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
