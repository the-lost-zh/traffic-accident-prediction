import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set common random seeds used by training and tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

