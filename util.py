import numpy as np


def random_init_grid(grid_size: int, 
                     q: float = 0.3701, seed=None) -> np.ndarray:
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    assert 0 <= q <= 1
    return rng.choice([0, 1], p=[1-q, q], size=[grid_size, grid_size])
