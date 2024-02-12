import sys, os
import numpy as np
from collections import Counter
from gol import SporeLife
from gol import ALIVE, DEAD, SPORE
from util import random_init_grid


def lifetime_distribution(state: int, sl: SporeLife, t_max: int, t_trans: int,
                          ignore_transient_dynamics: bool = True) -> Counter:
    """
    Generate the lifetime distribution for a given state. The given SporeLife
    is stepped until t_max (skipping the transient time t_trans) and the
    distibution how long any cell in the grid stayed in the state state before
    transitioning is returned as a counter. Optionally, ignore cells that
    transition to state within the transient time.
    """
    lifetime_grid = np.zeros((sl.N, sl.N), dtype=np.intc)
    sl.step_until(t_trans)
    # Mask to exclude cells that are state from the get-go
    if ignore_transient_dynamics:
        trans_mask = (sl.grid != state)
    else:
        trans_mask = np.ones((sl.N, sl.N), dtype=np.bool_)
    data = Counter() # initialize distribution
    while sl.t < t_max:
        old_grid = sl.grid.copy()
        new_grid = sl.step()
        # Cells that are *still* in state add to lifetime grid
        lifetime_grid += (trans_mask & 
                          ((old_grid == state) & (new_grid == state)))
        # Add lifetimes of cells that left state and reset lifetime grid
        _mask = (trans_mask & 
                 ((old_grid == state) & (new_grid != state)))
        data.update(lifetime_grid[_mask])
        lifetime_grid[_mask] = 0
        # Update transient mask: set mask values to true for cells initially in
        # state that left state.
        trans_mask |= (np.logical_not(trans_mask) & (new_grid != state)) # FIXME
    return data
