import sys, os
import numpy as np
import multiprocessing
from gol import SporeLife
from gol import ALIVE, DEAD, SPORE
from util import random_init_grid, save_data

from time import time
TT = 0

BASE_PATH = "data/spore-life/state-transitions/transition-table/"
PARAMS = (grid_size, q, t_max, t_trans, runs) = (
          300, 0.3701, 10_000, 200, 30)
ALPHAS = np.linspace(0, 1, 50)

# FIXME: Boost performance for count computation
def count_transitions(sl: SporeLife) -> np.ndarray:
    """
    Find counts for SporeLife rule table:
        STATE | ALIVE NEIGHBORS
        D     | D D D A D D D D D
        A     | D S A A D D D D D
        S     | S S A A S S S S S
    Each element contains the respective counts of ALIVE neighbors.
    """
    global TT
    counts = np.zeros((3, 9))
    c = sl.life_neighborhood_grid
    # Fill table with counts
    t0 = time()
    counts[DEAD] = [np.sum((sl.grid == DEAD) & (c == n)) for n in range(9)]
    counts[ALIVE] = [np.sum((sl.grid == ALIVE) & (c == n)) for n in range(9)]
    counts[SPORE] = [np.sum((sl.grid == SPORE) & (c == n)) for n in range(9)]
    TT += time() - t0
    return counts


def count_transitions_time_avg(sl: SporeLife, t_max: int, t_trans: int):
    assert sl.t + t_trans <= t_max
    sl.step_until(t_trans)
    data = np.zeros((3, 9))
    while sl.t < t_max:
        data += count_transitions(sl)
        sl.step()
    return data / (t_max - t_trans)


def count_transitions_run_avg(alpha, grid_size, q, t_max, t_trans, runs,
                              progress_updates=True):
    data = np.zeros((3, 9))
    for i in range(runs):
        if progress_updates:
            sys.stdout.write(f"\r{round(i/runs * 100, 1)}%")
            sys.stdout.flush()
        sl = SporeLife(random_init_grid(grid_size, q), alpha=alpha)
        data += count_transitions_time_avg(sl, t_max, t_trans)
    return data / runs


def _f(alpha):
    print(f"\nalpha = {alpha}")
    header = f"transitions.py -- Counts for different transitions in transition table, rows are DEAD, ALIVE, SPORE, cols are number of ALIVE neighbors, data averaged over time and runs -- alpha = {alpha}, (grid_size, q, t_max, t_trans, runs) = {str(PARAMS)}"
    data = count_transitions_run_avg(alpha, *PARAMS)
    save_data(data, param=alpha, header=header, base_path=BASE_PATH, prefix="alpha-")


if __name__ == "__main__":
    save_data(ALPHAS, prefix="alpha-range", header="transitions.py -- Counts for different transitions in transition table -- Alpha values for which data is stored",
              base_path=BASE_PATH)
    with multiprocessing.Pool(processes=6) as pool:
        # Use imap_unordered to apply the function to each value of alpha in 
        # parallel and yield the results as they are ready, regardless of the
        # order 
        for _ in pool.imap_unordered(_f, ALPHAS):
            pass
