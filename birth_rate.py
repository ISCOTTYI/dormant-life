import sys, os
import numpy as np
import multiprocessing
from gol import CellularAutomaton, SporeLife
from gol import ALIVE, DEAD, SPORE
from util import random_init_grid, save_data


BASE_PATH = "data/spore-life/state-transitions/birth-rate"
PARAMS = (grid_size, q, t_max, t_trans, runs) = (
          100, 0.3701, 10_000, 200, 200)
ALPHAS = np.linspace(0, 1, 50)


def count_transitions(pre_grid: np.ndarray, post_grid: np.ndarray,
                      transition: tuple[int]) -> int:
    assert len(transition) == 2
    assert pre_grid.shape == post_grid.shape
    return np.sum(np.all(
        (pre_grid == transition[0], post_grid == transition[1]), axis=0))


def births_time_series(sl: SporeLife, t_max, t_trans):
    data = np.zeros(t_max+1-t_trans)
    sl.step_until(t_trans)
    while sl.t < t_max:
        pre_grid = sl.grid.copy()
        post_grid = sl.step()
        data[sl.t-t_trans] = (
            count_transitions(pre_grid, post_grid, (DEAD, ALIVE))
            + count_transitions(pre_grid, post_grid, (SPORE, ALIVE))
        )
    return data


def births_time_series_statistics(alpha, grid_size, q, t_max, t_trans, runs,
                                  progress_updates=True):
    data = np.zeros((runs, t_max+1-t_trans))
    for i in range(runs):
        if progress_updates:
            sys.stdout.write(f"\r{round(i/runs * 100, 1)}%")
            sys.stdout.flush()
        sl = SporeLife(random_init_grid(grid_size, q), alpha=alpha)
        data[i] = births_time_series(sl, t_max, t_trans)
    return data


def _f(alpha):
    print(f"\nalpha = {alpha}")
    header = f"birth_rate.py -- Birth counts (number of transitions to ALIVE) time series for SporeLife -- alpha = {alpha}, (grid_size, q, t_max, t_trans, runs) = {str(PARAMS)}"
    data = births_time_series_statistics(alpha, *PARAMS)
    save_data(data, param=alpha, header=header, base_path=BASE_PATH, prefix="alpha-")


if __name__ == "__main__":
    save_data(ALPHAS, prefix="alpha-range", header="birth_rate.py -- Birth counts (number of transitions to ALIVE) time series for SporeLife -- Alpha values for which data is stored",
              base_path=BASE_PATH)
    with multiprocessing.Pool(processes=5) as pool:
        # Use imap_unordered to apply the function to each value of alpha in 
        # parallel and yield the results as they are ready, regardless of the
        # order 
        for _ in pool.imap_unordered(_f, ALPHAS):
            pass
