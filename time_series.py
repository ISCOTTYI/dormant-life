import sys, os
import numpy as np
import multiprocessing
from gol import SporeLife
from util import random_init_grid, save_data


PARAMS = (grid_size, q, t_max, runs) = (
          20, 0.3701, 10_000, 200)
BASE_PATH = f"data/spore-life/time-series/grid-size-{grid_size}"
ALPHAS = np.linspace(0, 1, 50)

def alive_dorm_time_series(sl: SporeLife,
                           t_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the time series for the given SporeLife sl until t_max returning
    the number of alive cells and dorm cells on the way.
    """
    assert 0 <= sl.t < t_max
    t0 = sl.t
    alive_data, dorm_data = np.zeros(t_max - sl.t), np.zeros(t_max - sl.t)
    while sl.t < t_max:
        alive_data[sl.t-t0] = sl.alive_count
        dorm_data[sl.t-t0] = sl.spore_count
        sl.step()
    return alive_data, dorm_data


def time_series_statistics(alpha: float, grid_size: int, q: float, t_max: int,
                           runs: int, progress_updates: bool = True):
    """
    Compute ALIVE and SPORE time series for DormantLife on grid_size x grid_size
    grid with initial alive probability q. Returns runs time series as data
    arrays, one for ALIVE one for SPORE
    """
    alive_data, spore_data = np.zeros((runs, t_max)), np.zeros((runs, t_max))
    for i in range(runs):
        if progress_updates:
            sys.stdout.write(f"\r{round(i/runs * 100, 1)}%")
            sys.stdout.flush()
        sl = SporeLife(random_init_grid(grid_size, q), alpha=alpha)
        alive_data[i], spore_data[i] = alive_dorm_time_series(sl, t_max)
    return alive_data, spore_data


def _f(alpha):
    print(f"\nalpha = {alpha}")
    alive_header = f"time_series.py -- Number of ALIVE cells over time in SporeLife -- alpha = {alpha}, (grid_size, q, t_max, runs) = {str(PARAMS)}"
    spore_header = f"time_series.py -- Number of SPORE cells over time in SporeLife -- alpha = {alpha}, (grid_size, q, t_max, runs) = {str(PARAMS)}"
    alive_data, spore_data = time_series_statistics(alpha, *PARAMS)
    save_data(alive_data, param=alpha, header=alive_header, base_path=BASE_PATH,
              prefix="alpha-", sub_path="alive")
    save_data(spore_data, param=alpha, header=spore_header, base_path=BASE_PATH,
              prefix="alpha-", sub_path="spore")


if __name__ == "__main__":
    save_data(ALPHAS, prefix="alpha-range", header="time_series.py -- Number of ALIVE / SPORE cells over time in SporeLife -- Alpha values for which data is stored",
              base_path=BASE_PATH)
    
    with multiprocessing.Pool(processes=6) as pool:
        # Use imap_unordered to apply the function to each value of alpha in 
        # parallel and yield the results as they are ready, regardless of the
        # order 
        for _ in pool.imap_unordered(_f, ALPHAS):
            pass
