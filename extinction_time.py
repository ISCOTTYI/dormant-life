import sys, os
import numpy as np
import multiprocessing
from gol import SporeLife
from util import random_init_grid, save_data


BASE_PATH = "./data/spore-life/extinction-time"
PARAMS = (grid_size, q, t_max, runs, equal_step_limit) = (
          30, 0.3701, 1_000_000, 3, 100)
ALPHAS = np.linspace(0, 0.5, 30)


def find_extinction_time(sl: SporeLife, t_max: int,
                         equal_step_limit: int = 100) -> int:
    """
    Find the time step where the DormantLife dl goes extinct, where extinction
    is characterized by the number of alive cells staying constant for at least
    equal_step_limit steps. If sl goes extinct after t_max, -1 is returned.
    """
    assert 0 < equal_step_limit < t_max
    assert sl.t == 0
    equal_step_counter = 0
    while sl.t <= t_max:
        if equal_step_counter >= equal_step_limit:
            return sl.t - equal_step_limit
        old_alive_count = sl.alive_count
        sl.step()
        if old_alive_count == sl.alive_count:
            equal_step_counter += 1
        else:
            equal_step_counter = 0
    return -1


def extinction_time_stastistics(alpha, grid_size, q, t_max, runs, equal_step_limit,
                                progress_updates: bool = False) -> np.array:
    """
    Find extinction times for SporeLife with grid_size x grid_size and
    inital probability for ALIVE cells q. Returns runs realizations as
    data array.
    """
    data = np.zeros(runs)
    for i in range(runs):
        if progress_updates:
            sys.stdout.write(f"\r{round(i/runs * 100, 1)}%")
            sys.stdout.flush()
        sl = SporeLife(random_init_grid(grid_size, q), alpha=alpha)
        data[i] = find_extinction_time(sl, t_max, equal_step_limit)
    return data


def _f(alpha):
    print(f"\nalpha = {alpha}")
    header = f"extinction_time.py -- Extinction times for spore life -- alpha = {alpha}, (grid_size, q, t_max, runs, equal_step_limit) = {str(PARAMS)}"
    data = extinction_time_stastistics(alpha, *PARAMS, progress_updates=True)
    save_data(data, param=alpha, header=header, base_path=BASE_PATH, prefix="alpha-")


if __name__ == "__main__":
    save_data(ALPHAS, prefix="alpha-range", header="extinction_time.py -- Extinction times for spore life -- Alpha values for which data is stored",
              base_path=BASE_PATH)

    # Use the with statement to create and close the pool
    with multiprocessing.Pool(processes=5) as pool:
        # Use imap_unordered to apply the function to each value of alpha in parallel
        # and yield the results as they are ready, regardless of the order
        for _ in pool.imap_unordered(_f, ALPHAS):
            pass

