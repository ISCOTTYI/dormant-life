import sys, os
import numpy as np
import multiprocessing
from gol import DormantLife


GRID_SIZE = 20
BASE_PATH = f"data/dormant-life/time-series/grid-size-{GRID_SIZE}"
os.makedirs(BASE_PATH, exist_ok=True)


def alive_dorm_time_series(dl: DormantLife,
                           t_max: int) -> tuple[np.array, np.array]:
    """
    Computes the time series for the given DormantLife dl until t_max returning
    the number of alive cells and dorm cells on the way.
    """
    assert 0 <= dl.t < t_max
    alive_data, dorm_data = np.zeros(t_max+1 - dl.t), np.zeros(t_max+1 - dl.t)
    i = 0
    while dl.t <= t_max:
        alive_data[i] = dl.alive_count
        dorm_data[i] = dl.dorm_count
        dl.step()
        i += 1
    return alive_data, dorm_data


def dormant_life_time_series(grid_size: int,
                             q: float,
                             alpha: float,
                             t_max: int,
                             runs: int,
                             progress_updates: bool = False) -> tuple[np.array, np.array]:
    """
    Compute ALIVE and DORM time series for DormantLife on grid_size x grid_size
    grid with initial alive probability q. Returns runs time series as data
    arrays, one for ALIVE one for DORM
    """
    alive_data, dorm_data = np.zeros((runs, t_max+1)), np.zeros((runs, t_max+1))
    for i in range(runs):
        if progress_updates:
            sys.stdout.write(f"\r{round(i/runs * 100, 1)}%")
            sys.stdout.flush()
        init_grid = np.random.choice([0, 1], p=[1-q, q], size=[grid_size, grid_size])
        dl = DormantLife(init_grid, alpha=alpha)
        alive_data[i], dorm_data[i] = alive_dorm_time_series(dl, t_max)
    return alive_data, dorm_data


def save_data(alive_data, dorm_data, alpha, header: str):
    fname = f"alpha-{str(alpha)[0]}p{str(alpha)[2:]}.dat"
    os.makedirs(os.path.join(BASE_PATH, "alive"), exist_ok=True)
    np.savetxt(os.path.join(BASE_PATH, "alive", fname),
               (alive_data), header=header)
    os.makedirs(os.path.join(BASE_PATH, "dorm"), exist_ok=True)
    np.savetxt(os.path.join(BASE_PATH, "dorm", fname),
               (dorm_data), header=header)


def compute(alpha):
    print(f"\nalpha = {alpha}")
    parameters = (grid_size, q, alpha, t_max, runs) = (
        GRID_SIZE, 0.3701, alpha, 10_000, 1000
    )
    alive_data, dorm_data = dormant_life_time_series(*parameters,
                                                     progress_updates=True)
    save_data(alive_data, dorm_data, alpha,
              header=f"(grid_size, q, alpha, t_max, runs) = {str(parameters)}")


if __name__ == "__main__":
    alphas = np.round(np.linspace(0, 1, 50), 3)
    np.savetxt(os.path.join(BASE_PATH, "alpha-range.dat"),
               (alphas), header="Alpha values for which data is stored.")
    
    with multiprocessing.Pool(processes=4) as pool:
        # Use imap_unordered to apply the function to each value of alpha in 
        # parallel and yield the results as they are ready, regardless of the
        # order 
        for _ in pool.imap_unordered(compute, alphas):
            pass
