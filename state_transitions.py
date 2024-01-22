import sys, os
import numpy as np
import multiprocessing
from gol import CellularAutomaton, DormantLife
from gol import ALIVE, DEAD, DORM
from util import random_init_grid
import matplotlib.pyplot as plt


BASE_PATH = "data/dormant-life/state-transitions/to-alive-transitions"
os.makedirs(BASE_PATH, exist_ok=True)


def count_transitions(pre_grid: np.ndarray, post_grid: np.ndarray,
                      transition: tuple[int]) -> int:
    assert len(transition) == 2
    assert pre_grid.shape == post_grid.shape
    return np.sum(np.all(
        (pre_grid == transition[0], post_grid == transition[1]), axis=0))


def step_and_count_transitions(ca: CellularAutomaton,
                               transitions: tuple[tuple[int, int]]) -> tuple[int]:
    """
    Perform one step of the passed cellular automaton and count transitions.
    Transitions are passed as a tuple of length 2 of states and the respective
    count is returned in the i-th element of the return tuple.
    """
    pre_grid = ca.grid.copy()
    post_grid = ca.step()
    return (count_transitions(pre_grid, post_grid, t) for t in transitions)


def dl_compute_alpha(grid_size: int, alpha: float, runs: int, t_max: int,
                     t_trans: int, transitions: tuple[tuple[int, int]],
                     progress_updates=True):
    data = np.zeros((runs, (t_max+1-t_trans)))
    for i in range(runs):
        if progress_updates:
            sys.stdout.write(f"\r{round(i/runs * 100, 1)}%")
            sys.stdout.flush()
        dl = DormantLife(random_init_grid(grid_size), alpha=alpha)
        dl.step_until(t_trans)
        while dl.t <= t_max:
            data[i, dl.t-t_trans-1] = sum(step_and_count_transitions(dl, transitions))
    return data


def compute(alpha):
    print(f"\nalpha = {alpha}")
    parameters = (grid_size, alpha, runs, t_max, t_trans, transitions) = (
        100, alpha, 200, 10_000, 200, ((DEAD, ALIVE), (DORM, ALIVE))
    )
    data = dl_compute_alpha(*parameters, progress_updates=True)
    fname = f"alpha-{str(alpha)[0]}p{str(alpha)[2:]}.dat"
    np.savetxt(os.path.join(BASE_PATH, fname), (data),
               header=f"(grid_size, alpha, runs, t_max, t_trans, transitions) = {str(parameters)}")


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
