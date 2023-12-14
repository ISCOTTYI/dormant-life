import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("science")
from gol import CellularAutomaton, DormantLife


def find_extinction_time(ca: CellularAutomaton, t_max: int,
                         equal_step_limit: int = 100):
    """
    Find the time step where the DormantLife dl goes extinct, where extinction
    is characterized by the number of alive cells staying constant for at least
    equal_step_limit steps.
    """
    assert 0 < equal_step_limit < t_max
    assert ca.t == 0
    equal_step_counter = 0
    while ca.t <= t_max:
        if equal_step_counter >= equal_step_limit:
            return ca.t - equal_step_limit
        old_alive_count = ca.alive_count
        ca.step()
        if old_alive_count == ca.alive_count:
            equal_step_counter += 1
        else:
            equal_step_counter = 0
    return -1


def dormant_life_extinction_times(grid_size: int,
                                  q: float,
                                  alpha: float,
                                  t_max: int,
                                  runs: int,
                                  equal_step_limit: int = 100,
                                  progress_updates: bool = False) -> np.array:
    """
    Find extinction times for Dormant Life with grid_size x grid_size and
    inital probability for ALIVE cells q. Returns runs realizations as
    data array.
    """
    data = np.zeros(runs)
    for i in range(runs):
        if progress_updates:
            sys.stdout.write(f"\r{round(i/runs * 100, 1)}%")
            sys.stdout.flush()
        init_grid = np.random.choice([0, 1], p=[1-q, q], size=[grid_size, grid_size])
        dl = DormantLife(init_grid, alpha=alpha)
        data[i] = find_extinction_time(dl, t_max, equal_step_limit)
    return data


if __name__ == "__main__":
    alphas = np.round(np.linspace(0, 0.4, 30), 3)
    np.savetxt("./data/dormant-life/extinction-time/alpha-range.dat",
               (alphas), header="Alpha values for which data is stored.")
    for alpha in alphas:
        print(f"alpha = {alpha}")
        fname = f"alpha-p{str(alpha)[2:]}"
        parameters = (grid_size, q, alpha, t_max, runs) = (
            30, 0.3701, alpha, 100_000, 5000
        )
        data = dormant_life_extinction_times(*parameters, progress_updates=True)
        np.savetxt("./data/dormant-life/extinction-time/alpha-p35.dat",
                   (data),
                   header=f"(grid_size, q, alpha, t_max, runs) = {str(parameters)}")
