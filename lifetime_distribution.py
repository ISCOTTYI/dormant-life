import sys, os
import numpy as np
from collections import Counter
from gol import CellularAutomaton, DormantLife
from gol import ALIVE, DEAD, DORM
from util import random_init_grid
import matplotlib.pyplot as plt


def lifetime_distribution(state: int, dl: DormantLife, t_max: int, t_trans: int,
                          ignore_transient_dynamics: bool = True) -> Counter:
    """
    Generate the lifetime distribution for a given state. The given DormantLife
    is stepped until t_max (skipping the transient time t_trans) and the
    distibution how long any cell in the grid stayed in the state state before
    transitioning is returned as a counter. Optionally, ignore cells that
    transition to state within the transient time.
    """
    lifetime_grid = np.zeros((dl.N, dl.N), dtype=np.intc)
    dl.step_until(t_trans)
    # Mask to exclude cells that are state from the get-go
    if ignore_transient_dynamics:
        trans_mask = dl.grid != state
    else:
        trans_mask = np.ones((dl.N, dl.N), dtype=np.bool_)
    data = Counter() # initialize distribution
    while dl.t <= t_max:
        old_grid = dl.grid.copy()
        new_grid = dl.step()
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


def tau_distribution(dl: DormantLife, t_max: int, t_trans: int,
                     ignore_transient_dynamics: bool = True) -> Counter:
    tau_grid = np.zeros((dl.N, dl.N), dtype=np.intc)
    dl.step_until(t_trans)
    if ignore_transient_dynamics:
        trans_mask = dl.grid != DORM
    else:
        trans_mask = np.ones((dl.N, dl.N), dtype=np.bool_)
    data = Counter()
    while dl.t <= t_max:
        old_grid = dl.grid.copy()
        new_grid = dl.step()
        # Count tau
        tau_grid += (trans_mask & ((old_grid == DORM) & (new_grid == DORM)))
        # Update distribution
        data.update(tau_grid[(trans_mask & (new_grid == DORM))])
        # Reset tau grid
        # NOTE: Could also add trans_mask & in front, but since tau grid is not updated for cells that are excluded through trans_mask it does not make a difference
        tau_grid[(old_grid == DORM) & (new_grid != DORM)] = 0
        # Update trans_mask
        trans_mask |= (np.logical_not(trans_mask) & (new_grid != DORM))
    return data


if __name__ == "__main__":
    # # CHANGE N
    # t_max = 10_000
    # fig,(ax, axx)=plt.subplots(figsize=(3*7.2, 3*3.2), ncols=2)
    # alpha = .4
    # for N in [30, 300]:
    #     init_grid = random_init_grid(N, seed=100)
    #     dl = DormantLife(init_grid, alpha, seed=100)
    #     distr = lifetime_distribution(DORM, dl, t_max, 150, ignore_transient_dynamics=1)
    #     ax.scatter(distr.keys(), distr.values(), label=N)
    #     dl = DormantLife(init_grid, alpha, seed=100)
    #     axx.plot(dl.dorm_count_time_series(t_max))
    # # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.legend()
    # axx.legend()


    # # CHANGE ALPHA
    # t_max = 10_000
    # fig,ax=plt.subplots(ncols=2)
    # for alpha in [1, .1]:
    #     init_grid = random_init_grid(100, seed=100)
    #     dl = DormantLife(init_grid, alpha, seed=100)
    #     distr = lifetime_distribution(DORM, dl, t_max, 150, ignore_transient_dynamics=1)
    #     ax[0].scatter(distr.keys(), distr.values(), label=alpha)
    #     dl = DormantLife(init_grid, alpha, seed=100)
    #     ax[1].plot(dl.dorm_count_time_series(t_max))
    #     print(sum(list(distr.values())))
    # # ax[0].set_xscale("log")
    # ax[0].set_yscale("log")
    # ax[0].legend()
    # ax[1].legend()
    

    # TAU DISTRIBUTION
    t_max, t_trans = 10_000, 200
    fig, ax = plt.subplots()
    ax.set(yscale="log")
    for alpha in [1, .8, .5, .2]:
        init_grid = random_init_grid(100, seed=100)
        dl = DormantLife(init_grid, alpha, seed=100)
        distr = tau_distribution(dl, t_max, t_trans)
        vals, counts = np.fromiter(distr.keys(), dtype=np.intc), np.fromiter(distr.values(), dtype=np.intc)
        ax.scatter(vals, counts/(t_max-t_trans), label=alpha)
    ax.legend()
    plt.show()
