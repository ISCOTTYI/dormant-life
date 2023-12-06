import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import binom
plt.style.use("science")
from gol import GameOfLife, DormantLife, CellularAutomaton


def alive_cells(GoL: CellularAutomaton, grid_size: int, runs: int, t_max: int,
                base_seed: int = None, init_p_ALIVE: float = 0.37017384,
                init_p_DORM: float = 0.0) -> np.array:
    assert 0 <= init_p_ALIVE <= 1
    assert 0 <= init_p_DORM <= 1
    assert init_p_DORM + init_p_ALIVE <= 1
    data = np.zeros((runs, t_max + 1))
    if base_seed is None:
        base_seed = np.random.randint(1)
    for i in range(runs):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        p = [1-init_p_ALIVE-init_p_DORM, init_p_ALIVE, init_p_DORM]
        init_grid = rng.choice([0, 1, 2], p=p, size=[grid_size, grid_size])
        gol = GoL(init_grid)
        for j in range(t_max + 1):
            data[i, j] = gol.alive_count
            gol.step()
    return np.mean(data, axis=0)


if __name__ == "__main__":
    t_max = 2000
    times = np.arange(t_max + 1)
    runs = 100
    base_seed = 100
    init_p_ALIVE = 0.37017384
    
    dl_30x30 = alive_cells(DormantLife, 30, runs, t_max, base_seed, init_p_ALIVE)
    dl_10x10 = alive_cells(DormantLife, 10, runs, t_max, base_seed, init_p_ALIVE)
    dl_6x6 = alive_cells(DormantLife, 6, runs, t_max, base_seed, init_p_ALIVE)

    fig, (axl, axr) = plt.subplots(figsize=(8, 3), ncols=2)
    axl.set(xlabel=r"$t$", ylabel=r"$\#\text{ALIVE}$",
            box_aspect=3/4, title="(a)")
    gol, = axl.plot(times, alive_cells(GameOfLife, 30, runs, t_max, base_seed, init_p_ALIVE),
                    label="Game of Life", color="tab:blue")
    dl, = axl.plot(times, dl_30x30,
                    label="Dormant Life", color="tab:orange")
    handles = [dl, gol]
    axl.legend(handles=handles, labels=[h.get_label() for h in handles])
    axr.set(xlabel=r"$t$", ylabel=r"ALIVE fraction", box_aspect=3/4, title="(b)")
    axr.plot(times, dl_30x30/dl_30x30[0], label=r"$N = 30$", color="tab:orange")
    axr.plot(times, dl_10x10/dl_10x10[0],
             label=r"$N = 10$", color=mpl.colormaps["Oranges"](0.4))
    # axr.plot(times, alive_cells(DormantLife, 4, runs, t_max, base_seed, .3, .3),
    #          label=r"$N = 4, mit DORM$", color=mpl.colormaps["Oranges"](0.4))
    axr.plot(times, dl_6x6/dl_6x6[0],
             label=r"$N = 6$", color=mpl.colormaps["Oranges"](0.2))
    axr.legend(loc=(.25, .15))
    fig.savefig("./img/extinciton_times", dpi=500)
    plt.show()
    