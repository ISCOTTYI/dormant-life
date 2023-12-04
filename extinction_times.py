import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use("science")
from gol import GameOfLife, DormantLife, CellularAutomaton


def alive_cells(GoL: CellularAutomaton, grid_size: int,
                runs: int, t_max: int , base_seed: int = None) -> np.array:
    data = np.zeros((runs, t_max + 1))
    if base_seed is None:
        base_seed = np.random.randint(1)
    for i in range(runs):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        init_grid = rng.choice([0, 1], p=[0.80, 0.20], size=[grid_size, grid_size])
        gol = GoL(init_grid)
        N_alive_0 = gol.alive_count
        for j in range(t_max + 1):
            data[i, j] = gol.alive_count / N_alive_0
            gol.step()
    return np.mean(data, axis=0)


if __name__ == "__main__":
    t_max = 2000
    times = np.arange(t_max + 1)
    runs = 100
    base_seed = 100
    dl_30x30 = alive_cells(DormantLife, 30, runs, t_max, base_seed)
    fig, (axl, axr) = plt.subplots(figsize=(7, 3), ncols=2, sharey=True)
    axl.set(xlabel=r"$t$", ylabel=r"$\#\text{ALIVE}/\#\text{ALIVE}_0$",
            box_aspect=3/4, title="(a)")
    axl.plot(times, dl_30x30, label="Dormant Life", color="tab:orange")
    axl.plot(times, alive_cells(GameOfLife, 30, runs, t_max, base_seed),
             label="Game of Life", color="tab:blue")
    axl.legend()
    axr.set(xlabel=r"$t$", box_aspect=3/4, title="(b)")
    axr.plot(times, dl_30x30, label=r"$N = 30$", color="tab:orange")
    axr.plot(times, alive_cells(DormantLife, 10, runs, t_max, base_seed),
             label=r"$N = 10$", color=mpl.colormaps["Oranges"](0.4))
    axr.plot(times, alive_cells(DormantLife, 5, runs, t_max, base_seed),
             label=r"$N = 5$", color=mpl.colormaps["Oranges"](0.3))
    axr.legend()
    plt.show()
    