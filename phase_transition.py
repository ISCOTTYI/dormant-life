import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use("science")
from gol import DormantLife


def DL_alive_cells(grid_size: int, runs: int, 
                   t_max: int, p: float, base_seed: int = None) -> np.array:
    data = np.zeros((runs, t_max + 1))
    if base_seed is None:
        base_seed = np.random.randint(1)
    for i in range(runs):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        init_grid = rng.choice([0, 1], p=[0.80, 0.20], size=[grid_size, grid_size])
        gol = DormantLife(init_grid)
        N_alive_0 = gol.alive_count
        for j in range(t_max + 1):
            data[i, j] = gol.alive_count / N_alive_0
            gol.step(p=p)
    return data


if __name__ == "__main__":
    grid_size = 30
    runs = 100
    t_max = 2000
    times = np.arange(t_max + 1)
    base_seed = 100
    fig, ax = plt.subplots()
    ax.set(xlabel=r"$t$", ylabel=r"$\#\text{ALIVE}/\#\text{ALIVE}_0$")
    ax.plot(times, np.mean(DL_alive_cells(grid_size, runs, t_max, 0, base_seed), axis=0),
            label=r"$p = 0$ (Game of Life)")
    ax.plot(times, np.mean(DL_alive_cells(grid_size, runs, t_max, .01, base_seed), axis=0),
            label=r"$p = 0.01$")
    ax.plot(times, np.mean(DL_alive_cells(grid_size, runs, t_max, .1, base_seed), axis=0),
            label=r"$p = 0.1$")
    ax.plot(times, np.mean(DL_alive_cells(grid_size, runs, t_max, .5, base_seed), axis=0),
            label=r"$p = 0.5$")
    ax.plot(times, np.mean(DL_alive_cells(grid_size, runs, t_max, 1, base_seed), axis=0),
            label=r"$p = 1$ (Dormant Life)")
    ax.legend()
    fig.savefig("./img/phase_transition", dpi=500)


    t_max = 5000
    ps = np.linspace(0, 1, 25)
    N_alive_final = [
        np.mean(
            DL_alive_cells(grid_size, runs, t_max, p, base_seed)[:, -1]
        ) for p in ps
    ]
    np.savetxt("./data/phase_transition_p.dat", (ps, N_alive_final), header="Probability DORM -> ALIVE transition with 2 neighbors, fraction of ALIVE cells after t_max=5000. grid_size = 30, runs = 100, base_seed = 100, init grid 80/20 DEAD/ALIVE")
    fig, ax = plt.subplots()
    ax.set(xlabel=r"$p$", ylabel=r"$\#\text{ALIVE}/\#\text{ALIVE}_0(t_\text{max})$")
    ax.plot(ps, N_alive_final)
    fig.savefig("./img/phase_transition2", dpi=500)
