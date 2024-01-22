import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from gol import GameOfLife, DormantLife, DEAD, ALIVE, DORM
from lifetime_distribution import lifetime_distribution

if __name__ == "__main__":
    # init_grid = np.random.choice([0, 1], p=[0.80, 0.20], size=[30, 30])
    init_grid = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ])
    gol = GameOfLife(init_grid)
    dl = DormantLife(init_grid, alpha=.3)
    colors = ["white", "tab:blue", "lightblue"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), ncols=2)
    ax[0].set(title=r"Game of Life: $N_\text{alive} = %d$"%gol.alive_count)
    ax[1].set(title=r"Dormant Life: $N_\text{alive} = %d$"%dl.alive_count)
    mat_gol = ax[0].matshow(gol.grid, cmap=cmap, vmin=0, vmax=2)
    mat_dl = ax[1].matshow(dl.grid, cmap=cmap, vmin=0, vmax=2)
    def update(frame):
        print(dl.p_grid)
        mat_gol.set_data(gol.grid)
        mat_dl.set_data(dl.grid)
        ax[0].set(title=r"Game of Life: $N_\text{alive} = %d$"%gol.alive_count)
        ax[1].set(title=r"Dormant Life: $N_\text{alive} = %d$"%dl.alive_count)
        gol.step()
        dl.step()
    ani = animation.FuncAnimation(fig, update, interval=1000, save_count=100)
    plt.show()
