import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap
from gol import GameOfLife, SporeLife, DEAD, ALIVE, SPORE
from util import random_init_grid

if __name__ == "__main__":
    init_grid = random_init_grid(30, seed=104)
    gol = GameOfLife(init_grid, periodic_boundary=True)
    dl = SporeLife(init_grid, alpha=1, periodic_boundary=True)
    colors = ["white", "tab:orange", "tab:blue"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), ncols=2)
    ax[0].set(title=r"Game of Life: $N_\text{alive} = %d$"%gol.alive_count,
            #   xlim=(-1,31), ylim=(-1,31),
              xticklabels=[], yticklabels=[])
    ax[1].set(title=r"Dormant Life: $N_\text{alive} = %d$"%dl.alive_count,
            #   xlim=(-1,31), ylim=(-1,31),
              xticklabels=[], yticklabels=[])
    mat_gol = ax[0].matshow(gol.grid, cmap=cmap, vmin=0, vmax=2)
    mat_dl = ax[1].matshow(dl.grid, cmap=cmap, vmin=0, vmax=2)
    def update(frame):
        print(frame)
        mat_gol.set_data(gol.grid)
        mat_dl.set_data(dl.grid)
        ax[0].set(title=r"Game of Life: $N_\text{alive} = %d$"%gol.alive_count)
        ax[1].set(title=r"Spore Life ($\alpha = 1$): $N_\text{alive} = %d$"%dl.alive_count)
        gol.step()
        dl.step(overcrowd_dormancy=True)
    ani = animation.FuncAnimation(fig, update, interval=10, save_count=100, frames=300)
    # Save the animation as a .gif file
    fig.tight_layout()
    ani.save('./img/spore_life.gif', writer='ffmpeg', fps=30, dpi=500)
    plt.show()
