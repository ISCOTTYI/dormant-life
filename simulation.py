import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap
from gol import GameOfLife, SporeLife, DEAD, ALIVE, SPORE
from util import random_init_grid

if __name__ == "__main__":
    # init_grid = random_init_grid(30, seed=104)
    init_grid = random_init_grid(30, q=0.17)
    gol = GameOfLife(init_grid)
    dl = SporeLife(init_grid, alpha=1)
    colors = ["white", "tab:orange", "tab:blue"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), ncols=2)
    time_text = fig.text(0.0, 0.0, f"time = {0}", fontsize=12) 
    ax[0].set(title=r"Game of Life: $N_\text{alive} = %d$"%gol.alive_count,
            #   xlim=(-1,31), ylim=(-1,31),
              xticklabels=[], yticklabels=[])
    ax[1].set(title=r"Dormant Life: $N_\text{alive} = %d$"%dl.alive_count,
            #   xlim=(-1,31), ylim=(-1,31),
              xticklabels=[], yticklabels=[])
    mat_gol = ax[0].matshow(gol.grid, cmap=cmap, vmin=0, vmax=2)
    mat_dl = ax[1].matshow(dl.grid, cmap=cmap, vmin=0, vmax=2)
    def update(frame):
        # print(frame)
        mat_gol.set_data(gol.grid)
        mat_dl.set_data(dl.grid)
        ax[0].set(title=r"Game of Life: $N_A = %d$"%gol.alive_count)
        ax[1].set(title=r"Spore Life ($\alpha = 1$): $N_A = %d$"%dl.alive_count)
        gol.step(scramble=True)
        dl.step(scramble=True)
        time_text.set_text(f"time = {frame}")
    ani = animation.FuncAnimation(fig, update, interval=10, save_count=100, frames=10000)
    # Save the animation as a .gif file
    # fig.tight_layout()
    # ani.save('./img/spore-life-new-time.mp4', writer='ffmpeg', fps=30, dpi=500)
    plt.show()
