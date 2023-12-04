import numpy as np
from scipy.ndimage import convolve

# https://gist.githubusercontent.com/electronut/5836145/raw/4e557aae3a4b5dd8962ef8c5827790213d9781ec/conway.py


DEAD = 0
ALIVE = 1
DORM = 2


class CellularAutomaton():
    def __init__(self, init_grid: np.ndarray, states: np.array, seed: int):
        # Ensure that init_grid is quadratic and only filled with states
        assert (len(init_grid.shape) == 2
                and init_grid.shape[0] == init_grid.shape[1])
        assert np.all(np.isin(init_grid, states))
        self.grid = init_grid
        self.N = init_grid.shape[0] # board size
        assert self.N > 2 # Cannot deal with 2x2

        # convolution kernel
        self.conv_ker = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
    
    @property
    def alive_count(self):
        return np.count_nonzero(self.grid == ALIVE)
    
    def reinit_grid(self):
        raise NotImplementedError("Instance of CllularAutomaton may not be initialized!")
    
    def step(self):
        raise NotImplementedError("Instance of CllularAutomaton does not implement rules!")


class GameOfLife(CellularAutomaton):
    def __init__(self, init_grid:np.ndarray, seed:int=None):
        # 0: dead, 1: alive
        self.states = np.array([DEAD, ALIVE])
        super().__init__(init_grid, self.states, seed)
    
    def reinit_grid(self, p_alive):
        assert 0 <= p_alive <= 1
        p_dead = 1 - p_alive
        dims = (self.N, self.N)
        self.grid = self.rng.choice(self.states, p=(p_dead, p_alive), size=dims)
    
    def step(self):
        ngrid = self.grid.copy()
        # Create array with 8-neighbor sums by convolution, using periodic
        # boundary conditions.
        c = convolve(self.grid, self.conv_ker, mode="wrap")
        # Apply rules of game of life
        ngrid[(self.grid == ALIVE) & ((c < 2) | (c > 3))] = DEAD
        ngrid[(self.grid == DEAD) & (c == 3)] = ALIVE
        self.grid = ngrid
        return ngrid
    

class DormantLife(CellularAutomaton):
    def __init__(self, init_grid:np.array, seed:int=None):
        # 0: dead, 1: alive, 2: dormant
        self.states = np.array([DEAD, ALIVE, DORM])
        super().__init__(init_grid, self.states, seed)
    
    def reinit_grid(self, p_alive, p_dorm):
        assert 0 <= p_alive <= 1 and 0 <= p_dorm <= 1 and p_alive + p_dorm < 1
        p_dead = 1 - p_alive - p_dorm
        dims = (self.N, self.N)
        prob = (p_dead, p_alive, p_dorm)
        self.grid = self.rng.choice(self.states, p=prob, size=dims)

    def step(self, p=1):
        """
        Perform a step in DormantLife. If probability p < 1, transition from
        DORMANT -> ALIVE with 2 ALIVE neighbors is stochastic.
        """
        assert 0 <= p <= 1
        ngrid = self.grid.copy()
        # Create array with 8-neighbor ALIVE counts by convolution, using
        # periodic boundary conditions.
        alive_grid = (self.grid == ALIVE).astype(int)
        c = convolve(alive_grid, self.conv_ker, mode="wrap")
        # Apply rules of game of life w/ dormancy
        # DEAD awake
        ngrid[(self.grid == DEAD) & (c == 3)] = ALIVE
        # DORMANT awake
        # Overlay grid with stochastic grid that determines transition probaiblity
        # for every cell, iff part of rules.
        p_grid = self.rng.choice([0, 1], p=[1-p, p], size=(self.N, self.N))
        ngrid[(self.grid == DORM) & ((p_grid == 1) & (c == 2))] = ALIVE
        ngrid[(self.grid == DORM) & (c == 3)] = ALIVE
        # ALIVE dies
        ngrid[(self.grid == ALIVE) & ((c < 1) | (c > 3))] = DEAD
        # ALIVE goes DORMANT
        ngrid[(self.grid == ALIVE) & (c == 1)] = DORM
        self.grid = ngrid
        return ngrid
