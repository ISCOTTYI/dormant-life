import numpy as np
from scipy.ndimage import convolve


DEAD = 0
ALIVE = 1
SPORE = 2


class CellularAutomaton():
    """
    Base class for game of life models.
    """
    def __init__(self, init_grid: np.ndarray, states: np.array, seed: int,
                 periodic_boundary: bool):
        # Ensure that init_grid is quadratic and only filled with states
        assert (len(init_grid.shape) == 2
                and init_grid.shape[0] == init_grid.shape[1])
        assert np.all(np.isin(init_grid, states))
        self.grid = init_grid
        self.t = 0
        self.N = init_grid.shape[0] # board size
        assert self.N > 2 # Cannot deal with 2x2

        # convolution kernel
        self.conv_ker = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
        self.periodic_boundary = periodic_boundary
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
    
    def count_state(self, state: int):
        return np.count_nonzero(self.grid == state)
    
    def compute_neighbor_counts_for(self, state: int, periodic_boundary=True):
        filtered_grid = (self.grid == state).astype(np.intc)
        mode = "wrap" if periodic_boundary else "constant"
        c = convolve(filtered_grid, self.conv_ker, mode=mode, cval=0)
        return c
    
    def reinit_grid(self):
        raise NotImplementedError("Instance of CellularAutomaton may not be initialized!")
    
    def step(self, silent: bool = False) -> np.ndarray:
        raise NotImplementedError("Instance of CellularAutomaton does not implement rules!")
    
    def step_until(self, t: int) -> np.ndarray:
        assert self.t <= t
        while self.t < t:
            self.step()
        return self.grid
    
    def state_count_time_series(self, t_max: int, state: int) -> np.ndarray:
        """
        Step the system until t_max and return the counts of state at each time
        step on the way.
        """
        t0 = self.t
        assert t0 < t_max
        data = np.zeros(t_max+1 - t0)
        while self.t <= t_max:
            data[self.t - t0] = self.count_state(state)
            self.step()
        return data
            

class GameOfLife(CellularAutomaton):
    def __init__(self, init_grid: np.ndarray, seed: int = None,
                 periodic_boundary: bool = True):
        # 0: dead, 1: alive
        self.states = np.array([DEAD, ALIVE])
        super().__init__(init_grid, self.states, seed, periodic_boundary)
        self.alive_neighbor_counts = None
    
    @property
    def alive_count(self):
        return self.count_state(ALIVE)
    
    def alive_count_time_series(self, t_max: int) -> np.array:
        return self.state_count_time_series(t_max, ALIVE)
        
    def reinit_grid(self, p_alive):
        assert 0 <= p_alive <= 1
        p_dead = 1 - p_alive
        dims = (self.N, self.N)
        self.grid = self.rng.choice(self.states, p=(p_dead, p_alive), size=dims)
    
    def step(self, silent: bool = False) -> np.ndarray:
        """
        Perform a step in Game of Life.
        A silent step is only computed and returned but does not count as a time
        step and is not stored.
        """
        ngrid = self.grid.copy()
        # Create array with 8-neighbor sums by convolution, using periodic
        # boundary conditions.
        self.alive_neighbor_counts = c = self.compute_neighbor_counts_for(
            ALIVE, periodic_boundary=self.periodic_boundary)
        # Apply rules of game of life
        ngrid[(self.grid == ALIVE) & ((c < 2) | (c > 3))] = DEAD
        ngrid[(self.grid == DEAD) & (c == 3)] = ALIVE
        # Update grid and time
        if not silent:
            self.grid = ngrid
            self.t += 1
        return ngrid
    

class SporeLife(CellularAutomaton):
    def __init__(self, init_grid: np.ndarray, alpha: float = 1,
                 seed: int = None, periodic_boundary: bool = True):
        """
        For alpha = 1 we get deterministic SporeLife, for alpha = 0 we get Game
        of Life.
        """
        # 0: dead, 1: alive, 2: spore
        self.states = np.array([DEAD, ALIVE, SPORE])
        super().__init__(init_grid, self.states, seed, periodic_boundary)
        
        self.alive_neighbor_counts = None

        assert 0 <= alpha <= 1
        self.alpha = alpha
    
    @property
    def alive_count(self):
        return self.count_state(ALIVE)
    
    @property
    def spore_count(self):
        return self.count_state(SPORE)
    
    def alive_count_time_series(self, t_max: int) -> np.ndarray:
        return self.state_count_time_series(t_max, ALIVE)
    
    def spore_count_time_series(self, t_max: int) -> np.ndarray:
        return self.state_count_time_series(t_max, SPORE)
    
    def reinit_grid(self, p_alive: float, p_dorm: float):
        assert 0 <= p_alive <= 1 and 0 <= p_dorm <= 1 and p_alive + p_dorm < 1
        p_dead = 1 - p_alive - p_dorm
        dims = (self.N, self.N)
        prob = (p_dead, p_alive, p_dorm)
        self.grid = self.rng.choice(self.states, p=prob, size=dims)
    
    def deterministic_step(self, silent: bool = False) -> np.ndarray:
        """
        Perform a step in SporeLife without stochasticity, i.e. ignore the given
        alpha and pretend that alpha = 1.
        A silent step is only computed and returned but does not count as a time
        step and is not stored.
        """
        ngrid = self.grid.copy()
        # Create array with 8-neighbor ALIVE counts by convolution, using
        # periodic boundary conditions.
        self.alive_neighbor_counts = c = self.compute_neighbor_counts_for(
            ALIVE, periodic_boundary=self.periodic_boundary)
        # Apply rules of game of life w/ dormancy
        # DEAD awake
        ngrid[(self.grid == DEAD)
              & (c == 3)] = ALIVE
        # DORMANT awake
        ngrid[(self.grid == SPORE)
              & ((c == 2) | (c == 3))] = ALIVE
        # ALIVE dies
        ngrid[(self.grid == ALIVE)
              & ((c < 1) | (c > 3))] = DEAD
        # ALIVE goes DORMANT
        ngrid[(self.grid == ALIVE)
              & (c == 1)] = SPORE
        # Update grid and time
        if not silent:
            self.grid = ngrid
            self.t += 1
        return ngrid

    def step(self, silent: bool = False) -> np.ndarray:
        """
        Perform a (possibly stochastic) step in SporeLife.
        A silent step is only computed and returned but does not count as a time
        step and is not stored.
        """
        ngrid = self.deterministic_step(silent=True)
        # Randomly kill SPOREs in ngrid based on alpha
        decision_grid = self.rng.random((self.N, self.N))
        ngrid[(ngrid == SPORE)
              & (decision_grid < (1-self.alpha))] = DEAD
        # Update grid and time
        if not silent:
            self.grid = ngrid
            self.t += 1
        return ngrid
