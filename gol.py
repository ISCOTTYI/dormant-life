import numpy as np
from scipy.ndimage import convolve


DEAD = 0
ALIVE = 1
DORM = 2


class CellularAutomaton():
    """
    Base class for game of life models.
    """
    def __init__(self, init_grid: np.ndarray, states: np.array, seed: int):
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
        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
    
    def count_state(self, state: int):
        return np.count_nonzero(self.grid == state)
    
    def compute_neighbor_counts_for(self, state: int):
        filtered_grid = (self.grid == state).astype(int)
        c = convolve(filtered_grid, self.conv_ker, mode="wrap")
        return c
    
    def transitions_from(self, from_grid: np.ndarray,
                         from_state: int, to_state: int) -> int:
        """
        Count the transitions from from_state to to_state between from_grid to
        the momentary grid.
        """
        return np.sum(np.all(
            (from_grid == from_state), (self.grid == to_state), axis=0))
    
    def reinit_grid(self):
        raise NotImplementedError("Instance of CellularAutomaton may not be initialized!")
    
    def step(self):
        raise NotImplementedError("Instance of CellularAutomaton does not implement rules!")
    
    def state_count_time_series(self, t_max: int, state: int) -> np.array:
        """
        Step the system until t_max and return the counts of state at each time
        step on the way.
        """
        assert t_max > self.t
        data = np.zeros(t_max+1 - self.t)
        i = 0
        while self.t <= t_max:
            data[i] = self.count_state(state)
            self.step()
            i += 1
        return data
            

class GameOfLife(CellularAutomaton):
    def __init__(self, init_grid: np.ndarray, seed: int = None):
        # 0: dead, 1: alive
        self.states = np.array([DEAD, ALIVE])
        super().__init__(init_grid, self.states, seed)
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
    
    def step(self):
        ngrid = self.grid.copy()
        # Create array with 8-neighbor sums by convolution, using periodic
        # boundary conditions.
        self.alive_neighbor_counts = c = self.compute_neighbor_counts_for(ALIVE)
        # Apply rules of game of life
        ngrid[(self.grid == ALIVE) & ((c < 2) | (c > 3))] = DEAD
        ngrid[(self.grid == DEAD) & (c == 3)] = ALIVE
        self.grid = ngrid
        self.t += 1
        return ngrid
    

class DormantLife(CellularAutomaton):
    def __init__(self, init_grid:np.ndarray, alpha: float = 1, seed: int=None):
        # 0: dead, 1: alive, 2: dormant
        self.states = np.array([DEAD, ALIVE, DORM])
        super().__init__(init_grid, self.states, seed)
        
        self.p_grid = np.ones((self.N, self.N))
        
        self.alive_neighbor_counts = None

        assert 0 <= alpha <= 1
        self.alpha = alpha
    
    @property
    def alive_count(self):
        return self.count_state(ALIVE)
    
    @property
    def dorm_count(self):
        return self.count_state(DORM)
    
    def alive_count_time_series(self, t_max: int) -> np.array:
        return self.state_count_time_series(t_max, ALIVE)
    
    def dorm_count_time_series(self, t_max: int) -> np.array:
        return self.state_count_time_series(t_max, DORM)
    
    def reinit_grid(self, p_alive: float, p_dorm: float):
        assert 0 <= p_alive <= 1 and 0 <= p_dorm <= 1 and p_alive + p_dorm < 1
        p_dead = 1 - p_alive - p_dorm
        dims = (self.N, self.N)
        prob = (p_dead, p_alive, p_dorm)
        self.grid = self.rng.choice(self.states, p=prob, size=dims)

    def step(self):
        """
        Perform a step in DormantLife. For alpha = 1 we get deterministic
        DormantLife, for alpha = 0 we get Game of Life.
        """
        self.p_grid *= self.alpha # decay
        ngrid = self.grid.copy()
        # Create array with 8-neighbor ALIVE counts by convolution, using
        # periodic boundary conditions.
        self.alive_neighbor_counts = c = self.compute_neighbor_counts_for(ALIVE)
        # Apply rules of game of life w/ dormancy
        # DEAD awake
        ngrid[(self.grid == DEAD)
              & (c == 3)] = ALIVE
        # DORMANT awake
        # Awakening with 2 ALIVE neighbors happens only with probability given
        # by p_grid. If alpha = 1, p_grid is always 1 and we get non-stochastic
        # DormantLife, if alpha = 0, p_grid is always 0 and we get GoL.
        decision_grid = self.rng.random((self.N, self.N))
        ngrid[(self.grid == DORM)
              & ((c == 2) & (decision_grid < self.p_grid))] = ALIVE
        ngrid[(self.grid == DORM)
              & (c == 3)] = ALIVE
        # ALIVE dies
        ngrid[(self.grid == ALIVE)
              & ((c < 1) | (c > 3))] = DEAD
        # ALIVE goes DORMANT
        ngrid[(self.grid == ALIVE)
              & (c == 1)] = DORM
        # Decay probability reset
        self.p_grid[(self.grid == ALIVE)
                    & (c == 1)] = 1
        self.grid = ngrid
        self.t += 1
        return ngrid
