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
    
    def neighborhood_grid(self, state: int, periodic_boundary=True):
        filtered_grid = (self.grid == state).astype(np.intc)
        mode = "wrap" if periodic_boundary else "constant"
        c = convolve(filtered_grid, self.conv_ker, mode=mode, cval=0)
        return c

    def scramble(self, grid=None):
        """
        Scrambles the grid, i.e. randomly rearanges the cells. If no grid is
        passed, uses stored grid, else, uses passed grid.
        """
        if grid is None:
            grid = self.grid
            self.grid = None
        flat = grid.flatten()
        self.rng.shuffle(flat)
        grid = flat.reshape((self.N, self.N))
        if self.grid is None:
            self.grid = grid
        return grid
    
    def reinit_grid(self):
        raise NotImplementedError("Instance of CellularAutomaton may not be initialized!")
    
    def step(self, silent: bool = False) -> np.ndarray:
        raise NotImplementedError("Instance of CellularAutomaton does not implement rules!")
    
    def step_until(self, t: int) -> np.ndarray:
        assert self.t <= t
        while self.t < t:
            self.step()
        return self.grid
    
    def state_count_time_series(self, t_max: int, state: int,
                                **kwargs) -> np.ndarray:
        """
        Step the system until t_max and return the counts of state at each time
        step on the way.
        """
        t0 = self.t
        assert t0 < t_max
        data = np.zeros(t_max+1 - t0)
        while self.t <= t_max:
            data[self.t - t0] = self.count_state(state)
            self.step(**kwargs)
        return data
            

class GameOfLife(CellularAutomaton):
    def __init__(self, init_grid: np.ndarray, seed: int = None,
                 periodic_boundary: bool = True):
        # 0: dead, 1: alive
        self.states = np.array([DEAD, ALIVE])
        super().__init__(init_grid, self.states, seed, periodic_boundary)
        self.life_neighborhood_grid = self.neighborhood_grid(ALIVE, self.periodic_boundary)
    
    @property
    def alive_count(self):
        return self.count_state(ALIVE)
    
    def alive_count_time_series(self, t_max: int, **kwargs) -> np.array:
        return self.state_count_time_series(t_max, ALIVE, **kwargs)
        
    def reinit_grid(self, p_alive):
        assert 0 <= p_alive <= 1
        p_dead = 1 - p_alive
        dims = (self.N, self.N)
        self.grid = self.rng.choice(self.states, p=(p_dead, p_alive), size=dims)
    
    def step(self, silent: bool = False, scramble: bool = False,
             overcrowd_birth_p: float = None) -> np.ndarray:
        """
        Perform a step in Game of Life.
        A silent step is only computed and returned but does not count as a time
        step and is not stored.
        """
        ngrid = self.grid.copy()
        # Create array with 8-neighbor sums by convolution, using periodic
        # boundary conditions.
        c = self.life_neighborhood_grid
        # Apply rules of game of life
        ngrid[(self.grid == ALIVE) & ((c < 2) | (c > 3))] = DEAD
        # Birth with 4 ALIVE neighbors with probability overcrowd_birth_p
        if overcrowd_birth_p is not None:
            decision_grid = self.rng.random((self.N, self.N))
            ngrid[(self.grid == DEAD) & (c == 4)
                  & (decision_grid < overcrowd_birth_p)] = ALIVE
        ngrid[(self.grid == DEAD) & (c == 3)] = ALIVE
        # Scramble
        if scramble:
            ngrid = self.scramble(ngrid)
        # Update grid and time
        if not silent:
            self.grid = ngrid
            self.life_neighborhood_grid = self.neighborhood_grid(
                ALIVE, self.periodic_boundary)
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

        self.life_neighborhood_grid = self.neighborhood_grid(ALIVE, self.periodic_boundary)
        
        assert 0 <= alpha <= 1
        self.alpha = alpha
    
    @property
    def alive_count(self):
        return self.count_state(ALIVE)
    
    @property
    def spore_count(self):
        return self.count_state(SPORE)
    
    def alive_count_time_series(self, t_max: int, **kwargs) -> np.ndarray:
        return self.state_count_time_series(t_max, ALIVE, **kwargs)
    
    def spore_count_time_series(self, t_max: int, **kwargs) -> np.ndarray:
        return self.state_count_time_series(t_max, SPORE, **kwargs)
    
    def reinit_grid(self, p_alive: float, p_dorm: float):
        assert 0 <= p_alive <= 1 and 0 <= p_dorm <= 1 and p_alive + p_dorm < 1
        p_dead = 1 - p_alive - p_dorm
        dims = (self.N, self.N)
        prob = (p_dead, p_alive, p_dorm)
        self.grid = self.rng.choice(self.states, p=prob, size=dims)
    
    def deterministic_step(self, silent: bool = False,
                           overcrowd_dormancy: bool = False,
                           scramble: bool = False,
                           overcrowd_birth_p: float = None) -> np.ndarray:
        """
        Perform a step in SporeLife without stochasticity, i.e. ignore the given
        alpha and pretend that alpha = 1.
        A silent step is only computed and returned but does not count as a time
        step and is not stored.
        If overcrowd_dormancy is true, an ALIVE cell with 4 ALIVE neighbors goes
        dormant, else it just dies.
        If scramble is true, scrambles the grid after performing the updates.
        """
        ngrid = self.grid.copy()
        # Create array with 8-neighbor ALIVE counts by convolution, using
        # periodic boundary conditions.
        c = self.life_neighborhood_grid
        # Apply rules of game of life w/ dormancy
        # DEAD awake
        ngrid[(self.grid == DEAD)
              & (c == 3)] = ALIVE
        # DORMANT awake
        ngrid[(self.grid == SPORE)
              & ((c == 2) | (c == 3))] = ALIVE
        # Birth with 4 ALIVE neighbors with probability overcrowd_birth_p
        if overcrowd_birth_p is not None:
            decision_grid = self.rng.random((self.N, self.N))
            ngrid[((self.grid == DEAD) | (self.grid == SPORE)) & (c == 4)
                  & (decision_grid < overcrowd_birth_p)] = ALIVE
        if overcrowd_dormancy:
            overcrowd_lim = 4
        else:
            overcrowd_lim = 3
        # ALIVE dies
        ngrid[(self.grid == ALIVE)
              & ((c < 1) | (c > overcrowd_lim))] = DEAD
        # ALIVE goes DORMANT
        if overcrowd_dormancy:
            ngrid[(self.grid == ALIVE)
                & ((c == 1) | (c == 4))] = SPORE
        else:
            ngrid[(self.grid == ALIVE)
                & (c == 1)] = SPORE
        # Scramble
        if scramble:
            ngrid = self.scramble(ngrid)
        # Update grid and time
        if not silent:
            self.grid = ngrid
            self.life_neighborhood_grid = self.neighborhood_grid(
                ALIVE, self.periodic_boundary)
            self.t += 1
        return ngrid

    def step(self, silent: bool = False,
             overcrowd_dormancy: bool = False,
             scramble: bool = False,
             overcrowd_birth_p: float = None) -> np.ndarray:
        """
        Perform a (possibly stochastic) step in SporeLife.
        A silent step is only computed and returned but does not count as a time
        step and is not stored.
        If overcrowd_dormancy is true, an ALIVE cell with 4 ALIVE neighbors goes
        dormant, else it just dies.
        """
        ngrid = self.deterministic_step(silent=True,
                                        overcrowd_dormancy=overcrowd_dormancy,
                                        scramble=scramble,
                                        overcrowd_birth_p=overcrowd_birth_p)
        # Randomly kill SPOREs in ngrid based on alpha
        decision_grid = self.rng.random((self.N, self.N))
        ngrid[(ngrid == SPORE)
              & (decision_grid < (1-self.alpha))] = DEAD
        # Update grid and time
        if not silent:
            self.grid = ngrid
            self.life_neighborhood_grid = self.neighborhood_grid(
                ALIVE, self.periodic_boundary)
            self.t += 1
        return ngrid
