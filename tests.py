import unittest
import numpy as np
from gol import GameOfLife, DormantLife, ALIVE, DORM, DEAD


class TestDormantLife(unittest.TestCase):
    def test_ALIVE_DORM_conversion(self):
        test_grid = np.array([
            [DEAD, ALIVE, DEAD],
            [DEAD, ALIVE, DEAD],
            [DEAD, DEAD, DEAD]
        ])
        gol = DormantLife(test_grid)
        grid_step = gol.step()
        res = np.array([
            [DEAD, DORM, DEAD],
            [DEAD, DORM, DEAD],
            [DEAD, DEAD, DEAD]
        ])
        np.testing.assert_array_equal(res, grid_step)
    
    def test_periodic_boundary(self):
        test_grid = np.array([
            [ALIVE, DEAD, ALIVE],
            [DEAD, ALIVE, DEAD],
            [DEAD, DEAD, ALIVE]
        ])
        gol = DormantLife(test_grid)
        grid_step = gol.step()
        np.testing.assert_array_equal(test_grid, grid_step)
    
    def test_stochasticity_DORM_ALIVE_2_neighbors(self):
        test_grid = np.array([
            [DEAD, ALIVE, DEAD],
            [DEAD, DORM, DEAD],
            [DEAD, ALIVE, DEAD]
        ])
        gol_0 = DormantLife(test_grid)
        grid_step_0 = gol_0.step(p=0)
        res_0 = np.array([
            [DEAD, DORM, DEAD],
            [DEAD, DORM, DEAD],
            [DEAD, DORM, DEAD]
        ])
        np.testing.assert_array_equal(res_0, grid_step_0)
        gol_1 = DormantLife(test_grid)
        grid_step_1 = gol_1.step(p=1)
        res_1 = np.array([
            [DEAD, DORM, DEAD],
            [DEAD, ALIVE, DEAD],
            [DEAD, DORM, DEAD]
        ])
        np.testing.assert_array_equal(res_1, grid_step_1)


if __name__ == '__main__':
    unittest.main()
