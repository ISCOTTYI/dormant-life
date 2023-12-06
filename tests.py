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
    
    def test_limit_cases_stochasticity(self):
        test_grid = np.array([
            [DEAD, ALIVE, DEAD],
            [DEAD, DORM, DEAD],
            [DEAD, ALIVE, DEAD]
        ])
        # alpha = 0 should get us GameOfLife step with DEAD == DORM.
        gol = DormantLife(test_grid)
        gol_grid_step = gol.step(alpha=0)
        gol_res = np.array([
            [DEAD, DORM, DEAD],
            [DEAD, DORM, DEAD],
            [DEAD, DORM, DEAD]
        ])
        np.testing.assert_array_equal(gol_res, gol_grid_step)
        # alpha = 1 should get us determinstic DormantLife step.
        dl = DormantLife(test_grid)
        dl_grid_step = dl.step(alpha=1)
        dl_res = np.array([
            [DEAD, DORM, DEAD],
            [DEAD, ALIVE, DEAD],
            [DEAD, DORM, DEAD]
        ])
        np.testing.assert_array_equal(dl_res, dl_grid_step)

    def test_p_grid_decay(self):
        gol = DormantLife(np.full((3, 3), DEAD))
        gol.step(alpha=0.5)
        np.testing.assert_array_equal(gol.p_grid, np.full((3, 3), 0.5))
    
    def test_p_grid_reset(self):
        test_grid = np.array([
            [ALIVE, ALIVE, DEAD],
            [DEAD, DEAD, DEAD],
            [DEAD, DEAD, DEAD]
        ])
        gol = DormantLife(test_grid)
        gol.step(alpha=0.5)
        res_p_grid = np.full((3, 3), 0.5); res_p_grid[0, 0:2] = 1
        np.testing.assert_array_equal(gol.p_grid, res_p_grid)
    
    def test_stochastic_update(self):
        test_grid = np.array([
            [DORM, ALIVE, DEAD],
            [DORM, ALIVE, DEAD],
            [DEAD, DEAD, DEAD]
        ])
        gol = DormantLife(test_grid, seed=1)
        grid_step = gol.step(alpha=0.6)
        res = np.array([
            [ALIVE, DORM, DEAD],
            [DORM, DORM, DEAD],
            [DEAD, DEAD, DEAD]
        ])
        np.testing.assert_array_equal(grid_step, res)


if __name__ == '__main__':
    unittest.main()
