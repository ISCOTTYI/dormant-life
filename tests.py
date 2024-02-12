import unittest
import numpy as np
from gol import CellularAutomaton, SporeLife
from gol import ALIVE, SPORE, DEAD


class TestCellularAutomaton(unittest.TestCase):
    """
    Tests for the CellularAutomaton base class.
    """
    def test_count_state(self):
        test_grid = np.array([
            [ALIVE, DEAD, DEAD],
            [DEAD, DEAD, ALIVE],
            [DEAD, ALIVE, ALIVE]
        ])
        ca = CellularAutomaton(test_grid, (DEAD, ALIVE), None, True)
        self.assertEqual(4, ca.count_state(ALIVE))
    
    def test_count_state_zero(self):
        test_grid = np.full((3, 3), DEAD)
        ca = CellularAutomaton(test_grid, (DEAD, ALIVE), None, True)
        self.assertEqual(0, ca.count_state(ALIVE))
        self.assertEqual(0, ca.count_state(SPORE))


class TestSporeLifeRules(unittest.TestCase):
    """
    Testing the step function for SporeLife, i.e. the rules of SporeLife.
    """
    def test_DEAD_awake_rule(self):
        test_grid = np.array([
            [DEAD, DEAD, DEAD],
            [DEAD, DEAD, ALIVE],
            [DEAD, ALIVE, ALIVE]
        ])
        sl = SporeLife(test_grid, periodic_boundary=False)
        grid_step = sl.step()
        res = np.array([
            [DEAD, DEAD, DEAD],
            [DEAD, ALIVE, ALIVE],
            [DEAD, ALIVE, ALIVE]
        ])
        np.testing.assert_array_equal(grid_step, res)
    
    def test_SPORE_awake_rule(self):
        test_grid = np.array([
            [DEAD, ALIVE, SPORE], # SPORE with 2 ALIVE neighbors
            [DEAD, SPORE, ALIVE], # SPORE with 3 ALIVE neighbors
            [DEAD, ALIVE, DEAD]
        ])
        sl = SporeLife(test_grid, periodic_boundary=False)
        grid_step = sl.step()
        res = np.array([
            [DEAD, SPORE, ALIVE],
            [DEAD, ALIVE, ALIVE],
            [DEAD, SPORE, DEAD]
        ])
        np.testing.assert_array_equal(grid_step, res)
    
    def test_ALIVE_dies_rule(self):
        test_grid = np.array([
            [DEAD, DEAD, DEAD, DEAD],
            [ALIVE, DEAD, DEAD, ALIVE], # At 1, 0 ALIVE cell with 0 ALIVE neighbors
            [DEAD, DEAD, ALIVE, ALIVE], # At 2, 2 and 2, 3 ALIVE cell with 4 ALIVE neighbors
            [DEAD, DEAD, ALIVE, ALIVE]
        ])
        sl = SporeLife(test_grid, periodic_boundary=False)
        grid_step = sl.step()
        res = np.array([
            [DEAD, DEAD, DEAD, DEAD],
            [DEAD, DEAD, ALIVE, ALIVE],
            [DEAD, ALIVE, DEAD, DEAD],
            [DEAD, DEAD, ALIVE, ALIVE]
        ])
        np.testing.assert_array_equal(grid_step, res)
    
    def test_ALIVE_goes_dormant_rule(self):
        test_grid = np.array([
            [DEAD, ALIVE, DEAD],
            [DEAD, ALIVE, DEAD],
            [DEAD, DEAD, DEAD]
        ])
        sl = SporeLife(test_grid, periodic_boundary=False)
        grid_step = sl.step()
        res = np.array([
            [DEAD, SPORE, DEAD],
            [DEAD, SPORE, DEAD],
            [DEAD, DEAD, DEAD]
        ])
        np.testing.assert_array_equal(grid_step, res)
    
    def test_periodic_boundary(self):
        test_grid = np.array([
            [DEAD, DEAD, DEAD],
            [DEAD, DEAD, ALIVE],
            [DEAD, ALIVE, ALIVE]
        ])
        sl = SporeLife(test_grid, periodic_boundary=True)
        grid_step = sl.step()
        np.testing.assert_array_equal(grid_step, np.full((3, 3), ALIVE))
    
    def test_silent_step(self):
        test_grid = np.array([
            [DEAD, DEAD, DEAD],
            [DEAD, DEAD, ALIVE],
            [DEAD, ALIVE, ALIVE]
        ])
        sl = SporeLife(test_grid, periodic_boundary=False)
        silent_step = sl.step(silent=True)
        res = np.array([
            [DEAD, DEAD, DEAD],
            [DEAD, ALIVE, ALIVE],
            [DEAD, ALIVE, ALIVE]
        ])
        np.testing.assert_array_equal(silent_step, res)
        np.testing.assert_array_equal(sl.grid, test_grid)
        self.assertEqual(sl.t, 0)
    
    def test_multistep(self):
        test_grid = np.array([ # blinker
            [DEAD, DEAD, DEAD],
            [DEAD, SPORE, ALIVE],
            [DEAD, ALIVE, SPORE]
        ])
        res = np.array([
            [DEAD, DEAD, DEAD],
            [DEAD, ALIVE, SPORE],
            [DEAD, SPORE, ALIVE]
        ])
        sl = SporeLife(test_grid, periodic_boundary=False)
        np.testing.assert_array_equal(sl.step(), res)
        np.testing.assert_array_equal(sl.step(), test_grid)


class TestSporeLifeStochasticity(unittest.TestCase):
    def test_game_of_life_limit(self):
        test_grid = np.array([
            [DEAD, ALIVE, DEAD],
            [DEAD, ALIVE, DEAD],
            [DEAD, DEAD, DEAD]
        ])
        sl = SporeLife(test_grid, alpha=0, periodic_boundary=False)
        grid_step = sl.step()
        np.testing.assert_array_equal(grid_step, np.full((3, 3), DEAD))
    
    def test_stochastic_step(self):
        test_grid = np.array([
            [DEAD, ALIVE, DEAD],
            [DEAD, ALIVE, DEAD],
            [SPORE, SPORE, SPORE]
        ])
        sl = SporeLife(test_grid, alpha=.3, periodic_boundary=False, seed=100)
        """
        decision_grid = [
            [0.83498163, 0.59655403, 0.28886324],
            [0.04295157, 0.9736544 , 0.5964717 ],
            [0.79026316, 0.91033938, 0.68815445]
        ]
        """
        grid_step = sl.step()
        res = np.array([
            [DEAD, DEAD, DEAD],
            [DEAD, SPORE, DEAD],
            [SPORE, SPORE, DEAD]
        ])
        np.testing.assert_array_equal(grid_step, res)
    
#     # def test_transitions(self):
#     #     test_grid = np.array([
#     #         [DEAD, ALIVE, SPORE],
#     #         [DEAD, ALIVE, SPORE],
#     #         [DEAD, DEAD, DEAD]
#     #     ])
#     #     gol = DormantLife(test_grid)
#     #     grid_step = gol.step()
#     #     self.assertEqual(gol.transitions_from(test_grid, ALIVE, SPORE), 2)
#     #     self.assertEqual(gol.transitions_from(test_grid, SPORE, ALIVE), 2)
#     #     self.assertEqual(gol.transitions_from(test_grid, ALIVE, DEAD), 0)
#     #     self.assertEqual(gol.transitions_from(test_grid, DEAD, ALIVE), 0)
        

from lifetime_distribution import lifetime_distribution

class TestLifetimeDistribution(unittest.TestCase):
    def test_lifetime_measuring(self):
        test_grid = np.array([
                [DEAD, ALIVE, DEAD],
                [DEAD, ALIVE, ALIVE],
                [DEAD, DEAD, DEAD]
        ])
        sl = SporeLife(test_grid)
        self.assertDictEqual(
            lifetime_distribution(ALIVE, sl, 3, 0, ignore_transient_dynamics=0),
            {0: 6, 1: 3})
    
    def test_lifetime_no_dynamics(self):
        test_grid = np.array([
            [DEAD, ALIVE, ALIVE],
            [DEAD, ALIVE, ALIVE],
            [DEAD, DEAD, DEAD]
        ])
        sl = SporeLife(test_grid)
        self.assertDictEqual(
            lifetime_distribution(ALIVE, sl, 3, 0, ignore_transient_dynamics=0),
            {})
    
    def test_lifetime_blinker(self):
        test_grid = np.array([
            [DEAD, ALIVE, SPORE],
            [DEAD, ALIVE, SPORE],
            [DEAD, DEAD, DEAD]
        ])
        sl = SporeLife(test_grid)
        self.assertDictEqual(
            lifetime_distribution(ALIVE, sl, 2, 0, ignore_transient_dynamics=1),
            {0: 2})
    
    def test_longer_lifetimes(self):
        test_grid = np.array([
            [ALIVE, DEAD, ALIVE],
            [ALIVE, DEAD, ALIVE],
            [DEAD, ALIVE, DEAD]
        ])
        sl = SporeLife(test_grid, periodic_boundary=False)
        self.assertDictEqual(
            lifetime_distribution(ALIVE, sl, 7, 0, ignore_transient_dynamics=0),
            {0:7, 1:2, 2:1})
        sl = SporeLife(test_grid, periodic_boundary=False)
        self.assertDictEqual(
            lifetime_distribution(SPORE, sl, 7, 0, ignore_transient_dynamics=0),
            {0:4})
    
    def test_ignore_transient_dynamics(self):
        test_grid = np.array([
                [DEAD, ALIVE, DEAD],
                [DEAD, ALIVE, ALIVE],
                [DEAD, DEAD, DEAD]
        ])
        dl = SporeLife(test_grid)
        self.assertDictEqual(
            lifetime_distribution(ALIVE, dl, 3, 0, ignore_transient_dynamics=1),
            {0: 6})


if __name__ == "__main__":
    unittest.main()
