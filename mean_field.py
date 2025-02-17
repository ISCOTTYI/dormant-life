# mean field theory for spore life
import random
from scipy.stats import binom
import matplotlib.pyplot as plt

class MFSporeLife:
    def __init__(self, init, alpha=1, n_neighbors=8, damping=0):
        self.t = 0
        assert 0 <= alpha <= 1
        self.alpha = alpha
        assert n_neighbors > 0
        self.n_neighbors = n_neighbors
        assert sum(init) == 1
        self.alive_densities = [init[0],] # rho_A
        self.spore_densities = [init[1],] # rho_I
        self.dead_densities = [(1. - init[0] - init[1]),] # rho_D
        self.damp = damping
    
    @property
    def rho_a(self):
        return self.alive_densities[-1]

    @property
    def rho_i(self):
        return self.spore_densities[-1]

    @property
    def rho_d(self):
        return self.dead_densities[-1]
    
    def _neighborhood_prob(self):
        # Probability that [index] number of active neighbors
        rho_a = self.rho_a if self.rho_a > 1e-4 else 0.0
        _P = lambda n: binom.pmf(n, self.n_neighbors, rho_a)
        P = [_P(n) for n in range(self.n_neighbors)]
        return P
    
    def _update_densities(self, rho_a, rho_i, rho_d):
        self.alive_densities.append(rho_a)
        self.spore_densities.append(rho_i)
        self.dead_densities.append(rho_d)
    
    def step(self):
        P = self._neighborhood_prob()
        # Kill spores randomly via alpha
        killed_rho_i = self.rho_i * self.alpha
        # Update densities
        new_rho_a = self.rho_a * (P[2] + P[3]) + killed_rho_i * (P[2] + P[3]) + self.rho_d * P[3]
        new_rho_i = killed_rho_i * (1. - P[2] - P[3]) + self.rho_a * P[1]
        # Damp
        new_rho_a = (1 - self.damp) * new_rho_a + self.damp * self.rho_a
        new_rho_i = (1 - self.damp) * new_rho_i + self.damp * killed_rho_i
        new_rho_d = 1. - new_rho_a - new_rho_i
        self._update_densities(new_rho_a, new_rho_i, new_rho_d)
        self.t += 1
        return new_rho_a, new_rho_i, new_rho_d
    
    def step_until(self, t_max):
        while self.t <= t_max:
            self.step()
        return self.alive_densities, self.spore_densities, self.dead_densities


if __name__ == "__main__":
    rhoLife  = 0.35 + 0.1*random.uniform(0,1)
    # rhoSpore = random.uniform(0,1)*(1.0-rhoLife)
    rhoSpore = 1.0 - rhoLife
    rhoDead  = 1.0 - rhoLife - rhoSpore
    print(rhoLife, rhoSpore, rhoDead)
    mf_sl = MFSporeLife((rhoLife, rhoSpore, 1-rhoLife-rhoSpore), alpha=0.001)
    rho_a, rho_i, rho_d = mf_sl.step_until(100)
    plt.plot(rho_a, label="A", c="tab:orange")
    plt.plot(rho_i, label="I", c="tab:blue")
    plt.legend()
    plt.show()
