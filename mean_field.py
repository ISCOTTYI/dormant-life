# mean field theory for spore life
import random
from scipy.stats import binom
import matplotlib.pyplot as plt

class MFSporeLife:
    def __init__(self, init, alpha=1, n_neighbors=8):
        self.t = 0
        assert 0 <= alpha <= 1
        self.alpha = alpha
        assert n_neighbors > 0
        self.n_neighbors = n_neighbors
        assert sum(init) == 1
        self.alive_densities = [init[0],] # rho_A
        self.spore_densities = [init[1],] # rho_I
        self.dead_densities = [(1. - init[0] - init[1]),] # rho_D
    
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
        _P = lambda n: binom.pmf(n, self.n_neighbors, self.rho_a)
        P = [_P(n) for n in range(self.n_neighbors)]
        return P
    
    def _update_densities(self, rho_a, rho_i, rho_d):
        self.alive_densities.append(rho_a)
        self.spore_densities.append(rho_i)
        self.dead_densities.append(rho_d)
    
    def step(self):
        P = self._neighborhood_prob()
        # Kill spores randomly via alpha
        rho_i = self.rho_i * self.alpha
        # Update densities
        rho_a = self.rho_a * (P[2] + P[3]) + rho_i * (P[2] + P[3]) + self.rho_d * P[3]
        rho_i = rho_i * (1. - P[2] - P[3]) + self.rho_a * P[1]
        rho_d = 1. - rho_a - rho_i
        self._update_densities(rho_a, rho_i, rho_d)
        self.t += 1
        return rho_a, rho_i, rho_d
    
    def step_until(self, t_max):
        while self.t <= t_max:
            self.step()
        return self.alive_densities, self.spore_densities, self.dead_densities



# # ------------------------
# # --- one update step
# # ------------------------
# def oneUpdate(rhoLife, rhoDead, alpha, nNeighbors=8):
#   # nProb = nNeighbors + 1
#   rhoSpore = 1.0 - rhoLife - rhoDead
#   print(f'{rhoLife:7.4f} {rhoDead:7.4f} {rhoSpore:7.4f}')
#   print("---")
# #
#   pLife = [binom.pmf(nn, nNeighbors, rhoLife) for nn in range(nNeighbors)]
# #
#   rhoSpore *= alpha                         # not decaying
#   newSpore = rhoSpore * (1.0 - pLife[2] - pLife[3])
#   newSpore += rhoLife * pLife[1]
# #
#   newLife = rhoLife * (pLife[2] + pLife[3]) # having a good day
#   newLife += rhoSpore * (pLife[2] + pLife[3])
#   newLife += rhoDead * pLife[3]
# #
# #   newSpore *= alpha # alpha decay 
#   newDead = 1.0 - newLife - newSpore
#   print(f'{newLife:7.4f} {newDead:7.4f} {newSpore:7.4f}')
#   return newLife, newDead, newSpore

# ------------------------
# --- main
# ------------------------

# rhoLife  = 0.35 + 0.1*random.uniform(0,1)
# rhoSpore = random.uniform(0,1)*(1.0-rhoLife)
# rhoSpore = 1.0 - rhoLife
# rhoDead  = 1.0 - rhoLife - rhoSpore
# alpha    = 1
# nIter = 100

# rhoLifes = [rhoLife]
# rhoSpores = [rhoSpore]

# for iIter in range(nIter):
# #   print(f'{rhoLife:7.4f} {rhoDead:7.4f} {rhoSpore:7.4f}')
#   rhoLife, rhoDead, rhoSpore = oneUpdate(rhoLife, rhoDead, alpha, nNeighbors=8)
# #   rhoLife, rhoDead, rhoSpore = oneUpdate(rhoLife, rhoSpore, alpha, nNeighbors=8)
#   rhoLifes.append(rhoLife)
#   rhoSpores.append(rhoSpore)

# plt.plot(rhoLifes, label="A", c="tab:orange")
# plt.plot(rhoSpores, label="I", c="tab:blue")
# plt.legend()
# plt.show()

if __name__ == "__main__":
    rhoLife  = 0.35 + 0.1*random.uniform(0,1)
    rhoSpore = random.uniform(0,1)*(1.0-rhoLife)
    rhoSpore = 1.0 - rhoLife
    rhoDead  = 1.0 - rhoLife - rhoSpore
    mf_sl = MFSporeLife((rhoLife, rhoSpore, rhoDead), alpha=1)
    rho_a, rho_i, rho_d = mf_sl.step_until(100)
    plt.plot(rho_a, label="A", c="tab:orange")
    plt.plot(rho_i, label="I", c="tab:blue")
    plt.legend()
    plt.show()
