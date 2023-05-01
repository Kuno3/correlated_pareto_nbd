import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal, expon

class pareto_nbd_simulator():

    def __init__(self, N, T=None, dim=3, features=None, B=None, Gamma=None, seed=1234):
        """
        N: number of data
        T: measurement period from first purchase
        dim: dimesion of features
        features: features
        B: coefficient of features
        Gamma: covariance of rho and mu
        """

        np.random.seed(seed)

        self.N = N

        if T is None:
            self.T = 100 * np.ones(N)
        else:
            self.T = T

        if features is None:
            self.features = np.concatenate([np.random.randn(self.N, dim-1), np.ones(self.N).reshape((self.N, 1))], 1) # type: ignore
        else:
            self.features = features
        
        if B is None:
            self.B = np.random.randn(dim, 2)
        else:
            self.B = B
        
        if Gamma is None:
            self.Gamma = np.eye(2, dtype=float)
        else:
            self.Gamma = Gamma

    def simulate(self):
        """
        rho: expectation of purchase frequency
        mu: expectation of (1 / time to leave)
        x: (Number of purchases) - 1
        t: elapsed period from first purchase to last purchase
        """
                
        mean_log_rho_mu = self.features.dot(self.B)
        log_rho_mu = np.array([multivariate_normal(mean_log_rho_mu[i, :], self.Gamma).rvs() for i in range(self.N)]) # type: ignore
        self.rho = np.exp(log_rho_mu[:, 0])
        self.mu = np.exp(log_rho_mu[:, 1])

        y = expon(scale=1/self.mu).rvs()

        self.x = np.zeros(self.N)
        self.t = np.zeros(self.N)
        for i in range(self.N):
            while 1:
                range_t = expon(scale=1/self.rho[i]).rvs()
                if self.t[i] + range_t < min(self.T[i], y[i]):
                    self.t[i] += range_t
                    self.x[i] += 1
                else:
                    break