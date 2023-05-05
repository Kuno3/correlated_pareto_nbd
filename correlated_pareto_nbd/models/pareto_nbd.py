import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, invwishart, matrix_normal
import numba
from tqdm import tqdm

class pareto_nbd():
    def __init__(self, T, t, x, features, Lambda0=None, nu0=None, V0=None):
        """
        T: measurement period from first purchase
        t: elapsed period from first purchase to last purchase
        x: (Number of purchases) - 1
        """
        
        self.T = T
        self.t = t
        self.x = x
        self.features = features

        self.N = len(T)
        self.dim = features.shape[1]
        
        if Lambda0 is None:
            self.Lambda0 = 0.00001 * np.eye(self.dim)
        else:
            self.Lambda0 = Lambda0

        if nu0 is None:
            self.nu0 = self.dim - 1
        else:
            self.nu0 = nu0

        if V0 is None:
            self.V0 = 0.00001 * np.eye(2)
        else:
            self.V0 = V0

    @staticmethod
    @numba.jit
    def E_delta_rho(rho, mu, T, t, x):
        recency = T - t
        return - x + rho / (rho+mu) + rho * t - rho * (1-rho*recency) * np.exp(-(rho+mu)*recency) / (mu+rho*np.exp(-(rho+mu)*recency))
    
    @staticmethod
    @numba.jit
    def E_delta_mu(rho, mu, T, t, x):
        recency = T - t
        return mu * (T + 1 / (rho+mu) - (1 + mu * recency) / (mu+rho*np.exp(-(rho+mu)*recency)))

    @staticmethod
    @numba.jit
    def E(rho, mu, T, t, x):
        recency = T - t
        return - x * np.log(rho) + np.log(rho+mu) + (rho+mu) * t - np.log(mu+rho*np.exp(-(rho+mu)*recency))
    
    def fit(self, num_sample=1000, L=100, epsilon=0.001, thinning=1):
        """
        num_sample: the number of samples you want
        L: Number of leapfrogs
        epsilon: step size
        thinning: how often to record samples
        """

        rho_list = []
        mu_list = []
        B_list = []
        Gamma_list = []
        lp_list = []

        log_rho_mu = np.random.randn(self.N, 2)
        rho = np.exp(log_rho_mu[:, 0])
        mu = np.exp(log_rho_mu[:, 1])
        B = np.zeros((self.dim, 2))
        Gamma = np.eye(2, dtype=float)

        for cnt in tqdm(range(num_sample)):
            for cnt_2 in range(thinning):
                # rhoとmuのサンプリング
                r = np.random.randn(self.N, 2)

                mean_log_rho_mu = self.features.dot(B)
                H_old = pareto_nbd.E(rho, mu, self.T, self.t, self.x) \
                    + ((log_rho_mu-mean_log_rho_mu) * np.linalg.solve(Gamma, (log_rho_mu-mean_log_rho_mu).T).T).sum(axis=1) / 2 \
                    + (r**2).sum(axis=1) / 2

                r -= epsilon/2 * (
                    np.exp(log_rho_mu).T * np.vstack([
                        pareto_nbd.E_delta_rho(rho, mu, self.T, self.t, self.x),
                        pareto_nbd.E_delta_mu(rho, mu, self.T, self.t, self.x)
                    ]) + np.linalg.solve(Gamma, (log_rho_mu-mean_log_rho_mu).T)).T
                log_rho_mu_new = log_rho_mu + epsilon * r
                rho_new = np.exp(log_rho_mu_new[:, 0])
                mu_new = np.exp(log_rho_mu_new[:, 1])

                for _ in range(L-1):
                    r -= epsilon * (
                        np.exp(log_rho_mu_new).T * np.vstack([
                            pareto_nbd.E_delta_rho(rho_new, mu_new, self.T, self.t, self.x),
                            pareto_nbd.E_delta_mu(rho_new, mu_new, self.T, self.t, self.x)
                        ]) + np.linalg.solve(Gamma, (log_rho_mu_new-mean_log_rho_mu).T)).T
                    log_rho_mu_new = log_rho_mu + epsilon * r
                    rho_new = np.exp(log_rho_mu_new[:, 0])
                    mu_new = np.exp(log_rho_mu_new[:, 1])

                r -= epsilon/2 * (
                        np.exp(log_rho_mu_new).T * np.vstack([
                            pareto_nbd.E_delta_rho(rho_new, mu_new, self.T, self.t, self.x),
                            pareto_nbd.E_delta_mu(rho_new, mu_new, self.T, self.t, self.x)
                        ]) + np.linalg.solve(Gamma, (log_rho_mu_new-mean_log_rho_mu).T)).T
                H_new =  pareto_nbd.E(rho_new, mu_new, self.T, self.t, self.x)\
                    + ((log_rho_mu_new-mean_log_rho_mu) * np.linalg.solve(Gamma, (log_rho_mu_new-mean_log_rho_mu).T).T).sum(axis=1) / 2\
                    + (r**2).sum(axis=1) / 2

                threshold = np.log(np.random.rand(self.N))
                accept = (threshold < H_old-H_new).astype(float)
                log_rho_mu[:, 0] = np.nan_to_num(accept * log_rho_mu_new[:, 0]) + (1-accept) * log_rho_mu[:, 0]
                log_rho_mu[:, 1] = np.nan_to_num(accept * log_rho_mu_new[:, 1]) + (1-accept) * log_rho_mu[:, 1]
                rho = np.exp(log_rho_mu[:, 0])
                mu = np.exp(log_rho_mu[:, 1])

                # BとGammaのサンプリング
                # 参考：https://en.wikipedia.org/wiki/Bayesian_multivariate_linear_regression
                mean_log_rho_mu = self.features.dot(B)
                Lambda_n = self.features.T.dot(self.features) + self.Lambda0
                B_n = np.linalg.solve(Lambda_n, self.features.T.dot(log_rho_mu))
                nu_n = self.nu0 + self.N
                V_n = self.V0 + (log_rho_mu - self.features.dot(B_n)).T.dot(log_rho_mu - self.features.dot(B_n)) + B_n.T.dot(self.Lambda0).dot(B_n)

                Gamma = invwishart(nu_n, scale=V_n).rvs()
                B = matrix_normal(mean=B_n.T, rowcov=Gamma, colcov=np.linalg.solve(Lambda_n, np.eye(len(Lambda_n)))).rvs().T # type: ignore

            mean_log_rho_mu = self.features.dot(B)
            lp = - pareto_nbd.E(rho, mu, self.T, self.t, self.x).sum()
            lp += - ((log_rho_mu-mean_log_rho_mu) * np.linalg.solve(Gamma, (log_rho_mu-mean_log_rho_mu).T).T).sum() / 2 - self.N * np.log(np.linalg.det(Gamma)) / 2
            lp += multivariate_normal(np.zeros(len(B.T.flatten())), cov=np.kron(Gamma, np.linalg.inv(self.Lambda0))).logpdf(B.T.flatten()) # type: ignore
            lp += invwishart(self.nu0, scale=self.V0).logpdf(Gamma)

            # リストへの格納
            rho_list.append(rho)
            mu_list.append(mu)
            B_list.append(B.copy())
            Gamma_list.append(Gamma.copy())
            lp_list.append(lp)

        self.rho_samples = np.array(rho_list)
        self.mu_samples = np.array(mu_list)
        self.B_samples = np.array(B_list)
        self.Gamma_samples = np.array(Gamma_list)
        self.lp = np.array(lp_list)