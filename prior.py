from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from utils import multiple_formatter
from functools import partial



class GP:
    def __init__(self, lengthscale: float, signal_variance: float, noise_variance: float, constant: float=0) -> None:
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.constant = constant

    def kernel(self, X1: float, X2: float) -> float:
        result = np.square(self.signal_variance) * np.exp(-0.5 * np.square((X1 - X2) / self.lengthscale))
        return result + self.constant

    def k(self, Xs1: np.ndarray, Xs2: np.ndarray) -> np.ndarray:
        assert len(list(Xs1.shape)) == 1
        assert len(list(Xs2.shape)) == 1
        result = np.zeros(shape=(Xs1.shape[0], Xs2.shape[0]))
        for row_id, X1 in enumerate(Xs1):
            for col_id, X2 in enumerate(Xs2):
                result[row_id, col_id] = self.kernel(X1, X2)
        return result

    def posterior(self, X: np.ndarray, Y: np.ndarray, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        n_star = len(X_star)

        assert Y.shape == (n,)

        k_X_X = self.k(X, X)
        assert k_X_X.shape == (n, n)

        inv = self.inv(X)
        assert inv.shape == (n, n)

        k_X_star_X = self.k(X_star, X)
        assert k_X_star_X.shape == (n_star, n)
        weights = k_X_star_X

        mean = np.matmul(np.matmul(k_X_star_X, inv), Y)
        assert mean.shape == (n_star,)

        k_X_star_X_star = self.k(X_star, X_star)
        assert k_X_star_X_star.shape == (n_star, n_star)

        variance = k_X_star_X_star - np.matmul(np.matmul(k_X_star_X, inv), k_X_star_X.T)
        assert variance.shape == (n_star, n_star)

        return mean, variance, weights

    def inv(self, X):
        n = len(X)
        sigma = np.eye(len(X)) * np.square(self.noise_variance)
        assert sigma.shape == (n, n)

        result = np.linalg.inv(self.k(X, X) + sigma)
        assert result.shape == (n, n)
        return result

    def data_fit_term(self, X, Y):
        result = - 0.5 * np.matmul(np.matmul(Y.T, self.inv(X)), Y)
        return result

    def model_complexity_term(self, X):
        k_X_X = self.k(X, X)
        noise_cov = np.eye(len(X)) * np.square(self.noise_variance)
        det = np.linalg.det(k_X_X + noise_cov)
        result = -0.5 * np.log(det)
        return result

    def objective(self, X, Y):
        return self.data_fit_term(X, Y) + self.model_complexity_term(X)

        # data_fit = data_fit_term(X, Y, lengthscale, sigma, noise_sigma)

#     model_complexity = model_complexity_term(X, Y, lengthscale, sigma, noise_sigma)





# def kernel(x1: float, x2: float, l: float, sigma: float) -> float:
#     result = sigma**2 * np.exp(-0.5 * np.square(x1-x2) / np.square(l))
#     return result
#
#
# def k(xs1: np.ndarray, xs2: np.ndarray, l: float, sigma: float) -> np.ndarray:
#     assert len(list(xs1.shape)) == 1
#     assert len(list(xs2.shape)) == 1
#     result = np.zeros(shape=(xs1.shape[0], xs2.shape[0]))
#     for row_id, x1 in enumerate(xs1):
#         for col_id, x2 in enumerate(xs2):
#             result[row_id, col_id] = kernel(x1, x2, l, sigma)
#     return result
#
#
# def posterior(X_star, X, Y, l, sigma, noise_sigma):
#     k_X_X = k(X, X, l, sigma)
#     k_X_star_X = k(X_star, X, l, sigma)
#     k_X_star_X_star = k(X_star, X_star, l, sigma)
#
#     if noise_sigma is not None:
#         noise_cov = np.eye(len(X)) * (noise_sigma**2)
#     else:
#         noise_cov = 0
#     inv = np.linalg.inv(k_X_X + noise_cov)
#
#     mean = np.matmul(np.matmul(k_X_star_X, inv), Y)
#     var = k_X_star_X_star - np.matmul(np.matmul(k_X_star_X, inv), k_X_star_X.T)
#     additional = {
#         'k_X_star_X': k_X_star_X
#     }
#     return mean, var, additional
#
#
# def data_fit_term(X, Y, l, sigma, noise_sigma):
#     k_X_X = k(X, X, l, sigma)
#     if noise_sigma is not None:
#         noise_cov = np.eye(len(X)) * (noise_sigma**2)
#     else:
#         noise_cov = 0
#     inv = np.linalg.inv(k_X_X + noise_cov)
#     result = - 0.5 * np.matmul(np.matmul(Y.T, inv), Y)
#     return result
#
#
# def model_complexity_term(X, Y, l, sigma, noise_sigma):
#     k_X_X = k(X, X, l, sigma)
#     if noise_sigma is not None:
#         noise_cov = np.eye(len(X)) * (noise_sigma ** 2)
#     else:
#         noise_cov = 0
#     det = np.linalg.det(k_X_X + noise_cov)
#
#     result = -0.5 * np.log(det)
#     return result