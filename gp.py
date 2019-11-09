from typing import Tuple
import numpy as np


class Kernel:
    """Abstract kernel class"""
    def covariance(self, X1: float, X2: float) -> float:
        """Return the covariance between location X1 and X2."""
        raise NotImplementedError()


class SquaredExponentialKernel(Kernel):
    def __init__(self, lengthscale: float, signal_variance: float) -> None:
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance

    def covariance(self, X1: float, X2: float) -> float:
        result = np.square(self.signal_variance) * np.exp(-0.5 * np.square((X1 - X2) / self.lengthscale))
        return result


class GP:
    """GP class."""
    def __init__(self, kernel: Kernel, noise_variance: float) -> None:
        self.kernel = kernel
        self.noise_variance = noise_variance

    def k(self, Xs1: np.ndarray, Xs2: np.ndarray) -> np.ndarray:
        """Covariance matrix between locations Xs1 and Xs2. Xs1 and Xs2 are 1D arrays."""
        assert len(list(Xs1.shape)) == 1
        assert len(list(Xs2.shape)) == 1
        result = np.zeros(shape=(Xs1.shape[0], Xs2.shape[0]))
        for row_id, X1 in enumerate(Xs1):
            for col_id, X2 in enumerate(Xs2):
                result[row_id, col_id] = self.kernel.covariance(X1, X2)
        return result

    def posterior(self, X: np.ndarray, Y: np.ndarray,
                  X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the posterior mean and covariance,
           with the weights used in the posterior mean for plotting."""
        n = len(X)  # Number of training data points.
        n_star = len(X_star) # Number of testing data points.

        assert Y.shape == (n,)

        # Compute k(X, X).
        k_X_X = self.k(X, X)
        assert k_X_X.shape == (n, n)

        # inv is (k(X, X) + Σ)⁻¹.
        inv = self.inv(X)
        assert inv.shape == (n, n)

        # Compute k(X_star, X).
        k_X_star_X = self.k(X_star, X)
        assert k_X_star_X.shape == (n_star, n)
        weights = k_X_star_X

        # Compute k(X_star, X_star).
        k_X_star_X_star = self.k(X_star, X_star)
        assert k_X_star_X_star.shape == (n_star, n_star)

        # Compute posterior mean k(X_star, X) (k(X, X) + Σ)⁻¹ Y.
        mean = np.matmul(np.matmul(k_X_star_X, inv), Y)
        assert mean.shape == (n_star,)

        # Compute posterior covariance:
        # k(X_star, X_star) - k(X_star, X) (k(X, X) + Σ)⁻¹ k(X_star, X)ᵀ
        covariance = k_X_star_X_star - np.matmul(np.matmul(k_X_star_X, inv), k_X_star_X.T)
        assert covariance.shape == (n_star, n_star)

        return mean, covariance, weights

    def inv(self, X: np.ndarray) -> np.ndarray:
        """Return the value of (k(X, X)+Σ)⁻¹."""
        n = len(X)
        sigma = np.eye(len(X)) * np.square(self.noise_variance)
        assert sigma.shape == (n, n)

        result = np.linalg.inv(self.k(X, X) + sigma)
        assert result.shape == (n, n)
        return result

    def data_fit_term(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return value of the data fit term -½ Yᵀ (k(X, X)+Σ)⁻¹ Y"""
        result = - 0.5 * np.matmul(np.matmul(Y.T, self.inv(X)), Y)
        return result

    def model_complexity_term(self, X: np.ndarray) -> np.ndarray:
        """Return value of the model complexity term -½log|k(X, X)+Σ|"""
        k_X_X = self.k(X, X)
        noise_cov = np.eye(len(X)) * np.square(self.noise_variance)
        det = np.linalg.det(k_X_X + noise_cov)
        result = -0.5 * np.log(det)
        return result

    def objective(self, X, Y):
        """Return value of the objective function"""
        return self.data_fit_term(X, Y) + self.model_complexity_term(X)
