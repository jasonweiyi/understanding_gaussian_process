from gp import Kernel
import numpy as np


class ConstantKernel(Kernel):
    def __init__(self, variance) -> None:
        self.variance = variance

    def covariance(self, X1: float, X2: float) -> float:
        return self.variance


class SumKernel(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel) -> None:
        self.kernels = [kernel1, kernel2]

    def covariance(self, X1: float, X2: float) -> float:
        result = sum([k.covariance(X1, X2) for k in self.kernels])
        return result


class ProductKernel(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel) -> None:
        self.kernels = [kernel1, kernel2]

    def covariance(self, X1: float, X2: float) -> float:
        result = np.prod([k.covariance(X1, X2) for k in self.kernels])
        return result


class LinearKernel(Kernel):
    def __init__(self, variance):
        self.variance = variance

    def covariance(self, X1: float, X2: float) -> float:
        result = self.variance * X1 * X2
        return result


class Matern12(Kernel):
    def __init__(self, lengthscale: float, signal_variance: float) -> None:
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance

    def covariance(self, X1: float, X2: float) -> float:
        result = np.square(self.signal_variance) * np.exp(- np.sqrt(np.square(X1 - X2)) / self.lengthscale)
        return result


class Matern52(Kernel):
    def __init__(self, lengthscale: float, signal_variance: float) -> None:
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance

    def covariance(self, X1: float, X2: float) -> float:
        r = np.sqrt(np.square(X1-X2)) / self.lengthscale
        a = 1. + np.sqrt(5.) * r + 5./3. * r*r
        result = np.square(self.signal_variance) * a * np.exp(- np.sqrt(5.) * r)
        return result


class PeriodicKernel(Kernel):
    def __init__(self, lengthscale: float, signal_variance: float, period: float) -> None:
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance
        self.period = period

    def covariance(self, X1: float, X2: float) -> float:
        r = np.abs(X1-X2)
        a = np.square(np.sin(np.pi * r / self.period))
        result = np.square(self.signal_variance) * np.exp(-2*a/np.square(self.lengthscale))
        return result


