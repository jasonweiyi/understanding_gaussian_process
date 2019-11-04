from gp import Kernel


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
