from gp import GP, ExponentialSquaredKernel
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from kernels import ConstantKernel, LinearKernel, SumKernel, ProductKernel, Matern12, Matern52, \
    PeriodicKernel
from utils import multiple_formatter

# Set values to model parameters.
lengthscale = 1
signal_variance = 1.
noise_variance = 0.01

# Create the GP.
e_kernel = ExponentialSquaredKernel(
    lengthscale=lengthscale*2, signal_variance=signal_variance)


c_kernel = ConstantKernel(variance=1)

l_kernel = LinearKernel(variance=1)

p_kernel = PeriodicKernel(
    lengthscale=lengthscale, signal_variance=signal_variance, period=np.pi * 0.25)

e2_kernel = ExponentialSquaredKernel(
    lengthscale=lengthscale * 1000, signal_variance=signal_variance)

m5_kernel = Matern52(lengthscale=lengthscale*10, signal_variance=signal_variance * 10)
m1_kernel = Matern12(lengthscale=lengthscale, signal_variance=signal_variance)


# kernel = SumKernel(m5_kernel, m1_kernel)
# kernel = p_kernel
kernel = m5_kernel

# kernel = m5_kernel
kernel = ProductKernel(m1_kernel, p_kernel)
kernel = SumKernel(m1_kernel, p_kernel)
# kernel = p_kernel
# kernel = ProductKernel(e_kernel, l_kernel)
# kernel = SumKernel(m5_kernel, m1_kernel)
# kernel = l_kernel
# kernel = m5_kernel

kernels = [#SumKernel(m1_kernel, p_kernel),
           ProductKernel(m1_kernel, p_kernel),
           m1_kernel,
           p_kernel]
names = ['Matern1/2 x Periodic', 'Matern1/2', 'Periodic']

colors = ['red', 'blue', 'green']


fig, ax = plt.subplots(3, 1, sharex=True)

for id_, kernel in enumerate(kernels):

    # kernel = e_kernel
    gp = GP(kernel=kernel, noise_variance=noise_variance)
    n = 200
    x = np.linspace(0, 3*np.pi, n)
    # x = np.array([1, 2, 3])
    mean = np.zeros(n)
    cov = gp.k(x, x)

    print(cov)

    # Draw samples from the GP prior.
    probabilities = []
    samples = []
    jitter = np.eye(n) * 1e-3
    for _ in range(1):
        y = multivariate_normal.rvs(mean=mean, cov=cov + jitter)
        # Add a jitter to the covariance matrix for numerical stability.
        prob = multivariate_normal.pdf(y, mean=mean, cov=cov + jitter)

        samples.append(y)
        probabilities.append(prob)

    # Normalize sample probabilities into [0, 1].
    probabilities = np.array(probabilities)
    min_prob, max_prob = np.min(probabilities), np.max(probabilities)
    probabilities = (probabilities - min_prob) / (max_prob - min_prob)

    # Plotting.
    for y, prob in zip(samples, probabilities):
        ax[id_].plot(x, y, alpha=1, color=colors[id_], label=names[id_])
        ax[id_].legend(loc="upper right")
    ax[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax[0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax[0].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))


plt.show()
