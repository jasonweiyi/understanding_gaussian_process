from prior import GP
import numpy as np
import matplotlib.pyplot as plt


lengthscale = 1.
signal_variance = 1.
noise_variance = 0.1
gp1 = GP(lengthscale=lengthscale, signal_variance=signal_variance, noise_variance=noise_variance, constant=0)
gp2 = GP(lengthscale=lengthscale, signal_variance=signal_variance, noise_variance=noise_variance, constant=1)

n = 60
X = np.linspace(0, 2*np.pi, n)


for gp, color in zip([gp1, gp2], ['blue', 'red']):
    mean = np.zeros(n)
    cov = gp.k(X, X)
    for s in range(10):
        y = np.random.multivariate_normal(mean, cov, 1)
        plt.plot(X, y[0], color=color)
plt.show()


