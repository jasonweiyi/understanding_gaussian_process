import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from utils import multiple_formatter
from functools import partial


def kernel(x1: float, x2: float, l: float, sigma: float) -> float:
    result = sigma**2 * np.exp(-0.5 * (x1-x2)**2 / (l**2))
    return result


def m(xs: np.ndarray) -> np.ndarray:
    """
    Zero mean function.
    :param xs: the X locations.
    :return: the mean vector for those locations.
    """
    return np.zeros(xs.shape)


def km(xs1: np.ndarray, xs2: np.ndarray, l: float, sigma: float) -> np.ndarray:
    assert len(list(xs1.shape)) == 1
    assert len(list(xs2.shape)) == 1
    result = np.zeros(shape=(xs1.shape[0], xs2.shape[0]))
    for row_id, x1 in enumerate(xs1):
        for col_id, x2 in enumerate(xs2):
            result[row_id, col_id] = kernel(x1, x2, l, sigma)
    return result


def posterior(X_star, X, Y, l, sigma, noise_sigma):
    k_X_X = km(X, X, l, sigma)
    k_X_star_X = km(X_star, X, l, sigma)
    k_X_star_X_star = km(X_star, X_star, l, sigma)

    if noise_sigma is not None:
        noise_cov = np.eye(len(X)) * (noise_sigma**2)
    else:
        noise_cov = 0
    inv = np.linalg.inv(k_X_X + noise_cov)

    mean = np.matmul(np.matmul(k_X_star_X, inv), Y)
    var = k_X_star_X_star - np.matmul(np.matmul(k_X_star_X, inv), k_X_star_X.T)
    return mean, var

#print(kernel(x1=1, x2=2, l=2, sigma=1))

n = 50

X = np.linspace(0, 2*np.pi, n)
Y = np.sin(X)
noise = np.random.normal(0, 0.1, n)
Y += noise


colors = ['red', 'blue']
for index, l in enumerate([np.pi/100, np.pi/2]):

    sigma = 1

    k = partial(km, l=l, sigma=sigma)

    # plt.scatter(X, Y)
    # plt.show()


    samples = []
    probs = []
    sample_number = 1
    for i in range(sample_number):
        sample = np.random.multivariate_normal(m(X), k(X, X))
        samples.append(sample)

        prob = multivariate_normal.pdf(
            sample, mean=m(X), cov=k(X, X) + np.eye(n) * 1e-6,
            allow_singular=False)
        probs.append(prob)

        # plt.plot(X, sample)


    probs = np.array(probs)
    normalized_prob = probs - np.min(probs)
    normalized_prob /= np.max(normalized_prob)

    print(normalized_prob)

    for prob, sample in zip(normalized_prob, samples):
        # alpha = prob * 10
        alpha = 1
        plt.plot(X, sample, alpha=alpha, markersize=5, marker='o', color=colors[index], label='l={}π'.format(0.01 if index==0 else 0.5))




ax = plt.gca()
ax.legend()
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

plt.show()

# xs = np.array([1, 2, 3, 4])
# xs1 = np.array([1, 2, 3])
# xs2 = np.array([1, 2, 3, 4])
# print(kernel_matrix(xs1, xs2, l=2, sigma=1))

