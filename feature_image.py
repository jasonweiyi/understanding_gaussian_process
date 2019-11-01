import numpy as np
from generate_training_data import generate_points
from prior import GP
import matplotlib.pyplot as plt
from utils import multiple_formatter


training_start= np.pi * 0
training_end = np.pi * 2

training_data = generate_points(training_start, training_end, section=7, points_quota=[2, 5], noise=0.01)

X, Y = training_data
l = 0.5
sigma = 1.
noise_sigma = 0.01


X_star = np.linspace(training_start, training_end + 0.2*np.pi, 50)


# Parameter learning
# fits = []
# complex = []
# ls = []
# for lengthscale in np.linspace(0.01, 3, 2000):
#     ls.append(lengthscale)
#     data_fit = data_fit_term(X, Y, lengthscale, sigma, noise_sigma)
#     model_complexity = model_complexity_term(X, Y, lengthscale, sigma, noise_sigma)
#     fits.append(data_fit)
#     complex.append(complex)
#     print(lengthscale, data_fit, model_complexity, data_fit + model_complexity)
#
# fits = np.array(fits)
# complex = np.array(complex)
# ls = np.array(ls)






# X_star = X
gp = GP(lengthscale=l, signal_variance=sigma, noise_variance=noise_sigma)
post_m, post_var, weights = gp.posterior(X, Y, X_star)

s = [60] * len(X)

st = [60] * len(X_star)


post_var = np.diagonal(post_var)
plt.plot(X_star, post_m, color='red', linewidth=2)

plt.fill_between(X_star,
                 post_m - 1.96 * np.sqrt(post_var),
                 post_m + 1.96 * np.sqrt(post_var),
                 color='C0', alpha=0.5)


plt.scatter(X_star, post_m, marker='.', s=st, color='red', alpha=1)
# plt.scatter(X, Y, marker='x', s=s, color='blue')


# Highlight weights in posterior mean.
# x_star = 25
# plt.scatter(X_star[x_star], post_m[x_star], s=[400], marker='.', color='red')
#
# weights = weights[x_star]
# min_w = min(weights)
# max_w = max(weights)
#
# weights = (weights - min_w) / max_w
marker_size = weights * 300 + 30

plt.scatter(X, Y, marker='x', s=60, color='blue')



ax = plt.gca()
ax.legend()
plt.xlim(min(X[0], X_star[0]), max(X[-1], X_star[-1]))
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

plt.show()