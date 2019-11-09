import numpy as np
from generate_training_data import generate_points
from gp import GP, SquaredExponentialKernel
import matplotlib.pyplot as plt
from utils import multiple_formatter


# Generate training data.
training_start= np.pi * 0
training_end = np.pi * 2
training_data = generate_points(training_start, training_end, section=7, quota=[2, 5], noise=0.01)
X, Y = training_data

lengthscale = 2.1
signal_variance = 1.
noise_variance = 0.01


# Setup testing locations.
# You can change the testing locations here.
X_star = np.linspace(training_start, training_end + 4*np.pi, 50)

# Compute posterior mean and variance.
kernel = SquaredExponentialKernel(lengthscale=lengthscale, signal_variance=signal_variance)
gp = GP(kernel=kernel, noise_variance=noise_variance)
post_m, post_variance, weights = gp.posterior(X, Y, X_star)

# Plot posterior mean and variance.
post_variance = np.diagonal(post_variance)
plt.plot(X_star, post_m, color='red')
plt.scatter(X_star, post_m, marker='.', s=[60] * len(X_star), color='red', alpha=0.7)
plt.fill_between(X_star,
                 post_m - 1.96 * np.sqrt(post_variance),
                 post_m + 1.96 * np.sqrt(post_variance),
                 color='C0', alpha=0.2)


# Highlight weights in posterior mean.
# If you want to highlight the weights of the training point to the test location
# highlight_x_star, toggle should_highlight to True. You can change which test location
# to highlight by changing the value of highlight_x_star.
should_highlight = False
highlight_x_star = 25

if should_highlight:
    plt.scatter(X_star[highlight_x_star], post_m[highlight_x_star],
                s=[400], marker='.', color='red')
    weights = weights[highlight_x_star]
    min_w = min(weights)
    max_w = max(weights)

    weights = (weights - min_w) / max_w
    marker_size = weights * 300 + 30
    plt.scatter(X, Y, marker='x', s=marker_size, color='blue')

ax = plt.gca()
ax.legend()
plt.xlim(min(X[0], X_star[0]), max(X[-1], X_star[-1]))
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

plt.show()
