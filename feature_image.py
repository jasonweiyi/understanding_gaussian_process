# Code to generate the feature image of the Medium article.
import numpy as np
import matplotlib.pyplot as plt

from gp import GP, SquaredExponentialKernel

# Load the image of the wave.
img = plt.imread("wave.jpg")
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.imshow(img)
x = range(800)

# Data points in the images.
data = [(223, 330),
        (244, 344),
        (279, 355),
        (331, 369),
        (392, 378),
        (432, 384),
        (468, 387),
        (514, 389),
        (565, 396),
        (595, 395),
        (638, 383),
        (680, 373),
        (707, 363),
        (745, 344),
        (784, 331),
        (829, 312),
        (863, 291)]

X = np.array([p[0] for p in data])
Y = np.array([p[1] for p in data])

# Normalize the Y dimension.
mean_Y = np.mean(Y)
std_Y = np.std(Y)
Y = (Y - mean_Y) / std_Y


# Fit a Gaussian Process to the data points.
lengthscale = 40
signal_variance = 3.
noise_variance = 0.1
X_star = np.linspace(0, 960, 50)
kernel = SquaredExponentialKernel(lengthscale=lengthscale, signal_variance=signal_variance)
gp = GP(kernel, noise_variance=noise_variance)
post_m, post_var, weights = gp.posterior(X, Y, X_star)


# Plot results.
color = 'yellow'
ax.plot(X_star, post_m*std_Y + mean_Y, color=color)
ax.scatter(X, Y*std_Y+mean_Y, s=30, color=color)

post_var = np.diagonal(post_var)

plt.fill_between(X_star,
                 (post_m - 1.96 * np.sqrt(post_var)) * std_Y + mean_Y,
                 (post_m + 1.96 * np.sqrt(post_var)) * std_Y + mean_Y,
                 color=color, alpha=0.2)

plt.xlim(0, 960)
plt.ylim(500, 100)
plt.axis('off')
plt.show()
