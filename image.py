import numpy as np
import matplotlib.pyplot as plt

from prior import GP

img = plt.imread("/home/yiwei/wave.jpg")
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.imshow(img)
x = range(800)


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
mean_Y = np.mean(Y)
std_Y = np.std(Y)
Y = (Y - mean_Y) / std_Y

l = 40
sigma = 3.
noise_sigma = 0.1


X_star = np.linspace(0, 960, 50)

gp = GP(lengthscale=l, signal_variance=sigma, noise_variance=noise_sigma)
post_m, post_var, weights = gp.posterior(X, Y, X_star)


ax.plot(X_star, post_m*std_Y + mean_Y, color='red')
ax.scatter(X, Y*std_Y+mean_Y, s=30, color='firebrick')

post_var = np.diagonal(post_var)

plt.fill_between(X_star,
                 (post_m - 1.96 * np.sqrt(post_var)) * std_Y + mean_Y,
                 (post_m + 1.96 * np.sqrt(post_var)) * std_Y + mean_Y,
                 color='red', alpha=0.2)

plt.xlim(0, 960)
plt.ylim(500, 100)
plt.axis('off')
plt.show()
# plt.savefig("/home/yiwei/wave_gp.jpg",bbox_inches='tight')

# plt.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off

# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off


# plt.show()
