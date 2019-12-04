from generate_training_data import generate_points
from gp import ExponentialSquaredKernel, GP
import numpy as np
import matplotlib.pyplot as plt

# Generate training data.



# Grid search for optimal lengthscale values, while keeping
# signal_variance and noise_variance unchanged.
print('n\tl\td\tm\to')
for noise_variance in [0.05, 0.1, 0.5, 1, 2, 3, 5, 10]:
    data_fits = []
    model_complexities = []
    lengthscales = []
    for lengthscale in np.linspace(0.001, 3, 100):
        # lengthscale += 0.1

        kernel = ExponentialSquaredKernel(
            lengthscale=lengthscale, signal_variance=1.)

        gp = GP(kernel, noise_variance=0.1)
        X, Y = generate_points(start=np.pi * 0, end=np.pi * 2, noise=noise_variance)

        data_fit = gp.data_fit_term(X, Y)
        model_complexity = gp.model_complexity_term(X)
        objective = gp.objective(X, Y)

        lengthscales.append(lengthscale)
        data_fits.append(data_fit)
        model_complexities.append(model_complexity)

    # Find the lengthscale that gives the maximum objective.
    objectives = np.array(data_fits) + np.array(model_complexities)
    optimal_lengthscale_id = np.argmax(objectives)
    max_objective = objectives[optimal_lengthscale_id]

    d = [noise_variance, lengthscales[optimal_lengthscale_id], data_fits[optimal_lengthscale_id], model_complexities[optimal_lengthscale_id], objectives[optimal_lengthscale_id]]

    line = ''
    for v in d:
        line += str(round(v, 7)) + '\t'
    print(line)
    # print(lengthscales[optimal_lengthscale_id])

# Plotting.
plt.figure(figsize=(20, 12))
plt.plot(lengthscales, data_fits, color='green', linewidth=2, label='Data fit')
plt.plot(lengthscales, model_complexities, color='blue', linewidth=2, label='Model complexity')
plt.plot(lengthscales, objectives, color='red', linewidth=5, label='Objective')
plt.scatter(lengthscales[optimal_lengthscale_id], max_objective, color='red', s=[300], marker='o')
plt.legend(loc=2, prop={'size': 14})

plt.show()
