import csv
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('c:\\users\\jasonw\\Desktop\\objective_values.csv', delimiter=' ')

l = data[:, 0]
data_fit = data[:, 1]
model_complex = data[:, 2]

objective = data_fit + model_complex
max_point = np.argmax(objective)

print(l[max_point])
plt.plot(l, data_fit, color='green', linewidth=2, label='Data fit term')
plt.plot(l, model_complex, color='blue', linewidth=2, label='Model complexity term')
plt.plot(l, data_fit + model_complex, color='red', linewidth=5, label='Objective function = data fit + model complexity')
plt.xlim(min(l), max(l))
plt.ylim(min(data_fit), max(objective)+ 80)
plt.scatter(l[max_point], objective[max_point], color='red', s=[300], marker='o')
plt.legend(loc=2, prop={'size': 14})


plt.show()



