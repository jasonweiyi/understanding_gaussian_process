from gp import GP, ExponentialSquaredKernel
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from kernels import ConstantKernel, LinearKernel, SumKernel, ProductKernel, Matern12, Matern52
from utils import multiple_formatter

import gpflow
import numpy as np
import tensorflow as tf
import matplotlib



# Set values to model parameters.
lengthscale = 1
signal_variance = 1.
noise_variance = 0.1




k = gpflow.kernels.Matern52(input_dim=1)



def plotkernelsample(k, xmin=0, xmax=3):
    xx = np.linspace(xmin, xmax, 300)[:,None]
    K = k.K(xx)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cov = sess.run(K)
        plt.plot(xx, np.random.multivariate_normal(np.zeros(300), cov, 5).T)
        plt.set_title('Samples ' + k.__class__.__name__)
        plt.show()


plotkernelsample(k)

