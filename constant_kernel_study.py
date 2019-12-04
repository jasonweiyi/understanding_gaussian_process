import numpy as np


j = 1e-6
for i in range(1, 5):
    jitter = np.eye(i) * j

    B = np.ones((i, i)) + jitter

    B_inv = np.linalg.inv(B)

    C = np.ones((1, i))


    C_B_inv = np.matmul(C, B_inv)

    A = np.array([1])

    sigma = A - np.matmul(C_B_inv, C.T)
    print(i, C_B_inv, sigma)




