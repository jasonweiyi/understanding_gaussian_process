import numpy as np


j = 1e-6
for i in range(1, 3):
    jitter = np.eye(i) * j

    X = np.array(list(range(i, 0, -1)))
    X = np.expand_dims(X, axis=1)
    B = np.matmul(X, X.T)

    B = B + jitter

    B_inv = np.linalg.inv(B)

    C = (i+1) * X.T


    C_B_inv = np.matmul(C, B_inv)

    A = np.array([(i+1) * (i+1)])

    sigma = A - np.matmul(C_B_inv, C.T)
    print(i, C_B_inv, sigma)




