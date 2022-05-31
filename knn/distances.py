import numpy as np


def euclidean_distance(x, y):
    m = y.shape[0]
    n = x.shape[0]
    y_2 = np.zeros((m))
    x_2 = np.zeros((n))
    for i in range(m):
        y_2[i] = np.array([np.dot(y[i], y[i])])
    for i in range(n):
        x_2[i] = np.array([np.dot(x[i], x[i])])
    A = -2*x@y.T
    return np.sqrt(A + x_2[:, np.newaxis]*np.ones((n, m)) + y_2*np.ones((n, m)))
    raise NotImplementedError()


def cosine_distance(x, y):
    m = y.shape[0]
    n = x.shape[0]
    x = np.array(x)
    y = np.array(y)
    y_2 = np.zeros((m))
    x_2 = np.zeros((n))
    for i in range(m):
        y_2[i] = np.array([np.dot(y[i], y[i])])
    for i in range(n):
        x_2[i] = np.array([np.dot(x[i], x[i])])
    A = x@y.T
    return np.ones((n, m)) - A / np.sqrt((x_2[:, np.newaxis] * np.ones((n, m))) * y_2)
    raise NotImplementedError()