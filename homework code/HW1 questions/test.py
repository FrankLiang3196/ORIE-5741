import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from fontTools.misc.bezierTools import epsilon
from scipy.spatial.distance import cdist


if __name__ == '__main__':
    x = np.array([[1, 0],
                  [0, 0],
                  [0, 1],
                  [2, 0],
                  [1, 1],
                  [2, 2]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    theta = np.array([1, 1])

    N, m = x.shape

    x_pad = np.concatenate((x, np.ones((N, 1))), axis=1)
    m += 1

    P = cvxopt_matrix(np.eye(m), tc='d')
    P[2, 2] = 0
    G = cvxopt_matrix(- y.reshape(-1, 1) * x_pad, tc='d')

    h = cvxopt_matrix(-1 * np.ones(N), tc='d')
    q = cvxopt_matrix(np.zeros(m), tc='d')

    solution = cvxopt_solvers.qp(P=P, q=q, G=G, h=h)

    w = np.array(solution['x'][:-1])
    b = np.array(solution['x'][-1])

    eps = 1e-3
    dist = np.abs(np.matmul(x, w) + b) / np.linalg.norm(w)
    min_dist = min(dist)

    negative_vectors = []
    positive_vectors = []

    for i, point in enumerate(x):
        if abs(dist[i] - min_dist) < eps:
            if np.dot(point, w) + b < 0:
                negative_vectors.append(point)
            else:
                positive_vectors.append(point)

    negative_vectors = np.array(negative_vectors)
    positive_vectors = np.array(positive_vectors)

    b_negative = np.dot(w.reshape(2), negative_vectors[0])
    b_positive = np.dot(w.reshape(2), positive_vectors[0])

    print(negative_vectors, b_negative)

