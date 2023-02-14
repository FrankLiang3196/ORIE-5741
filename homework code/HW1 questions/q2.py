import cvxopt
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from scipy.spatial.distance import cdist

def solve_SVM_primal(x, y, regularization=None):
    """ solve the primal problem of SVM
    :param x: dataset x
    :param y: dataset y
    :pram regularization: the C
    :return w: the weights
    :return b: the bias
    """
    N, m = x.shape

    # TODO Q2(a):
    # step 1: convert the input so we can get offset
    # TODO: pad the input
    x_pad = np.concatenate((x, np.ones((N, 1))), axis=1)
    m += 1

    # step 2: formalize the problem
    # TODO: prepare input for the cvxopt solver
    # tutorial: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    if regularization is None:
        # Q2(a)
        P = cvxopt_matrix(np.eye(m), tc='d')
        G = cvxopt_matrix(- y.reshape(-1, 1) * x_pad, tc='d')
        h = cvxopt_matrix(-1 * np.ones(N), tc='d')
        q = cvxopt_matrix(np.zeros(m), tc='d')

    else:
        # Q2(c)
        pass

    # step 3: solve the problem using cvxopt
    # TODO: call the cvxopt solver
    solution = cvxopt_solvers.qp(P=P, G=G, h=h, q=q)['x']

    # step 4: convert the result and return
    # TODO: get the w and b from the solver's solution
    w = np.array(solution[:-1])
    b = np.array(solution[-1])
    print(f'weights: {w}; bias: {b}')
    return w, b


def solve_SVM_dual(affinities, y, regularization, folds=5):
    """ solver the dual problem of SVM
    :param affinities: the affinity matrix, 
        where A[i,j] is K(x_i, x_j), 
        K is the kernel function
    :param y: dataset y
    :param regularization: the C
    :param folds: number of folds for cross validation
    :return w: the weights
    :return b: the bias
    """
    N = affinities.shape[0]
    N_test = N // folds
    N_train = N - N_test

    accuracy_records = []

    idxs = np.arange(N)
    np.random.shuffle(idxs)
    for iter in range(folds):
        # TODO Q2(d)(ii) and Q2(d)(iii):
        # step 0: split the data into train and test set

        # step 1: formalize the problem
        # TODO: prepare input for the cvxopt solver
        # tutorial: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

        # step 2: solve the problem using cvxopt
        # TODO: call the cvxopt solver

        # step 3: get the offset
        # TODO:

        # step 4: fit training data
        # TODO: compute train accuracy
        train_accuracy = 0.0
        print(f'iter {iter}: train accuracy={train_accuracy}')

        # step 5: fit test data and record the accuracy
        # TODO: compute the test accuracy
        accuracy = 0.0
        print(f'iter {iter}: test accuracy={accuracy}')
        accuracy_records.append(accuracy)

    print(f'average test accuracy: {np.mean(accuracy_records)}')


def get_support_vectors(x, y, w, b, eps=1e-3):
    """ compute the support vectors
    :param x: the datapoints' x
    :param y: the datapoints' y
    :param w, b: y=sign(wx+b) is the decision boundary
    :param eps: a==b if |a-b| < eps
    :return positive_vectors: support vectors for positive examples
    :return positive_boundary (margin line where the positive support vectors lie on): (w, b) for positive examples
    :return negative_vectors: support vectors for negative examples
    :return negative_boundary (margin line where the negative support vectors lie on): (w, b) for the negative examples
    """
    # TODO Q2(b):
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

    b_negative = - np.dot(w.reshape(2), negative_vectors[0])
    b_positive = - np.dot(w.reshape(2), positive_vectors[0])

    negative_vectors = np.array(negative_vectors)
    positive_vectors = np.array(positive_vectors)

    positive_boundary = (w, b_positive)
    negative_boundary = (w, b_negative)

    return positive_vectors, positive_boundary, \
        negative_vectors, negative_boundary


def get_affinity_matrix(X, method, **kwargs):
    """ apply kernel function to the data
    :param X: the input data (Nxd)
    :param method: 'product' or 'rbf'
    :param M: affinity matrix (NxN)
    """
    if method == 'product':
        products = cdist(X, X, lambda u, v: np.dot(u, v))
        return products
    elif method == 'rbf':
        # TODO Q2(d)(i): implement the kernel function
        raise NotImplementedError
    else:
        raise NotImplementedError(f'Unknown kernel type {method}')

