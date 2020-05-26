# IMPLEMENTATION OF THE FRANK-WOLFE ALGORITHM
# In this file, we define the functions required for the implementation of the algorithm.

# %% Define the function f(x)
def f(x, A, y):
    """
    Function that computes the f(x) defined as: |Ax - y|^2
    :param x: (p, 1) numpy vector
                Data vector
    :param A: (n, p) numpy matrix
                Design matrix of the LASSO problem
    :param y: (n, ) numpy vector
                Target vector of the LASSO problem
    :return: f(x), scalar, evaluated as defined above
    """
    return np.matmul((np.matmul(A, x) - y).transpose(), np.matmul(A, x) - y)


# %% Define function to compute gradient
def gradient(x, A, y):
    """
    Function that computes the gradient of f(x) defined as: 2 * (A.T(Ax - y))
    :param x: same as defined in previous function
    :param A: same as defined in previous function
    :param y: same as defined in previous function
    :return:
    """
    return 2 * np.matmul(A.transpose(), np.matmul(A, x) - y)


# %% Define function to find a point in the Oracle
def fwOracle(gradient, l):
    """
    Function that computes the Frank-Wolfe Oracle defined as:
        argmin(s) <gradient(f(x)), s> where s in the feasible
        set D and < > is the inner product.
    :param gradient: (p, 1) numpy vector
                Should be the gradient of f(x)
    :param l: (lambda) a scalar > 0
                Penalty parameter of the LASSO problem
    :return: s: (p, 1) numpy vector
                FW Oracle as defined above
    """
    # Initialize the zero vector
    s = np.zeros((p, 1))

    # Check if all coordinates of x are 0
    # If they are, then the Oracle contains zero vector
    if (gradient != 0).sum() == 0:
        return s

    # Otherwise, follow the following steps
    else:
        # Compute the (element-wise) absolute value of x
        a = abs(gradient)
        # Find the first coordinate that has the maximum absolute value
        i = np.nonzero(a == max(a))[0][0]
        # Compute s
        s[i] = - np.sign(gradient[i]) * l
        return s


#%% Define function for applying the Frank-Wolfe algorithm to solve LASSO problem
def frankWolfeLASSO(A, y, l=500, tol=0.0001, K=15000):
    """
    Function that applies the Frank-Wolfe algorithm on a LASSO problem, given
    the required x, A, y and K.
    :param A: (n, p) numpy matrix
                Design matrix of the LASSO problem
    :param y: (n, ) numpy vector
                Target vector of the LASSO problem
    :param K: integer > 0
                Maximum number of iterations
    :param l: integer > 0
                Regularization parameter
    :param tol: float > 0
                    Tolerance rate of the error ||f(x_k) - f(x_(k-1))||
    :return: data: f(x): K-dimensional numpy vector
                argmin(D) of f
            diffx: (K-1)-dimensional numpy vector
                difference ||f(x_k) - f(x_(k-1))||
            k: integer > 0
                The number of iterations made
    """
    # Initialise:
    # x : sequence of K data points
    #       (each being a p-dimensional vector of features)
    # s : sequence of K "oracles"
    #       (each being a p-dimensional vector)
    # rho : step-size sequence having K elements
    # data : K-dimensional vector of resulting data points
    # data : (K-1)-dimensional vector of the difference f(x_k) - f(x_(k-1))
    # x[0] and s[0] to p-dimensional vectors of zeros (starting points)
    x = [None] * K
    s = [None] * K
    rho = [None] * K
    data = [None] * K
    diffx = [None] * K
    p = np.shape(A)[1]
    x[0] = np.zeros((p, 1))
    s[0] = np.zeros((p, 1))

    # Apply the Frank-Wolfe Algorithm
    for k in range(1, K):
        rho[k] = 2 / (2 + k)
        s[k] = fwOracle(gradient(x[k - 1], A, y), l)
        x[k] = (1 - rho[k]) * x[k - 1] + rho[k] * s[k]
        data[k] = f(x[k], A, y)
        if k > 1:
            diffx[k - 1] = data[k] - data[k - 1]
            if tol >= abs(diffx[k - 1]): break

    # Return
    return data, diffx, k
