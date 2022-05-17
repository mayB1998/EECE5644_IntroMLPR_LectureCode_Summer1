"""Learning algorithms and models."""

import matplotlib.pyplot as plt
import numpy as np

# Suppress scientific notation
np.set_printoptions(suppress=True)

from modules import data_utils


def analytical_ls_solution(X, y):
    # Analytical solution is (X^T*X)^-1 * X^T * y
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def gradient_descent(loss_func, theta0, X, y, *args, **kwargs):
    """  Mini-batch GD for LS regression. Stochastic GD if batch_size=1.

    Args:
        loss_func: Loss function handle to optimize over using GD.
        theta0: Initial parameters vector, of shape (n + 1).
        X: Design matrix (added bias units), shape (N, n + 1), where n is the feature dimensionality.
        y: Labels for regression problem, of shape (N, 1).
        opts: Options for total sweeps over data (max_epochs), and parameters, like learning rate and momentum.

    Returns:
        theta: Final weights solution converged to after `iterations`, of shape [n].
        trace: Arrays of loss and weight updates, of shape [iterations, -1].
    """

    # Default options
    max_epoch = kwargs['max_epoch'] if 'max_epoch' in kwargs else 200
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.1
    epsilon = kwargs['tolerance'] if 'tolerance' in kwargs else 1e-6

    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 10

    # Turn the data into batches
    X_batch, y_batch = data_utils.batchify(X, y, batch_size)
    num_batches = len(y_batch)
    print("%d batches of size %d\n" % (num_batches, batch_size))

    theta = theta0

    trace = {}
    trace['loss'] = []
    trace['theta'] = []

    # Main loop:
    for epoch in range(1, max_epoch + 1):
        print("epoch %d\n" % epoch)
        for b in range(num_batches):
            X_b = X_batch[b]
            y_b = y_batch[b]
            # print("epoch %d batch %d\n" % (epoch, b))

            mse, gradient = loss_func(theta, X_b, y_b, *args)

            # Steepest descent update
            theta = theta - alpha * gradient

            # Storing the history of the parameters and loss values (MSE)
            trace['loss'].append(mse)
            trace['theta'].append(theta)

            # Terminating Condition is based on how close we are to minimum (gradient = 0)
            if np.linalg.norm(gradient) < epsilon:
                print("Gradient Descent has converged")
                break

        # Also break epochs loop
        if np.linalg.norm(gradient) < epsilon:
            break

    return theta, trace


def perform_lda(X, labels, C=2, plot_vec=True):
    """  Fisher's Linear Discriminant Analysis (LDA) on data from two classes (C=2).

    Note: you can generalize this implementation to multiple classes by finding a
    projection matrix W (rather than vector) that reduces dimensionality n inputs
    to a multidimensional projection z=W'*x (e.g. z of dimension C-1). Now we have
    a crude but quick way of achieving class-separability-preserving linear dimensionality
    reduction using the Fisher LDA objective as a measure of class separability.

    Args:
        X: Real-valued matrix of samples with shape [N, n], N for sample count and n for dimensionality.
        labels: Class labels per sample received as an [N, 1] column.
        C: Number classes, explicitly clarifying that we're doing binary classification here.
        plot_vec: If you want the option of directly plotting the linear projection vector.

    Returns:
        w: Fisher's LDA project vector, shape [n, 1].
        z: Scalar LDA projections of input samples, shape [N, 1].
    """

    # Estimate mean vectors and covariance matrices from samples
    # Note that reshape ensures my return mean vectors are of 2D shape (column vectors nx1)
    mu = np.array([np.mean(X[labels == i], axis=0).reshape(-1, 1) for i in range(C)])
    cov = np.array([np.cov(X[labels == i].T) for i in range(C)])

    # Determine between class and within class scatter matrix
    Sb = (mu[1] - mu[0]).dot((mu[1] - mu[0]).T)
    Sw = cov[0] + cov[1]

    # Regular eigenvector problem for matrix Sw^-1 Sb
    lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
    idx = lambdas.argsort()[::-1]
    # Extract corresponding sorted eigenvectors
    U = U[:, idx]
    # First eigenvector is now associated with the maximum eigenvalue, mean it is our LDA solution weight vector
    w = U[:, 0]

    # Scalar LDA projections in matrix form
    z = X.dot(w)

    if plot_vec:
        # All the variables we need to get set for plotting
        mid_point = (mu[0] + mu[1]) / 2
        slope = w[1] / w[0]
        c = mid_point[1] - slope * mid_point[0]

        xmax = np.max(X[:, 0])
        xmin = np.min(X[:, 0])
        x = np.linspace(xmin + 1, xmax + 1, 100)

        fig = plt.figure(figsize=(12, 12))

        x0 = X[labels == 0]
        x1 = X[labels == 1]
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(x0[:, 0], x0[:, 1], 'b.', x1[:, 0], x1[:, 1], 'r+')
        ax1.plot(x, slope * x + c, c='orange')
        ax1.legend(["C0", "C1", "wLDA"])

        ax2 = fig.add_subplot(2, 1, 2)
        z0 = z[labels == 0]
        z1 = z[labels == 1]
        ax2.plot(z0, np.zeros(len(z0)), 'b.', z1, np.zeros(len(z1)), 'r+')
        plt.show()

    return w, z
