"""Example of Least Squares (LS) regression using Stochastic Gradient Descent (GD)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from modules import data_utils, loss_functions, models

np.random.seed(7)

N = 1000  # Number of samples
batch_size = 10
order = 1  # Order of features to "engineer"
theta_true = np.array([4, -5.4])  # True parameter vector
bias_true = 7.7  # True bias value

# Each theta are now coordinate matrices determined by the theta_space
theta_space = np.arange(-15, 15, 0.4)
Th0, Th1 = np.meshgrid(theta_space, theta_space)

# Options for stochastic gradient descent (SGD)
opts = {}
opts['max_epoch'] = 50
opts['alpha'] = 0.05
opts['tolerance'] = 1e-3
opts['batch_size'] = 16


# Plot residual sum squared (RSS) error surface over parameter space
def create_rss_surface(X, y):
    # Compute RSS error for each weight combination
    # Split out into vector form for convenience in reshaping and feeding into the RSS surface plot
    rss = np.array(
        [sum((th0 * X[:, 0] + th1 * X[:, 1] - y) ** 2) for th0, th1 in zip(np.ravel(Th0), np.ravel(Th1))])
    rss = rss.reshape(Th0.shape)

    fig_surface = plt.figure(figsize=(10, 10))
    ax_surface = fig_surface.add_subplot(111, projection='3d')
    # Plot contour surface of loss function with 50 levels
    cs3d = ax_surface.contour3D(Th0, Th1, rss, 50, cmap=cm.coolwarm)

    ax_surface.set_xlabel(r"$\theta_0$", fontsize=18)
    ax_surface.set_ylabel(r"$\theta_1$", fontsize=18)
    ax_surface.set_zlabel("RSS Loss", fontsize=18)
    ax_surface.view_init(elev=25., azim=20)

    return rss, ax_surface


def main():
    X, y = data_utils.synthetic_data(theta_true, bias_true, N)

    print("Input data and labels shape:")
    print(X.shape)
    print(y.shape)
    plt.scatter(X[:, 0], y, s=1)
    plt.scatter(X[:, 1], y, s=1)

    print("Feature engineering if n > 1:")
    X_aug = data_utils.data_augmentation(X, order)

    rss, ax_surface = create_rss_surface(X, y)

    print("Analytical least squares solution:")
    theta_opt = models.analytical_ls_solution(X_aug, y)
    rss_opt = sum((X_aug.dot(theta_opt) - y) ** 2)
    mse_opt = (1 / N) * rss_opt
    print("Analytical Theta: ", theta_opt)
    print("MSE: ", mse_opt)

    ax_surface.plot(xs=theta_opt[1], ys=theta_opt[2], zs=rss_opt, marker='x', color='r', ms=10, mew=5, label="Opt")
    ax_surface.legend()

    # Initialize weights by sampling a normal distribution with mean 0 and standard deviation of 1
    theta0 = np.random.randn(3)
    theta_gd, trace = models.gradient_descent(loss_functions.lin_reg_loss, theta0, X_aug, y, **opts)

    print("Mini-batch GD Theta: ", theta_gd)
    print("MSE: ", trace['loss'][-1])

    # Convert our trace of parameter and loss function values into NumPy "history" arrays:
    theta_hist = np.asarray(trace['theta'])
    mse_hist = np.array(trace['loss'])

    fig_rss_traj, ax_rss = plt.subplots(figsize=(10, 10))

    # Parameter trajectory as a contour plot
    ax_rss.set_title("RSS Trajectory", fontsize=32);
    cs = plt.contour(Th0, Th1, rss)
    ax_rss.plot(theta_opt[1], theta_opt[2], 'x', color='r', ms=10, mew=5, label="Opt");

    ax_rss.plot(theta_hist[:, 1], theta_hist[:, 2], 'bo-', lw=2, label="b_size={}".format(opts['batch_size']));
    ax_rss.legend()

    # 3D plot along loss function surface
    ax_surface.plot(xs=theta_hist[:, 1], ys=theta_hist[:, 2], zs=N * mse_hist, c='b',
                    label="b_size={}".format(opts['batch_size']))
    ax_surface.legend()

    plt.show()


if __name__ == '__main__':
    main()
