"""Utilities for manipulating or generating data."""

from sys import float_info  # Threshold smallest positive floating value

import numpy as np


def data_augmentation(X, n):
    X_aug = np.ones(shape=(len(X), 1))

    for order in range(1, n + 1):
        X_aug = np.column_stack((X_aug, X ** order))

    return X_aug


# Generating synthetic data
def synthetic_data(theta, b, N):
    # Generate y = Xw + b + noise
    X = np.random.normal(0, 1, (N, len(theta)))
    y = X.dot(theta) + b + np.random.normal(0, 0.01, N)
    return X, y


# Breaks the matrix X and vector y into batches
def batchify(X, y, batch_size):
    N = X.shape[0]
    X_batch = []
    y_batch = []
    # Iterate over N in batch_size steps, last batch may be < batch_size
    for i in range(0, N, batch_size):
        nxt = min(i + batch_size, N + 1)
        X_batch.append(X[i:nxt, :])
        y_batch.append(y[i:nxt])

    return X_batch, y_batch


# Generate ROC curve samples
def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))

    sorted_score = sorted(discriminant_score)

    # Use tau values that will account for every possible classification split
    taus = ([sorted_score[0] - float_info.epsilon] +
            sorted_score +
            [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= t for t in taus]

    ind10 = [np.argwhere((d == 1) & (label == 0)) for d in decisions]
    p10 = [len(inds) / Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d == 1) & (label == 1)) for d in decisions]
    p11 = [len(inds) / Nlabels[1] for inds in ind11]

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))

    return roc, taus
