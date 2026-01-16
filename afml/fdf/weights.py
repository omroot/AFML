"""
Weight computation for fractional differentiation.

This module provides functions to compute the weight sequence used in
fractional differentiation. The weights are derived from the binomial
series expansion of the backshift operator (1-B)^d.

Reference: AFML Chapter 5, Sections 5.2-5.4
"""

from typing import Optional
import numpy as np


def compute_weights(
    diff_order: float,
    window_size: int,
) -> np.ndarray:
    """
    Compute weights for fractional differentiation using expanding window.

    Computes the weight sequence w_k for k = 0, 1, ..., size-1 using
    the iterative formula: w_k = -w_{k-1} * (d - k + 1) / k

    Parameters
    ----------
    diff_order : float
        The fractional differentiation order (d). Can be any real number.
        - d = 1.0: First difference (standard differencing)
        - d = 0.5: Half differentiation
        - d = 0.0: No differencing (original series)

    window_size : int
        Number of weights to compute. Determines the memory length
        of the fractional differentiation.

    Returns
    -------
    np.ndarray
        Weight array of shape (window_size, 1), ordered from oldest
        to newest observation (w_{size-1}, ..., w_1, w_0).

    Notes
    -----
    The weights are computed from the binomial series expansion:

        (1-B)^d = sum_{k=0}^{inf} C(d,k) * (-B)^k

    where C(d,k) is the generalized binomial coefficient.

    The iterative formula avoids computing factorials directly:

        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    For integer d, only d+1 weights are non-zero. For non-integer d,
    weights decay slowly and never reach exactly zero.

    References
    ----------
    AFML Chapter 5, Snippet 5.1: getWeights function

    Examples
    --------
    >>> weights = compute_weights(diff_order=0.5, window_size=10)
    >>> weights.shape
    (10, 1)
    >>> weights[-1]  # w_0 = 1.0
    array([[1.]])
    """
    weights = [1.0]

    for lag in range(1, window_size):
        # Iterative formula: w_k = -w_{k-1} * (d - k + 1) / k
        next_weight = -weights[-1] / lag * (diff_order - lag + 1)
        weights.append(next_weight)

    # Reverse to get oldest-to-newest order and reshape for dot product
    weights_array = np.array(weights[::-1]).reshape(-1, 1)

    return weights_array


def compute_weights_ffd(
    diff_order: float,
    weight_threshold: float = 1e-5,
) -> np.ndarray:
    """
    Compute weights for fixed-width window fractional differentiation (FFD).

    Unlike compute_weights(), this function computes weights until they
    fall below a threshold, resulting in a fixed window size regardless
    of the series length. This is the practical implementation for
    real-world applications.

    Parameters
    ----------
    diff_order : float
        The fractional differentiation order (d). Must be non-negative.
        - d = 1.0: First difference
        - 0 < d < 1: Partial differentiation (preserves some memory)

    weight_threshold : float, default=1e-5
        Stop computing weights when |w_k| < threshold.
        Smaller values include more weights (longer memory).

    Returns
    -------
    np.ndarray
        Weight array of shape (num_weights, 1), ordered from oldest
        to newest observation. Length is determined by when weights
        become negligible.

    Notes
    -----
    The FFD method addresses a key limitation of the expanding window
    approach: as the series grows, the window size grows without bound,
    making computation increasingly expensive.

    By truncating weights below a threshold, we get:
    1. Constant computational cost per observation
    2. Consistent memory length across the series
    3. Minimal loss of precision (weights below threshold are negligible)

    The weight decay rate depends on d:
    - Larger d: faster decay, shorter window
    - Smaller d: slower decay, longer window

    References
    ----------
    AFML Chapter 5, Snippet 5.3: getWeights_FFD function

    Examples
    --------
    >>> weights = compute_weights_ffd(diff_order=0.5, weight_threshold=1e-5)
    >>> len(weights)  # Window size depends on d and threshold
    12
    >>> weights[-1]  # w_0 = 1.0
    array([[1.]])
    """
    weights = [1.0]
    lag = 1

    while True:
        # Iterative formula: w_k = -w_{k-1} * (d - k + 1) / k
        next_weight = -weights[-1] / lag * (diff_order - lag + 1)

        # Stop when weight becomes negligible
        if abs(next_weight) < weight_threshold:
            break

        weights.append(next_weight)
        lag += 1

    # Reverse to get oldest-to-newest order and reshape for dot product
    weights_array = np.array(weights[::-1]).reshape(-1, 1)

    return weights_array


def get_weight_convergence(
    diff_order: float,
    num_lags: int = 100,
) -> np.ndarray:
    """
    Get the weight sequence to analyze convergence properties.

    Useful for visualizing how quickly weights decay for different
    differentiation orders.

    Parameters
    ----------
    diff_order : float
        The fractional differentiation order (d).
    num_lags : int, default=100
        Number of lags to compute.

    Returns
    -------
    np.ndarray
        1D array of weights (w_0, w_1, ..., w_{num_lags-1}).
        Note: This is in lag order, not reversed like compute_weights.

    Examples
    --------
    >>> weights = get_weight_convergence(diff_order=0.5, num_lags=50)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(weights)  # Visualize weight decay
    """
    weights = [1.0]

    for lag in range(1, num_lags):
        next_weight = -weights[-1] / lag * (diff_order - lag + 1)
        weights.append(next_weight)

    return np.array(weights)
