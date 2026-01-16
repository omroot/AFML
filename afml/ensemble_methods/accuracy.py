"""
Bagging classifier accuracy calculations.

This module provides functions to compute the theoretical accuracy of
bagging classifiers based on individual estimator accuracy, number of
estimators, and number of classes.

Reference: AFML Chapter 6, Section 6.3.2
"""

from typing import Optional, Tuple
import numpy as np
from scipy.special import comb


def compute_bagging_accuracy(
    num_estimators: int,
    individual_accuracy: float,
    num_classes: int = 2,
) -> float:
    """
    Compute the theoretical accuracy of a bagging classifier.

    Calculates the probability that the bagging classifier makes a correct
    prediction through majority voting, given the accuracy of individual
    estimators.

    Parameters
    ----------
    num_estimators : int
        Number of estimators (N) in the bagging ensemble.
    individual_accuracy : float
        Accuracy (p) of each individual classifier. Must be in (0, 1).
    num_classes : int, default=2
        Number of classes (k) in the classification problem.

    Returns
    -------
    float
        Probability that the bagging classifier makes a correct prediction.
        This is a lower bound based on the necessary condition X > N/k.

    Notes
    -----
    For majority voting to succeed, we need more than N/k votes for the
    correct class (necessary but not sufficient condition).

    The probability is computed as:

        P[X > N/k] = 1 - P[X <= N/k] = 1 - sum_{i=0}^{floor(N/k)} C(N,i) * p^i * (1-p)^{N-i}

    For sufficiently large N, if p > 1/k (better than random), then:
        P[X > N/k] > p

    This means the bagging classifier's accuracy exceeds the average
    accuracy of individual classifiers.

    References
    ----------
    AFML Chapter 6, Snippet 6.1: Accuracy of the Bagging Classifier

    Examples
    --------
    >>> # Binary classifier with 100 estimators, each 55% accurate
    >>> accuracy = compute_bagging_accuracy(
    ...     num_estimators=100,
    ...     individual_accuracy=0.55,
    ...     num_classes=2
    ... )
    >>> print(f"Bagging accuracy: {accuracy:.4f}")
    Bagging accuracy: 0.8414

    >>> # 3-class classifier
    >>> accuracy = compute_bagging_accuracy(
    ...     num_estimators=100,
    ...     individual_accuracy=1/3,  # Random guessing
    ...     num_classes=3
    ... )
    >>> print(f"Accuracy at random: {accuracy:.4f}")
    """
    if not 0 < individual_accuracy < 1:
        raise ValueError(
            f"individual_accuracy must be in (0, 1), got {individual_accuracy}"
        )
    if num_estimators < 1:
        raise ValueError(f"num_estimators must be >= 1, got {num_estimators}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    # Compute P[X > N/k] = 1 - P[X <= N/k]
    # where X ~ Binomial(N, p)
    threshold = int(num_estimators / num_classes)

    cumulative_prob = 0.0
    for i in range(threshold + 1):
        # P[X = i] = C(N, i) * p^i * (1-p)^(N-i)
        prob_i = (
            comb(num_estimators, i, exact=True)
            * (individual_accuracy ** i)
            * ((1 - individual_accuracy) ** (num_estimators - i))
        )
        cumulative_prob += prob_i

    bagging_accuracy = 1 - cumulative_prob

    return bagging_accuracy


def compute_accuracy_improvement(
    num_estimators: int,
    individual_accuracy: float,
    num_classes: int = 2,
) -> Tuple[float, float, float]:
    """
    Compute the accuracy improvement from bagging.

    Parameters
    ----------
    num_estimators : int
        Number of estimators in the ensemble.
    individual_accuracy : float
        Accuracy of each individual classifier.
    num_classes : int, default=2
        Number of classes.

    Returns
    -------
    tuple of (float, float, float)
        - individual_accuracy: The input individual accuracy
        - bagging_accuracy: The computed bagging ensemble accuracy
        - improvement: The absolute improvement (bagging - individual)

    Examples
    --------
    >>> ind, bag, imp = compute_accuracy_improvement(100, 0.55, 2)
    >>> print(f"Individual: {ind:.2%}, Bagging: {bag:.2%}, Improvement: {imp:.2%}")
    """
    bagging_accuracy = compute_bagging_accuracy(
        num_estimators, individual_accuracy, num_classes
    )
    improvement = bagging_accuracy - individual_accuracy

    return individual_accuracy, bagging_accuracy, improvement


def analyze_accuracy_vs_estimators(
    individual_accuracy: float,
    num_classes: int = 2,
    max_estimators: int = 100,
    step: int = 5,
) -> np.ndarray:
    """
    Analyze how bagging accuracy changes with number of estimators.

    Parameters
    ----------
    individual_accuracy : float
        Accuracy of each individual classifier.
    num_classes : int, default=2
        Number of classes.
    max_estimators : int, default=100
        Maximum number of estimators to analyze.
    step : int, default=5
        Step size for number of estimators.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) where each row is
        [num_estimators, bagging_accuracy].

    Examples
    --------
    >>> results = analyze_accuracy_vs_estimators(0.55, num_classes=2)
    >>> # Plot results
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(results[:, 0], results[:, 1])
    """
    results = []

    for n in range(step, max_estimators + 1, step):
        accuracy = compute_bagging_accuracy(n, individual_accuracy, num_classes)
        results.append([n, accuracy])

    return np.array(results)


def compute_accuracy_heatmap(
    accuracy_range: Tuple[float, float] = (0.2, 0.8),
    estimator_range: Tuple[int, int] = (1, 101),
    num_classes: int = 2,
    accuracy_steps: int = 20,
    estimator_steps: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a heatmap of bagging accuracy over accuracy and estimator ranges.

    Parameters
    ----------
    accuracy_range : tuple of (float, float), default=(0.2, 0.8)
        Range of individual accuracies to analyze.
    estimator_range : tuple of (int, int), default=(1, 101)
        Range of number of estimators to analyze.
    num_classes : int, default=2
        Number of classes.
    accuracy_steps : int, default=20
        Number of steps in accuracy range.
    estimator_steps : int, default=20
        Number of steps in estimator range.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        - accuracies: 1D array of individual accuracy values
        - estimators: 1D array of estimator counts
        - heatmap: 2D array of bagging accuracies

    Examples
    --------
    >>> acc, est, heatmap = compute_accuracy_heatmap()
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(heatmap, aspect='auto')
    """
    accuracies = np.linspace(accuracy_range[0], accuracy_range[1], accuracy_steps)
    estimators = np.linspace(
        estimator_range[0], estimator_range[1], estimator_steps, dtype=int
    )

    heatmap = np.zeros((len(estimators), len(accuracies)))

    for i, n in enumerate(estimators):
        for j, p in enumerate(accuracies):
            heatmap[i, j] = compute_bagging_accuracy(int(n), p, num_classes)

    return accuracies, estimators, heatmap


def minimum_estimators_for_accuracy(
    individual_accuracy: float,
    target_accuracy: float,
    num_classes: int = 2,
    max_search: int = 1000,
) -> Optional[int]:
    """
    Find the minimum number of estimators to achieve a target accuracy.

    Parameters
    ----------
    individual_accuracy : float
        Accuracy of each individual classifier.
    target_accuracy : float
        Desired bagging ensemble accuracy.
    num_classes : int, default=2
        Number of classes.
    max_search : int, default=1000
        Maximum number of estimators to search.

    Returns
    -------
    int or None
        Minimum number of estimators needed, or None if not achievable
        within max_search.

    Examples
    --------
    >>> n = minimum_estimators_for_accuracy(0.55, 0.90, num_classes=2)
    >>> print(f"Need {n} estimators for 90% accuracy")
    """
    if individual_accuracy <= 1 / num_classes:
        # Cannot achieve better than random with poor individual classifiers
        return None

    for n in range(1, max_search + 1):
        accuracy = compute_bagging_accuracy(n, individual_accuracy, num_classes)
        if accuracy >= target_accuracy:
            return n

    return None
