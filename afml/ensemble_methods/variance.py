"""
Variance reduction analysis for bagging ensembles.

This module provides functions to analyze and visualize how bagging
reduces forecast variance, and the impact of estimator correlation.

Reference: AFML Chapter 6, Section 6.3.1
"""

from typing import Tuple
import numpy as np


def compute_bagging_variance(
    num_estimators: int,
    avg_variance: float,
    avg_correlation: float,
) -> float:
    """
    Compute the variance of a bagged prediction.

    The variance of the ensemble forecast depends on the number of
    estimators, their individual variances, and their correlations.

    Parameters
    ----------
    num_estimators : int
        Number of estimators (N) in the ensemble.
    avg_variance : float
        Average variance (σ²) of individual estimator predictions.
    avg_correlation : float
        Average correlation (ρ̄) between estimator predictions.
        Must be in [0, 1].

    Returns
    -------
    float
        Variance of the bagged prediction.

    Notes
    -----
    The variance formula is:

        V[1/N * Σφᵢ] = σ̄² * (ρ̄ + (1-ρ̄)/N)

    Key insights:
    - When ρ̄ = 0 (uncorrelated): V → σ̄²/N (maximum reduction)
    - When ρ̄ = 1 (perfectly correlated): V = σ̄² (no reduction)
    - As N → ∞: V → σ̄² * ρ̄ (lower bound)

    This is why sequential bootstrapping (Chapter 4) is important:
    it produces more independent samples, reducing ρ̄.

    References
    ----------
    AFML Chapter 6, Section 6.3.1: Variance Reduction

    Examples
    --------
    >>> # 10 estimators with unit variance and 0.5 correlation
    >>> var = compute_bagging_variance(
    ...     num_estimators=10,
    ...     avg_variance=1.0,
    ...     avg_correlation=0.5
    ... )
    >>> print(f"Bagged variance: {var:.4f}")
    Bagged variance: 0.5500
    """
    if not 0 <= avg_correlation <= 1:
        raise ValueError(
            f"avg_correlation must be in [0, 1], got {avg_correlation}"
        )
    if num_estimators < 1:
        raise ValueError(f"num_estimators must be >= 1, got {num_estimators}")
    if avg_variance < 0:
        raise ValueError(f"avg_variance must be >= 0, got {avg_variance}")

    # V = σ̄² * (ρ̄ + (1-ρ̄)/N)
    variance = avg_variance * (avg_correlation + (1 - avg_correlation) / num_estimators)

    return variance


def compute_variance_reduction_ratio(
    num_estimators: int,
    avg_correlation: float,
) -> float:
    """
    Compute the variance reduction ratio from bagging.

    Parameters
    ----------
    num_estimators : int
        Number of estimators in the ensemble.
    avg_correlation : float
        Average correlation between estimator predictions.

    Returns
    -------
    float
        Ratio of bagged variance to individual variance.
        Values < 1 indicate variance reduction.

    Examples
    --------
    >>> ratio = compute_variance_reduction_ratio(10, 0.3)
    >>> print(f"Variance is {ratio:.2%} of individual")
    """
    return avg_correlation + (1 - avg_correlation) / num_estimators


def compute_standard_deviation(
    num_estimators: int,
    avg_std: float,
    avg_correlation: float,
) -> float:
    """
    Compute the standard deviation of a bagged prediction.

    Parameters
    ----------
    num_estimators : int
        Number of estimators in the ensemble.
    avg_std : float
        Average standard deviation (σ̄) of individual estimators.
    avg_correlation : float
        Average correlation (ρ̄) between estimator predictions.

    Returns
    -------
    float
        Standard deviation of the bagged prediction.

    Examples
    --------
    >>> std = compute_standard_deviation(10, 1.0, 0.5)
    >>> print(f"Bagged std: {std:.4f}")
    """
    variance = compute_bagging_variance(
        num_estimators, avg_std ** 2, avg_correlation
    )
    return np.sqrt(variance)


def analyze_variance_vs_correlation(
    num_estimators: int,
    avg_variance: float = 1.0,
    correlation_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze how bagging variance changes with estimator correlation.

    Parameters
    ----------
    num_estimators : int
        Number of estimators in the ensemble.
    avg_variance : float, default=1.0
        Average variance of individual estimators.
    correlation_steps : int, default=50
        Number of correlation values to analyze.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - correlations: Array of correlation values [0, 1]
        - variances: Array of corresponding bagged variances

    Examples
    --------
    >>> corr, var = analyze_variance_vs_correlation(num_estimators=20)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(corr, var)
    >>> plt.xlabel('Average Correlation')
    >>> plt.ylabel('Bagged Variance')
    """
    correlations = np.linspace(0, 1, correlation_steps)
    variances = np.array([
        compute_bagging_variance(num_estimators, avg_variance, rho)
        for rho in correlations
    ])

    return correlations, variances


def analyze_variance_vs_estimators(
    avg_correlation: float,
    avg_variance: float = 1.0,
    max_estimators: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze how bagging variance changes with number of estimators.

    Parameters
    ----------
    avg_correlation : float
        Average correlation between estimator predictions.
    avg_variance : float, default=1.0
        Average variance of individual estimators.
    max_estimators : int, default=100
        Maximum number of estimators to analyze.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - estimators: Array of estimator counts
        - variances: Array of corresponding bagged variances

    Examples
    --------
    >>> est, var = analyze_variance_vs_estimators(avg_correlation=0.3)
    >>> print(f"Variance converges to: {var[-1]:.4f}")
    """
    estimators = np.arange(1, max_estimators + 1)
    variances = np.array([
        compute_bagging_variance(n, avg_variance, avg_correlation)
        for n in estimators
    ])

    return estimators, variances


def compute_variance_heatmap(
    correlation_range: Tuple[float, float] = (0.0, 1.0),
    estimator_range: Tuple[int, int] = (5, 30),
    avg_variance: float = 1.0,
    correlation_steps: int = 50,
    estimator_steps: int = 26,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a heatmap of bagged variance over correlation and estimator ranges.

    This recreates Figure 6.1 from AFML.

    Parameters
    ----------
    correlation_range : tuple of (float, float), default=(0.0, 1.0)
        Range of average correlations to analyze.
    estimator_range : tuple of (int, int), default=(5, 30)
        Range of number of estimators to analyze.
    avg_variance : float, default=1.0
        Average variance of individual estimators.
    correlation_steps : int, default=50
        Number of steps in correlation range.
    estimator_steps : int, default=26
        Number of steps in estimator range.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        - correlations: 1D array of correlation values
        - estimators: 1D array of estimator counts
        - heatmap: 2D array of standard deviations (sqrt of variance)

    Examples
    --------
    >>> corr, est, heatmap = compute_variance_heatmap()
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(heatmap, aspect='auto', origin='lower')
    >>> plt.colorbar(label='Standard Deviation')
    """
    correlations = np.linspace(
        correlation_range[0], correlation_range[1], correlation_steps
    )
    estimators = np.linspace(
        estimator_range[0], estimator_range[1], estimator_steps, dtype=int
    )

    # Compute standard deviation (not variance) for visualization
    heatmap = np.zeros((len(estimators), len(correlations)))

    for i, n in enumerate(estimators):
        for j, rho in enumerate(correlations):
            variance = compute_bagging_variance(int(n), avg_variance, rho)
            heatmap[i, j] = np.sqrt(variance)

    return correlations, estimators, heatmap


def estimate_correlation_from_predictions(
    predictions: np.ndarray,
) -> float:
    """
    Estimate average correlation from ensemble predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Array of shape (n_estimators, n_samples) containing predictions
        from each estimator.

    Returns
    -------
    float
        Average pairwise correlation between estimator predictions.

    Examples
    --------
    >>> # Predictions from 5 estimators on 100 samples
    >>> preds = np.random.randn(5, 100)
    >>> avg_corr = estimate_correlation_from_predictions(preds)
    """
    n_estimators = predictions.shape[0]

    if n_estimators < 2:
        return 1.0

    # Compute correlation matrix
    corr_matrix = np.corrcoef(predictions)

    # Average off-diagonal correlations
    mask = ~np.eye(n_estimators, dtype=bool)
    avg_correlation = np.mean(corr_matrix[mask])

    return avg_correlation


def optimal_num_estimators(
    avg_correlation: float,
    target_variance_ratio: float,
    avg_variance: float = 1.0,
    max_search: int = 10000,
) -> int:
    """
    Find the number of estimators needed to achieve target variance reduction.

    Parameters
    ----------
    avg_correlation : float
        Average correlation between estimator predictions.
    target_variance_ratio : float
        Target ratio of bagged variance to individual variance.
    avg_variance : float, default=1.0
        Average variance of individual estimators.
    max_search : int, default=10000
        Maximum number of estimators to search.

    Returns
    -------
    int
        Number of estimators needed, or max_search if not achievable.

    Notes
    -----
    The minimum achievable variance ratio is ρ̄ (when N → ∞).
    If target_variance_ratio < ρ̄, the target is not achievable.

    Examples
    --------
    >>> n = optimal_num_estimators(avg_correlation=0.3, target_variance_ratio=0.4)
    >>> print(f"Need {n} estimators")
    """
    # Check if target is achievable
    min_ratio = avg_correlation
    if target_variance_ratio < min_ratio:
        return max_search  # Not achievable

    # Solve: ρ̄ + (1-ρ̄)/N = target
    # N = (1-ρ̄) / (target - ρ̄)
    if target_variance_ratio > avg_correlation:
        n = (1 - avg_correlation) / (target_variance_ratio - avg_correlation)
        return max(1, int(np.ceil(n)))

    return max_search
