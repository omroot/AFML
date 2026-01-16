"""
Orthogonal features via PCA.

This module provides functions to orthogonalize features using PCA,
which helps address substitution effects in feature importance analysis.
It also provides tools to compare PCA ranking with feature importance ranking.

Reference: AFML Chapter 8, Section 8.4.2, Snippets 8.5-8.6
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import weightedtau, kendalltau, spearmanr, pearsonr


def compute_eigenvectors(
    correlation_matrix: np.ndarray,
    variance_threshold: float = 0.95,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute eigenvalues and eigenvectors from correlation matrix.

    Performs eigendecomposition and reduces dimensionality by keeping
    only the principal components that explain at least `variance_threshold`
    of the total variance.

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix (Z'Z where Z is standardized features).
    variance_threshold : float, default=0.95
        Minimum cumulative variance to retain. Components are kept
        until this threshold is reached.

    Returns
    -------
    eigenvalues : pd.Series
        Eigenvalues sorted in descending order, indexed by PC name.
    eigenvectors : pd.DataFrame
        Eigenvector matrix where columns are principal components.

    Notes
    -----
    The eigendecomposition satisfies: Z'Z @ W = W @ Λ
    where Λ is diagonal matrix of eigenvalues and W is orthonormal.

    References
    ----------
    AFML Chapter 8, Snippet 8.5: Computation of Orthogonal Features

    Examples
    --------
    >>> Z = (X - X.mean()) / X.std()  # Standardize
    >>> corr = np.dot(Z.T, Z)
    >>> eigenvalues, eigenvectors = compute_eigenvectors(corr, variance_threshold=0.95)
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

    # Sort in descending order (eigh returns ascending)
    sort_idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Keep only positive eigenvalues
    positive_mask = eigenvalues > 0
    eigenvalues = eigenvalues[positive_mask]
    eigenvectors = eigenvectors[:, positive_mask]

    # Create named series/dataframe
    n_components = len(eigenvalues)
    pc_names = [f'PC_{i+1}' for i in range(n_components)]

    eigenvalues = pd.Series(eigenvalues, index=pc_names)
    eigenvectors = pd.DataFrame(
        eigenvectors,
        index=correlation_matrix.columns if hasattr(correlation_matrix, 'columns')
              else range(correlation_matrix.shape[0]),
        columns=pc_names
    )

    # Reduce dimension based on variance threshold
    cumulative_variance = eigenvalues.cumsum() / eigenvalues.sum()
    n_keep = (cumulative_variance <= variance_threshold).sum() + 1
    n_keep = min(n_keep, len(eigenvalues))

    eigenvalues = eigenvalues.iloc[:n_keep]
    eigenvectors = eigenvectors.iloc[:, :n_keep]

    return eigenvalues, eigenvectors


def compute_orthogonal_features(
    X: pd.DataFrame,
    variance_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Transform features to orthogonal (PCA) space.

    Given a features DataFrame, standardizes the data and transforms
    it to orthogonal principal components. This eliminates linear
    correlations between features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with observations as rows and features as columns.
    variance_threshold : float, default=0.95
        Minimum cumulative variance to retain in reduced dimensionality.

    Returns
    -------
    orthogonal_features : pd.DataFrame
        Transformed features (principal components) as P = Z @ W.
    eigenvalues : pd.Series
        Eigenvalues for each principal component.
    eigenvectors : pd.DataFrame
        Eigenvector matrix for transforming back to original space.

    Notes
    -----
    The transformation process:
    1. Standardize: Z = (X - μ) / σ
    2. Compute correlation: C = Z'Z
    3. Eigendecompose: C @ W = W @ Λ
    4. Transform: P = Z @ W

    The orthogonality can be verified: P'P = Λ (diagonal)

    Why standardize?
    - Centering ensures PC1 is oriented in main data direction
    - Scaling makes PCA focus on correlations, not variances

    References
    ----------
    AFML Chapter 8, Snippet 8.5: Computation of Orthogonal Features

    Examples
    --------
    >>> ortho_X, eigenvalues, eigenvectors = compute_orthogonal_features(X)
    >>> # Verify orthogonality
    >>> print(np.allclose(ortho_X.T @ ortho_X, np.diag(eigenvalues)))
    """
    # Standardize features (center and scale)
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

    # Compute correlation matrix (dot product of standardized features)
    correlation_matrix = pd.DataFrame(
        np.dot(X_standardized.T, X_standardized),
        index=X.columns,
        columns=X.columns
    )

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = compute_eigenvectors(
        correlation_matrix,
        variance_threshold=variance_threshold
    )

    # Transform to orthogonal space: P = Z @ W
    orthogonal_features = pd.DataFrame(
        np.dot(X_standardized, eigenvectors),
        index=X.index,
        columns=eigenvectors.columns
    )

    return orthogonal_features, eigenvalues, eigenvectors


def compute_weighted_kendall_tau(
    feature_importance: np.ndarray,
    pca_rank: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute weighted Kendall's tau between feature importance and PCA ranking.

    This measures the consistency between the supervised importance ranking
    (from MDI/MDA/SFI) and the unsupervised PCA ranking. High correlation
    suggests the pattern found by ML is not entirely overfit.

    Parameters
    ----------
    feature_importance : np.ndarray
        Feature importance values (higher = more important).
    pca_rank : np.ndarray
        PCA rank of each feature (1 = first principal component).

    Returns
    -------
    tau : float
        Weighted Kendall's tau correlation coefficient.
    p_value : float
        Two-sided p-value for the correlation.

    Notes
    -----
    We use weighted Kendall's tau because we care more about rank
    concordance among the most important features. We don't care
    much about rank concordance among irrelevant (noisy) features.

    The comparison is done with inverse PCA rank (1/rank) because
    weightedtau gives higher weight to higher values.

    References
    ----------
    AFML Chapter 8, Snippet 8.6: Weighted Kendall's Tau

    Examples
    --------
    >>> feature_imp = np.array([0.55, 0.33, 0.07, 0.05])
    >>> pca_rank = np.array([1, 2, 4, 3])
    >>> tau, p = compute_weighted_kendall_tau(feature_imp, pca_rank)
    >>> print(f"Weighted Kendall's tau: {tau:.4f}")
    """
    # Use inverse PCA rank so higher values = more principal
    # Convert to float to allow negative powers
    inverse_pca_rank = np.asarray(pca_rank, dtype=float) ** -1

    # Compute weighted Kendall's tau
    tau, p_value = weightedtau(feature_importance, inverse_pca_rank)

    return tau, p_value


def compute_importance_pca_correlation(
    feature_importance: pd.Series,
    eigenvalues: pd.Series,
) -> pd.DataFrame:
    """
    Compute multiple correlation metrics between importance and PCA eigenvalues.

    Parameters
    ----------
    feature_importance : pd.Series
        Feature importance indexed by feature name.
    eigenvalues : pd.Series
        PCA eigenvalues indexed by PC name.

    Returns
    -------
    pd.DataFrame
        DataFrame with correlation metrics and p-values.

    Notes
    -----
    Computes four correlation metrics:
    - Kendall's tau (rank correlation)
    - Spearman's rho (rank correlation)
    - Pearson's r (linear correlation)
    - Weighted Kendall's tau (prioritizes important features)

    Examples
    --------
    >>> correlations = compute_importance_pca_correlation(importance['mean'], eigenvalues)
    >>> print(correlations)
    """
    # Both must be same length and aligned
    if len(feature_importance) != len(eigenvalues):
        raise ValueError(
            f"feature_importance ({len(feature_importance)}) and eigenvalues "
            f"({len(eigenvalues)}) must have same length"
        )

    imp_values = feature_importance.values
    eig_values = eigenvalues.values

    # Compute correlations
    kendall_tau, kendall_p = kendalltau(imp_values, eig_values)
    spearman_rho, spearman_p = spearmanr(imp_values, eig_values)
    pearson_r, pearson_p = pearsonr(imp_values, eig_values)

    # For weighted Kendall, use rank
    pca_rank = np.arange(1, len(eig_values) + 1)
    weighted_tau, weighted_p = compute_weighted_kendall_tau(imp_values, pca_rank)

    results = pd.DataFrame({
        'correlation': [kendall_tau, spearman_rho, pearson_r, weighted_tau],
        'p_value': [kendall_p, spearman_p, pearson_p, weighted_p],
    }, index=['kendall_tau', 'spearman_rho', 'pearson_r', 'weighted_kendall_tau'])

    return results


def get_pca_feature_ranking(
    X: pd.DataFrame,
    variance_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Get PCA-based feature ranking.

    Ranks features by their contribution to the principal components,
    weighted by the eigenvalues (variance explained).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    variance_threshold : float, default=0.95
        Variance threshold for PCA.

    Returns
    -------
    pd.DataFrame
        Features ranked by PCA importance with scores.

    Examples
    --------
    >>> pca_ranking = get_pca_feature_ranking(X)
    >>> print(pca_ranking.head())
    """
    _, eigenvalues, eigenvectors = compute_orthogonal_features(X, variance_threshold)

    # Compute weighted importance: sum of |loading| * eigenvalue
    weighted_loading = np.abs(eigenvectors) * eigenvalues.values

    pca_importance = weighted_loading.sum(axis=1)
    pca_importance = pca_importance / pca_importance.sum()  # Normalize

    result = pd.DataFrame({
        'pca_importance': pca_importance,
        'rank': range(1, len(pca_importance) + 1)
    })

    result = result.sort_values('pca_importance', ascending=False)
    result['rank'] = range(1, len(result) + 1)

    return result
