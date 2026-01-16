"""
Synthetic data generation for testing feature importance methods.

This module provides functions to generate synthetic datasets with known
feature properties (informative, redundant, noise) for testing and
validating feature importance methods.

Reference: AFML Chapter 8, Section 8.6, Snippet 8.7
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_synthetic_dataset(
    n_samples: int = 10000,
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic classification dataset with known feature types.

    Creates a dataset with three types of features:
    1. **Informative**: Features used to determine the label
    2. **Redundant**: Random linear combinations of informative features
    3. **Noise**: Features with no bearing on the label

    Parameters
    ----------
    n_samples : int, default=10000
        Number of observations to generate.
    n_features : int, default=40
        Total number of features.
    n_informative : int, default=10
        Number of informative features.
    n_redundant : int, default=10
        Number of redundant features (linear combos of informative).
    random_state : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns named by type:
        - 'I_0', 'I_1', ... for informative
        - 'R_0', 'R_1', ... for redundant
        - 'N_0', 'N_1', ... for noise
    metadata : pd.DataFrame
        DataFrame with observation metadata including:
        - 'label': Target label (0 or 1)
        - 'weight': Sample weight (default 1.0)
        - 't1': Label end time (for purged CV)

    Notes
    -----
    The number of noise features is computed as:
        n_noise = n_features - n_informative - n_redundant

    The dataset is generated using sklearn's make_classification with
    shuffle=False to maintain feature ordering.

    References
    ----------
    AFML Chapter 8, Snippet 8.7: Creating a Synthetic Dataset

    Examples
    --------
    >>> X, meta = generate_synthetic_dataset(
    ...     n_samples=5000,
    ...     n_features=30,
    ...     n_informative=10,
    ...     n_redundant=5
    ... )
    >>> print(f"X shape: {X.shape}")
    >>> print(f"Feature types: {X.columns.tolist()[:5]}...")
    >>> print(f"Noise features: {30 - 10 - 5}")
    """
    # Calculate number of noise features
    n_noise = n_features - n_informative - n_redundant

    if n_noise < 0:
        raise ValueError(
            f"n_informative ({n_informative}) + n_redundant ({n_redundant}) "
            f"exceeds n_features ({n_features})"
        )

    # Generate synthetic data using sklearn
    X_array, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
        shuffle=False,  # Keep feature ordering
    )

    # Create datetime index (business days ending today)
    date_index = pd.DatetimeIndex(
        pd.bdate_range(
            periods=n_samples,
            freq=pd.tseries.offsets.BDay(),
            end=pd.Timestamp.today()
        )
    )

    # Create feature names by type
    feature_names = (
        [f'I_{i}' for i in range(n_informative)] +
        [f'R_{i}' for i in range(n_redundant)] +
        [f'N_{i}' for i in range(n_noise)]
    )

    # Create X DataFrame
    X = pd.DataFrame(X_array, index=date_index, columns=feature_names)

    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'label': y,
        'weight': 1.0 / n_samples,  # Equal weights summing to 1
        't1': date_index,  # Label end time (same as index for simplicity)
    }, index=date_index)

    return X, metadata


def get_feature_types(feature_names: list) -> pd.Series:
    """
    Extract feature types from standardized feature names.

    Parameters
    ----------
    feature_names : list
        List of feature names in format 'TYPE_NUMBER' (e.g., 'I_0', 'R_5').

    Returns
    -------
    pd.Series
        Series mapping feature name to type ('I', 'R', or 'N').

    Examples
    --------
    >>> types = get_feature_types(['I_0', 'I_1', 'R_0', 'N_0', 'N_1'])
    >>> print(types)
    I_0    I
    I_1    I
    R_0    R
    N_0    N
    N_1    N
    dtype: object
    """
    return pd.Series({name: name[0] for name in feature_names})


def analyze_importance_by_type(
    importance: pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:
    """
    Analyze feature importance grouped by feature type.

    Parameters
    ----------
    importance : pd.DataFrame
        Feature importance DataFrame with 'mean' column.
    feature_names : list
        Feature names in standardized format.

    Returns
    -------
    pd.DataFrame
        Summary statistics by feature type (I=informative, R=redundant, N=noise).

    Examples
    --------
    >>> X, meta = generate_synthetic_dataset()
    >>> importance = compute_mdi_importance(clf, X.columns.tolist())
    >>> type_analysis = analyze_importance_by_type(importance, X.columns.tolist())
    >>> print(type_analysis)
    """
    feature_types = get_feature_types(feature_names)

    # Add type to importance DataFrame
    importance_with_type = importance.copy()
    importance_with_type['type'] = importance_with_type.index.map(feature_types)

    # Group by type and compute statistics
    type_summary = importance_with_type.groupby('type')['mean'].agg([
        'count', 'mean', 'std', 'min', 'max', 'sum'
    ])

    # Add descriptive names
    type_names = {
        'I': 'Informative',
        'R': 'Redundant',
        'N': 'Noise'
    }
    type_summary.index = type_summary.index.map(type_names)

    return type_summary


def compute_importance_accuracy(
    importance: pd.DataFrame,
    feature_names: list,
    threshold_method: str = 'median',
) -> dict:
    """
    Evaluate how well importance method identifies informative features.

    Computes metrics like precision, recall, and accuracy for detecting
    informative (and redundant) features vs noise features.

    Parameters
    ----------
    importance : pd.DataFrame
        Feature importance with 'mean' column.
    feature_names : list
        Feature names in standardized format.
    threshold_method : str, default='median'
        Method to determine importance threshold:
        - 'median': Use median importance
        - 'uniform': Use 1/n_features (expected if all features equal)

    Returns
    -------
    dict
        Dictionary with evaluation metrics:
        - 'precision': Important features that are truly informative/redundant
        - 'recall': Informative/redundant features that are marked important
        - 'accuracy': Overall classification accuracy
        - 'n_informative_detected': Count of I/R features above threshold
        - 'n_noise_detected': Count of N features above threshold

    Examples
    --------
    >>> accuracy = compute_importance_accuracy(importance, X.columns.tolist())
    >>> print(f"Precision: {accuracy['precision']:.2%}")
    >>> print(f"Recall: {accuracy['recall']:.2%}")
    """
    feature_types = get_feature_types(feature_names)

    # Determine threshold
    if threshold_method == 'median':
        threshold = importance['mean'].median()
    else:  # uniform
        threshold = 1.0 / len(feature_names)

    # Classify features as "important" based on threshold
    predicted_important = importance['mean'] > threshold

    # True labels: informative (I) and redundant (R) are "truly important"
    actual_important = feature_types.isin(['I', 'R'])

    # Compute metrics
    true_positive = (predicted_important & actual_important).sum()
    false_positive = (predicted_important & ~actual_important).sum()
    false_negative = (~predicted_important & actual_important).sum()
    true_negative = (~predicted_important & ~actual_important).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    accuracy = (true_positive + true_negative) / len(feature_names)

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
        'threshold': threshold,
        'n_informative_detected': true_positive,
        'n_redundant_detected': true_positive,  # Same as informative for I+R
        'n_noise_detected': false_positive,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_negative': true_negative,
    }


def generate_financial_synthetic_data(
    n_samples: int = 1000,
    n_informative: int = 5,
    n_redundant: int = 3,
    n_noise: int = 10,
    label_span_days: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic financial data with overlapping labels.

    This is more realistic for financial applications as it includes
    label times that span multiple days, requiring purged CV.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of observations.
    n_informative : int, default=5
        Number of informative features.
    n_redundant : int, default=3
        Number of redundant features.
    n_noise : int, default=10
        Number of noise features.
    label_span_days : int, default=10
        Number of days each label spans into the future.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    metadata : pd.DataFrame
        Metadata with label, weight, and label times (t1).

    Examples
    --------
    >>> X, meta = generate_financial_synthetic_data(
    ...     n_samples=500,
    ...     label_span_days=5
    ... )
    >>> # Use with purged CV
    >>> cv = PurgedKFold(n_splits=5, label_times=meta['t1'])
    """
    n_features = n_informative + n_redundant + n_noise

    X, metadata = generate_synthetic_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
    )

    # Update t1 to span label_span_days into the future
    metadata['t1'] = metadata.index + pd.Timedelta(days=label_span_days)

    return X, metadata
