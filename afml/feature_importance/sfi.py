"""
Single Feature Importance (SFI).

This module provides SFI feature importance calculation. SFI evaluates
each feature in isolation, measuring its predictive power independently
of other features.

Reference: AFML Chapter 8, Section 8.4.1, Snippet 8.4
"""

from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.base import clone

from afml.cross_validation import PurgedKFold


def compute_sfi_importance(
    classifier: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    label_times: Optional[pd.Series] = None,
    n_splits: int = 10,
    embargo_pct: float = 0.0,
    scoring: str = 'accuracy',
) -> pd.DataFrame:
    """
    Compute Single Feature Importance (SFI).

    SFI is an out-of-sample feature importance method that evaluates
    each feature in isolation. Unlike MDI and MDA, SFI is not affected
    by substitution effects because only one feature is used at a time.

    Parameters
    ----------
    classifier : estimator
        A classifier implementing fit() and predict()/predict_proba().
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray or pd.Series
        Target labels of shape (n_samples,).
    sample_weight : np.ndarray or pd.Series, optional
        Sample weights.
    label_times : pd.Series, optional
        Label times for purged CV.
    n_splits : int, default=10
        Number of cross-validation folds.
    embargo_pct : float, default=0.0
        Embargo percentage for purged CV.
    scoring : str, default='accuracy'
        Scoring method: 'neg_log_loss' or 'accuracy'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'mean': Mean CV score for each feature
        - 'std': Standard deviation of CV score

    Notes
    -----
    Important considerations for SFI:

    1. **No substitution effects**: Each feature is evaluated alone,
       so correlated features don't affect each other's importance.

    2. **Any classifier**: Works with any classifier.

    3. **Misses joint effects**: A feature might only be useful in
       combination with others. SFI will miss such hierarchical importance.

    4. **Can find all unimportant**: Unlike MDI, SFI can conclude that
       all features are unimportant (if none predicts well alone).

    5. **Complementary to MDI/MDA**: Use SFI alongside MDI and MDA
       to get a complete picture of feature importance.

    References
    ----------
    AFML Chapter 8, Snippet 8.4: Implementation of SFI

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier(n_estimators=100)
    >>> importance = compute_sfi_importance(
    ...     clf, X, y,
    ...     label_times=label_times,
    ...     scoring='accuracy'
    ... )
    >>> print(importance.sort_values('mean', ascending=False).head())
    """
    # Validate scoring method
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise ValueError(
            f"scoring must be 'neg_log_loss' or 'accuracy', got '{scoring}'"
        )

    # Convert to arrays and get feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    if isinstance(y, pd.Series):
        y = y.values

    if sample_weight is not None and isinstance(sample_weight, pd.Series):
        sample_weight = sample_weight.values

    if sample_weight is None:
        sample_weight = np.ones(len(y))

    # Create cross-validator
    if label_times is not None:
        cv = PurgedKFold(
            n_splits=n_splits,
            label_times=label_times,
            embargo_pct=embargo_pct,
        )
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=False)

    # Storage for scores
    results = pd.DataFrame(columns=['mean', 'std'])

    # Evaluate each feature independently
    for feat_idx, feat_name in enumerate(feature_names):
        # Extract single feature
        X_single = X[:, feat_idx].reshape(-1, 1)

        # Cross-validation scores for this single feature
        fold_scores = []

        for train_idx, test_idx in cv.split(X_single):
            # Get data
            X_train = X_single[train_idx]
            X_test = X_single[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            train_weights = sample_weight[train_idx]
            test_weights = sample_weight[test_idx]

            # Clone and fit classifier
            clf = clone(classifier)
            clf.fit(X_train, y_train, sample_weight=train_weights)

            # Score
            if scoring == 'neg_log_loss':
                y_prob = clf.predict_proba(X_test)
                score = -log_loss(
                    y_test, y_prob,
                    sample_weight=test_weights,
                    labels=clf.classes_
                )
            else:  # accuracy
                y_pred = clf.predict(X_test)
                score = accuracy_score(
                    y_test, y_pred,
                    sample_weight=test_weights
                )

            fold_scores.append(score)

        # Store results
        results.loc[feat_name, 'mean'] = np.mean(fold_scores)
        results.loc[feat_name, 'std'] = np.std(fold_scores) * (len(fold_scores) ** -0.5)

    return results


def compute_sfi_for_feature_subset(
    classifier: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    feature_subset: List[int],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    label_times: Optional[pd.Series] = None,
    n_splits: int = 10,
    embargo_pct: float = 0.0,
    scoring: str = 'accuracy',
) -> float:
    """
    Compute CV score for a subset of features.

    This extends SFI to evaluate groups of features together,
    which can capture joint effects missed by single-feature SFI.

    Parameters
    ----------
    classifier : estimator
        Classifier to use.
    X : np.ndarray or pd.DataFrame
        Full feature matrix.
    y : np.ndarray or pd.Series
        Target labels.
    feature_subset : List[int]
        Indices of features to include.
    sample_weight : array-like, optional
        Sample weights.
    label_times : pd.Series, optional
        Label times for purged CV.
    n_splits : int, default=10
        Number of CV folds.
    embargo_pct : float, default=0.0
        Embargo percentage.
    scoring : str, default='accuracy'
        Scoring method.

    Returns
    -------
    float
        Mean CV score for the feature subset.

    Examples
    --------
    >>> # Evaluate features 0, 2, and 5 together
    >>> score = compute_sfi_for_feature_subset(
    ...     clf, X, y,
    ...     feature_subset=[0, 2, 5],
    ...     label_times=label_times
    ... )
    >>> print(f"Score with features 0, 2, 5: {score:.4f}")
    """
    # Convert to arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if sample_weight is not None and isinstance(sample_weight, pd.Series):
        sample_weight = sample_weight.values

    if sample_weight is None:
        sample_weight = np.ones(len(y))

    # Extract feature subset
    X_subset = X[:, feature_subset]

    # Create cross-validator
    if label_times is not None:
        cv = PurgedKFold(
            n_splits=n_splits,
            label_times=label_times,
            embargo_pct=embargo_pct,
        )
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=False)

    # Cross-validation
    fold_scores = []

    for train_idx, test_idx in cv.split(X_subset):
        X_train = X_subset[train_idx]
        X_test = X_subset[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        train_weights = sample_weight[train_idx]
        test_weights = sample_weight[test_idx]

        clf = clone(classifier)
        clf.fit(X_train, y_train, sample_weight=train_weights)

        if scoring == 'neg_log_loss':
            y_prob = clf.predict_proba(X_test)
            score = -log_loss(y_test, y_prob, sample_weight=test_weights, labels=clf.classes_)
        else:
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred, sample_weight=test_weights)

        fold_scores.append(score)

    return np.mean(fold_scores)


def get_sfi_feature_ranking(
    classifier: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    label_times: Optional[pd.Series] = None,
    n_splits: int = 10,
    embargo_pct: float = 0.0,
    scoring: str = 'accuracy',
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get ranked list of features by SFI importance.

    Parameters
    ----------
    classifier : estimator
        Classifier to use.
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    y : np.ndarray or pd.Series
        Target labels.
    label_times : pd.Series, optional
        Label times for purged CV.
    n_splits : int, default=10
        Number of CV folds.
    embargo_pct : float, default=0.0
        Embargo percentage.
    scoring : str, default='accuracy'
        Scoring method.
    top_n : int, optional
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Ranked features with CV score statistics.
    """
    importance = compute_sfi_importance(
        classifier, X, y,
        label_times=label_times,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        scoring=scoring,
    )

    importance = importance.sort_values('mean', ascending=False)

    if top_n is not None:
        importance = importance.head(top_n)

    importance['rank'] = range(1, len(importance) + 1)

    return importance
