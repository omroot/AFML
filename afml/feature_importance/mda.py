"""
Mean Decrease Accuracy (MDA) feature importance.

This module provides MDA (permutation importance) feature importance
calculation. MDA measures the decrease in model performance when each
feature is randomly shuffled.

Reference: AFML Chapter 8, Section 8.3.2, Snippet 8.3
"""

from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from afml.cross_validation import PurgedKFold


def compute_mda_importance(
    classifier: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    label_times: Optional[pd.Series] = None,
    n_splits: int = 10,
    embargo_pct: float = 0.0,
    scoring: str = 'neg_log_loss',
) -> Tuple[pd.DataFrame, float]:
    """
    Compute Mean Decrease Accuracy (MDA) feature importance.

    MDA is a slow, out-of-sample feature importance method that works
    with any classifier. It measures the decrease in performance when
    each feature column is randomly permuted (shuffled).

    Parameters
    ----------
    classifier : estimator
        A classifier implementing fit() and predict()/predict_proba().
        Will be cloned for each fold.
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray or pd.Series
        Target labels of shape (n_samples,).
    sample_weight : np.ndarray or pd.Series, optional
        Sample weights for fitting and scoring.
    label_times : pd.Series, optional
        Label times for purged CV. Required for financial data.
    n_splits : int, default=10
        Number of cross-validation folds.
    embargo_pct : float, default=0.0
        Embargo percentage for purged CV.
    scoring : str, default='neg_log_loss'
        Scoring method: 'neg_log_loss' or 'accuracy'.

    Returns
    -------
    importance : pd.DataFrame
        DataFrame with columns:
        - 'mean': Mean importance across all folds
        - 'std': Standard deviation of importance
    baseline_score : float
        Mean baseline score (without permutation).

    Notes
    -----
    Important considerations for MDA:

    1. **Any classifier**: Works with any classifier, not just trees.

    2. **Scoring flexibility**: Can use accuracy, log-loss, F1, etc.
       Better name would be "permutation importance".

    3. **Substitution effects**: Like MDI, correlated features will
       have diluted importance (shuffling one leaves the other intact).

    4. **Can be negative**: A feature can have negative importance if
       it hurts model performance (should be removed).

    5. **Purged CV required**: Must use purged and embargoed CV to
       prevent information leakage in financial applications.

    References
    ----------
    AFML Chapter 8, Snippet 8.3: MDA Feature Importance

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier(n_estimators=100)
    >>> importance, baseline = compute_mda_importance(
    ...     clf, X, y,
    ...     label_times=label_times,
    ...     n_splits=5,
    ...     scoring='accuracy'
    ... )
    >>> print(importance.sort_values('mean', ascending=False).head())
    """
    # Validate scoring method
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise ValueError(
            f"scoring must be 'neg_log_loss' or 'accuracy', got '{scoring}'"
        )

    # Convert to numpy arrays and get feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    if isinstance(y, pd.Series):
        y = y.values

    if sample_weight is not None and isinstance(sample_weight, pd.Series):
        sample_weight = sample_weight.values

    # Default sample weights
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
    # baseline_scores[fold] = score without permutation
    # permuted_scores[fold, feature] = score with feature permuted
    baseline_scores = pd.Series(dtype=float)
    permuted_scores = pd.DataFrame(columns=feature_names)

    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Get train/test data
        X_train, X_test = X[train_idx], X[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        train_weights = sample_weight[train_idx]
        test_weights = sample_weight[test_idx]

        # Fit classifier
        classifier.fit(X_train, y_train, sample_weight=train_weights)

        # Compute baseline score (no permutation)
        if scoring == 'neg_log_loss':
            y_prob = classifier.predict_proba(X_test)
            baseline_score = -log_loss(
                y_test, y_prob,
                sample_weight=test_weights,
                labels=classifier.classes_
            )
        else:  # accuracy
            y_pred = classifier.predict(X_test)
            baseline_score = accuracy_score(
                y_test, y_pred,
                sample_weight=test_weights
            )

        baseline_scores.loc[fold_idx] = baseline_score

        # Compute score with each feature permuted
        for feature_idx, feature_name in enumerate(feature_names):
            # Make a copy and permute the feature
            X_test_permuted = X_test.copy()
            np.random.shuffle(X_test_permuted[:, feature_idx])

            # Score with permuted feature
            if scoring == 'neg_log_loss':
                y_prob_permuted = classifier.predict_proba(X_test_permuted)
                permuted_score = -log_loss(
                    y_test, y_prob_permuted,
                    sample_weight=test_weights,
                    labels=classifier.classes_
                )
            else:  # accuracy
                y_pred_permuted = classifier.predict(X_test_permuted)
                permuted_score = accuracy_score(
                    y_test, y_pred_permuted,
                    sample_weight=test_weights
                )

            permuted_scores.loc[fold_idx, feature_name] = permuted_score

    # Compute importance as decrease in performance
    # importance = (baseline - permuted) / max_possible_improvement
    importance_per_fold = baseline_scores.values.reshape(-1, 1) - permuted_scores.values

    # Normalize by maximum possible score
    if scoring == 'neg_log_loss':
        # Max score is 0 (perfect log-loss)
        # importance = (baseline - permuted) / (-permuted) = 1 - baseline/permuted
        importance_per_fold = importance_per_fold / (-permuted_scores.values)
    else:
        # Max score is 1 (perfect accuracy)
        # importance = (baseline - permuted) / (1 - permuted)
        importance_per_fold = importance_per_fold / (1 - permuted_scores.values)

    # Convert to DataFrame
    importance_df = pd.DataFrame(
        importance_per_fold,
        columns=feature_names
    )

    # Compute mean and std
    mean_importance = importance_df.mean(axis=0)
    std_importance = importance_df.std(axis=0) * (importance_df.shape[0] ** -0.5)

    result = pd.concat(
        {'mean': mean_importance, 'std': std_importance},
        axis=1
    )

    return result, baseline_scores.mean()


def compute_mda_importance_simple(
    classifier: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
    scoring: str = 'accuracy',
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Compute MDA importance without cross-validation (simpler interface).

    This is useful when you already have a train/test split and want
    to quickly compute permutation importance.

    Parameters
    ----------
    classifier : estimator
        A classifier to fit.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    feature_names : list, optional
        Feature names.
    scoring : str, default='accuracy'
        Scoring method.
    n_repeats : int, default=10
        Number of times to repeat permutation for each feature.

    Returns
    -------
    pd.DataFrame
        Importance with mean and std columns.

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> clf.fit(X_train, y_train)
    >>> importance = compute_mda_importance_simple(
    ...     clf, X_train, y_train, X_test, y_test
    ... )
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

    # Fit classifier
    classifier.fit(X_train, y_train)

    # Baseline score
    if scoring == 'neg_log_loss':
        y_prob = classifier.predict_proba(X_test)
        baseline = -log_loss(y_test, y_prob, labels=classifier.classes_)
    else:
        y_pred = classifier.predict(X_test)
        baseline = accuracy_score(y_test, y_pred)

    # Compute importance for each feature
    importance_scores = {name: [] for name in feature_names}

    for _ in range(n_repeats):
        for feat_idx, feat_name in enumerate(feature_names):
            X_test_perm = X_test.copy()
            np.random.shuffle(X_test_perm[:, feat_idx])

            if scoring == 'neg_log_loss':
                y_prob_perm = classifier.predict_proba(X_test_perm)
                score_perm = -log_loss(y_test, y_prob_perm, labels=classifier.classes_)
            else:
                y_pred_perm = classifier.predict(X_test_perm)
                score_perm = accuracy_score(y_test, y_pred_perm)

            importance_scores[feat_name].append(baseline - score_perm)

    # Compute statistics
    result = pd.DataFrame({
        'mean': {k: np.mean(v) for k, v in importance_scores.items()},
        'std': {k: np.std(v) for k, v in importance_scores.items()},
    })

    return result


def get_mda_feature_ranking(
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
    Get ranked list of features by MDA importance.

    Parameters
    ----------
    classifier : estimator
        Classifier to evaluate.
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
        Ranked features with importance statistics.
    """
    importance, _ = compute_mda_importance(
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
