"""
Cross-validation scoring functions for financial data.

This module provides scoring functions that properly handle sample weights
and address known bugs in sklearn's cross_val_score.

Reference: AFML Chapter 7, Section 7.5
"""

from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import BaseCrossValidator

from afml.cross_validation.purged_kfold import PurgedKFold


def cv_score(
    classifier: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    scoring: str = "neg_log_loss",
    label_times: Optional[pd.Series] = None,
    cv: Optional[BaseCrossValidator] = None,
    n_splits: int = 3,
    embargo_pct: float = 0.0,
) -> np.ndarray:
    """
    Perform cross-validation with proper handling of sample weights.

    This function addresses known sklearn bugs where cross_val_score
    passes weights to fit() but not to the scoring function.

    Parameters
    ----------
    classifier : estimator
        A classifier implementing fit() and predict()/predict_proba().
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray or pd.Series
        Target labels of shape (n_samples,).
    sample_weight : np.ndarray or pd.Series, optional
        Sample weights for both fitting and scoring.
    scoring : str, default='neg_log_loss'
        Scoring method: 'neg_log_loss' or 'accuracy'.
    label_times : pd.Series, optional
        Label times for purged CV. Required if cv is None.
    cv : BaseCrossValidator, optional
        Cross-validator object. If None, creates PurgedKFold.
    n_splits : int, default=3
        Number of CV splits (used if cv is None).
    embargo_pct : float, default=0.0
        Embargo percentage (used if cv is None).

    Returns
    -------
    np.ndarray
        Array of scores for each fold.

    Raises
    ------
    ValueError
        If scoring method is not supported.

    Notes
    -----
    This function fixes two known sklearn bugs:
    1. Scoring functions not receiving classes_ information
    2. cross_val_score passing weights to fit but not to scoring

    Use this function instead of sklearn's cross_val_score for
    financial applications.

    References
    ----------
    AFML Chapter 7, Snippet 7.4: Using the PurgedKFold Class

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier(n_estimators=100)
    >>> scores = cv_score(
    ...     clf, X, y,
    ...     sample_weight=weights,
    ...     scoring='neg_log_loss',
    ...     label_times=label_times,
    ...     n_splits=5,
    ...     embargo_pct=0.01
    ... )
    >>> print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
    """
    # Validate scoring method
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError(
            f"scoring must be 'neg_log_loss' or 'accuracy', got '{scoring}'"
        )

    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if sample_weight is not None and isinstance(sample_weight, pd.Series):
        sample_weight = sample_weight.values

    # Create default sample weights if not provided
    if sample_weight is None:
        sample_weight = np.ones(len(y))

    # Create cross-validator if not provided
    if cv is None:
        if label_times is None:
            raise ValueError("label_times required when cv is None")
        cv = PurgedKFold(
            n_splits=n_splits,
            label_times=label_times,
            embargo_pct=embargo_pct,
        )

    # Perform cross-validation
    scores = []

    for train_indices, test_indices in cv.split(X):
        # Get train/test data
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        # Get sample weights
        train_weights = sample_weight[train_indices]
        test_weights = sample_weight[test_indices]

        # Fit classifier
        classifier.fit(X_train, y_train, sample_weight=train_weights)

        # Compute score
        if scoring == "neg_log_loss":
            # Get predicted probabilities
            y_prob = classifier.predict_proba(X_test)

            # Compute log loss with sample weights
            # Note: We pass labels explicitly to handle the sklearn bug
            score = -log_loss(
                y_test,
                y_prob,
                sample_weight=test_weights,
                labels=classifier.classes_,
            )
        else:  # accuracy
            # Get predictions
            y_pred = classifier.predict(X_test)

            # Compute accuracy with sample weights
            score = accuracy_score(
                y_test,
                y_pred,
                sample_weight=test_weights,
            )

        scores.append(score)

    return np.array(scores)


def cv_score_with_predictions(
    classifier: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    label_times: Optional[pd.Series] = None,
    cv: Optional[BaseCrossValidator] = None,
    n_splits: int = 3,
    embargo_pct: float = 0.0,
) -> dict:
    """
    Perform cross-validation and return both scores and predictions.

    Parameters
    ----------
    classifier : estimator
        A classifier implementing fit() and predict_proba().
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    y : np.ndarray or pd.Series
        Target labels.
    sample_weight : np.ndarray or pd.Series, optional
        Sample weights.
    label_times : pd.Series, optional
        Label times for purged CV.
    cv : BaseCrossValidator, optional
        Cross-validator.
    n_splits : int, default=3
        Number of splits.
    embargo_pct : float, default=0.0
        Embargo percentage.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'scores_log_loss': Log loss for each fold
        - 'scores_accuracy': Accuracy for each fold
        - 'predictions': Out-of-fold predictions (indices, y_true, y_pred, y_prob)
        - 'feature_importances': Feature importances if available

    Examples
    --------
    >>> results = cv_score_with_predictions(clf, X, y, label_times=label_times)
    >>> print(f"Log Loss: {results['scores_log_loss'].mean():.4f}")
    >>> print(f"Accuracy: {results['scores_accuracy'].mean():.4f}")
    """
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if sample_weight is not None and isinstance(sample_weight, pd.Series):
        sample_weight = sample_weight.values

    if sample_weight is None:
        sample_weight = np.ones(len(y))

    # Create cross-validator
    if cv is None:
        if label_times is None:
            raise ValueError("label_times required when cv is None")
        cv = PurgedKFold(
            n_splits=n_splits,
            label_times=label_times,
            embargo_pct=embargo_pct,
        )

    # Storage
    scores_log_loss = []
    scores_accuracy = []
    all_predictions = []
    feature_importances = []

    for fold_idx, (train_indices, test_indices) in enumerate(cv.split(X)):
        # Get train/test data
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        train_weights = sample_weight[train_indices]
        test_weights = sample_weight[test_indices]

        # Fit classifier
        classifier.fit(X_train, y_train, sample_weight=train_weights)

        # Get predictions
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)

        # Compute scores
        score_ll = -log_loss(
            y_test, y_prob,
            sample_weight=test_weights,
            labels=classifier.classes_
        )
        score_acc = accuracy_score(y_test, y_pred, sample_weight=test_weights)

        scores_log_loss.append(score_ll)
        scores_accuracy.append(score_acc)

        # Store predictions
        for i, idx in enumerate(test_indices):
            all_predictions.append({
                'fold': fold_idx,
                'index': idx,
                'y_true': y_test[i],
                'y_pred': y_pred[i],
                'y_prob': y_prob[i].tolist(),
            })

        # Store feature importances if available
        if hasattr(classifier, 'feature_importances_'):
            feature_importances.append(classifier.feature_importances_)

    results = {
        'scores_log_loss': np.array(scores_log_loss),
        'scores_accuracy': np.array(scores_accuracy),
        'predictions': pd.DataFrame(all_predictions),
    }

    if feature_importances:
        results['feature_importances'] = np.mean(feature_importances, axis=0)

    return results


def compare_cv_methods(
    classifier: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    label_times: pd.Series,
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Compare different CV methods on the same data.

    Compares standard k-fold, standard k-fold with shuffle, and
    purged k-fold to demonstrate the impact of leakage.

    Parameters
    ----------
    classifier : estimator
        Classifier to evaluate.
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    y : np.ndarray or pd.Series
        Target labels.
    label_times : pd.Series
        Label times for purged CV.
    sample_weight : np.ndarray or pd.Series, optional
        Sample weights.
    n_splits : int, default=5
        Number of CV splits.
    embargo_pct : float, default=0.01
        Embargo percentage for purged CV.

    Returns
    -------
    pd.DataFrame
        Comparison of CV methods with accuracy statistics.

    Examples
    --------
    >>> comparison = compare_cv_methods(clf, X, y, label_times)
    >>> print(comparison)
    """
    from sklearn.model_selection import KFold, cross_val_score

    # Convert to numpy
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X
    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = y

    results = []

    # Standard K-Fold (no shuffle) - sklearn
    cv_standard = KFold(n_splits=n_splits, shuffle=False)
    scores_standard = cross_val_score(classifier, X_arr, y_arr, cv=cv_standard)
    results.append({
        'method': 'Standard K-Fold (no shuffle)',
        'mean_accuracy': scores_standard.mean(),
        'std_accuracy': scores_standard.std(),
        'min_accuracy': scores_standard.min(),
        'max_accuracy': scores_standard.max(),
    })

    # Standard K-Fold (with shuffle) - sklearn
    cv_shuffle = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_shuffle = cross_val_score(classifier, X_arr, y_arr, cv=cv_shuffle)
    results.append({
        'method': 'Standard K-Fold (shuffle)',
        'mean_accuracy': scores_shuffle.mean(),
        'std_accuracy': scores_shuffle.std(),
        'min_accuracy': scores_shuffle.min(),
        'max_accuracy': scores_shuffle.max(),
    })

    # Purged K-Fold
    scores_purged = cv_score(
        classifier, X, y,
        sample_weight=sample_weight,
        scoring='accuracy',
        label_times=label_times,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
    )
    results.append({
        'method': f'Purged K-Fold ({embargo_pct:.0%} embargo)',
        'mean_accuracy': scores_purged.mean(),
        'std_accuracy': scores_purged.std(),
        'min_accuracy': scores_purged.min(),
        'max_accuracy': scores_purged.max(),
    })

    return pd.DataFrame(results)
