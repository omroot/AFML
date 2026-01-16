"""
Bagging classifier setup for financial applications.

This module provides helper functions and factory methods for creating
bagging classifiers that properly handle the non-IID nature of financial
data.

Reference: AFML Chapter 6, Sections 6.3.3 and 6.7
"""

from typing import Any, Optional, Union
import numpy as np
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold


def create_bagging_classifier(
    base_estimator: Optional[Any] = None,
    num_estimators: int = 100,
    max_samples: Union[int, float] = 1.0,
    max_features: Union[int, float] = 1.0,
    bootstrap: bool = True,
    bootstrap_features: bool = False,
    oob_score: bool = False,
    warm_start: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: int = 0,
) -> BaggingClassifier:
    """
    Create a bagging classifier with sensible defaults for financial data.

    Parameters
    ----------
    base_estimator : estimator, optional
        The base estimator to fit on random subsets. If None, uses
        DecisionTreeClassifier with entropy criterion.
    num_estimators : int, default=100
        Number of base estimators in the ensemble.
    max_samples : int or float, default=1.0
        Number of samples to draw for each base estimator.
        - If int, draw that many samples
        - If float, draw that fraction of the training set
        For financial data with overlapping labels, set this to the
        average uniqueness (avgU) from Chapter 4.
    max_features : int or float, default=1.0
        Number of features to draw for each base estimator.
        - If int, draw that many features
        - If float, draw that fraction of features
    bootstrap : bool, default=True
        Whether to sample with replacement.
    bootstrap_features : bool, default=False
        Whether to sample features with replacement.
    oob_score : bool, default=False
        Whether to use out-of-bag samples for accuracy estimation.
        Note: OOB accuracy is often inflated for financial data.
    warm_start : bool, default=False
        Whether to reuse previous fit results.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all processors.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    BaggingClassifier
        Configured bagging classifier.

    Notes
    -----
    For financial data with overlapping labels (non-IID), you should:
    1. Set max_samples to the average uniqueness between samples
    2. Use StratifiedKFold with shuffle=False for cross-validation
    3. Ignore OOB accuracy (it's likely inflated)

    References
    ----------
    AFML Chapter 6, Section 6.3.3: Observation Redundancy

    Examples
    --------
    >>> # Basic usage
    >>> clf = create_bagging_classifier(num_estimators=100)

    >>> # With average uniqueness from Chapter 4
    >>> avg_uniqueness = 0.15  # From sample_weights module
    >>> clf = create_bagging_classifier(
    ...     num_estimators=1000,
    ...     max_samples=avg_uniqueness,
    ...     n_jobs=-1
    ... )
    """
    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(
            criterion="entropy",
            max_features="sqrt",
            class_weight="balanced",
        )

    return BaggingClassifier(
        estimator=base_estimator,
        n_estimators=num_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        bootstrap_features=bootstrap_features,
        oob_score=oob_score,
        warm_start=warm_start,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )


def create_bagging_regressor(
    base_estimator: Optional[Any] = None,
    num_estimators: int = 100,
    max_samples: Union[int, float] = 1.0,
    max_features: Union[int, float] = 1.0,
    bootstrap: bool = True,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> BaggingRegressor:
    """
    Create a bagging regressor with sensible defaults for financial data.

    Parameters
    ----------
    base_estimator : estimator, optional
        The base estimator. If None, uses DecisionTreeRegressor.
    num_estimators : int, default=100
        Number of base estimators.
    max_samples : int or float, default=1.0
        Samples per estimator.
    max_features : int or float, default=1.0
        Features per estimator.
    bootstrap : bool, default=True
        Sample with replacement.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Random seed.

    Returns
    -------
    BaggingRegressor
        Configured bagging regressor.
    """
    if base_estimator is None:
        base_estimator = DecisionTreeRegressor(max_features="sqrt")

    return BaggingRegressor(
        estimator=base_estimator,
        n_estimators=num_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def create_scalable_svm_classifier(
    num_estimators: int = 100,
    max_samples: Union[int, float] = 0.1,
    max_iter: int = 100000,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    class_weight: str = "balanced",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> BaggingClassifier:
    """
    Create a scalable SVM classifier using bagging.

    SVMs don't scale well with sample size. By using bagging with
    early stopping, we can train on large datasets efficiently.

    Parameters
    ----------
    num_estimators : int, default=100
        Number of SVM base estimators.
    max_samples : int or float, default=0.1
        Fraction of samples per SVM. Smaller values = faster training.
    max_iter : int, default=100000
        Maximum iterations per SVM (early stopping).
    kernel : str, default='rbf'
        SVM kernel type.
    C : float, default=1.0
        Regularization parameter.
    gamma : str, default='scale'
        Kernel coefficient.
    class_weight : str, default='balanced'
        Class weights for imbalanced data.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Random seed.

    Returns
    -------
    BaggingClassifier
        Bagging classifier with SVM base estimators.

    Notes
    -----
    This approach trades off individual SVM accuracy for:
    1. Faster training (smaller subsets, early stopping)
    2. Parallelization (multiple SVMs in parallel)
    3. Lower variance (bagging effect)

    References
    ----------
    AFML Chapter 6, Section 6.7: Bagging for Scalability

    Examples
    --------
    >>> clf = create_scalable_svm_classifier(
    ...     num_estimators=100,
    ...     max_samples=0.1,
    ...     n_jobs=-1
    ... )
    >>> clf.fit(X_train, y_train)
    """
    base_svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    return BaggingClassifier(
        estimator=base_svm,
        n_estimators=num_estimators,
        max_samples=max_samples,
        max_features=1.0,
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def create_scalable_svm_regressor(
    num_estimators: int = 100,
    max_samples: Union[int, float] = 0.1,
    max_iter: int = 100000,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> BaggingRegressor:
    """
    Create a scalable SVM regressor using bagging.

    Parameters
    ----------
    num_estimators : int, default=100
        Number of SVR base estimators.
    max_samples : int or float, default=0.1
        Fraction of samples per SVR.
    max_iter : int, default=100000
        Maximum iterations per SVR.
    kernel : str, default='rbf'
        SVR kernel type.
    C : float, default=1.0
        Regularization parameter.
    gamma : str, default='scale'
        Kernel coefficient.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Random seed.

    Returns
    -------
    BaggingRegressor
        Bagging regressor with SVR base estimators.
    """
    base_svr = SVR(
        kernel=kernel,
        C=C,
        gamma=gamma,
        max_iter=max_iter,
    )

    return BaggingRegressor(
        estimator=base_svr,
        n_estimators=num_estimators,
        max_samples=max_samples,
        max_features=1.0,
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def get_financial_cv_splitter(
    n_splits: int = 5,
    shuffle: bool = False,
) -> StratifiedKFold:
    """
    Get a cross-validation splitter appropriate for financial data.

    For financial data with time-dependent observations, we should NOT
    shuffle before splitting, as this would leak future information.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. A low number is preferred to avoid placing
        very similar samples in both train and test sets.
    shuffle : bool, default=False
        Whether to shuffle. Should be False for financial data.

    Returns
    -------
    StratifiedKFold
        Configured cross-validation splitter.

    Notes
    -----
    For financial applications:
    1. Use shuffle=False to respect temporal order
    2. Use low n_splits to avoid test set contamination
    3. Ignore OOB accuracy from bagging classifiers

    References
    ----------
    AFML Chapter 6, Section 6.3.3: Observation Redundancy

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> cv = get_financial_cv_splitter(n_splits=5)
    >>> scores = cross_val_score(clf, X, y, cv=cv)
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle)


def estimate_max_samples(
    avg_uniqueness: float,
    min_samples: float = 0.01,
    max_samples: float = 1.0,
) -> float:
    """
    Estimate the appropriate max_samples parameter from average uniqueness.

    Parameters
    ----------
    avg_uniqueness : float
        Average uniqueness of samples from Chapter 4's sample weights.
    min_samples : float, default=0.01
        Minimum fraction of samples (floor).
    max_samples : float, default=1.0
        Maximum fraction of samples (ceiling).

    Returns
    -------
    float
        Recommended max_samples value for bagging.

    Notes
    -----
    If each observation at t is labeled according to the return between
    t and t+100, we should sample ~1% of observations per bagged estimator.

    The average uniqueness provides a principled way to set this parameter.

    Examples
    --------
    >>> from afml.sample_weights import get_average_uniqueness
    >>> avg_u = get_average_uniqueness(events, close_prices)
    >>> max_samp = estimate_max_samples(avg_u)
    >>> clf = create_bagging_classifier(max_samples=max_samp)
    """
    return np.clip(avg_uniqueness, min_samples, max_samples)
