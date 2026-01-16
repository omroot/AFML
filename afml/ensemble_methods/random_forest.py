"""
Random Forest setup for financial applications.

This module provides helper functions and factory methods for creating
Random Forest classifiers that properly handle the non-IID nature of
financial data and avoid common overfitting pitfalls.

Reference: AFML Chapter 6, Section 6.4 and Snippet 6.2
"""

from typing import Any, Optional, Union
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def create_random_forest_classifier(
    num_estimators: int = 1000,
    criterion: str = "entropy",
    max_depth: Optional[int] = None,
    min_samples_split: Union[int, float] = 2,
    min_samples_leaf: Union[int, float] = 1,
    min_weight_fraction_leaf: float = 0.0,
    max_features: Union[int, float, str] = "sqrt",
    max_leaf_nodes: Optional[int] = None,
    bootstrap: bool = True,
    oob_score: bool = False,
    class_weight: Optional[str] = "balanced_subsample",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: int = 0,
) -> RandomForestClassifier:
    """
    Create a Random Forest classifier with defaults suited for financial data.

    Parameters
    ----------
    num_estimators : int, default=1000
        Number of trees in the forest.
    criterion : str, default='entropy'
        Split criterion. 'entropy' for information gain, 'gini' for Gini impurity.
    max_depth : int, optional
        Maximum depth of trees. None means unlimited.
    min_samples_split : int or float, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int or float, default=1
        Minimum samples required at a leaf node.
    min_weight_fraction_leaf : float, default=0.0
        Minimum weighted fraction of samples at a leaf.
        Set to 0.05 for early stopping to prevent overfitting.
    max_features : int, float, or str, default='sqrt'
        Number of features to consider at each split.
        'sqrt' is recommended to decorrelate trees.
    max_leaf_nodes : int, optional
        Maximum number of leaf nodes.
    bootstrap : bool, default=True
        Whether to bootstrap samples.
    oob_score : bool, default=False
        Whether to compute OOB score. Note: Often inflated for financial data.
    class_weight : str, optional, default='balanced_subsample'
        Class weights. 'balanced_subsample' helps with imbalanced classes.
    n_jobs : int, optional
        Parallel jobs. -1 uses all processors.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    RandomForestClassifier
        Configured Random Forest classifier.

    Notes
    -----
    For financial data with non-IID observations:
    1. Use 'balanced_subsample' for class_weight to handle imbalanced classes
    2. Set min_weight_fraction_leaf to ~0.05 for early stopping
    3. Consider using bagged_random_forest_classifier() instead for
       better control over sample uniqueness

    References
    ----------
    AFML Chapter 6, Section 6.4: Random Forest
    AFML Chapter 6, Snippet 6.2: Three ways of setting up an RF

    Examples
    --------
    >>> clf = create_random_forest_classifier(
    ...     num_estimators=1000,
    ...     min_weight_fraction_leaf=0.05,  # Early stopping
    ...     n_jobs=-1
    ... )
    """
    return RandomForestClassifier(
        n_estimators=num_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        bootstrap=bootstrap,
        oob_score=oob_score,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )


def create_random_forest_regressor(
    num_estimators: int = 1000,
    criterion: str = "squared_error",
    max_depth: Optional[int] = None,
    min_samples_split: Union[int, float] = 2,
    min_samples_leaf: Union[int, float] = 1,
    min_weight_fraction_leaf: float = 0.0,
    max_features: Union[int, float, str] = "sqrt",
    bootstrap: bool = True,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> RandomForestRegressor:
    """
    Create a Random Forest regressor with defaults suited for financial data.

    Parameters
    ----------
    num_estimators : int, default=1000
        Number of trees.
    criterion : str, default='squared_error'
        Split criterion.
    max_depth : int, optional
        Maximum tree depth.
    min_samples_split : int or float, default=2
        Minimum samples to split.
    min_samples_leaf : int or float, default=1
        Minimum samples at leaf.
    min_weight_fraction_leaf : float, default=0.0
        Minimum weighted fraction at leaf.
    max_features : int, float, or str, default='sqrt'
        Features per split.
    bootstrap : bool, default=True
        Bootstrap samples.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Random seed.

    Returns
    -------
    RandomForestRegressor
        Configured Random Forest regressor.
    """
    return RandomForestRegressor(
        n_estimators=num_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def create_bagged_decision_tree_classifier(
    num_estimators: int = 1000,
    avg_uniqueness: float = 1.0,
    criterion: str = "entropy",
    max_features: Union[int, float, str] = "sqrt",
    class_weight: str = "balanced",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> BaggingClassifier:
    """
    Create a bagged decision tree classifier with controlled sample size.

    This is Method 3 from Snippet 6.2: BaggingClassifier on DecisionTreeClassifier
    where max_samples is set to the average uniqueness.

    Parameters
    ----------
    num_estimators : int, default=1000
        Number of decision trees.
    avg_uniqueness : float, default=1.0
        Average uniqueness between samples (from Chapter 4).
        Controls max_samples to prevent overfitting with overlapping labels.
    criterion : str, default='entropy'
        Split criterion.
    max_features : str, default='sqrt'
        Features per split.
    class_weight : str, default='balanced'
        Class weights.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Random seed.

    Returns
    -------
    BaggingClassifier
        Bagging classifier with decision tree base estimators.

    Notes
    -----
    This approach differs from RandomForestClassifier in that:
    1. max_samples is controlled (not fixed to dataset size)
    2. Better handles non-IID financial data

    References
    ----------
    AFML Chapter 6, Snippet 6.2: Method 3

    Examples
    --------
    >>> # Using average uniqueness from sample_weights module
    >>> clf = create_bagged_decision_tree_classifier(
    ...     num_estimators=1000,
    ...     avg_uniqueness=0.15,
    ...     n_jobs=-1
    ... )
    """
    base_tree = DecisionTreeClassifier(
        criterion=criterion,
        max_features=max_features,
        class_weight=class_weight,
    )

    return BaggingClassifier(
        estimator=base_tree,
        n_estimators=num_estimators,
        max_samples=avg_uniqueness,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def create_bagged_random_forest_classifier(
    num_estimators: int = 1000,
    avg_uniqueness: float = 1.0,
    criterion: str = "entropy",
    max_features: Union[int, float, str] = 1.0,
    class_weight: str = "balanced_subsample",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> BaggingClassifier:
    """
    Create a bagged Random Forest classifier with controlled sample size.

    This is Method 4 from Snippet 6.2: BaggingClassifier on RandomForestClassifier
    where max_samples is set to the average uniqueness.

    Parameters
    ----------
    num_estimators : int, default=1000
        Number of single-tree Random Forests (effectively trees).
    avg_uniqueness : float, default=1.0
        Average uniqueness between samples.
    criterion : str, default='entropy'
        Split criterion.
    max_features : float, default=1.0
        Features per bagging iteration (each RF uses all features).
    class_weight : str, default='balanced_subsample'
        Class weights.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Random seed.

    Returns
    -------
    BaggingClassifier
        Bagging classifier with single-tree RF base estimators.

    Notes
    -----
    Each base estimator is a RandomForestClassifier with n_estimators=1
    and bootstrap=False (bagging handles the bootstrapping).

    This provides maximum control over the sampling process while
    retaining RF's feature randomization.

    References
    ----------
    AFML Chapter 6, Snippet 6.2: Method 4

    Examples
    --------
    >>> clf = create_bagged_random_forest_classifier(
    ...     num_estimators=1000,
    ...     avg_uniqueness=0.15,
    ...     n_jobs=-1
    ... )
    """
    base_rf = RandomForestClassifier(
        n_estimators=1,  # Single tree per RF
        criterion=criterion,
        bootstrap=False,  # Bagging handles bootstrapping
        class_weight=class_weight,
    )

    return BaggingClassifier(
        estimator=base_rf,
        n_estimators=num_estimators,
        max_samples=avg_uniqueness,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def create_early_stopping_rf_classifier(
    num_estimators: int = 1000,
    min_weight_fraction_leaf: float = 0.05,
    criterion: str = "entropy",
    max_features: Union[int, float, str] = "sqrt",
    class_weight: str = "balanced_subsample",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> RandomForestClassifier:
    """
    Create a Random Forest with early stopping via min_weight_fraction_leaf.

    This is Method 2 from Snippet 6.2: Setting min_weight_fraction_leaf
    to a sufficiently large value to converge OOB to cross-validation accuracy.

    Parameters
    ----------
    num_estimators : int, default=1000
        Number of trees.
    min_weight_fraction_leaf : float, default=0.05
        Minimum weighted fraction at leaf. Higher values = more regularization.
        5% is recommended as a starting point.
    criterion : str, default='entropy'
        Split criterion.
    max_features : str, default='sqrt'
        Features per split.
    class_weight : str, default='balanced_subsample'
        Class weights.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Random seed.

    Returns
    -------
    RandomForestClassifier
        Random Forest with early stopping regularization.

    Notes
    -----
    Setting min_weight_fraction_leaf to ~5% forces trees to stop
    growing early, preventing overfitting to redundant samples.

    Tune this parameter so that OOB accuracy converges to
    cross-validation accuracy (with shuffle=False).

    References
    ----------
    AFML Chapter 6, Section 6.4: Method 2

    Examples
    --------
    >>> clf = create_early_stopping_rf_classifier(
    ...     min_weight_fraction_leaf=0.05,
    ...     n_jobs=-1
    ... )
    """
    return RandomForestClassifier(
        n_estimators=num_estimators,
        criterion=criterion,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def create_low_max_features_rf_classifier(
    num_estimators: int = 1000,
    max_features: int = 1,
    criterion: str = "entropy",
    class_weight: str = "balanced_subsample",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> RandomForestClassifier:
    """
    Create a Random Forest with very low max_features for decorrelation.

    This is Method 1 from Snippet 6.2: Setting max_features to a lower
    value to force discrepancy between trees.

    Parameters
    ----------
    num_estimators : int, default=1000
        Number of trees.
    max_features : int, default=1
        Features per split. Low values force tree diversity.
    criterion : str, default='entropy'
        Split criterion.
    class_weight : str, default='balanced_subsample'
        Class weights.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Random seed.

    Returns
    -------
    RandomForestClassifier
        Random Forest with extreme feature subsampling.

    Notes
    -----
    Extremely low max_features (even 1) forces each tree to make
    very different splits, reducing correlation between trees.

    This can help when samples are highly redundant, as it
    artificially creates diversity.

    References
    ----------
    AFML Chapter 6, Section 6.4: Method 1
    """
    return RandomForestClassifier(
        n_estimators=num_estimators,
        criterion=criterion,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def get_rf_feature_importance(
    rf_classifier: Union[RandomForestClassifier, RandomForestRegressor],
) -> dict:
    """
    Extract feature importances from a fitted Random Forest.

    Parameters
    ----------
    rf_classifier : RandomForestClassifier or RandomForestRegressor
        A fitted Random Forest model.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'importances': Array of feature importances
        - 'std': Standard deviation of importances across trees

    Notes
    -----
    Feature importance in RF is computed as Mean Decrease in Impurity (MDI).
    See Chapter 8 for more sophisticated importance measures.

    Examples
    --------
    >>> clf = create_random_forest_classifier()
    >>> clf.fit(X_train, y_train)
    >>> importance = get_rf_feature_importance(clf)
    >>> print(importance['importances'])
    """
    return {
        "importances": rf_classifier.feature_importances_,
        "std": rf_classifier.estimators_[0].feature_importances_.std()
        if hasattr(rf_classifier, "estimators_")
        else 0.0,
    }
