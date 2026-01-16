"""
Hyperparameter search with purged cross-validation.

This module provides functions for hyperparameter optimization using
grid search and randomized search with purged k-fold cross-validation,
which is essential for preventing information leakage in financial data.

Reference: AFML Chapter 9, Sections 9.2-9.3, Snippets 9.1 and 9.3
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline

from afml.cross_validation import PurgedKFold
from afml.hpo.pipeline import SampleWeightPipeline


def select_scoring_metric(labels: np.ndarray) -> str:
    """
    Select appropriate scoring metric based on label distribution.

    For meta-labeling applications (binary labels with potential class
    imbalance), use F1 score. For other applications, use neg_log_loss
    as it considers prediction probabilities.

    Parameters
    ----------
    labels : np.ndarray
        Target labels.

    Returns
    -------
    str
        Scoring metric name ('f1' or 'neg_log_loss').

    Notes
    -----
    Why F1 for meta-labeling?
    - If there's a large number of negative cases, a classifier that
      predicts all negatives achieves high accuracy but zero recall
    - F1 corrects for this by considering both precision and recall

    Why neg_log_loss for investment strategies?
    - Accuracy treats all errors equally regardless of confidence
    - Log loss penalizes high-confidence errors more heavily
    - Investment profits depend on both label AND confidence

    References
    ----------
    AFML Chapter 9, Section 9.4

    Examples
    --------
    >>> # Meta-labeling (binary)
    >>> y_meta = np.array([0, 1, 0, 0, 1])
    >>> print(select_scoring_metric(y_meta))  # 'f1'
    >>>
    >>> # Multi-class or general
    >>> y_general = np.array([0, 1, 2, 0, 1])
    >>> print(select_scoring_metric(y_general))  # 'neg_log_loss'
    """
    unique_labels = set(np.unique(labels))

    # Meta-labeling: binary classification with labels {0, 1}
    if unique_labels == {0, 1}:
        return "f1"
    else:
        # General classification: symmetric scoring
        return "neg_log_loss"


def fit_hyperparameters(
    features: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, pd.Series],
    label_times: pd.Series,
    estimator: Any,
    param_grid: Dict[str, List[Any]],
    n_splits: int = 3,
    embargo_pct: float = 0.0,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    bagging_config: Optional[Tuple[int, float, float]] = None,
    sample_weight: Optional[np.ndarray] = None,
    random_search_iterations: int = 0,
    random_state: Optional[int] = None,
    verbose: int = 0,
) -> Union[Any, Pipeline]:
    """
    Fit classifier with hyperparameter tuning using purged cross-validation.

    Performs either grid search or randomized search to find optimal
    hyperparameters, using purged k-fold CV to prevent information leakage.
    Optionally bags the tuned estimator for improved robustness.

    Parameters
    ----------
    features : array-like of shape (n_samples, n_features)
        Training features.
    labels : array-like of shape (n_samples,)
        Target labels.
    label_times : pd.Series
        Series where index is observation start time and values are
        observation end times. Required for purged CV.
    estimator : estimator object
        The estimator to tune. Can be a classifier or a Pipeline.
    param_grid : dict
        Dictionary with parameter names as keys and lists of values
        (for grid search) or distributions (for randomized search) as values.
    n_splits : int, default=3
        Number of cross-validation folds.
    embargo_pct : float, default=0.0
        Percentage of observations to embargo after test set.
    scoring : str, default=None
        Scoring metric. If None, automatically selected based on labels.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all CPUs).
    bagging_config : tuple of (n_estimators, max_samples, max_features), default=None
        If provided, the tuned estimator will be bagged:
        - n_estimators: Number of base estimators
        - max_samples: Fraction of samples to draw
        - max_features: Fraction of features to draw
        If None or n_estimators <= 0, no bagging is applied.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights for training.
    random_search_iterations : int, default=0
        If > 0, use RandomizedSearchCV with this many iterations.
        If 0, use GridSearchCV.
    random_state : int, default=None
        Random state for reproducibility.
    verbose : int, default=0
        Verbosity level for search progress.

    Returns
    -------
    fitted_model : estimator or Pipeline
        The fitted model with best hyperparameters. If bagging is applied,
        returns a Pipeline containing the bagged estimator.

    Notes
    -----
    The function:
    1. Creates PurgedKFold CV with specified embargo
    2. Performs grid or randomized search to find best parameters
    3. Optionally bags the tuned estimator
    4. Fits the final model on all data

    For meta-labeling (binary labels {0,1}), F1 scoring is used automatically.
    For other cases, neg_log_loss is used as it better reflects investment
    performance.

    References
    ----------
    AFML Chapter 9, Snippets 9.1 and 9.3

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from afml.hpo.search import fit_hyperparameters
    >>> from afml.hpo.distributions import log_uniform
    >>>
    >>> # Grid search
    >>> param_grid = {
    ...     'C': [0.01, 0.1, 1, 10, 100],
    ...     'gamma': [0.01, 0.1, 1, 10, 100],
    ... }
    >>> model = fit_hyperparameters(
    ...     X, y, label_times,
    ...     estimator=SVC(probability=True),
    ...     param_grid=param_grid,
    ...     n_splits=5,
    ... )
    >>>
    >>> # Randomized search with log-uniform sampling
    >>> param_distributions = {
    ...     'C': log_uniform(a=1e-2, b=1e2),
    ...     'gamma': log_uniform(a=1e-2, b=1e2),
    ... }
    >>> model = fit_hyperparameters(
    ...     X, y, label_times,
    ...     estimator=SVC(probability=True),
    ...     param_grid=param_distributions,
    ...     random_search_iterations=25,
    ... )
    """
    # Convert to numpy arrays if needed
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(labels, pd.Series):
        labels = labels.values

    # Auto-select scoring metric if not provided
    if scoring is None:
        scoring = select_scoring_metric(labels)

    # Create purged k-fold cross-validation
    purged_cv = PurgedKFold(
        n_splits=n_splits,
        label_times=label_times,
        embargo_pct=embargo_pct,
    )

    # Prepare fit parameters for sample weights
    fit_params = {}
    if sample_weight is not None:
        # Handle Pipeline vs single estimator
        if isinstance(estimator, Pipeline):
            last_step_name = estimator.steps[-1][0]
            fit_params[f"{last_step_name}__sample_weight"] = sample_weight
        else:
            fit_params["sample_weight"] = sample_weight

    # Choose search strategy
    if random_search_iterations == 0:
        # Grid search
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=purged_cv,
            n_jobs=n_jobs,
            refit=False,  # We'll refit manually
            verbose=verbose,
        )
    else:
        # Randomized search
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=random_search_iterations,
            scoring=scoring,
            cv=purged_cv,
            n_jobs=n_jobs,
            refit=False,  # We'll refit manually
            random_state=random_state,
            verbose=verbose,
        )

    # Fit the search
    search.fit(features, labels, **fit_params)

    # Get best estimator with optimal parameters
    best_estimator = search.best_estimator_
    if best_estimator is None:
        # refit=False means we need to clone and set params manually
        from sklearn.base import clone
        best_estimator = clone(estimator).set_params(**search.best_params_)

    # Apply bagging if configured
    if bagging_config is not None and bagging_config[0] > 0:
        n_estimators, max_samples, max_features = bagging_config

        # Get the base estimator (last step if Pipeline)
        if isinstance(best_estimator, Pipeline):
            base_estimator = SampleWeightPipeline(steps=best_estimator.steps)
        else:
            base_estimator = best_estimator

        # Create bagging classifier
        bagged_estimator = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=int(n_estimators),
            max_samples=float(max_samples),
            max_features=float(max_features),
            n_jobs=n_jobs,
            random_state=random_state,
        )

        # Fit bagged estimator on all data
        if sample_weight is not None:
            # Route sample_weight to the base estimator
            if isinstance(base_estimator, Pipeline):
                last_step_name = base_estimator.steps[-1][0]
                weight_key = f"estimator__{last_step_name}__sample_weight"
            else:
                weight_key = "estimator__sample_weight"
            bagged_estimator.fit(features, labels, **{weight_key: sample_weight})
        else:
            bagged_estimator.fit(features, labels)

        # Wrap in Pipeline for consistent interface
        final_model = Pipeline([("bag", bagged_estimator)])

    else:
        # No bagging, just fit the best estimator
        if sample_weight is not None:
            if isinstance(best_estimator, Pipeline):
                last_step_name = best_estimator.steps[-1][0]
                fit_params = {f"{last_step_name}__sample_weight": sample_weight}
            else:
                fit_params = {"sample_weight": sample_weight}
            best_estimator.fit(features, labels, **fit_params)
        else:
            best_estimator.fit(features, labels)

        final_model = best_estimator

    return final_model


def grid_search_cv(
    features: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, pd.Series],
    label_times: pd.Series,
    estimator: Any,
    param_grid: Dict[str, List[Any]],
    n_splits: int = 5,
    embargo_pct: float = 0.0,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    sample_weight: Optional[np.ndarray] = None,
    verbose: int = 0,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Perform grid search with purged cross-validation.

    Simplified interface for grid search that returns the search results
    along with the best estimator.

    Parameters
    ----------
    features : array-like
        Training features.
    labels : array-like
        Target labels.
    label_times : pd.Series
        Label times for purging.
    estimator : estimator object
        The estimator to tune.
    param_grid : dict
        Parameter grid to search.
    n_splits : int, default=5
        Number of CV folds.
    embargo_pct : float, default=0.0
        Embargo percentage.
    scoring : str, default=None
        Scoring metric.
    n_jobs : int, default=-1
        Number of parallel jobs.
    sample_weight : array-like, default=None
        Sample weights.
    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    best_estimator : estimator
        Fitted estimator with best parameters.
    best_params : dict
        Best parameter combination.
    best_score : float
        Best CV score.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> param_grid = {
    ...     'n_estimators': [100, 200, 500],
    ...     'max_depth': [3, 5, 10, None],
    ... }
    >>> best_model, best_params, best_score = grid_search_cv(
    ...     X, y, label_times,
    ...     estimator=RandomForestClassifier(random_state=42),
    ...     param_grid=param_grid,
    ... )
    >>> print(f"Best params: {best_params}")
    >>> print(f"Best CV score: {best_score:.4f}")
    """
    # Convert to numpy arrays if needed
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(labels, pd.Series):
        labels = labels.values

    # Auto-select scoring metric
    if scoring is None:
        scoring = select_scoring_metric(labels)

    # Create purged CV
    purged_cv = PurgedKFold(
        n_splits=n_splits,
        label_times=label_times,
        embargo_pct=embargo_pct,
    )

    # Create grid search
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=purged_cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
    )

    # Prepare fit params
    fit_params = {}
    if sample_weight is not None:
        if isinstance(estimator, Pipeline):
            last_step_name = estimator.steps[-1][0]
            fit_params[f"{last_step_name}__sample_weight"] = sample_weight
        else:
            fit_params["sample_weight"] = sample_weight

    # Fit
    grid_search.fit(features, labels, **fit_params)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def randomized_search_cv(
    features: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, pd.Series],
    label_times: pd.Series,
    estimator: Any,
    param_distributions: Dict[str, Any],
    n_iter: int = 25,
    n_splits: int = 5,
    embargo_pct: float = 0.0,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    sample_weight: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    verbose: int = 0,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Perform randomized search with purged cross-validation.

    Simplified interface for randomized search that returns the search
    results along with the best estimator.

    Parameters
    ----------
    features : array-like
        Training features.
    labels : array-like
        Target labels.
    label_times : pd.Series
        Label times for purging.
    estimator : estimator object
        The estimator to tune.
    param_distributions : dict
        Parameter distributions to sample from.
    n_iter : int, default=25
        Number of parameter settings to sample.
    n_splits : int, default=5
        Number of CV folds.
    embargo_pct : float, default=0.0
        Embargo percentage.
    scoring : str, default=None
        Scoring metric.
    n_jobs : int, default=-1
        Number of parallel jobs.
    sample_weight : array-like, default=None
        Sample weights.
    random_state : int, default=None
        Random state for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    best_estimator : estimator
        Fitted estimator with best parameters.
    best_params : dict
        Best parameter combination.
    best_score : float
        Best CV score.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from afml.hpo.distributions import log_uniform
    >>> param_distributions = {
    ...     'C': log_uniform(a=1e-2, b=1e2),
    ...     'gamma': log_uniform(a=1e-2, b=1e2),
    ... }
    >>> best_model, best_params, best_score = randomized_search_cv(
    ...     X, y, label_times,
    ...     estimator=SVC(probability=True),
    ...     param_distributions=param_distributions,
    ...     n_iter=50,
    ... )
    """
    # Convert to numpy arrays if needed
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(labels, pd.Series):
        labels = labels.values

    # Auto-select scoring metric
    if scoring is None:
        scoring = select_scoring_metric(labels)

    # Create purged CV
    purged_cv = PurgedKFold(
        n_splits=n_splits,
        label_times=label_times,
        embargo_pct=embargo_pct,
    )

    # Create randomized search
    rand_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=purged_cv,
        n_jobs=n_jobs,
        refit=True,
        random_state=random_state,
        verbose=verbose,
    )

    # Prepare fit params
    fit_params = {}
    if sample_weight is not None:
        if isinstance(estimator, Pipeline):
            last_step_name = estimator.steps[-1][0]
            fit_params[f"{last_step_name}__sample_weight"] = sample_weight
        else:
            fit_params["sample_weight"] = sample_weight

    # Fit
    rand_search.fit(features, labels, **fit_params)

    return rand_search.best_estimator_, rand_search.best_params_, rand_search.best_score_


def get_cv_results_dataframe(
    search_cv: Union[GridSearchCV, RandomizedSearchCV],
) -> pd.DataFrame:
    """
    Convert CV results to a pandas DataFrame for analysis.

    Parameters
    ----------
    search_cv : GridSearchCV or RandomizedSearchCV
        Fitted search object.

    Returns
    -------
    pd.DataFrame
        DataFrame with CV results, sorted by mean test score.

    Examples
    --------
    >>> # After fitting grid search
    >>> results_df = get_cv_results_dataframe(grid_search)
    >>> print(results_df.head())
    """
    results = pd.DataFrame(search_cv.cv_results_)

    # Select relevant columns
    param_cols = [c for c in results.columns if c.startswith("param_")]
    score_cols = ["mean_test_score", "std_test_score", "rank_test_score"]
    time_cols = ["mean_fit_time", "std_fit_time"]

    relevant_cols = param_cols + score_cols + time_cols
    results = results[[c for c in relevant_cols if c in results.columns]]

    # Sort by rank
    results = results.sort_values("rank_test_score")

    return results
