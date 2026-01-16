"""
Hyperparameter Optimization module for financial machine learning.

This module provides tools for hyperparameter tuning that properly handle
the non-IID nature of financial data through purged cross-validation.

Key Components:
---------------

**Search Functions**
- `fit_hyperparameters`: Main function for HPO with optional bagging
- `grid_search_cv`: Grid search with purged k-fold CV
- `randomized_search_cv`: Randomized search with purged k-fold CV

**Distributions**
- `log_uniform`: Log-uniform distribution for parameters like C and gamma
- `int_log_uniform`: Integer log-uniform for n_estimators, etc.

**Pipeline**
- `SampleWeightPipeline`: Enhanced Pipeline that handles sample_weight

Key Concepts:
-------------

**Why Purged CV for HPO?**
Standard CV methods cause information leakage in financial data due to
overlapping labels. Using purged CV ensures that the hyperparameter
selection process doesn't overfit to leaked information.

**Scoring Metrics**
- `f1`: Use for meta-labeling (imbalanced binary classification)
- `neg_log_loss`: Use for investment strategies (considers confidence)
- `accuracy`: Generally NOT recommended for financial applications

**Log-Uniform Sampling**
For parameters like SVC's C or RBF's gamma that don't respond linearly:
- Uniform[0, 100]: 99% of samples > 1 (inefficient)
- Log-uniform: equal probability in each order of magnitude

**Bagging Tuned Estimators**
After finding optimal hyperparameters, bagging the estimator often
improves out-of-sample performance and reduces variance.

Reference: AFML Chapter 9

Examples
--------
>>> from afml.hpo import (
...     grid_search_cv,
...     randomized_search_cv,
...     log_uniform,
... )
>>> from sklearn.svm import SVC
>>>
>>> # Grid search with purged CV
>>> param_grid = {
...     'C': [0.01, 0.1, 1, 10, 100],
...     'gamma': [0.01, 0.1, 1, 10, 100],
... }
>>> best_model, best_params, score = grid_search_cv(
...     X, y, label_times,
...     estimator=SVC(probability=True),
...     param_grid=param_grid,
...     n_splits=5,
...     embargo_pct=0.01,
... )
>>>
>>> # Randomized search with log-uniform sampling
>>> param_distributions = {
...     'C': log_uniform(a=1e-2, b=1e2),
...     'gamma': log_uniform(a=1e-2, b=1e2),
... }
>>> best_model, best_params, score = randomized_search_cv(
...     X, y, label_times,
...     estimator=SVC(probability=True),
...     param_distributions=param_distributions,
...     n_iter=50,
... )
"""

# Distributions
from afml.hpo.distributions import (
    LogUniformDistribution,
    log_uniform,
    IntLogUniformDistribution,
    int_log_uniform,
)

# Pipeline
from afml.hpo.pipeline import (
    SampleWeightPipeline,
    create_pipeline_with_estimator,
)

# Search functions
from afml.hpo.search import (
    fit_hyperparameters,
    grid_search_cv,
    randomized_search_cv,
    select_scoring_metric,
    get_cv_results_dataframe,
)

__all__ = [
    # Distributions
    "LogUniformDistribution",
    "log_uniform",
    "IntLogUniformDistribution",
    "int_log_uniform",
    # Pipeline
    "SampleWeightPipeline",
    "create_pipeline_with_estimator",
    # Search functions
    "fit_hyperparameters",
    "grid_search_cv",
    "randomized_search_cv",
    "select_scoring_metric",
    "get_cv_results_dataframe",
]
