"""
Ensemble Methods module for financial machine learning.

This module provides functions and factory methods for creating and
analyzing ensemble classifiers (bagging, random forests) that properly
handle the non-IID nature of financial data.

Key Concepts:
- **Bias-Variance Tradeoff**: ML errors come from bias, variance, and noise
- **Bagging**: Reduces variance through bootstrap aggregation
- **Random Forest**: Bagging + random feature subsampling
- **Financial Considerations**: Overlapping labels, inflated OOB accuracy

Key insight: In finance, overfitting (variance) is often a bigger concern
than underfitting (bias) due to low signal-to-noise ratio. Bagging is
generally preferred over boosting for this reason.

Reference: AFML Chapter 6
"""

from afml.ensemble_methods.accuracy import (
    compute_bagging_accuracy,
    compute_accuracy_improvement,
    analyze_accuracy_vs_estimators,
    compute_accuracy_heatmap,
    minimum_estimators_for_accuracy,
)
from afml.ensemble_methods.variance import (
    compute_bagging_variance,
    compute_variance_reduction_ratio,
    compute_standard_deviation,
    analyze_variance_vs_correlation,
    analyze_variance_vs_estimators,
    compute_variance_heatmap,
    estimate_correlation_from_predictions,
    optimal_num_estimators,
)
from afml.ensemble_methods.bagging import (
    create_bagging_classifier,
    create_bagging_regressor,
    create_scalable_svm_classifier,
    create_scalable_svm_regressor,
    get_financial_cv_splitter,
    estimate_max_samples,
)
from afml.ensemble_methods.random_forest import (
    create_random_forest_classifier,
    create_random_forest_regressor,
    create_bagged_decision_tree_classifier,
    create_bagged_random_forest_classifier,
    create_early_stopping_rf_classifier,
    create_low_max_features_rf_classifier,
    get_rf_feature_importance,
)

__all__ = [
    # Accuracy calculations
    "compute_bagging_accuracy",
    "compute_accuracy_improvement",
    "analyze_accuracy_vs_estimators",
    "compute_accuracy_heatmap",
    "minimum_estimators_for_accuracy",
    # Variance analysis
    "compute_bagging_variance",
    "compute_variance_reduction_ratio",
    "compute_standard_deviation",
    "analyze_variance_vs_correlation",
    "analyze_variance_vs_estimators",
    "compute_variance_heatmap",
    "estimate_correlation_from_predictions",
    "optimal_num_estimators",
    # Bagging factory functions
    "create_bagging_classifier",
    "create_bagging_regressor",
    "create_scalable_svm_classifier",
    "create_scalable_svm_regressor",
    "get_financial_cv_splitter",
    "estimate_max_samples",
    # Random Forest factory functions
    "create_random_forest_classifier",
    "create_random_forest_regressor",
    "create_bagged_decision_tree_classifier",
    "create_bagged_random_forest_classifier",
    "create_early_stopping_rf_classifier",
    "create_low_max_features_rf_classifier",
    "get_rf_feature_importance",
]
