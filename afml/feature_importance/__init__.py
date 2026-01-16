"""
Feature Importance module for financial machine learning.

This module provides feature importance methods that help understand
what features contribute to a classifier's predictive power. It implements
three complementary approaches from AFML Chapter 8:

1. **MDI (Mean Decrease Impurity)**: Fast, in-sample method for tree-based
   classifiers. Measures how much each feature contributes to reducing
   impurity (Gini/entropy) across all trees.

2. **MDA (Mean Decrease Accuracy)**: Slow, out-of-sample method that works
   with any classifier. Measures performance drop when each feature is
   randomly permuted.

3. **SFI (Single Feature Importance)**: Out-of-sample method that evaluates
   each feature in isolation. Not affected by substitution effects.

Key Concepts:
- **Substitution effects**: Correlated features dilute each other's importance
  in MDI and MDA (similar to multicollinearity in regression)
- **Orthogonalization**: Using PCA to create uncorrelated features before
  importance analysis helps address substitution effects
- **PCA-Importance correlation**: High correlation between PCA ranking
  (unsupervised) and importance ranking (supervised) suggests pattern is
  not entirely overfit

Marcos' First Law of Backtesting:
    "Backtesting is not a research tool. Feature importance is."

Reference: AFML Chapter 8
"""

# Mean Decrease Impurity (MDI)
from afml.feature_importance.mdi import (
    compute_mdi_importance,
    compute_mdi_importance_clustered,
    get_mdi_feature_ranking,
)

# Mean Decrease Accuracy (MDA)
from afml.feature_importance.mda import (
    compute_mda_importance,
    compute_mda_importance_simple,
    get_mda_feature_ranking,
)

# Single Feature Importance (SFI)
from afml.feature_importance.sfi import (
    compute_sfi_importance,
    compute_sfi_for_feature_subset,
    get_sfi_feature_ranking,
)

# Orthogonal features (PCA)
from afml.feature_importance.orthogonal import (
    compute_eigenvectors,
    compute_orthogonal_features,
    compute_weighted_kendall_tau,
    compute_importance_pca_correlation,
    get_pca_feature_ranking,
)

# Synthetic data generation
from afml.feature_importance.synthetic import (
    generate_synthetic_dataset,
    generate_financial_synthetic_data,
    get_feature_types,
    analyze_importance_by_type,
    compute_importance_accuracy,
)

# Plotting utilities
from afml.feature_importance.plotting import (
    plot_feature_importance,
    plot_importance_comparison,
    plot_importance_heatmap,
    plot_importance_vs_pca,
    plot_importance_by_type,
)

__all__ = [
    # MDI functions
    "compute_mdi_importance",
    "compute_mdi_importance_clustered",
    "get_mdi_feature_ranking",
    # MDA functions
    "compute_mda_importance",
    "compute_mda_importance_simple",
    "get_mda_feature_ranking",
    # SFI functions
    "compute_sfi_importance",
    "compute_sfi_for_feature_subset",
    "get_sfi_feature_ranking",
    # Orthogonal features
    "compute_eigenvectors",
    "compute_orthogonal_features",
    "compute_weighted_kendall_tau",
    "compute_importance_pca_correlation",
    "get_pca_feature_ranking",
    # Synthetic data
    "generate_synthetic_dataset",
    "generate_financial_synthetic_data",
    "get_feature_types",
    "analyze_importance_by_type",
    "compute_importance_accuracy",
    # Plotting
    "plot_feature_importance",
    "plot_importance_comparison",
    "plot_importance_heatmap",
    "plot_importance_vs_pca",
    "plot_importance_by_type",
]
