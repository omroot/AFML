"""
Mean Decrease Impurity (MDI) feature importance.

This module provides MDI feature importance calculation for tree-based
classifiers. MDI measures how much each feature contributes to decreasing
impurity across all trees in an ensemble.

Reference: AFML Chapter 8, Section 8.3.1, Snippet 8.2
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


def compute_mdi_importance(
    fitted_classifier: Union[RandomForestClassifier, BaggingClassifier],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute Mean Decrease Impurity (MDI) feature importance.

    MDI is a fast, in-sample feature importance method specific to
    tree-based classifiers. It measures how much each feature contributes
    to decreasing impurity (e.g., Gini or entropy) across all decision trees.

    Parameters
    ----------
    fitted_classifier : RandomForestClassifier or BaggingClassifier
        A fitted tree-based ensemble classifier. Must have `estimators_`
        attribute containing individual trees.
    feature_names : List[str]
        Names of features corresponding to columns in the training data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'mean': Mean importance across all trees
        - 'std': Standard deviation of importance

    Notes
    -----
    Important considerations for MDI:

    1. **Masking effects**: Set `max_features=1` when training to ensure
       every feature gets a chance to reduce impurity at some level.

    2. **Zero importances**: Features with 0 importance were not randomly
       selected. These are replaced with NaN before averaging.

    3. **In-sample only**: MDI is computed from training data. Every feature
       will have some importance, even noise features.

    4. **Substitution effects**: Correlated features will have their
       importance diluted (split between them).

    5. **Bounded output**: Importances sum to 1 and are bounded [0, 1].

    References
    ----------
    AFML Chapter 8, Snippet 8.2: MDI Feature Importance

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier(n_estimators=100, max_features=1)
    >>> clf.fit(X_train, y_train)
    >>> importance = compute_mdi_importance(clf, feature_names)
    >>> print(importance.sort_values('mean', ascending=False).head())
    """
    # Extract feature importances from each tree in the ensemble
    importance_per_tree = {}

    for tree_idx, tree in enumerate(fitted_classifier.estimators_):
        # Handle both RandomForest (trees directly) and Bagging (trees in base_estimator)
        if hasattr(tree, 'feature_importances_'):
            tree_importance = tree.feature_importances_
        elif hasattr(tree, 'tree_'):
            # For individual DecisionTree wrapped in Bagging
            tree_importance = tree.feature_importances_
        else:
            raise ValueError(
                f"Tree {tree_idx} does not have feature_importances_ attribute"
            )

        importance_per_tree[tree_idx] = tree_importance

    # Create DataFrame with importance per tree
    importance_df = pd.DataFrame.from_dict(importance_per_tree, orient='index')
    importance_df.columns = feature_names

    # Replace 0 with NaN (0 means feature wasn't selected, not that it's unimportant)
    # This is critical when using max_features=1
    importance_df = importance_df.replace(0, np.nan)

    # Compute mean and std across trees
    # Use ddof=1 for sample std, multiply by sqrt(n) for std of mean
    mean_importance = importance_df.mean(axis=0)
    std_importance = importance_df.std(axis=0) * (importance_df.shape[0] ** -0.5)

    # Combine into result DataFrame
    result = pd.concat(
        {'mean': mean_importance, 'std': std_importance},
        axis=1
    )

    # Normalize so importances sum to 1
    result['mean'] = result['mean'] / result['mean'].sum()

    return result


def compute_mdi_importance_clustered(
    fitted_classifier: Union[RandomForestClassifier, BaggingClassifier],
    feature_names: List[str],
    cluster_labels: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute MDI importance with optional clustering to address substitution effects.

    When features are grouped into clusters, the importance of each cluster
    is computed by summing the importances of its constituent features.
    This can help identify groups of related features.

    Parameters
    ----------
    fitted_classifier : RandomForestClassifier or BaggingClassifier
        A fitted tree-based ensemble classifier.
    feature_names : List[str]
        Names of features.
    cluster_labels : pd.Series, optional
        Series mapping feature names to cluster labels. If None, returns
        standard per-feature importance.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean and std importance, either per-feature or per-cluster.

    Examples
    --------
    >>> # Group features by type
    >>> clusters = pd.Series({
    ...     'price': 'price_features',
    ...     'volume': 'volume_features',
    ...     'ma_10': 'price_features',
    ...     'vol_ma': 'volume_features',
    ... })
    >>> cluster_importance = compute_mdi_importance_clustered(clf, features, clusters)
    """
    # Get per-feature importance
    feature_importance = compute_mdi_importance(fitted_classifier, feature_names)

    if cluster_labels is None:
        return feature_importance

    # Aggregate by cluster
    feature_importance['cluster'] = feature_importance.index.map(cluster_labels)

    cluster_importance = feature_importance.groupby('cluster').agg({
        'mean': 'sum',
        'std': lambda x: np.sqrt((x ** 2).sum())  # Propagate uncertainty
    })

    return cluster_importance


def get_mdi_feature_ranking(
    fitted_classifier: Union[RandomForestClassifier, BaggingClassifier],
    feature_names: List[str],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get ranked list of features by MDI importance.

    Parameters
    ----------
    fitted_classifier : RandomForestClassifier or BaggingClassifier
        A fitted tree-based ensemble classifier.
    feature_names : List[str]
        Names of features.
    top_n : int, optional
        Number of top features to return. If None, returns all features.

    Returns
    -------
    pd.DataFrame
        Ranked features with mean importance and standard deviation.

    Examples
    --------
    >>> ranking = get_mdi_feature_ranking(clf, feature_names, top_n=10)
    >>> print("Top 10 features by MDI:")
    >>> print(ranking)
    """
    importance = compute_mdi_importance(fitted_classifier, feature_names)
    importance = importance.sort_values('mean', ascending=False)

    if top_n is not None:
        importance = importance.head(top_n)

    # Add rank column
    importance['rank'] = range(1, len(importance) + 1)

    return importance
