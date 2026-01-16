"""
Cross-Validation module for financial machine learning.

This module provides cross-validation tools that properly handle the
non-IID nature of financial data through purging and embargo.

Key Concepts:
- **Leakage**: Training set contains information also in testing set
- **Purging**: Remove training observations whose labels overlap with test
- **Embargo**: Remove training observations immediately after test set

Why Standard CV Fails in Finance:
1. Observations are not IID (serial correlation, overlapping labels)
2. Standard CV causes leakage through overlapping information
3. Results appear good but don't generalize to new data

Reference: AFML Chapter 7
"""

from afml.cross_validation.purging import (
    get_train_times,
    find_overlapping_indices,
    count_overlapping_observations,
    get_purged_train_indices,
)
from afml.cross_validation.embargo import (
    get_embargo_times,
    compute_embargo_indices,
    apply_embargo_to_test_times,
    get_embargo_mask,
    get_embargoed_train_indices,
)
from afml.cross_validation.purged_kfold import (
    PurgedKFold,
    PurgedWalkForwardCV,
)
from afml.cross_validation.scoring import (
    cv_score,
    cv_score_with_predictions,
    compare_cv_methods,
)

__all__ = [
    # Purging functions
    "get_train_times",
    "find_overlapping_indices",
    "count_overlapping_observations",
    "get_purged_train_indices",
    # Embargo functions
    "get_embargo_times",
    "compute_embargo_indices",
    "apply_embargo_to_test_times",
    "get_embargo_mask",
    "get_embargoed_train_indices",
    # Cross-validation classes
    "PurgedKFold",
    "PurgedWalkForwardCV",
    # Scoring functions
    "cv_score",
    "cv_score_with_predictions",
    "compare_cv_methods",
]
