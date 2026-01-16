"""
Sample weights module for financial machine learning.

This module provides functions to address the non-IID nature of financial
observations through sample weighting, uniqueness estimation, and sequential
bootstrap sampling.

Reference: AFML Chapter 4
"""

from afml.sample_weights.concurrency import (
    get_num_concurrent_events,
    get_average_uniqueness,
    compute_sample_uniqueness,
)
from afml.sample_weights.bootstrap import (
    get_indicator_matrix,
    get_average_uniqueness_from_matrix,
    sequential_bootstrap,
)
from afml.sample_weights.attribution import (
    get_sample_weights_by_return,
    get_sample_weights_by_uniqueness,
    compute_sample_weights,
)
from afml.sample_weights.time_decay import (
    apply_time_decay,
    apply_exponential_decay,
    compute_decayed_sample_weights,
)
from afml.sample_weights.bootstrap import (
    compare_bootstrap_methods,
)

__all__ = [
    # Concurrency functions
    "get_num_concurrent_events",
    "get_average_uniqueness",
    "compute_sample_uniqueness",
    # Bootstrap functions
    "get_indicator_matrix",
    "get_average_uniqueness_from_matrix",
    "sequential_bootstrap",
    "compare_bootstrap_methods",
    # Attribution functions
    "get_sample_weights_by_return",
    "get_sample_weights_by_uniqueness",
    "compute_sample_weights",
    # Time decay functions
    "apply_time_decay",
    "apply_exponential_decay",
    "compute_decayed_sample_weights",
]
