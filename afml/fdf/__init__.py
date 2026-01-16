"""
Fractionally Differentiated Features module for financial machine learning.

This module provides functions for fractional differentiation of time series,
which addresses the stationarity vs. memory dilemma in financial ML.

The key insight: standard integer differencing (d=1) makes series stationary
but destroys memory/predictive signal. Fractional differentiation with d < 1
can achieve stationarity while preserving more memory.

Key Concepts:
- **Stationarity**: Required for most ML models (constant mean/variance)
- **Memory**: Historical information useful for prediction
- **FFD**: Fixed-width window fracdiff - practical implementation

Reference: AFML Chapter 5
"""

from afml.fdf.weights import (
    compute_weights,
    compute_weights_ffd,
    get_weight_convergence,
)
from afml.fdf.fracdiff import (
    fracdiff_expanding,
    fracdiff_ffd,
    fracdiff,
    get_ffd_window_size,
)
from afml.fdf.stationarity import (
    adf_test,
    find_minimum_d,
    analyze_stationarity_memory_tradeoff,
)

__all__ = [
    # Weight computation
    "compute_weights",
    "compute_weights_ffd",
    "get_weight_convergence",
    # Fractional differentiation
    "fracdiff_expanding",
    "fracdiff_ffd",
    "fracdiff",
    "get_ffd_window_size",
    # Stationarity analysis
    "adf_test",
    "find_minimum_d",
    "analyze_stationarity_memory_tradeoff",
]
