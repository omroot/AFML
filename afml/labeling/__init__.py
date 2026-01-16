"""
Labeling module for financial machine learning.

This module provides functions for labeling financial data for supervised learning,
including the triple-barrier method, meta-labeling, and dynamic volatility thresholds.

Based on Chapter 3 of "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

from afml.labeling.volatility import get_daily_volatility
from afml.labeling.barriers import (
    apply_triple_barrier,
    get_events,
    add_vertical_barrier,
)
from afml.labeling.labels import (
    get_labels,
    get_labels_side_and_size,
)
from afml.labeling.utils import drop_rare_labels

__all__ = [
    # Volatility
    "get_daily_volatility",
    # Barriers
    "apply_triple_barrier",
    "get_events",
    "add_vertical_barrier",
    # Labels
    "get_labels",
    "get_labels_side_and_size",
    # Utils
    "drop_rare_labels",
]
