"""
Embargo period for cross-validation.

This module provides functions to implement an embargo period that removes
training observations immediately following the test set. This prevents
leakage through serial correlation in features.

Reference: AFML Chapter 7, Section 7.4.2
"""

from typing import Union
import pandas as pd
import numpy as np


def get_embargo_times(
    times: pd.DatetimeIndex,
    embargo_pct: float,
) -> pd.Series:
    """
    Compute the embargo end time for each observation.

    For each observation, computes the time after which training can resume
    following an embargo period.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Datetime index of all observations.
    embargo_pct : float
        Embargo period as a fraction of total observations.
        A value of 0.01 means 1% of observations after each test
        observation will be embargoed.

    Returns
    -------
    pd.Series
        Series where index is the original observation time and value
        is the embargo end time. For observations near the end where
        the full embargo can't be applied, uses the last available time.

    Notes
    -----
    The embargo addresses leakage through serial correlation. Even after
    purging overlapping labels, features X_t and X_{t+1} may be correlated.
    By adding an embargo period h after the test set, we prevent training
    on observations whose features are too similar to test features.

    A typical value is embargo_pct ≈ 0.01 (1% of observations).

    References
    ----------
    AFML Chapter 7, Snippet 7.2: Embargo on Training Observations

    Examples
    --------
    >>> times = pd.date_range('2020-01-01', periods=1000, freq='D')
    >>> embargo_times = get_embargo_times(times, embargo_pct=0.01)
    >>> # For observation at index i, training can only use observations
    >>> # where their start time > embargo_times[i]
    """
    # Compute the embargo step size (number of observations)
    num_observations = len(times)
    embargo_step = int(num_observations * embargo_pct)

    if embargo_step == 0:
        # No embargo - each time maps to itself
        embargo_times = pd.Series(times, index=times)
    else:
        # Shift times by embargo_step
        # For early observations, map to the time embargo_step ahead
        # For late observations (within embargo_step of end), map to last time
        embargo_times = pd.Series(
            times[embargo_step:].tolist() + [times[-1]] * embargo_step,
            index=times
        )

    return embargo_times


def compute_embargo_indices(
    num_observations: int,
    embargo_pct: float,
) -> int:
    """
    Compute the number of observations to embargo.

    Parameters
    ----------
    num_observations : int
        Total number of observations.
    embargo_pct : float
        Embargo percentage (0.01 = 1%).

    Returns
    -------
    int
        Number of observations to skip after test set.

    Examples
    --------
    >>> n_embargo = compute_embargo_indices(1000, 0.01)
    >>> print(f"Embargo {n_embargo} observations after each test set")
    """
    return int(num_observations * embargo_pct)


def apply_embargo_to_test_times(
    test_times: pd.Series,
    embargo_times: pd.Series,
) -> pd.Series:
    """
    Extend test end times to include the embargo period.

    This is used before purging to ensure the embargo period is also
    considered as "overlapping" with the test set.

    Parameters
    ----------
    test_times : pd.Series
        Original test times (index=start, value=end).
    embargo_times : pd.Series
        Embargo end times from get_embargo_times().

    Returns
    -------
    pd.Series
        Extended test times that include embargo period.

    Examples
    --------
    >>> # Original test ends at 2020-03-15
    >>> # With embargo, we extend it further
    >>> extended_test = apply_embargo_to_test_times(test_times, embargo_times)
    """
    extended_times = test_times.copy()

    for test_start, test_end in test_times.items():
        # Find the embargo end time for the test end
        if test_end in embargo_times.index:
            embargo_end = embargo_times.loc[test_end]
            extended_times.loc[test_start] = embargo_end
        elif len(embargo_times) > 0:
            # If exact match not found, find nearest
            nearest_idx = embargo_times.index.get_indexer([test_end], method='nearest')[0]
            if nearest_idx >= 0:
                nearest_time = embargo_times.index[nearest_idx]
                extended_times.loc[test_start] = embargo_times.loc[nearest_time]

    return extended_times


def get_embargo_mask(
    all_indices: np.ndarray,
    test_end_index: int,
    embargo_size: int,
) -> np.ndarray:
    """
    Get mask of indices that fall within the embargo period.

    Parameters
    ----------
    all_indices : np.ndarray
        All observation indices.
    test_end_index : int
        Index where the test set ends.
    embargo_size : int
        Number of observations to embargo.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates embargoed indices.

    Examples
    --------
    >>> mask = get_embargo_mask(np.arange(100), test_end_index=50, embargo_size=5)
    >>> embargoed_indices = np.where(mask)[0]  # [51, 52, 53, 54, 55]
    """
    embargo_start = test_end_index + 1
    embargo_end = min(test_end_index + embargo_size + 1, len(all_indices))

    mask = np.zeros(len(all_indices), dtype=bool)
    mask[embargo_start:embargo_end] = True

    return mask


def get_embargoed_train_indices(
    all_indices: np.ndarray,
    test_indices: np.ndarray,
    embargo_size: int,
) -> np.ndarray:
    """
    Get training indices after removing test and embargoed observations.

    Parameters
    ----------
    all_indices : np.ndarray
        All observation indices.
    test_indices : np.ndarray
        Indices of test observations.
    embargo_size : int
        Number of observations to embargo after test set.

    Returns
    -------
    np.ndarray
        Training indices with test and embargoed observations removed.

    Examples
    --------
    >>> train_idx = get_embargoed_train_indices(
    ...     all_indices=np.arange(100),
    ...     test_indices=np.array([40, 41, 42, 43, 44]),
    ...     embargo_size=5
    ... )
    """
    # Start with all indices except test
    train_mask = np.ones(len(all_indices), dtype=bool)
    train_mask[test_indices] = False

    # Find embargo indices (after test set)
    if len(test_indices) > 0:
        test_end = test_indices.max()
        embargo_mask = get_embargo_mask(all_indices, test_end, embargo_size)
        train_mask &= ~embargo_mask

    return all_indices[train_mask]
