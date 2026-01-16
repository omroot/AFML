"""
Training set purging for cross-validation.

This module provides functions to purge (remove) training observations
whose labels overlap with testing observations, preventing information
leakage in cross-validation.

Reference: AFML Chapter 7, Section 7.4.1
"""

from typing import Optional
import pandas as pd
import numpy as np


def get_train_times(
    label_times: pd.Series,
    test_times: pd.Series,
) -> pd.Series:
    """
    Purge training observations that overlap with the test set.

    Given the times of all labels and the times of test labels, returns
    the times of training labels that do not overlap with any test label.

    Parameters
    ----------
    label_times : pd.Series
        Series where index is the observation start time (t0) and
        values are the observation end times (t1). Represents all
        available observations.
    test_times : pd.Series
        Series where index is the test observation start time and
        values are the test observation end times. Represents the
        test set observations.

    Returns
    -------
    pd.Series
        Purged training times - observations that don't overlap with test.

    Notes
    -----
    An observation i with label Y_i = f[[t_{i,0}, t_{i,1}]] overlaps with
    a test observation j with Y_j = f[[t_{j,0}, t_{j,1}]] if any of these
    conditions is met:

    1. t_{j,0} <= t_{i,0} <= t_{j,1}  (train starts within test)
    2. t_{j,0} <= t_{i,1} <= t_{j,1}  (train ends within test)
    3. t_{i,0} <= t_{j,0} <= t_{j,1} <= t_{i,1}  (train envelops test)

    This function removes all such overlapping observations from the
    training set to prevent leakage.

    References
    ----------
    AFML Chapter 7, Snippet 7.1: Purging Observations in the Training Set

    Examples
    --------
    >>> import pandas as pd
    >>> # All observation times (index=start, value=end)
    >>> label_times = pd.Series(
    ...     index=pd.date_range('2020-01-01', periods=100, freq='D'),
    ...     data=pd.date_range('2020-01-10', periods=100, freq='D')
    ... )
    >>> # Test set times
    >>> test_times = pd.Series(
    ...     index=[pd.Timestamp('2020-03-01')],
    ...     data=[pd.Timestamp('2020-03-15')]
    ... )
    >>> train_times = get_train_times(label_times, test_times)
    """
    # Start with all observations as potential training data
    train_times = label_times.copy(deep=True)

    # For each test observation, remove overlapping training observations
    for test_start, test_end in test_times.items():
        # Condition 1: Training observation starts within test period
        # t_{j,0} <= t_{i,0} <= t_{j,1}
        starts_within_test = train_times[
            (test_start <= train_times.index) & (train_times.index <= test_end)
        ].index

        # Condition 2: Training observation ends within test period
        # t_{j,0} <= t_{i,1} <= t_{j,1}
        ends_within_test = train_times[
            (test_start <= train_times) & (train_times <= test_end)
        ].index

        # Condition 3: Training observation envelops test period
        # t_{i,0} <= t_{j,0} <= t_{j,1} <= t_{i,1}
        envelops_test = train_times[
            (train_times.index <= test_start) & (test_end <= train_times)
        ].index

        # Remove all overlapping observations
        overlapping = starts_within_test.union(ends_within_test).union(envelops_test)
        train_times = train_times.drop(overlapping)

    return train_times


def find_overlapping_indices(
    label_times: pd.Series,
    test_times: pd.Series,
) -> pd.Index:
    """
    Find indices of observations that overlap with the test set.

    Parameters
    ----------
    label_times : pd.Series
        Series where index is observation start time and values are end times.
    test_times : pd.Series
        Series where index is test start time and values are test end times.

    Returns
    -------
    pd.Index
        Indices of observations that overlap with any test observation.

    Examples
    --------
    >>> overlapping = find_overlapping_indices(label_times, test_times)
    >>> print(f"Found {len(overlapping)} overlapping observations")
    """
    overlapping_indices = pd.Index([])

    for test_start, test_end in test_times.items():
        # Condition 1: starts within test
        starts_within = label_times[
            (test_start <= label_times.index) & (label_times.index <= test_end)
        ].index

        # Condition 2: ends within test
        ends_within = label_times[
            (test_start <= label_times) & (label_times <= test_end)
        ].index

        # Condition 3: envelops test
        envelops = label_times[
            (label_times.index <= test_start) & (test_end <= label_times)
        ].index

        overlapping_indices = overlapping_indices.union(
            starts_within.union(ends_within).union(envelops)
        )

    return overlapping_indices


def count_overlapping_observations(
    label_times: pd.Series,
    test_times: pd.Series,
) -> int:
    """
    Count the number of observations that would be purged.

    Useful for understanding the impact of purging on training set size.

    Parameters
    ----------
    label_times : pd.Series
        All observation times.
    test_times : pd.Series
        Test observation times.

    Returns
    -------
    int
        Number of observations that overlap with test set.

    Examples
    --------
    >>> n_purged = count_overlapping_observations(label_times, test_times)
    >>> print(f"Will purge {n_purged} observations ({n_purged/len(label_times):.1%})")
    """
    overlapping = find_overlapping_indices(label_times, test_times)
    return len(overlapping)


def get_purged_train_indices(
    label_times: pd.Series,
    test_indices: np.ndarray,
    all_times: pd.Series,
) -> np.ndarray:
    """
    Get training indices after purging overlapping observations.

    This is a more efficient version for use within CV splitters where
    we work with integer indices.

    Parameters
    ----------
    label_times : pd.Series
        Series with observation start times as index and end times as values.
    test_indices : np.ndarray
        Integer indices of test observations.
    all_times : pd.Series
        Full series of times to convert between indices and timestamps.

    Returns
    -------
    np.ndarray
        Integer indices of purged training observations.
    """
    # Get all indices
    all_indices = np.arange(len(label_times))

    # Get test times
    test_start_times = label_times.index[test_indices]
    test_end_times = label_times.iloc[test_indices]
    test_times = pd.Series(test_end_times.values, index=test_start_times)

    # Find overlapping indices
    overlapping = find_overlapping_indices(label_times, test_times)

    # Convert to integer indices
    overlapping_int = np.array([
        label_times.index.get_loc(idx) for idx in overlapping
    ])

    # Remove test indices and overlapping indices from training
    train_indices = np.setdiff1d(all_indices, test_indices)
    train_indices = np.setdiff1d(train_indices, overlapping_int)

    return train_indices
