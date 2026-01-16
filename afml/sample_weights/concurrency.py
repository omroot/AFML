"""
Concurrency and uniqueness estimation for overlapping labels.

This module provides functions to compute the number of concurrent labels
at each time point and estimate the uniqueness (non-overlap) of each label.

Reference: AFML Chapter 4, Sections 4.3-4.4
"""

from typing import Optional
import pandas as pd
import numpy as np


def get_num_concurrent_events(
    close_index: pd.DatetimeIndex,
    event_end_times: pd.Series,
    molecule: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Compute the number of concurrent events per bar.

    Two labels yi and yj are concurrent at time t when both are a function
    of at least one common return. This function counts how many labels
    are active (concurrent) at each time point.

    Parameters
    ----------
    close_index : pd.DatetimeIndex
        Index of price bars (timestamps).
    event_end_times : pd.Series
        A pandas Series where:
        - Index: Event start timestamps (when features are observed)
        - Values: Event end timestamps (when label is determined, i.e., t1)
    molecule : pd.DatetimeIndex, optional
        Subset of event indices to process. If None, uses all events.
        Used for parallel processing of subsets.

    Returns
    -------
    pd.Series
        A Series indexed by close_index timestamps containing the count
        of concurrent events at each time point.

    Notes
    -----
    For each time point t, we count how many events span that time.
    An event spans time t if its start time <= t <= its end time.

    This is used to understand how much "information overlap" exists
    in the dataset, which affects the independence assumption of ML models.

    References
    ----------
    AFML Chapter 4, Snippet 4.1: Estimating the Uniqueness of a Label

    Examples
    --------
    >>> import pandas as pd
    >>> close_idx = pd.date_range('2023-01-01', periods=10, freq='D')
    >>> t1 = pd.Series(
    ...     [close_idx[2], close_idx[4], close_idx[5]],
    ...     index=[close_idx[0], close_idx[1], close_idx[3]]
    ... )
    >>> concurrent = get_num_concurrent_events(close_idx, t1)
    """
    if molecule is None:
        molecule = event_end_times.index

    # Step 1: Find events that span the period [molecule[0], molecule[-1]]
    # Fill NaN end times with the last available close index
    event_end_times_filled = event_end_times.fillna(close_index[-1])

    # Filter to events that:
    # - End at or after the first molecule timestamp
    # - Start at or before the last molecule timestamp (via .loc)
    relevant_events = event_end_times_filled[
        event_end_times_filled >= molecule[0]
    ]
    relevant_events = relevant_events.loc[
        : event_end_times_filled.loc[molecule].max()
    ]

    # Step 2: Count events spanning each bar
    # Find indices in close_index corresponding to event boundaries
    end_indices = close_index.searchsorted(
        np.array([event_end_times_filled.index[0], event_end_times_filled.max()])
    )

    # Create count series for the relevant range
    count = pd.Series(
        0,
        index=close_index[end_indices[0] : end_indices[1] + 1],
    )

    # For each event, increment count for all bars it spans
    for event_start, event_end in relevant_events.items():
        count.loc[event_start:event_end] += 1

    return count.loc[molecule[0] : event_end_times_filled.loc[molecule].max()]


def get_average_uniqueness(
    event_end_times: pd.Series,
    num_concurrent_events: pd.Series,
    molecule: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Compute the average uniqueness of each label over its lifespan.

    A label's uniqueness at time t is defined as the reciprocal of the
    number of concurrent labels at that time: u_{t,i} = 1 / c_t.
    The average uniqueness is the mean of u_{t,i} over the label's lifespan.

    Parameters
    ----------
    event_end_times : pd.Series
        A pandas Series where:
        - Index: Event start timestamps
        - Values: Event end timestamps (t1)
    num_concurrent_events : pd.Series
        Number of concurrent events at each time point, as returned by
        get_num_concurrent_events().
    molecule : pd.DatetimeIndex, optional
        Subset of event indices to process. If None, uses all events.

    Returns
    -------
    pd.Series
        A Series indexed by event start timestamps containing the average
        uniqueness score (between 0 and 1) for each event.

    Notes
    -----
    Average uniqueness can be interpreted as the reciprocal of the harmonic
    average of ct over the event's lifespan. A uniqueness of 1 means the
    label has no overlap with any other label. Lower values indicate more
    overlap.

    This metric is NOT used for forecasting (would cause information leakage)
    but for weighting samples during training.

    References
    ----------
    AFML Chapter 4, Snippet 4.2: Estimating the Average Uniqueness of a Label

    Examples
    --------
    >>> # Continuing from get_num_concurrent_events example
    >>> uniqueness = get_average_uniqueness(t1, concurrent)
    """
    if molecule is None:
        molecule = event_end_times.index

    average_uniqueness = pd.Series(index=molecule, dtype=float)

    for event_start in molecule:
        event_end = event_end_times.loc[event_start]

        if pd.isna(event_end):
            continue

        # Get concurrent event counts over this event's lifespan
        concurrent_counts = num_concurrent_events.loc[event_start:event_end]

        # Uniqueness at each time point is 1/c_t
        uniqueness_values = 1.0 / concurrent_counts

        # Average uniqueness is the mean over the lifespan
        average_uniqueness.loc[event_start] = uniqueness_values.mean()

    return average_uniqueness


def compute_sample_uniqueness(
    close_prices: pd.Series,
    event_end_times: pd.Series,
    num_threads: int = 1,
) -> pd.DataFrame:
    """
    Compute both concurrent event counts and average uniqueness for all events.

    This is a convenience function that combines get_num_concurrent_events
    and get_average_uniqueness into a single call.

    Parameters
    ----------
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
    event_end_times : pd.Series
        A pandas Series where:
        - Index: Event start timestamps
        - Values: Event end timestamps (t1)
    num_threads : int, default=1
        Number of threads for parallel processing (currently unused,
        reserved for future implementation).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'num_concurrent': Number of concurrent events at event start
        - 'average_uniqueness': Average uniqueness score for each event

    Examples
    --------
    >>> import pandas as pd
    >>> close = pd.Series(
    ...     [100, 101, 102, 103, 104],
    ...     index=pd.date_range('2023-01-01', periods=5, freq='D')
    ... )
    >>> t1 = pd.Series(
    ...     [close.index[2], close.index[4]],
    ...     index=[close.index[0], close.index[1]]
    ... )
    >>> uniqueness_df = compute_sample_uniqueness(close, t1)
    """
    # Compute concurrent events
    num_concurrent = get_num_concurrent_events(
        close_index=close_prices.index,
        event_end_times=event_end_times,
    )

    # Remove duplicate indices (keep last)
    num_concurrent = num_concurrent.loc[
        ~num_concurrent.index.duplicated(keep="last")
    ]

    # Reindex to match close prices and forward fill
    num_concurrent = num_concurrent.reindex(close_prices.index).fillna(0)

    # Compute average uniqueness
    avg_uniqueness = get_average_uniqueness(
        event_end_times=event_end_times,
        num_concurrent_events=num_concurrent,
    )

    # Create output DataFrame
    output = pd.DataFrame(index=event_end_times.index)
    output["num_concurrent"] = num_concurrent.loc[event_end_times.index]
    output["average_uniqueness"] = avg_uniqueness

    return output
