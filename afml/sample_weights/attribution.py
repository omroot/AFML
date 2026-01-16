"""
Sample weight attribution based on returns and uniqueness.

This module provides functions to compute sample weights that account for
both the uniqueness of observations and the magnitude of returns.

Reference: AFML Chapter 4, Section 4.6
"""

from typing import Optional
import pandas as pd
import numpy as np


def get_sample_weights_by_return(
    event_end_times: pd.Series,
    num_concurrent_events: pd.Series,
    close_prices: pd.Series,
    molecule: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Compute sample weights based on absolute return attribution.

    Weights observations as a function of the absolute log returns that
    can be attributed uniquely to each event. Returns are divided by
    concurrency to attribute them fairly across overlapping events.

    Parameters
    ----------
    event_end_times : pd.Series
        A pandas Series where:
        - Index: Event start timestamps (t0)
        - Values: Event end timestamps (t1)
    num_concurrent_events : pd.Series
        Number of concurrent events at each time point.
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
    molecule : pd.DatetimeIndex, optional
        Subset of event indices to process. If None, uses all events.

    Returns
    -------
    pd.Series
        Sample weights indexed by event start timestamps, scaled so that
        the sum equals the number of observations.

    Notes
    -----
    The weight for event i is computed as:
        w_i = |sum_{t=t0}^{t1} (r_t / c_t)|

    where r_t is the log return at time t and c_t is the concurrency.

    This method gives higher weights to:
    1. Events with larger absolute returns (more informative)
    2. Events with lower overlap (more unique information)

    **Warning**: This method does not work well with a "neutral" (zero)
    label case. For neutral labels, lower returns should have higher
    weights. Consider dropping neutral cases or using a different
    weighting scheme.

    References
    ----------
    AFML Chapter 4, Snippet 4.10: Determination of Sample Weight by
    Absolute Return Attribution

    Examples
    --------
    >>> import pandas as pd
    >>> close = pd.Series(
    ...     [100, 101, 102, 101, 103],
    ...     index=pd.date_range('2023-01-01', periods=5, freq='D')
    ... )
    >>> t1 = pd.Series(
    ...     [close.index[2], close.index[4]],
    ...     index=[close.index[0], close.index[1]]
    ... )
    >>> # First compute concurrent events
    >>> from afml.sample_weights import get_num_concurrent_events
    >>> concurrent = get_num_concurrent_events(close.index, t1)
    >>> weights = get_sample_weights_by_return(t1, concurrent, close)
    """
    if molecule is None:
        molecule = event_end_times.index

    # Compute log returns (additive)
    log_returns = np.log(close_prices).diff()

    # Initialize weights
    weights = pd.Series(index=molecule, dtype=float)

    for event_start in molecule:
        event_end = event_end_times.loc[event_start]

        if pd.isna(event_end):
            weights.loc[event_start] = 0.0
            continue

        # Get returns and concurrency over event lifespan
        event_returns = log_returns.loc[event_start:event_end]
        event_concurrency = num_concurrent_events.loc[event_start:event_end]

        # Attributed return = return / concurrency at each time point
        attributed_returns = event_returns / event_concurrency

        # Weight is absolute sum of attributed returns
        weights.loc[event_start] = attributed_returns.sum()

    # Take absolute value
    weights = weights.abs()

    # Scale weights to sum to number of observations
    num_observations = weights.shape[0]
    weight_sum = weights.sum()

    if weight_sum > 0:
        weights = weights * num_observations / weight_sum

    return weights


def get_sample_weights_by_uniqueness(
    event_end_times: pd.Series,
    num_concurrent_events: pd.Series,
    molecule: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Compute sample weights based purely on uniqueness (average 1/concurrency).

    This is a simpler alternative to return attribution that only considers
    how unique each observation is, without factoring in return magnitude.

    Parameters
    ----------
    event_end_times : pd.Series
        A pandas Series where:
        - Index: Event start timestamps (t0)
        - Values: Event end timestamps (t1)
    num_concurrent_events : pd.Series
        Number of concurrent events at each time point.
    molecule : pd.DatetimeIndex, optional
        Subset of event indices to process. If None, uses all events.

    Returns
    -------
    pd.Series
        Sample weights indexed by event start timestamps, scaled so that
        the sum equals the number of observations.

    Notes
    -----
    The weight for event i is the average uniqueness over its lifespan:
        w_i = mean_{t=t0}^{t1} (1 / c_t)

    Use this method when:
    1. You want to weight only by uniqueness, not return magnitude
    2. You have labels that include a "neutral" case
    3. You want a simpler, more robust weighting scheme

    Examples
    --------
    >>> weights = get_sample_weights_by_uniqueness(t1, concurrent)
    """
    if molecule is None:
        molecule = event_end_times.index

    # Initialize weights
    weights = pd.Series(index=molecule, dtype=float)

    for event_start in molecule:
        event_end = event_end_times.loc[event_start]

        if pd.isna(event_end):
            weights.loc[event_start] = 0.0
            continue

        # Get concurrency over event lifespan
        event_concurrency = num_concurrent_events.loc[event_start:event_end]

        # Weight is average uniqueness (1/concurrency)
        uniqueness = 1.0 / event_concurrency
        weights.loc[event_start] = uniqueness.mean()

    # Scale weights to sum to number of observations
    num_observations = weights.shape[0]
    weight_sum = weights.sum()

    if weight_sum > 0:
        weights = weights * num_observations / weight_sum

    return weights


def compute_sample_weights(
    event_end_times: pd.Series,
    close_prices: pd.Series,
    use_returns: bool = True,
    num_threads: int = 1,
) -> pd.Series:
    """
    Convenience function to compute sample weights in a single call.

    Handles the computation of concurrent events and applies the chosen
    weighting method (return attribution or uniqueness only).

    Parameters
    ----------
    event_end_times : pd.Series
        A pandas Series where:
        - Index: Event start timestamps (t0)
        - Values: Event end timestamps (t1)
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
    use_returns : bool, default=True
        If True, weight by return attribution (recommended for binary labels).
        If False, weight by uniqueness only (recommended when neutral label exists).
    num_threads : int, default=1
        Number of threads for parallel processing (reserved for future use).

    Returns
    -------
    pd.Series
        Sample weights indexed by event start timestamps.

    Examples
    --------
    >>> weights = compute_sample_weights(t1, close, use_returns=True)
    >>> # Use weights in sklearn with sample_weight parameter
    >>> model.fit(X, y, sample_weight=weights)
    """
    from afml.sample_weights.concurrency import get_num_concurrent_events

    # Compute concurrent events
    num_concurrent = get_num_concurrent_events(
        close_index=close_prices.index,
        event_end_times=event_end_times,
    )

    # Handle duplicates and reindex
    num_concurrent = num_concurrent.loc[
        ~num_concurrent.index.duplicated(keep="last")
    ]
    num_concurrent = num_concurrent.reindex(close_prices.index).fillna(0)

    # Compute weights using chosen method
    if use_returns:
        weights = get_sample_weights_by_return(
            event_end_times=event_end_times,
            num_concurrent_events=num_concurrent,
            close_prices=close_prices,
        )
    else:
        weights = get_sample_weights_by_uniqueness(
            event_end_times=event_end_times,
            num_concurrent_events=num_concurrent,
        )

    return weights
