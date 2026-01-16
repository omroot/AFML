"""
Averaging active bets for reduced turnover.

This module provides functions to compute the average of active bet signals,
which helps reduce portfolio turnover and transaction costs.

Reference: AFML Chapter 10, Section 10.4, Snippet 10.2
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def compute_average_active_signals(
    signals: pd.Series,
    events: pd.DataFrame,
    num_threads: int = 1,
) -> pd.Series:
    """
    Compute time-weighted average of active signals.

    At any point in time, multiple bets may be active (overlapping holding
    periods). This function computes the average signal across all active
    bets at each timestamp.

    Parameters
    ----------
    signals : pd.Series
        Series of bet signals indexed by event start time.
        Values should be in range [-1, 1].
    events : pd.DataFrame
        DataFrame with event information. Must contain:
        - 't1': end time of each event (vertical barrier)
        Index should be the start time of each event.
    num_threads : int, default=1
        Number of threads for parallel computation.
        Currently not implemented (reserved for future use).

    Returns
    -------
    pd.Series
        Average signal at each unique timestamp, indexed by time.
        The index includes all event start and end times.

    Notes
    -----
    For each timestamp t, the average signal is:
        avg_signal[t] = sum(signals[i] for active bets i) / count(active bets)

    A bet is considered active at time t if:
        event_start[i] <= t < event_end[i]

    This averaging approach has several benefits:
    1. Reduces portfolio turnover and transaction costs
    2. Smooths position changes over time
    3. Provides more stable position sizing

    References
    ----------
    AFML Chapter 10, Section 10.4, Snippet 10.2

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample events with overlapping periods
    >>> events = pd.DataFrame({
    ...     't1': pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05'])
    ... }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    >>> signals = pd.Series([0.5, -0.3, 0.8], index=events.index)
    >>> avg_signals = compute_average_active_signals(signals, events)
    """
    # Validate inputs
    if 't1' not in events.columns:
        raise ValueError("events DataFrame must contain 't1' column")

    if len(signals) == 0:
        return pd.Series(dtype=float)

    # Get all unique timestamps (start and end times)
    all_times = set(signals.index)
    all_times.update(events['t1'].dropna())
    all_times = sorted(all_times)

    # Compute average signal at each timestamp
    avg_signals = pd.Series(index=all_times, dtype=float)

    for t in all_times:
        # Find active bets at time t
        # A bet is active if: start_time <= t < end_time
        active_mask = (signals.index <= t) & (events['t1'] > t)
        active_signals = signals[active_mask]

        if len(active_signals) > 0:
            avg_signals[t] = active_signals.mean()
        else:
            avg_signals[t] = 0.0

    return avg_signals


def get_signal_at_time(
    signals: pd.Series,
    events: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> float:
    """
    Get the average signal at a specific timestamp.

    Parameters
    ----------
    signals : pd.Series
        Series of bet signals indexed by event start time.
    events : pd.DataFrame
        DataFrame with 't1' column for event end times.
    timestamp : pd.Timestamp
        The timestamp at which to compute the average signal.

    Returns
    -------
    float
        Average signal at the given timestamp.

    Examples
    --------
    >>> signal = get_signal_at_time(signals, events, pd.Timestamp('2023-01-02'))
    """
    if 't1' not in events.columns:
        raise ValueError("events DataFrame must contain 't1' column")

    # Find active bets at the given timestamp
    active_mask = (signals.index <= timestamp) & (events['t1'] > timestamp)
    active_signals = signals[active_mask]

    if len(active_signals) > 0:
        return float(active_signals.mean())
    else:
        return 0.0


def compute_signal_with_decay(
    signals: pd.Series,
    events: pd.DataFrame,
    decay_factor: float = 1.0,
) -> pd.Series:
    """
    Compute average signals with time decay weighting.

    More recent signals receive higher weight in the average.

    Parameters
    ----------
    signals : pd.Series
        Series of bet signals indexed by event start time.
    events : pd.DataFrame
        DataFrame with 't1' column for event end times.
    decay_factor : float, default=1.0
        Decay rate. Higher values give more weight to recent signals.
        - decay_factor=0: equal weighting (same as compute_average_active_signals)
        - decay_factor>0: exponential decay favoring recent signals

    Returns
    -------
    pd.Series
        Decay-weighted average signal at each timestamp.

    Notes
    -----
    Weight for signal i at time t:
        w[i] = exp(-decay_factor * (t - start_time[i]) / (end_time[i] - start_time[i]))

    Examples
    --------
    >>> avg_signals = compute_signal_with_decay(signals, events, decay_factor=0.5)
    """
    if 't1' not in events.columns:
        raise ValueError("events DataFrame must contain 't1' column")

    if len(signals) == 0:
        return pd.Series(dtype=float)

    # Get all unique timestamps
    all_times = set(signals.index)
    all_times.update(events['t1'].dropna())
    all_times = sorted(all_times)

    # Compute weighted average at each timestamp
    avg_signals = pd.Series(index=all_times, dtype=float)

    for t in all_times:
        # Find active bets
        active_mask = (signals.index <= t) & (events['t1'] > t)

        if not active_mask.any():
            avg_signals[t] = 0.0
            continue

        active_indices = signals.index[active_mask]
        active_signals = signals[active_mask]
        active_end_times = events.loc[active_indices, 't1']

        # Compute time-based weights
        if decay_factor == 0:
            weights = np.ones(len(active_signals))
        else:
            # Compute fraction of holding period elapsed
            # Convert to Series of timedeltas and extract total seconds
            duration_timedeltas = pd.Series(active_end_times.values - active_indices)
            durations = duration_timedeltas.dt.total_seconds().values
            elapsed_timedeltas = pd.Series(t - active_indices)
            elapsed = elapsed_timedeltas.dt.total_seconds().values

            # Avoid division by zero
            durations = np.where(durations == 0, 1, durations)
            fraction_elapsed = elapsed / durations

            # Exponential decay weight
            weights = np.exp(-decay_factor * fraction_elapsed)

        # Weighted average
        avg_signals[t] = np.average(active_signals, weights=weights)

    return avg_signals


def resample_signals(
    signals: pd.Series,
    events: pd.DataFrame,
    freq: str = 'D',
    method: str = 'mean',
) -> pd.Series:
    """
    Resample average active signals to a regular frequency.

    Parameters
    ----------
    signals : pd.Series
        Series of bet signals indexed by event start time.
    events : pd.DataFrame
        DataFrame with 't1' column for event end times.
    freq : str, default='D'
        Resampling frequency (e.g., 'D' for daily, 'H' for hourly).
    method : str, default='mean'
        Aggregation method: 'mean', 'last', 'first', 'sum'.

    Returns
    -------
    pd.Series
        Resampled average signals at regular intervals.

    Examples
    --------
    >>> # Resample to daily frequency
    >>> daily_signals = resample_signals(signals, events, freq='D')
    """
    # First compute average active signals
    avg_signals = compute_average_active_signals(signals, events)

    if len(avg_signals) == 0:
        return pd.Series(dtype=float)

    # Resample to regular frequency
    if method == 'mean':
        resampled = avg_signals.resample(freq).mean()
    elif method == 'last':
        resampled = avg_signals.resample(freq).last()
    elif method == 'first':
        resampled = avg_signals.resample(freq).first()
    elif method == 'sum':
        resampled = avg_signals.resample(freq).sum()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean', 'last', 'first', or 'sum'.")

    return resampled.dropna()


def compute_turnover(
    signals: pd.Series,
    events: pd.DataFrame,
) -> pd.Series:
    """
    Compute signal turnover (change in position) over time.

    Parameters
    ----------
    signals : pd.Series
        Series of bet signals indexed by event start time.
    events : pd.DataFrame
        DataFrame with 't1' column for event end times.

    Returns
    -------
    pd.Series
        Absolute change in average signal at each timestamp.
        Higher values indicate more trading activity.

    Examples
    --------
    >>> turnover = compute_turnover(signals, events)
    >>> print(f"Average turnover: {turnover.mean():.4f}")
    """
    avg_signals = compute_average_active_signals(signals, events)
    turnover = avg_signals.diff().abs()
    return turnover.dropna()
