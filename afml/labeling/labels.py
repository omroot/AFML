"""
Label generation functions for supervised learning in financial ML.

This module provides functions to generate labels from barrier touch events,
supporting both standard labeling (learning side and size) and meta-labeling
(learning size given a known side from a primary model).

Reference: AFML Chapter 3, Sections 3.5-3.7
"""

from typing import Optional
import pandas as pd
import numpy as np


def get_labels_side_and_size(
    events: pd.DataFrame,
    close_prices: pd.Series,
    label_vertical_barrier_as_zero: bool = False,
) -> pd.DataFrame:
    """
    Generate labels for learning both side and size of bets.

    This function labels observations based on the return realized at the time
    of the first barrier touch. Use this when you don't have an underlying model
    to set the side of the position (long or short).

    Parameters
    ----------
    events : pd.DataFrame
        A DataFrame with event information, containing:
        - Index: Event start timestamps
        - 't1': Event end timestamps (first barrier touch times)
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
    label_vertical_barrier_as_zero : bool, default=False
        If True, label observations as 0 when the vertical barrier is touched
        first (i.e., when the index equals t1). If False, label based on the
        sign of the return.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'ret': Return realized at the time of first barrier touch
        - 'bin': Label in {-1, 0, 1} based on the sign of return

    Notes
    -----
    Labels are assigned as follows:
    - bin = 1: Positive return (upper barrier or positive vertical)
    - bin = -1: Negative return (lower barrier or negative vertical)
    - bin = 0: Zero return (or vertical barrier if label_vertical_barrier_as_zero=True)

    When learning both side and size, horizontal barriers must be symmetric
    since we cannot distinguish between profit-taking and stop-loss without
    knowing the position side.

    Examples
    --------
    >>> import pandas as pd
    >>> events = pd.DataFrame({
    ...     't1': pd.to_datetime(['2023-01-03', '2023-01-04'])
    ... }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
    >>> close = pd.Series([100, 102, 98, 105, 95],
    ...                   index=pd.date_range('2023-01-01', periods=5, freq='D'))
    >>> labels = get_labels_side_and_size(events, close)

    References
    ----------
    AFML Chapter 3, Snippet 3.5: Labeling for Side and Size
    """
    # Step 1: Filter out events with missing end times
    valid_events = events.dropna(subset=["t1"])

    if len(valid_events) == 0:
        return pd.DataFrame(columns=["ret", "bin"])

    # Step 2: Get prices aligned with event times
    # Union of start times and end times
    all_timestamps = valid_events.index.union(valid_events["t1"].values)
    all_timestamps = all_timestamps.drop_duplicates()

    # Reindex close prices to these timestamps (forward fill for any gaps)
    aligned_prices = close_prices.reindex(all_timestamps, method="bfill")

    # Step 3: Compute returns
    output = pd.DataFrame(index=valid_events.index)

    start_prices = aligned_prices.loc[valid_events.index]
    end_prices = aligned_prices.loc[valid_events["t1"].values].values

    output["ret"] = end_prices / start_prices.values - 1

    # Step 4: Generate labels based on return sign
    output["bin"] = np.sign(output["ret"])

    # Step 5: Optionally label vertical barrier touches as 0
    if label_vertical_barrier_as_zero:
        # Find events where the vertical barrier was touched
        # (i.e., event end time equals start time in the original events)
        vertical_touch_mask = valid_events.index.isin(valid_events["t1"].values)
        output.loc[vertical_touch_mask, "bin"] = 0

    return output


def get_labels(
    events: pd.DataFrame,
    close_prices: pd.Series,
) -> pd.DataFrame:
    """
    Generate labels with support for meta-labeling.

    This function computes labels that support both standard labeling and
    meta-labeling. When the 'side' column is present in events (meta-labeling),
    labels are binary {0, 1} indicating whether to take the bet or not.
    When 'side' is absent, labels are ternary {-1, 0, 1} based on return sign.

    Parameters
    ----------
    events : pd.DataFrame
        A DataFrame with event information, containing:
        - Index: Event start timestamps
        - 't1': Event end timestamps (first barrier touch times)
        - 'trgt': Target return (optional)
        - 'side': Position side from primary model (optional, for meta-labeling)

    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'ret': Return realized at the time of first barrier touch
                 (adjusted by side if meta-labeling)
        - 'bin': Label. For meta-labeling: {0, 1}. Otherwise: {-1, 0, 1}.

    Notes
    -----
    **Standard Labeling (no 'side' column)**:
    - bin = 1: Positive return
    - bin = -1: Negative return
    - bin = 0: Zero return

    **Meta-Labeling (with 'side' column)**:
    - bin = 1: Profitable bet (return * side > 0) - take the bet
    - bin = 0: Unprofitable bet (return * side <= 0) - pass on the bet

    The key insight of meta-labeling is that a secondary ML model learns
    whether to act on a primary model's prediction, rather than learning
    the direction itself. This can significantly improve F1 scores.

    Examples
    --------
    Standard labeling:
    >>> events = pd.DataFrame({
    ...     't1': pd.to_datetime(['2023-01-03'])
    ... }, index=pd.to_datetime(['2023-01-01']))
    >>> labels = get_labels(events, close_prices)

    Meta-labeling (with side from primary model):
    >>> events = pd.DataFrame({
    ...     't1': pd.to_datetime(['2023-01-03']),
    ...     'side': [1]  # Primary model says "long"
    ... }, index=pd.to_datetime(['2023-01-01']))
    >>> labels = get_labels(events, close_prices)

    References
    ----------
    AFML Chapter 3, Snippet 3.7: Expanding getBins for Meta-Labeling
    """
    # Step 1: Filter out events with missing end times
    valid_events = events.dropna(subset=["t1"])

    if len(valid_events) == 0:
        return pd.DataFrame(columns=["ret", "bin"])

    # Step 2: Get prices aligned with event times
    all_timestamps = valid_events.index.union(valid_events["t1"].values)
    all_timestamps = all_timestamps.drop_duplicates()
    aligned_prices = close_prices.reindex(all_timestamps, method="bfill")

    # Step 3: Compute returns
    output = pd.DataFrame(index=valid_events.index)

    start_prices = aligned_prices.loc[valid_events.index]
    end_prices = aligned_prices.loc[valid_events["t1"].values].values

    output["ret"] = end_prices / start_prices.values - 1

    # Step 4: Apply meta-labeling adjustment if side is provided
    is_meta_labeling = "side" in valid_events.columns

    if is_meta_labeling:
        # Adjust returns by position side
        # Positive adjusted return means the primary model's call was correct
        output["ret"] = output["ret"] * valid_events["side"]

    # Step 5: Generate labels
    output["bin"] = np.sign(output["ret"])

    if is_meta_labeling:
        # For meta-labeling: 0 = don't take bet, 1 = take bet
        # If return <= 0, the primary model was wrong, so don't bet
        output.loc[output["ret"] <= 0, "bin"] = 0

    return output


def get_labels_with_timing(
    events: pd.DataFrame,
    close_prices: pd.Series,
    include_barrier_type: bool = True,
) -> pd.DataFrame:
    """
    Generate labels with additional timing and barrier type information.

    This extended version of label generation includes information about
    which barrier was touched and the holding period.

    Parameters
    ----------
    events : pd.DataFrame
        A DataFrame with event information, containing:
        - Index: Event start timestamps
        - 't1': Event end timestamps (first barrier touch times)
        - 'trgt': Target return (optional)
        - 'side': Position side from primary model (optional)
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
    include_barrier_type : bool, default=True
        If True, attempts to classify which barrier type was touched.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'ret': Return realized at the time of first barrier touch
        - 'bin': Label ({-1, 0, 1} or {0, 1} for meta-labeling)
        - 'holding_period': Time between event start and first barrier touch
        - 'barrier_type': Type of barrier touched ('pt', 'sl', 'vertical')
                          (only if include_barrier_type=True and determinable)

    Notes
    -----
    The barrier type classification requires the target return ('trgt') to be
    present in events. If not available, barrier_type will be 'unknown'.
    """
    # Get base labels
    output = get_labels(events, close_prices)

    if len(output) == 0:
        return output

    valid_events = events.dropna(subset=["t1"])

    # Add holding period
    output["holding_period"] = valid_events["t1"] - valid_events.index

    # Add barrier type classification if possible
    if include_barrier_type:
        if "trgt" in valid_events.columns:
            target = valid_events["trgt"]

            # Classify based on return magnitude relative to target
            abs_ret = output["ret"].abs()

            # If return exceeds target, likely hit horizontal barrier
            # Otherwise, likely hit vertical barrier
            barrier_type = pd.Series("vertical", index=output.index)

            # Positive return exceeding target = profit taking
            pt_mask = (output["ret"] > 0) & (abs_ret >= target)
            barrier_type.loc[pt_mask] = "pt"

            # Negative return exceeding target = stop loss
            sl_mask = (output["ret"] < 0) & (abs_ret >= target)
            barrier_type.loc[sl_mask] = "sl"

            output["barrier_type"] = barrier_type
        else:
            output["barrier_type"] = "unknown"

    return output
