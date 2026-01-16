"""
Triple-barrier method implementation for labeling financial observations.

The triple-barrier method labels observations according to the first barrier touched
out of three barriers:
1. Upper horizontal barrier (profit-taking)
2. Lower horizontal barrier (stop-loss)
3. Vertical barrier (maximum holding period / expiration)

Reference: AFML Chapter 3, Section 3.4
"""

from typing import Optional, Tuple, Union, List
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def apply_triple_barrier(
    close_prices: pd.Series,
    events: pd.DataFrame,
    profit_taking_stop_loss: List[float],
    molecule: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Apply stop-loss and profit-taking barriers to detect first barrier touch.

    This function determines when (if ever) the stop-loss or profit-taking barriers
    are touched before the vertical barrier (expiration time t1).

    Parameters
    ----------
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
    events : pd.DataFrame
        A DataFrame containing event information with columns:
        - 't1': Timestamp of the vertical barrier (expiration time)
        - 'trgt': Target return (used to compute barrier widths)
        - 'side': Position side (+1 for long, -1 for short)
    profit_taking_stop_loss : List[float]
        A list of two non-negative floats [pt_multiplier, sl_multiplier]:
        - pt_multiplier: Factor that multiplies target to set profit-taking barrier
        - sl_multiplier: Factor that multiplies target to set stop-loss barrier
        If either is 0, that barrier is disabled.
    molecule : pd.DatetimeIndex
        Subset of event indices to process (used for parallel processing).

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same index as molecule, containing columns:
        - 't1': Original vertical barrier timestamp
        - 'sl': Timestamp when stop-loss was hit (NaT if not hit)
        - 'pt': Timestamp when profit-taking was hit (NaT if not hit)

    Notes
    -----
    The function is path-dependent: it examines the entire price path from
    event start to vertical barrier to determine which barrier was touched first.

    References
    ----------
    AFML Chapter 3, Snippet 3.2: Triple-Barrier Labeling Method
    """
    # Filter events to the subset we're processing
    events_subset = events.loc[molecule]
    output = events_subset[["t1"]].copy(deep=True)

    # Set up profit-taking threshold
    pt_multiplier = profit_taking_stop_loss[0]
    if pt_multiplier > 0:
        profit_taking_threshold = pt_multiplier * events_subset["trgt"]
    else:
        profit_taking_threshold = pd.Series(index=events_subset.index, dtype=float)

    # Set up stop-loss threshold
    sl_multiplier = profit_taking_stop_loss[1]
    if sl_multiplier > 0:
        stop_loss_threshold = -sl_multiplier * events_subset["trgt"]
    else:
        stop_loss_threshold = pd.Series(index=events_subset.index, dtype=float)

    # Iterate through each event to find barrier touches
    for event_start, vertical_barrier in events_subset["t1"].fillna(
        close_prices.index[-1]
    ).items():
        # Extract price path from event start to vertical barrier
        price_path = close_prices[event_start:vertical_barrier]

        # Compute returns along the path, adjusted for position side
        position_side = events_subset.at[event_start, "side"]
        path_returns = (price_path / close_prices[event_start] - 1) * position_side

        # Find earliest stop-loss touch
        sl_threshold = stop_loss_threshold[event_start]
        stop_loss_touches = path_returns[path_returns < sl_threshold]
        output.loc[event_start, "sl"] = (
            stop_loss_touches.index.min() if len(stop_loss_touches) > 0 else pd.NaT
        )

        # Find earliest profit-taking touch
        pt_threshold = profit_taking_threshold[event_start]
        profit_taking_touches = path_returns[path_returns > pt_threshold]
        output.loc[event_start, "pt"] = (
            profit_taking_touches.index.min()
            if len(profit_taking_touches) > 0
            else pd.NaT
        )

    return output


def _process_batch(
    batch_indices: pd.DatetimeIndex,
    close_prices: pd.Series,
    events: pd.DataFrame,
    profit_taking_stop_loss: List[float],
) -> pd.DataFrame:
    """Helper function to process a batch of events for parallel execution."""
    return apply_triple_barrier(
        close_prices=close_prices,
        events=events,
        profit_taking_stop_loss=profit_taking_stop_loss,
        molecule=batch_indices,
    )


def get_events(
    close_prices: pd.Series,
    timestamp_events: pd.DatetimeIndex,
    profit_taking_stop_loss: Union[float, List[float]],
    target_returns: pd.Series,
    min_return: float,
    num_threads: int = 1,
    vertical_barrier_times: Optional[pd.Series] = None,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Find the time of the first barrier touch for each event.

    This function implements the triple-barrier labeling method, which labels
    observations according to which of three barriers is touched first:
    profit-taking, stop-loss, or vertical (time) barrier.

    Parameters
    ----------
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
    timestamp_events : pd.DatetimeIndex
        Timestamps that seed each triple barrier. These are typically selected
        by sampling procedures (e.g., CUSUM filter).
    profit_taking_stop_loss : float or List[float]
        If float: Used as symmetric multiplier for both barriers.
        If list [pt, sl]: Separate multipliers for profit-taking and stop-loss.
        - pt: Factor multiplying target for upper (profit-taking) barrier
        - sl: Factor multiplying target for lower (stop-loss) barrier
        Set to 0 to disable respective barrier.
    target_returns : pd.Series
        Target returns expressed in absolute terms. Used to compute barrier widths.
    min_return : float
        Minimum target return required to run a triple-barrier search.
        Events with target below this threshold are filtered out.
    num_threads : int, default=1
        Number of threads for parallel processing.
        Set to 1 for sequential processing.
    vertical_barrier_times : pd.Series, optional
        Timestamps of vertical barriers (expiration times).
        If None, no vertical barrier is applied.
    side : pd.Series, optional
        Position side for each event (+1 for long, -1 for short).
        If None, assumes learning both side and size (symmetric barriers).
        If provided, enables meta-labeling with asymmetric barriers.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 't1': Timestamp at which the first barrier was touched
        - 'trgt': Target return used to generate horizontal barriers
        - 'side': Position side (only if side parameter was provided)

    Notes
    -----
    When side is None (learning both side and size):
    - Horizontal barriers are symmetric
    - Side is arbitrarily set to 1 (long) since barriers are symmetric

    When side is provided (meta-labeling):
    - Horizontal barriers can be asymmetric
    - The provided side determines profit-taking vs stop-loss direction

    References
    ----------
    AFML Chapter 3, Snippet 3.3: Getting the Time of First Touch
    AFML Chapter 3, Snippet 3.6: Expanding getEvents for Meta-Labeling
    """
    # Step 1: Filter targets to event timestamps and minimum return threshold
    filtered_targets = target_returns.loc[
        target_returns.index.intersection(timestamp_events)
    ]
    filtered_targets = filtered_targets[filtered_targets > min_return]

    if len(filtered_targets) == 0:
        raise ValueError(
            "No events remain after filtering by minimum return threshold. "
            "Consider lowering min_return or checking target_returns values."
        )

    # Step 2: Set up vertical barrier times
    if vertical_barrier_times is None:
        vertical_barriers = pd.Series(pd.NaT, index=filtered_targets.index)
    else:
        vertical_barriers = vertical_barrier_times.loc[
            vertical_barrier_times.index.intersection(filtered_targets.index)
        ]
        # Reindex to match filtered targets
        vertical_barriers = vertical_barriers.reindex(filtered_targets.index)

    # Step 3: Set up position sides and barrier multipliers
    if side is None:
        # Learning both side and size: use symmetric barriers
        position_sides = pd.Series(1.0, index=filtered_targets.index)
        # Convert to list format if needed
        if isinstance(profit_taking_stop_loss, (int, float)):
            barrier_multipliers = [profit_taking_stop_loss, profit_taking_stop_loss]
        else:
            # Use same value for both barriers (symmetric)
            barrier_multipliers = [
                profit_taking_stop_loss[0],
                profit_taking_stop_loss[0],
            ]
    else:
        # Meta-labeling: use provided sides and potentially asymmetric barriers
        position_sides = side.loc[side.index.intersection(filtered_targets.index)]
        position_sides = position_sides.reindex(filtered_targets.index)
        if isinstance(profit_taking_stop_loss, (int, float)):
            barrier_multipliers = [profit_taking_stop_loss, profit_taking_stop_loss]
        else:
            barrier_multipliers = list(profit_taking_stop_loss[:2])

    # Step 4: Create events DataFrame
    events_df = pd.concat(
        {
            "t1": vertical_barriers,
            "trgt": filtered_targets,
            "side": position_sides,
        },
        axis=1,
    ).dropna(subset=["trgt"])

    # Step 5: Apply triple-barrier method (with optional parallelization)
    if num_threads == 1:
        # Sequential processing
        barrier_touches = apply_triple_barrier(
            close_prices=close_prices,
            events=events_df,
            profit_taking_stop_loss=barrier_multipliers,
            molecule=events_df.index,
        )
    else:
        # Parallel processing
        batch_size = max(1, len(events_df) // num_threads)
        batches = [
            events_df.index[i : i + batch_size]
            for i in range(0, len(events_df), batch_size)
        ]

        process_func = partial(
            _process_batch,
            close_prices=close_prices,
            events=events_df,
            profit_taking_stop_loss=barrier_multipliers,
        )

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_func, batches))

        barrier_touches = pd.concat(results)

    # Step 6: Determine first barrier touch time
    # pd.min ignores NaN values
    first_touch_times = barrier_touches.dropna(how="all").min(axis=1)
    events_df["t1"] = first_touch_times

    # Step 7: Remove side column if not doing meta-labeling
    if side is None:
        events_df = events_df.drop("side", axis=1)

    return events_df


def add_vertical_barrier(
    timestamp_events: pd.DatetimeIndex,
    close_prices: pd.Series,
    num_days: int = 1,
) -> pd.Series:
    """
    Add vertical barriers (expiration times) to events.

    For each event timestamp, find the timestamp of the next price bar at or
    immediately after a specified number of days.

    Parameters
    ----------
    timestamp_events : pd.DatetimeIndex
        Timestamps of events that need vertical barriers.
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
        Used to find valid future timestamps.
    num_days : int, default=1
        Number of days for the vertical barrier (holding period).

    Returns
    -------
    pd.Series
        A pandas Series mapping event timestamps to their vertical barrier
        timestamps. Events near the end of the series (where vertical barrier
        would fall outside available data) are truncated.

    Notes
    -----
    The vertical barrier represents the maximum holding period for a position.
    If neither profit-taking nor stop-loss barriers are touched before this time,
    the position is closed at the vertical barrier.

    Examples
    --------
    >>> import pandas as pd
    >>> events = pd.DatetimeIndex(['2023-01-01', '2023-01-02'])
    >>> close = pd.Series([100, 101, 102, 103],
    ...                   index=pd.date_range('2023-01-01', periods=4, freq='D'))
    >>> barriers = add_vertical_barrier(events, close, num_days=2)

    References
    ----------
    AFML Chapter 3, Snippet 3.4: Adding a Vertical Barrier
    """
    # Find indices of bars at or after (event_time + num_days)
    target_times = timestamp_events + pd.Timedelta(days=num_days)
    barrier_indices = close_prices.index.searchsorted(target_times)

    # Filter out indices that fall outside the available data
    valid_mask = barrier_indices < close_prices.shape[0]
    valid_barrier_indices = barrier_indices[valid_mask]
    valid_event_timestamps = timestamp_events[: valid_barrier_indices.shape[0]]

    # Create series mapping events to their vertical barrier timestamps
    vertical_barrier_times = pd.Series(
        close_prices.index[valid_barrier_indices],
        index=valid_event_timestamps,
    )

    return vertical_barrier_times
