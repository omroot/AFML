"""
Sequential bootstrap for sampling with overlapping labels.

Standard bootstrap assumes IID observations, which leads to oversampling
when labels overlap. The sequential bootstrap adjusts sampling probabilities
to account for overlap, producing samples closer to IID.

Reference: AFML Chapter 4, Section 4.5
"""

from typing import Optional, List
import pandas as pd
import numpy as np


def get_indicator_matrix(
    bar_index: pd.DatetimeIndex,
    event_end_times: pd.Series,
) -> pd.DataFrame:
    """
    Build an indicator matrix showing which bars affect each label.

    Creates a binary matrix where entry (t, i) = 1 if bar t is within
    the time span of event i (from event start to event end).

    Parameters
    ----------
    bar_index : pd.DatetimeIndex
        Index of all price bars (timestamps).
    event_end_times : pd.Series
        A pandas Series where:
        - Index: Event start timestamps (t0)
        - Values: Event end timestamps (t1)

    Returns
    -------
    pd.DataFrame
        A binary DataFrame where:
        - Index: Bar timestamps
        - Columns: Event indices (0, 1, 2, ...)
        - Values: 1 if bar is within event span, 0 otherwise

    Notes
    -----
    This indicator matrix is the foundation for computing uniqueness
    and performing sequential bootstrap. Each column represents an event,
    and each row represents a time point.

    References
    ----------
    AFML Chapter 4, Snippet 4.3: Build an Indicator Matrix

    Examples
    --------
    >>> import pandas as pd
    >>> bar_idx = pd.date_range('2023-01-01', periods=7, freq='D')
    >>> t1 = pd.Series([bar_idx[2], bar_idx[3], bar_idx[5]],
    ...                index=[bar_idx[0], bar_idx[2], bar_idx[4]])
    >>> ind_matrix = get_indicator_matrix(bar_idx, t1)
    """
    # Initialize indicator matrix with zeros
    indicator_matrix = pd.DataFrame(
        0,
        index=bar_index,
        columns=range(event_end_times.shape[0]),
    )

    # For each event, set indicators to 1 for bars within its span
    for event_idx, (event_start, event_end) in enumerate(event_end_times.items()):
        if pd.notna(event_end):
            indicator_matrix.loc[event_start:event_end, event_idx] = 1

    return indicator_matrix


def get_average_uniqueness_from_matrix(
    indicator_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Compute average uniqueness for each event from the indicator matrix.

    For each event, computes the uniqueness at each time point (1/concurrency)
    and returns the average over the event's lifespan.

    Parameters
    ----------
    indicator_matrix : pd.DataFrame
        Binary indicator matrix as returned by get_indicator_matrix().

    Returns
    -------
    pd.Series
        Average uniqueness for each event (indexed by event column number).

    Notes
    -----
    The concurrency at time t is the sum of indicators across all events.
    Uniqueness at time t for event i is 1/concurrency if the event is active.
    Average uniqueness is the mean of these values over the event's span.

    References
    ----------
    AFML Chapter 4, Snippet 4.4: Compute Average Uniqueness

    Examples
    --------
    >>> ind_matrix = get_indicator_matrix(bar_idx, t1)
    >>> avg_uniqueness = get_average_uniqueness_from_matrix(ind_matrix)
    """
    # Concurrency: sum of indicators at each time point
    concurrency = indicator_matrix.sum(axis=1)

    # Uniqueness: indicator / concurrency (0 where event is not active)
    uniqueness = indicator_matrix.div(concurrency, axis=0)

    # Average uniqueness: mean over each event's active period
    # Only consider time points where event is active (indicator > 0)
    average_uniqueness = uniqueness[uniqueness > 0].mean()

    return average_uniqueness


def _compute_sequential_uniqueness(
    indicator_matrix: pd.DataFrame,
    selected_indices: List[int],
    candidate_idx: int,
) -> float:
    """
    Compute the average uniqueness of a candidate given already-selected events.

    This internal function calculates how unique a candidate event would be
    if added to the current selection, accounting for overlaps with
    already-selected events.

    Parameters
    ----------
    indicator_matrix : pd.DataFrame
        Binary indicator matrix.
    selected_indices : List[int]
        Column indices of already-selected events.
    candidate_idx : int
        Column index of the candidate event.

    Returns
    -------
    float
        Average uniqueness of the candidate given the current selection.
    """
    if len(selected_indices) == 0:
        # First draw: uniqueness is 1 (no overlap with empty set)
        candidate_indicators = indicator_matrix.iloc[:, candidate_idx]
        active_mask = candidate_indicators > 0
        if active_mask.sum() == 0:
            return 0.0
        return 1.0

    # Get indicators for selected events plus the candidate
    selected_columns = list(selected_indices) + [candidate_idx]
    subset_matrix = indicator_matrix.iloc[:, selected_columns]

    # Concurrency among selected events + candidate
    concurrency = subset_matrix.sum(axis=1)

    # Uniqueness of candidate at each time point
    candidate_indicators = indicator_matrix.iloc[:, candidate_idx]

    # Only compute uniqueness where candidate is active
    active_mask = candidate_indicators > 0
    if active_mask.sum() == 0:
        return 0.0

    uniqueness = candidate_indicators[active_mask] / concurrency[active_mask]

    return uniqueness.mean()


def sequential_bootstrap(
    indicator_matrix: pd.DataFrame,
    sample_length: Optional[int] = None,
    random_state: Optional[int] = None,
) -> List[int]:
    """
    Generate a sample using sequential bootstrap with overlap-aware probabilities.

    Unlike standard bootstrap (uniform sampling), sequential bootstrap adjusts
    probabilities after each draw to reduce the likelihood of selecting
    highly overlapping observations. This produces samples closer to IID.

    Parameters
    ----------
    indicator_matrix : pd.DataFrame
        Binary indicator matrix as returned by get_indicator_matrix().
    sample_length : int, optional
        Number of samples to draw. Defaults to the number of events.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    List[int]
        Indices of selected events (column indices from indicator_matrix).
        May contain duplicates (bootstrap with replacement).

    Notes
    -----
    The algorithm works as follows:
    1. First draw: uniform probability across all events
    2. Subsequent draws: probability proportional to average uniqueness
       given already-selected events
    3. Events with high overlap to selected events get lower probability
    4. Repeat until sample_length events are drawn

    Monte Carlo experiments show sequential bootstrap achieves ~0.7 median
    uniqueness vs ~0.6 for standard bootstrap.

    References
    ----------
    AFML Chapter 4, Snippet 4.5: Return Sample from Sequential Bootstrap

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> bar_idx = pd.date_range('2023-01-01', periods=7, freq='D')
    >>> t1 = pd.Series([bar_idx[2], bar_idx[3], bar_idx[5]],
    ...                index=[bar_idx[0], bar_idx[2], bar_idx[4]])
    >>> ind_matrix = get_indicator_matrix(bar_idx, t1)
    >>> sample = sequential_bootstrap(ind_matrix, random_state=42)
    >>> print(f"Selected events: {sample}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_events = indicator_matrix.shape[1]

    if sample_length is None:
        sample_length = num_events

    selected_indices: List[int] = []

    while len(selected_indices) < sample_length:
        # Compute average uniqueness for each candidate given current selection
        average_uniqueness = pd.Series(index=range(num_events), dtype=float)

        for candidate_idx in range(num_events):
            # Compute uniqueness including overlap with already-selected events
            # Note: This allows the same event to be selected again (bootstrap)
            temp_selected = selected_indices + [candidate_idx]
            subset_matrix = indicator_matrix.iloc[:, temp_selected]
            concurrency = subset_matrix.sum(axis=1)

            candidate_indicators = indicator_matrix.iloc[:, candidate_idx]
            active_mask = candidate_indicators > 0

            if active_mask.sum() == 0:
                average_uniqueness.loc[candidate_idx] = 0.0
            else:
                uniqueness = candidate_indicators[active_mask] / concurrency[active_mask]
                average_uniqueness.loc[candidate_idx] = uniqueness.mean()

        # Convert uniqueness to probabilities (normalize to sum to 1)
        total_uniqueness = average_uniqueness.sum()
        if total_uniqueness == 0:
            # Fallback to uniform if all uniqueness is zero
            probabilities = np.ones(num_events) / num_events
        else:
            probabilities = average_uniqueness.values / total_uniqueness

        # Draw next event based on computed probabilities
        selected_event = np.random.choice(
            indicator_matrix.columns,
            p=probabilities,
        )
        selected_indices.append(selected_event)

    return selected_indices


def compare_bootstrap_methods(
    indicator_matrix: pd.DataFrame,
    num_iterations: int = 100,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compare standard bootstrap vs sequential bootstrap uniqueness.

    Runs multiple iterations of both methods and returns statistics
    on the average uniqueness achieved by each.

    Parameters
    ----------
    indicator_matrix : pd.DataFrame
        Binary indicator matrix as returned by get_indicator_matrix().
    num_iterations : int, default=100
        Number of bootstrap samples to generate for each method.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Statistics comparing the two methods:
        - 'standard_uniqueness': Average uniqueness from standard bootstrap
        - 'sequential_uniqueness': Average uniqueness from sequential bootstrap

    Examples
    --------
    >>> comparison = compare_bootstrap_methods(ind_matrix, num_iterations=50)
    >>> print(comparison.describe())
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_events = indicator_matrix.shape[1]
    results = []

    for _ in range(num_iterations):
        # Standard bootstrap: uniform random sampling with replacement
        standard_sample = np.random.choice(
            indicator_matrix.columns,
            size=num_events,
            replace=True,
        )
        standard_matrix = indicator_matrix.iloc[:, standard_sample]
        standard_uniqueness = get_average_uniqueness_from_matrix(
            standard_matrix
        ).mean()

        # Sequential bootstrap
        sequential_sample = sequential_bootstrap(indicator_matrix)
        sequential_matrix = indicator_matrix.iloc[:, sequential_sample]
        sequential_uniqueness = get_average_uniqueness_from_matrix(
            sequential_matrix
        ).mean()

        results.append({
            "standard_uniqueness": standard_uniqueness,
            "sequential_uniqueness": sequential_uniqueness,
        })

    return pd.DataFrame(results)
