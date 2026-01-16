"""
Time decay factors for sample weights.

Markets are adaptive systems - as they evolve, older observations become
less relevant than newer ones. This module provides functions to apply
time decay to sample weights.

Reference: AFML Chapter 4, Section 4.7
"""

from typing import Union
import pandas as pd
import numpy as np


def apply_time_decay(
    sample_weights: pd.Series,
    decay_factor: float = 1.0,
) -> pd.Series:
    """
    Apply piecewise-linear time decay to sample weights.

    Multiplies sample weights by time-decay factors that give more weight
    to recent observations and less weight to older ones. The decay is
    applied based on cumulative uniqueness, not chronological time.

    Parameters
    ----------
    sample_weights : pd.Series
        Sample weights indexed by event timestamps, typically from
        get_sample_weights_by_return() or get_sample_weights_by_uniqueness().
    decay_factor : float, default=1.0
        Controls the shape of the decay function:
        - c = 1: No decay (all weights unchanged)
        - 0 < c < 1: Linear decay, all observations get positive weight
        - c = 0: Weights decay linearly to zero for oldest observations
        - -1 < c < 0: Oldest |c| fraction of observations get zero weight

    Returns
    -------
    pd.Series
        Time-decayed sample weights, indexed like input weights.

    Notes
    -----
    The decay is applied based on cumulative uniqueness (sum of weights),
    NOT chronological time. This prevents over-penalizing older observations
    when there is high redundancy among recent observations.

    The decay function is piecewise-linear: d(x) = max{0, a + b*x}

    Boundary conditions:
    - d(sum(weights)) = 1 (newest observation has no decay)
    - d(0) = c for c >= 0 (oldest observation's decay factor)
    - d(-c * sum(weights)) = 0 for c < 0 (zero weight cutoff)

    **Special cases:**
    - c = 1: d(x) = 1 for all x (no decay)
    - c = 0: d(x) = x / sum(weights) (linear decay to zero)
    - c = 0.5: d(0) = 0.5 (oldest gets half weight of newest)
    - c = -0.5: oldest 50% of cumulative uniqueness gets zero weight

    References
    ----------
    AFML Chapter 4, Snippet 4.11: Implementation of Time-Decay Factors

    Examples
    --------
    >>> import pandas as pd
    >>> weights = pd.Series([1.0, 1.2, 0.8, 1.5],
    ...                     index=pd.date_range('2023-01-01', periods=4, freq='D'))
    >>> # No decay
    >>> decayed = apply_time_decay(weights, decay_factor=1.0)
    >>> # Linear decay, oldest gets half weight
    >>> decayed = apply_time_decay(weights, decay_factor=0.5)
    >>> # Oldest 25% of cumulative uniqueness gets zero weight
    >>> decayed = apply_time_decay(weights, decay_factor=-0.25)
    """
    # Validate decay factor
    if decay_factor < -1 or decay_factor > 1:
        raise ValueError(
            f"decay_factor must be in range [-1, 1], got {decay_factor}"
        )

    # Sort by index (chronological order)
    sorted_weights = sample_weights.sort_index()

    # Compute cumulative uniqueness (x-axis for decay function)
    cumulative_uniqueness = sorted_weights.cumsum()
    total_uniqueness = cumulative_uniqueness.iloc[-1]

    if total_uniqueness == 0:
        return sorted_weights.copy()

    # Compute decay parameters based on boundary conditions
    if decay_factor >= 0:
        # Case: c in [0, 1]
        # d(0) = c, d(total) = 1
        # d = a + b*x => a = c, b = (1-c) / total
        slope = (1.0 - decay_factor) / total_uniqueness
        intercept = decay_factor
    else:
        # Case: c in (-1, 0)
        # d(-c * total) = 0, d(total) = 1
        # The oldest |c| fraction gets zero weight
        # d = a + b*x where d(total) = 1 and d(-c*total) = 0
        # Solving: b = 1 / ((1+c) * total), a = -b * (-c * total) = c / (1+c)
        slope = 1.0 / ((decay_factor + 1) * total_uniqueness)
        intercept = 1.0 - slope * total_uniqueness

    # Apply decay function: d(x) = max{0, intercept + slope * x}
    decay_values = intercept + slope * cumulative_uniqueness
    decay_values = decay_values.clip(lower=0)

    # Multiply weights by decay factors
    decayed_weights = sorted_weights * decay_values

    return decayed_weights


def apply_exponential_decay(
    sample_weights: pd.Series,
    half_life: float,
) -> pd.Series:
    """
    Apply exponential time decay to sample weights.

    An alternative to piecewise-linear decay that uses an exponential
    decay function based on cumulative uniqueness.

    Parameters
    ----------
    sample_weights : pd.Series
        Sample weights indexed by event timestamps.
    half_life : float
        The cumulative uniqueness at which the decay factor equals 0.5.
        Smaller values mean faster decay (older observations weighted less).

    Returns
    -------
    pd.Series
        Time-decayed sample weights.

    Notes
    -----
    The decay function is: d(x) = 2^(-(total - x) / half_life)

    where x is cumulative uniqueness and total is the sum of all weights.
    The newest observation has d = 1, and decay increases exponentially
    as we go back in time.

    Examples
    --------
    >>> # Half-life of 10 means weights halve every 10 units of uniqueness
    >>> decayed = apply_exponential_decay(weights, half_life=10.0)
    """
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")

    # Sort by index (chronological order)
    sorted_weights = sample_weights.sort_index()

    # Compute cumulative uniqueness
    cumulative_uniqueness = sorted_weights.cumsum()
    total_uniqueness = cumulative_uniqueness.iloc[-1]

    if total_uniqueness == 0:
        return sorted_weights.copy()

    # Compute decay: d(x) = 2^(-(total - x) / half_life)
    # At x = total, d = 1 (newest)
    # At x = total - half_life, d = 0.5
    time_from_end = total_uniqueness - cumulative_uniqueness
    decay_values = np.power(2, -time_from_end / half_life)

    # Multiply weights by decay factors
    decayed_weights = sorted_weights * decay_values

    return decayed_weights


def compute_decayed_sample_weights(
    sample_weights: pd.Series,
    decay_type: str = "linear",
    decay_factor: float = 1.0,
    half_life: float = None,
) -> pd.Series:
    """
    Apply time decay to sample weights using the specified method.

    Convenience function that supports both linear and exponential decay.

    Parameters
    ----------
    sample_weights : pd.Series
        Sample weights indexed by event timestamps.
    decay_type : str, default='linear'
        Type of decay to apply: 'linear' or 'exponential'.
    decay_factor : float, default=1.0
        Decay factor for linear decay (ignored if decay_type='exponential').
    half_life : float, optional
        Half-life for exponential decay (required if decay_type='exponential').

    Returns
    -------
    pd.Series
        Time-decayed sample weights.

    Examples
    --------
    >>> # Linear decay with c=0.5
    >>> decayed = compute_decayed_sample_weights(
    ...     weights, decay_type='linear', decay_factor=0.5
    ... )
    >>> # Exponential decay with half-life of 20
    >>> decayed = compute_decayed_sample_weights(
    ...     weights, decay_type='exponential', half_life=20.0
    ... )
    """
    if decay_type == "linear":
        return apply_time_decay(sample_weights, decay_factor=decay_factor)
    elif decay_type == "exponential":
        if half_life is None:
            raise ValueError("half_life is required for exponential decay")
        return apply_exponential_decay(sample_weights, half_life=half_life)
    else:
        raise ValueError(
            f"Unknown decay_type '{decay_type}'. Use 'linear' or 'exponential'."
        )
