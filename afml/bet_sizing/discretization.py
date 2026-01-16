"""
Size discretization for bet sizing.

This module provides functions to discretize continuous bet sizes into
discrete levels, reducing unnecessary trading from small signal changes.

Reference: AFML Chapter 10, Section 10.5, Snippet 10.3
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def discretize_signal(
    signal: Union[float, np.ndarray, pd.Series],
    step_size: float = 0.1,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Discretize continuous signal to nearest step size.

    Rounds the signal to the nearest multiple of step_size, reducing
    unnecessary trading from small signal changes.

    Parameters
    ----------
    signal : float or array-like
        Continuous signal(s) in range [-1, 1].
    step_size : float, default=0.1
        Discretization step size. Signal will be rounded to nearest
        multiple of this value.

    Returns
    -------
    float or array-like
        Discretized signal(s). Same type as input.

    Notes
    -----
    The discretization formula is:
        m* = round(m / d) * d

    Where:
        - m is the continuous signal
        - d is the step size
        - m* is the discretized signal

    Benefits of discretization:
    1. Reduces trading frequency and transaction costs
    2. Prevents overreaction to small signal changes
    3. Makes position sizes more interpretable (e.g., 10%, 20%, etc.)

    References
    ----------
    AFML Chapter 10, Section 10.5, Snippet 10.3

    Examples
    --------
    >>> # Discretize a single signal
    >>> signal = 0.37
    >>> discrete = discretize_signal(signal, step_size=0.1)
    >>> print(f"Discretized: {discrete}")  # 0.4

    >>> # Discretize array of signals
    >>> signals = np.array([0.12, 0.28, 0.55, 0.87])
    >>> discrete = discretize_signal(signals, step_size=0.2)
    >>> print(f"Discretized: {discrete}")  # [0.2, 0.2, 0.6, 0.8]
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    # Discretize: round to nearest multiple of step_size
    discretized = np.round(signal / step_size) * step_size

    # Clip to valid range [-1, 1]
    if isinstance(discretized, pd.Series):
        discretized = discretized.clip(-1, 1)
    elif isinstance(discretized, np.ndarray):
        discretized = np.clip(discretized, -1, 1)
    else:
        discretized = max(-1, min(1, discretized))

    return discretized


def get_discrete_levels(step_size: float = 0.1) -> np.ndarray:
    """
    Get all discrete signal levels for a given step size.

    Parameters
    ----------
    step_size : float, default=0.1
        Discretization step size.

    Returns
    -------
    np.ndarray
        Array of all possible discrete levels from -1 to 1.

    Examples
    --------
    >>> levels = get_discrete_levels(step_size=0.25)
    >>> print(levels)  # [-1. , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    # Generate levels from -1 to 1
    num_steps = int(2 / step_size) + 1
    levels = np.linspace(-1, 1, num_steps)

    # Round to avoid floating point issues
    levels = np.round(levels / step_size) * step_size

    return levels


def compute_required_trade(
    current_position: Union[float, np.ndarray, pd.Series],
    target_signal: Union[float, np.ndarray, pd.Series],
    step_size: float = 0.1,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Compute required trade to reach discretized target position.

    Parameters
    ----------
    current_position : float or array-like
        Current position(s) in range [-1, 1].
    target_signal : float or array-like
        Target signal(s) from the model.
    step_size : float, default=0.1
        Discretization step size.

    Returns
    -------
    float or array-like
        Required trade(s) to reach the discretized target.

    Examples
    --------
    >>> current = 0.3
    >>> target = 0.45
    >>> trade = compute_required_trade(current, target, step_size=0.1)
    >>> print(f"Trade: {trade}")  # 0.2 (to reach 0.5)
    """
    # Discretize target signal
    discrete_target = discretize_signal(target_signal, step_size)
    discrete_current = discretize_signal(current_position, step_size)

    # Compute required trade
    trade = discrete_target - discrete_current

    return trade


def should_trade(
    current_position: Union[float, np.ndarray, pd.Series],
    target_signal: Union[float, np.ndarray, pd.Series],
    step_size: float = 0.1,
    min_trade_size: Optional[float] = None,
) -> Union[bool, np.ndarray, pd.Series]:
    """
    Determine if trading is necessary based on discretized signals.

    Parameters
    ----------
    current_position : float or array-like
        Current position(s).
    target_signal : float or array-like
        Target signal(s) from the model.
    step_size : float, default=0.1
        Discretization step size.
    min_trade_size : float, optional
        Minimum trade size to trigger trading. If None, any non-zero
        trade will return True.

    Returns
    -------
    bool or array-like
        Whether trading should occur.

    Examples
    --------
    >>> should_trade(0.35, 0.42, step_size=0.1)  # False (both round to 0.4)
    >>> should_trade(0.35, 0.55, step_size=0.1)  # True (0.4 -> 0.6)
    """
    trade = compute_required_trade(current_position, target_signal, step_size)

    if min_trade_size is None:
        min_trade_size = step_size / 2

    if isinstance(trade, (np.ndarray, pd.Series)):
        return np.abs(trade) >= min_trade_size
    else:
        return abs(trade) >= min_trade_size


def adaptive_discretization(
    signal: Union[float, np.ndarray, pd.Series],
    signal_confidence: Union[float, np.ndarray, pd.Series],
    min_step: float = 0.05,
    max_step: float = 0.25,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Apply adaptive discretization based on signal confidence.

    High confidence signals use finer discretization (smaller steps),
    while low confidence signals use coarser discretization.

    Parameters
    ----------
    signal : float or array-like
        Continuous signal(s) in range [-1, 1].
    signal_confidence : float or array-like
        Confidence in the signal(s), in range [0, 1].
        Higher values indicate more confidence.
    min_step : float, default=0.05
        Minimum step size (used for high confidence).
    max_step : float, default=0.25
        Maximum step size (used for low confidence).

    Returns
    -------
    float or array-like
        Adaptively discretized signal(s).

    Notes
    -----
    Step size is computed as:
        step = max_step - confidence * (max_step - min_step)

    High confidence (confidence=1) -> step = min_step (fine discretization)
    Low confidence (confidence=0) -> step = max_step (coarse discretization)

    Examples
    --------
    >>> # High confidence: fine discretization
    >>> adaptive_discretization(0.37, confidence=0.9, min_step=0.05, max_step=0.25)
    >>> # Low confidence: coarse discretization
    >>> adaptive_discretization(0.37, confidence=0.2, min_step=0.05, max_step=0.25)
    """
    # Compute adaptive step size
    step_size = max_step - signal_confidence * (max_step - min_step)

    # Handle scalar and array cases
    if np.isscalar(signal) and np.isscalar(step_size):
        return discretize_signal(signal, step_size)

    # For arrays, we need element-wise discretization
    signal = np.asarray(signal)
    step_size = np.asarray(step_size)

    # Broadcast to same shape
    signal, step_size = np.broadcast_arrays(signal, step_size)

    # Discretize each element
    discretized = np.round(signal / step_size) * step_size
    discretized = np.clip(discretized, -1, 1)

    return discretized


def quantile_discretization(
    signals: Union[np.ndarray, pd.Series],
    num_levels: int = 10,
) -> Union[np.ndarray, pd.Series]:
    """
    Discretize signals based on quantiles of the signal distribution.

    Maps signals to discrete levels based on their rank in the distribution,
    rather than their absolute value.

    Parameters
    ----------
    signals : array-like
        Array of continuous signals.
    num_levels : int, default=10
        Number of discrete levels (including 0).

    Returns
    -------
    array-like
        Discretized signals based on quantile rank.

    Notes
    -----
    This approach ensures equal allocation across discrete levels,
    which may be desirable for portfolio construction.

    Examples
    --------
    >>> signals = np.random.randn(100)
    >>> discrete = quantile_discretization(signals, num_levels=5)
    >>> print(np.unique(discrete))  # Array of 5 unique values
    """
    is_series = isinstance(signals, pd.Series)
    if is_series:
        index = signals.index
        signals = signals.values

    # Compute quantile rank (0 to 1)
    quantile_rank = pd.Series(signals).rank(pct=True).values

    # Map to discrete levels
    levels = np.linspace(-1, 1, num_levels)
    indices = np.clip(
        np.floor(quantile_rank * num_levels).astype(int),
        0,
        num_levels - 1
    )
    discretized = levels[indices]

    if is_series:
        return pd.Series(discretized, index=index)
    return discretized


def symmetric_discretization(
    signal: Union[float, np.ndarray, pd.Series],
    step_size: float = 0.1,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Discretize signal ensuring symmetric treatment of positive/negative values.

    Parameters
    ----------
    signal : float or array-like
        Continuous signal(s).
    step_size : float, default=0.1
        Discretization step size.

    Returns
    -------
    float or array-like
        Symmetrically discretized signal(s).

    Notes
    -----
    This function ensures that:
        symmetric_discretization(-x) == -symmetric_discretization(x)

    This property is important for fair treatment of long and short positions.

    Examples
    --------
    >>> symmetric_discretization(0.37, step_size=0.1)   # 0.4
    >>> symmetric_discretization(-0.37, step_size=0.1)  # -0.4
    """
    # Discretize magnitude
    sign = np.sign(signal)
    magnitude = np.abs(signal)
    discrete_magnitude = np.round(magnitude / step_size) * step_size

    # Clip and apply sign
    discrete_magnitude = np.minimum(discrete_magnitude, 1.0)
    discretized = sign * discrete_magnitude

    return discretized
