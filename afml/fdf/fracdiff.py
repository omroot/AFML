"""
Fractional differentiation implementations.

This module provides functions to apply fractional differentiation to
time series data. Fractional differentiation offers a middle ground
between the memory-preserving but non-stationary original series and
the stationary but memoryless integer-differenced series.

Reference: AFML Chapter 5, Sections 5.3-5.4
"""

from typing import Optional, Union
import pandas as pd
import numpy as np
import warnings

from afml.fdf.weights import compute_weights, compute_weights_ffd


def fracdiff_expanding(
    series: Union[pd.Series, pd.DataFrame],
    diff_order: float,
    weight_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Apply fractional differentiation using expanding window.

    This is the theoretically exact implementation where each observation
    uses all available history. However, it's computationally expensive
    and can lead to look-ahead bias issues in backtesting.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Time series data with datetime index. If DataFrame, each column
        is differentiated independently.
    diff_order : float
        The fractional differentiation order (d). Typically 0 < d < 1
        for partial differentiation.
    weight_threshold : float, default=0.0
        Drop weights with |w| < threshold. Default 0 keeps all weights.
        Note: This still uses expanding window, just with small weights
        truncated.

    Returns
    -------
    pd.DataFrame
        Fractionally differentiated series. First observations are
        dropped where insufficient history exists.

    Notes
    -----
    The expanding window approach has a critical flaw: each observation
    uses a different amount of history. Early observations use short
    windows while late observations use very long windows. This creates:

    1. Non-uniform effective memory across the series
    2. Computational cost that grows with series length
    3. Potential look-ahead bias in out-of-sample testing

    For practical applications, use fracdiff_ffd() instead.

    References
    ----------
    AFML Chapter 5, Snippet 5.2: fracDiff function

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.Series([100, 101, 99, 102, 105],
    ...                    index=pd.date_range('2023-01-01', periods=5))
    >>> fd_prices = fracdiff_expanding(prices, diff_order=0.5)
    """
    # Convert Series to DataFrame for uniform handling
    if isinstance(series, pd.Series):
        series = series.to_frame()

    result = {}

    for col in series.columns:
        # Forward fill and drop NaN values
        col_series = series[[col]].ffill().dropna()

        fracdiff_values = pd.Series(dtype=float)

        for idx in range(col_series.shape[0]):
            # Compute weights for current window size
            window_size = idx + 1
            weights = compute_weights(diff_order, window_size)

            # Apply threshold to skip small weights
            if weight_threshold > 0:
                weights = weights[np.abs(weights.flatten()) >= weight_threshold]

            # Skip if we don't have enough history
            if len(weights) > window_size:
                continue

            # Get the window of values
            loc_end = col_series.index[idx]
            loc_start_idx = idx - len(weights) + 1
            loc_start = col_series.index[loc_start_idx]

            # Skip if original value is not finite
            if not np.isfinite(series.loc[loc_end, col]):
                continue

            # Apply weights via dot product
            window_values = col_series.loc[loc_start:loc_end].values
            fracdiff_values[loc_end] = np.dot(weights.T, window_values)[0, 0]

        result[col] = fracdiff_values

    return pd.concat(result, axis=1)


def fracdiff_ffd(
    series: Union[pd.Series, pd.DataFrame],
    diff_order: float,
    weight_threshold: float = 1e-5,
) -> pd.DataFrame:
    """
    Apply fractional differentiation using fixed-width window (FFD).

    This is the recommended implementation for practical applications.
    It uses a constant window size determined by when weights become
    negligible, ensuring consistent memory length across all observations.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Time series data with datetime index. If DataFrame, each column
        is differentiated independently.
    diff_order : float
        The fractional differentiation order (d). Typically 0 < d < 1
        for partial differentiation.
    weight_threshold : float, default=1e-5
        Threshold for weight truncation. Weights with |w| < threshold
        are dropped. This determines the effective window size.

    Returns
    -------
    pd.DataFrame
        Fractionally differentiated series. First (window_size - 1)
        observations are dropped where insufficient history exists.

    Notes
    -----
    The FFD method addresses the limitations of the expanding window:

    1. **Constant memory**: Every observation uses the same window size
    2. **Efficient computation**: O(1) per observation, not O(n)
    3. **No look-ahead bias**: Window size determined before processing

    The effective window size depends on d and threshold:
    - Lower d → slower weight decay → larger window
    - Lower threshold → more weights included → larger window

    Typical window sizes for threshold=1e-5:
    - d = 0.3: ~60 weights
    - d = 0.5: ~12 weights
    - d = 0.7: ~6 weights

    References
    ----------
    AFML Chapter 5, Snippet 5.3: fracDiff_FFD function

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame({'close': [100, 101, 99, 102, 105, 103]},
    ...                       index=pd.date_range('2023-01-01', periods=6))
    >>> fd_prices = fracdiff_ffd(prices, diff_order=0.5)
    >>> # Can also use with Series
    >>> fd_series = fracdiff_ffd(prices['close'], diff_order=0.4)
    """
    # Convert Series to DataFrame for uniform handling
    if isinstance(series, pd.Series):
        series = series.to_frame()

    # Compute weights (fixed for all observations)
    weights = compute_weights_ffd(diff_order, weight_threshold)
    window_size = len(weights)

    result = {}

    for col in series.columns:
        # Forward fill and drop NaN values
        col_series = series[[col]].ffill().dropna()

        fracdiff_values = pd.Series(dtype=float)

        # Start from index where we have enough history
        for idx in range(window_size - 1, col_series.shape[0]):
            loc_end = col_series.index[idx]
            loc_start_idx = idx - window_size + 1
            loc_start = col_series.index[loc_start_idx]

            # Skip if original value is not finite
            if not np.isfinite(series.loc[loc_end, col]):
                continue

            # Apply weights via dot product
            window_values = col_series.loc[loc_start:loc_end].values
            fracdiff_values[loc_end] = np.dot(weights.T, window_values)[0, 0]

        result[col] = fracdiff_values

    return pd.concat(result, axis=1)


def fracdiff(
    series: Union[pd.Series, pd.DataFrame],
    diff_order: float,
    method: str = "ffd",
    weight_threshold: float = 1e-5,
) -> pd.DataFrame:
    """
    Apply fractional differentiation to a time series.

    Convenience function that supports both expanding and fixed-width
    window methods.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Time series data with datetime index.
    diff_order : float
        The fractional differentiation order (d).
        - d = 0: No differencing (returns original series)
        - 0 < d < 1: Fractional differencing (preserves some memory)
        - d = 1: First difference (standard differencing)
    method : str, default='ffd'
        Differentiation method: 'ffd' (fixed-width) or 'expanding'.
    weight_threshold : float, default=1e-5
        Threshold for weight truncation.

    Returns
    -------
    pd.DataFrame
        Fractionally differentiated series.

    Raises
    ------
    ValueError
        If method is not 'ffd' or 'expanding'.

    Examples
    --------
    >>> # Using FFD method (recommended)
    >>> fd_prices = fracdiff(prices, diff_order=0.5, method='ffd')
    >>> # Using expanding window (for comparison)
    >>> fd_prices_exp = fracdiff(prices, diff_order=0.5, method='expanding')
    """
    if method == "ffd":
        return fracdiff_ffd(series, diff_order, weight_threshold)
    elif method == "expanding":
        return fracdiff_expanding(series, diff_order, weight_threshold)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'ffd' or 'expanding'."
        )


def get_ffd_window_size(
    diff_order: float,
    weight_threshold: float = 1e-5,
) -> int:
    """
    Get the window size that will be used for FFD with given parameters.

    Useful for understanding how many observations will be dropped
    and the effective memory length.

    Parameters
    ----------
    diff_order : float
        The fractional differentiation order (d).
    weight_threshold : float, default=1e-5
        Threshold for weight truncation.

    Returns
    -------
    int
        Number of observations in the FFD window.

    Examples
    --------
    >>> get_ffd_window_size(diff_order=0.5, weight_threshold=1e-5)
    12
    >>> get_ffd_window_size(diff_order=0.3, weight_threshold=1e-5)
    57
    """
    weights = compute_weights_ffd(diff_order, weight_threshold)
    return len(weights)
