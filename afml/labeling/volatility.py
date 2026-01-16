"""
Volatility estimation functions for dynamic threshold computation.

This module provides functions for computing volatility estimates that are used
to set dynamic profit-taking and stop-loss thresholds in the triple-barrier method.

Reference: AFML Chapter 3, Section 3.3
"""

import pandas as pd


def get_daily_volatility(
    close_prices: pd.Series,
    lookback_span: int = 100,
) -> pd.Series:
    """
    Compute daily volatility using exponentially weighted moving standard deviation.

    This function estimates daily volatility at intraday estimation points by computing
    the exponentially weighted moving standard deviation of daily returns. The volatility
    estimates can be used to set dynamic profit-taking and stop-loss limits that adapt
    to current market conditions.

    Parameters
    ----------
    close_prices : pd.Series
        A pandas Series of close prices with a DatetimeIndex.
        The index must be datetime-like to compute daily returns.
    lookback_span : int, default=100
        The span (in days) for the exponentially weighted moving standard deviation.
        A larger span gives more weight to historical observations, resulting in
        smoother volatility estimates.

    Returns
    -------
    pd.Series
        A pandas Series containing the daily volatility estimates, indexed to match
        the input close prices. The series is named 'daily_volatility'.

    Notes
    -----
    The function computes daily returns by finding the price from approximately
    one day ago and calculating the percentage change. It then applies an
    exponentially weighted moving standard deviation with the specified span.

    This approach is useful because:
    1. It adapts to changing market conditions (high/low volatility regimes)
    2. It provides smoother estimates than simple rolling windows
    3. It can be used at intraday frequency while still capturing daily volatility

    Examples
    --------
    >>> import pandas as pd
    >>> close = pd.Series([100, 101, 99, 102, 98],
    ...                   index=pd.date_range('2023-01-01', periods=5, freq='D'))
    >>> vol = get_daily_volatility(close, lookback_span=20)

    References
    ----------
    AFML Chapter 3, Snippet 3.1: Daily Volatility Estimates
    """
    # Find indices of prices from approximately 1 day ago
    one_day_ago_indices = close_prices.index.searchsorted(
        close_prices.index - pd.Timedelta(days=1)
    )

    # Filter to only include valid indices (where we have prior data)
    valid_indices = one_day_ago_indices[one_day_ago_indices > 0]

    # Create a series mapping current timestamps to their 1-day-ago timestamps
    num_valid = valid_indices.shape[0]
    current_timestamps = close_prices.index[close_prices.shape[0] - num_valid:]
    prior_timestamps = close_prices.index[valid_indices - 1]

    timestamp_mapping = pd.Series(
        prior_timestamps,
        index=current_timestamps,
    )

    # Compute daily returns
    try:
        current_prices = close_prices.loc[timestamp_mapping.index]
        prior_prices = close_prices.loc[timestamp_mapping.values].values
        daily_returns = current_prices / prior_prices - 1
    except Exception as e:
        raise ValueError(
            f"Error computing daily returns: {e}. "
            "Please confirm there are no duplicate indices in the price series."
        )

    # Compute exponentially weighted moving standard deviation
    daily_volatility = daily_returns.ewm(span=lookback_span).std()
    daily_volatility = daily_volatility.rename("daily_volatility")

    return daily_volatility
