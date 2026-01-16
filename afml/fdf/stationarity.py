"""
Stationarity testing and optimal differentiation order finding.

This module provides functions to test for stationarity and find
the minimum fractional differentiation order (d) that achieves
stationarity while preserving maximum memory.

Reference: AFML Chapter 5, Sections 5.5-5.6
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def adf_test(
    series: pd.Series,
    max_lag: Optional[int] = None,
    regression: str = "c",
    autolag: str = "AIC",
) -> Dict[str, float]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    The ADF test is used to determine if a time series is stationary.
    A stationary series has constant mean and variance over time.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    max_lag : int, optional
        Maximum lag to include in the test. If None, uses
        int(4*(nobs/100)^0.25) based on Schwert (1989).
    regression : str, default='c'
        Type of regression to include:
        - 'c': constant only (default)
        - 'ct': constant and trend
        - 'ctt': constant, trend, and trend squared
        - 'n': no constant or trend
    autolag : str, default='AIC'
        Method for automatic lag selection:
        - 'AIC': Akaike Information Criterion
        - 'BIC': Bayesian Information Criterion
        - 't-stat': Based on last lag significance

    Returns
    -------
    dict
        Dictionary containing:
        - 'adf_statistic': Test statistic
        - 'p_value': p-value for the test
        - 'lags_used': Number of lags used
        - 'num_observations': Number of observations used
        - 'critical_1%': Critical value at 1% significance
        - 'critical_5%': Critical value at 5% significance
        - 'critical_10%': Critical value at 10% significance
        - 'is_stationary': True if p_value < 0.05

    Notes
    -----
    The null hypothesis is that the series has a unit root (non-stationary).
    If p-value < significance level (typically 0.05), reject the null
    and conclude the series is stationary.

    Examples
    --------
    >>> result = adf_test(price_series)
    >>> if result['is_stationary']:
    ...     print("Series is stationary")
    >>> print(f"p-value: {result['p_value']:.4f}")
    """
    # Drop NaN values
    clean_series = series.dropna()

    if len(clean_series) < 20:
        raise ValueError(
            f"Series too short for ADF test: {len(clean_series)} observations"
        )

    # Run ADF test
    adf_result = adfuller(
        clean_series,
        maxlag=max_lag,
        regression=regression,
        autolag=autolag,
    )

    return {
        "adf_statistic": adf_result[0],
        "p_value": adf_result[1],
        "lags_used": adf_result[2],
        "num_observations": adf_result[3],
        "critical_1%": adf_result[4]["1%"],
        "critical_5%": adf_result[4]["5%"],
        "critical_10%": adf_result[4]["10%"],
        "is_stationary": adf_result[1] < 0.05,
    }


def find_minimum_d(
    series: Union[pd.Series, pd.DataFrame],
    d_values: Optional[List[float]] = None,
    weight_threshold: float = 1e-5,
    significance_level: float = 0.05,
    adf_regression: str = "c",
    adf_autolag: str = "AIC",
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Find the minimum d value that achieves stationarity.

    Iterates through candidate d values and finds the smallest one
    where the ADF test rejects the null hypothesis of non-stationarity.
    This preserves maximum memory while achieving stationarity.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Time series to analyze. If DataFrame, uses first column.
    d_values : list of float, optional
        Candidate d values to test. If None, uses np.arange(0, 1.05, 0.05).
    weight_threshold : float, default=1e-5
        Threshold for FFD weight truncation.
    significance_level : float, default=0.05
        Significance level for ADF test.
    adf_regression : str, default='c'
        Regression type for ADF test.
    adf_autolag : str, default='AIC'
        Autolag method for ADF test.

    Returns
    -------
    dict
        Dictionary containing:
        - 'min_d': Minimum d achieving stationarity (None if not found)
        - 'results': DataFrame with d values, ADF stats, and p-values
        - 'correlation': Correlation between original and fracdiff series
          at min_d (None if min_d not found)

    Notes
    -----
    The goal is to find the smallest d such that:
    1. The fractionally differentiated series is stationary (passes ADF)
    2. Maximum memory from the original series is preserved

    The correlation between original and fracdiff series indicates
    how much predictive information is retained.

    References
    ----------
    AFML Chapter 5, Snippet 5.4: plotMinFFD function

    Examples
    --------
    >>> result = find_minimum_d(price_series)
    >>> print(f"Minimum d: {result['min_d']}")
    >>> print(f"Memory preserved: {result['correlation']:.2%}")
    >>> # Plot the results
    >>> result['results'].plot(x='d', y='p_value')
    """
    from afml.fdf.fracdiff import fracdiff_ffd

    # Handle DataFrame input
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    # Default d values
    if d_values is None:
        d_values = np.arange(0.0, 1.05, 0.05).tolist()

    results = []
    min_d = None
    correlation_at_min_d = None

    for d in d_values:
        try:
            # Apply FFD
            fracdiff_series = fracdiff_ffd(
                series, diff_order=d, weight_threshold=weight_threshold
            )

            # Get the fractionally differentiated values
            fd_values = fracdiff_series.iloc[:, 0].dropna()

            if len(fd_values) < 20:
                results.append({
                    "d": d,
                    "adf_statistic": np.nan,
                    "p_value": np.nan,
                    "is_stationary": False,
                })
                continue

            # Run ADF test
            adf_result = adf_test(
                fd_values,
                regression=adf_regression,
                autolag=adf_autolag,
            )

            results.append({
                "d": d,
                "adf_statistic": adf_result["adf_statistic"],
                "p_value": adf_result["p_value"],
                "is_stationary": adf_result["p_value"] < significance_level,
            })

            # Track minimum d that achieves stationarity
            if min_d is None and adf_result["p_value"] < significance_level:
                min_d = d

                # Compute correlation with original series
                common_idx = series.index.intersection(fd_values.index)
                if len(common_idx) > 0:
                    correlation_at_min_d = series.loc[common_idx].corr(
                        fd_values.loc[common_idx]
                    )

        except Exception as e:
            results.append({
                "d": d,
                "adf_statistic": np.nan,
                "p_value": np.nan,
                "is_stationary": False,
            })

    results_df = pd.DataFrame(results)

    return {
        "min_d": min_d,
        "results": results_df,
        "correlation": correlation_at_min_d,
    }


def analyze_stationarity_memory_tradeoff(
    series: Union[pd.Series, pd.DataFrame],
    d_values: Optional[List[float]] = None,
    weight_threshold: float = 1e-5,
) -> pd.DataFrame:
    """
    Analyze the tradeoff between stationarity and memory preservation.

    For each d value, computes the ADF p-value (stationarity indicator)
    and correlation with original series (memory indicator).

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Time series to analyze.
    d_values : list of float, optional
        Candidate d values. If None, uses np.arange(0, 1.05, 0.05).
    weight_threshold : float, default=1e-5
        Threshold for FFD weight truncation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'd': Differentiation order
        - 'p_value': ADF test p-value
        - 'correlation': Correlation with original series
        - 'is_stationary': Whether series passes ADF at 5% level

    Notes
    -----
    This analysis helps visualize the fundamental tradeoff:
    - Higher d → more stationary but less memory
    - Lower d → more memory but less stationary

    The optimal d is the smallest value where p-value < 0.05
    (series becomes stationary).

    Examples
    --------
    >>> tradeoff = analyze_stationarity_memory_tradeoff(price_series)
    >>> # Plot the tradeoff
    >>> fig, ax1 = plt.subplots()
    >>> ax1.plot(tradeoff['d'], tradeoff['p_value'], 'b-', label='p-value')
    >>> ax2 = ax1.twinx()
    >>> ax2.plot(tradeoff['d'], tradeoff['correlation'], 'r-', label='correlation')
    """
    from afml.fdf.fracdiff import fracdiff_ffd

    # Handle DataFrame input
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    # Default d values
    if d_values is None:
        d_values = np.arange(0.0, 1.05, 0.05).tolist()

    results = []

    for d in d_values:
        try:
            # Apply FFD
            fracdiff_series = fracdiff_ffd(
                series, diff_order=d, weight_threshold=weight_threshold
            )

            fd_values = fracdiff_series.iloc[:, 0].dropna()

            if len(fd_values) < 20:
                results.append({
                    "d": d,
                    "p_value": np.nan,
                    "correlation": np.nan,
                    "is_stationary": False,
                })
                continue

            # ADF test
            adf_result = adf_test(fd_values)

            # Correlation with original
            common_idx = series.index.intersection(fd_values.index)
            if len(common_idx) > 0:
                correlation = series.loc[common_idx].corr(fd_values.loc[common_idx])
            else:
                correlation = np.nan

            results.append({
                "d": d,
                "p_value": adf_result["p_value"],
                "correlation": correlation,
                "is_stationary": adf_result["is_stationary"],
            })

        except Exception:
            results.append({
                "d": d,
                "p_value": np.nan,
                "correlation": np.nan,
                "is_stationary": False,
            })

    return pd.DataFrame(results)
