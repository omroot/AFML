"""
Dynamic position sizing with limit prices.

This module provides functions for dynamic bet sizing that adjusts
positions based on divergence from the market maker's forecast,
and computes limit prices at which positions should be updated.

Reference: AFML Chapter 10, Section 10.6, Snippet 10.4
"""

from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd


def bet_size_sigmoid(
    divergence: float,
    omega: float = 1.0,
) -> float:
    """
    Compute bet size using sigmoid function.

    The sigmoid function provides a smooth mapping from divergence
    to bet size, with controllable sensitivity via omega.

    Parameters
    ----------
    divergence : float
        Divergence between price and forecast. Positive values indicate
        undervaluation (price below forecast), negative values indicate
        overvaluation.
    omega : float, default=1.0
        Sensitivity parameter controlling how quickly bet size
        responds to divergence. Higher omega means slower response.
        - omega → 0: step function (immediate full position)
        - omega → ∞: very gradual position changes

    Returns
    -------
    float
        Bet size in range [-1, 1].

    Notes
    -----
    The sigmoid function is:
        m[ω, x] = x / sqrt(ω + x²)

    Where:
        - x is the divergence
        - ω is the sensitivity parameter
        - m is the resulting bet size

    Properties:
    - Bounded: m ∈ [-1, 1]
    - Continuous and differentiable
    - Monotonic: higher divergence → higher bet size
    - Symmetric: m(-x) = -m(x)
    - At x=0: m=0 (no divergence → no position)

    References
    ----------
    AFML Chapter 10, Section 10.6.1, Snippet 10.4

    Examples
    --------
    >>> # Small divergence with default sensitivity
    >>> bet_size_sigmoid(0.5, omega=1.0)  # ~0.45

    >>> # Same divergence with lower sensitivity (faster response)
    >>> bet_size_sigmoid(0.5, omega=0.1)  # ~0.85

    >>> # Same divergence with higher sensitivity (slower response)
    >>> bet_size_sigmoid(0.5, omega=10.0)  # ~0.16
    """
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}")

    # Sigmoid function: m = x / sqrt(omega + x^2)
    bet_size = divergence / np.sqrt(omega + divergence ** 2)

    return bet_size


def bet_size_power(
    divergence: float,
    omega: float = 1.0,
) -> float:
    """
    Compute bet size using power function.

    Alternative to sigmoid that provides different sensitivity characteristics.

    Parameters
    ----------
    divergence : float
        Divergence between price and forecast.
    omega : float, default=1.0
        Sensitivity parameter. Higher omega means slower response.

    Returns
    -------
    float
        Bet size in range [-1, 1].

    Notes
    -----
    The power function is:
        m[ω, x] = sign(x) * |x|^ω  for 0 < ω <= 1

    This provides:
    - Convex response for ω < 1 (aggressive at small divergences)
    - Linear response for ω = 1
    - Concave response for ω > 1 (conservative at small divergences)

    Examples
    --------
    >>> bet_size_power(0.5, omega=0.5)  # ~0.71 (aggressive)
    >>> bet_size_power(0.5, omega=1.0)  # 0.5 (linear)
    >>> bet_size_power(0.5, omega=2.0)  # 0.25 (conservative)
    """
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}")

    # Clip divergence to [-1, 1] first
    divergence = np.clip(divergence, -1, 1)

    # Power function: m = sign(x) * |x|^omega
    bet_size = np.sign(divergence) * np.abs(divergence) ** omega

    return bet_size


def compute_target_position(
    forecast: float,
    price: float,
    current_position: float,
    omega: float = 1.0,
    bet_func: str = 'sigmoid',
    max_position: float = 1.0,
) -> float:
    """
    Compute target position based on price divergence from forecast.

    Parameters
    ----------
    forecast : float
        The market maker's price forecast (fair value).
    price : float
        Current market price.
    current_position : float
        Current position size.
    omega : float, default=1.0
        Sensitivity parameter for bet sizing function.
    bet_func : str, default='sigmoid'
        Bet sizing function to use: 'sigmoid' or 'power'.
    max_position : float, default=1.0
        Maximum allowed position size (scales the output).

    Returns
    -------
    float
        Target position size.

    Notes
    -----
    The target position is computed as:
        target = max_position * bet_size(divergence, omega)

    Where divergence = forecast - price (positive when undervalued).

    Examples
    --------
    >>> # Stock is undervalued (price below forecast)
    >>> target = compute_target_position(
    ...     forecast=110, price=100, current_position=0.3, omega=1.0
    ... )
    >>> print(f"Target position: {target:.2f}")
    """
    # Compute divergence (positive = undervalued)
    divergence = forecast - price

    # Normalize divergence by price for percentage basis
    if price != 0:
        divergence_pct = divergence / price
    else:
        divergence_pct = divergence

    # Compute bet size
    if bet_func == 'sigmoid':
        bet_size = bet_size_sigmoid(divergence_pct, omega)
    elif bet_func == 'power':
        bet_size = bet_size_power(divergence_pct, omega)
    else:
        raise ValueError(f"Unknown bet_func: {bet_func}. Use 'sigmoid' or 'power'.")

    # Scale to max position
    target = max_position * bet_size

    return target


def inverse_price_sigmoid(
    target_position: float,
    forecast: float,
    omega: float = 1.0,
) -> float:
    """
    Compute price at which target position would be achieved (sigmoid).

    This is the inverse of the bet sizing function: given a target
    position, what price would generate that position?

    Parameters
    ----------
    target_position : float
        Desired position size in range [-1, 1].
    forecast : float
        The market maker's price forecast.
    omega : float, default=1.0
        Sensitivity parameter (must match the one used for bet sizing).

    Returns
    -------
    float
        Price at which the target position would be achieved.

    Notes
    -----
    For the sigmoid function m = x / sqrt(omega + x^2), the inverse is:
        x = m * sqrt(omega / (1 - m^2))

    Then: price = forecast - x * forecast (for percentage-based divergence)

    References
    ----------
    AFML Chapter 10, Section 10.6.2

    Examples
    --------
    >>> # At what price would we want a 0.5 position?
    >>> price = inverse_price_sigmoid(target_position=0.5, forecast=110, omega=1.0)
    >>> print(f"Target price: {price:.2f}")
    """
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}")

    # Clip target to avoid division by zero
    m = np.clip(target_position, -0.99999, 0.99999)

    # Inverse sigmoid: x = m * sqrt(omega / (1 - m^2))
    divergence_pct = m * np.sqrt(omega / (1 - m ** 2))

    # Convert back to price: price = forecast / (1 + divergence_pct)
    price = forecast / (1 + divergence_pct)

    return price


def compute_limit_price(
    target_position: float,
    current_position: float,
    forecast: float,
    omega: float = 1.0,
    side: int = 1,
) -> float:
    """
    Compute limit price for position update.

    Given current and target positions, compute the price at which
    we should place a limit order to adjust our position.

    Parameters
    ----------
    target_position : float
        Desired position size after the trade.
    current_position : float
        Current position size.
    forecast : float
        The market maker's price forecast.
    omega : float, default=1.0
        Sensitivity parameter.
    side : int, default=1
        Side of the trade: 1 for buy, -1 for sell.

    Returns
    -------
    float
        Limit price for the order.

    Notes
    -----
    For buying (side=1):
        - We want to buy at a price where our position would be target_position
        - This is the inverse of bet_size at target_position

    For selling (side=-1):
        - We want to sell at a price where our position would be target_position
        - Similarly computed from the inverse

    Examples
    --------
    >>> # Compute buy limit price to increase position from 0.2 to 0.5
    >>> limit = compute_limit_price(
    ...     target_position=0.5, current_position=0.2,
    ...     forecast=110, omega=1.0, side=1
    ... )
    >>> print(f"Buy limit: {limit:.2f}")
    """
    # Compute price at which target position would be achieved
    limit_price = inverse_price_sigmoid(target_position, forecast, omega)

    return limit_price


def compute_omega_from_price_range(
    max_divergence_pct: float,
    position_at_max: float = 0.95,
) -> float:
    """
    Compute omega parameter from desired price range behavior.

    Parameters
    ----------
    max_divergence_pct : float
        Maximum expected divergence (as percentage, e.g., 0.1 for 10%).
    position_at_max : float, default=0.95
        Desired position size at maximum divergence.

    Returns
    -------
    float
        Omega parameter that achieves the desired behavior.

    Notes
    -----
    For sigmoid function m = x / sqrt(omega + x^2), solving for omega:
        omega = x^2 * (1/m^2 - 1)

    Examples
    --------
    >>> # Want 95% position at 10% divergence
    >>> omega = compute_omega_from_price_range(0.1, 0.95)
    >>> print(f"Omega: {omega:.4f}")
    """
    if position_at_max <= 0 or position_at_max >= 1:
        raise ValueError("position_at_max must be in (0, 1)")

    x = max_divergence_pct
    m = position_at_max

    # omega = x^2 * (1/m^2 - 1)
    omega = x ** 2 * (1 / m ** 2 - 1)

    return omega


def get_position_schedule(
    forecast: float,
    price_range: Tuple[float, float],
    omega: float = 1.0,
    num_points: int = 100,
) -> pd.DataFrame:
    """
    Generate position schedule for a range of prices.

    Parameters
    ----------
    forecast : float
        The market maker's price forecast.
    price_range : tuple
        (min_price, max_price) range to evaluate.
    omega : float, default=1.0
        Sensitivity parameter.
    num_points : int, default=100
        Number of price points to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'price', 'divergence_pct', 'position'.

    Examples
    --------
    >>> schedule = get_position_schedule(forecast=100, price_range=(90, 110))
    >>> print(schedule.head())
    """
    prices = np.linspace(price_range[0], price_range[1], num_points)

    divergences = (forecast - prices) / prices
    positions = [bet_size_sigmoid(d, omega) for d in divergences]

    return pd.DataFrame({
        'price': prices,
        'divergence_pct': divergences,
        'position': positions,
    })


def get_limit_price_schedule(
    forecast: float,
    position_levels: np.ndarray,
    omega: float = 1.0,
) -> pd.DataFrame:
    """
    Generate limit price schedule for discrete position levels.

    Parameters
    ----------
    forecast : float
        The market maker's price forecast.
    position_levels : array-like
        Array of position levels to compute prices for.
    omega : float, default=1.0
        Sensitivity parameter.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'position', 'limit_price'.

    Examples
    --------
    >>> positions = np.array([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    >>> schedule = get_limit_price_schedule(forecast=100, position_levels=positions)
    >>> print(schedule)
    """
    limit_prices = [
        inverse_price_sigmoid(pos, forecast, omega)
        for pos in position_levels
    ]

    return pd.DataFrame({
        'position': position_levels,
        'limit_price': limit_prices,
    })


def dynamic_position_update(
    current_position: float,
    current_price: float,
    forecast: float,
    omega: float = 1.0,
    step_size: float = 0.1,
    max_position: float = 1.0,
) -> dict:
    """
    Compute dynamic position update with discretization.

    Combines dynamic bet sizing with discretization to determine
    if and how to update position.

    Parameters
    ----------
    current_position : float
        Current position size.
    current_price : float
        Current market price.
    forecast : float
        Price forecast.
    omega : float, default=1.0
        Sensitivity parameter.
    step_size : float, default=0.1
        Discretization step size.
    max_position : float, default=1.0
        Maximum position size.

    Returns
    -------
    dict
        Dictionary containing:
        - 'target_continuous': continuous target position
        - 'target_discrete': discretized target position
        - 'current_discrete': discretized current position
        - 'trade_required': whether trade is needed
        - 'trade_size': size of required trade
        - 'limit_price': limit price for the trade

    Examples
    --------
    >>> result = dynamic_position_update(
    ...     current_position=0.3,
    ...     current_price=95,
    ...     forecast=100,
    ...     omega=1.0,
    ...     step_size=0.1,
    ... )
    >>> print(f"Trade required: {result['trade_required']}")
    >>> print(f"Trade size: {result['trade_size']:.2f}")
    """
    # Compute continuous target position
    divergence_pct = (forecast - current_price) / current_price
    target_continuous = max_position * bet_size_sigmoid(divergence_pct, omega)

    # Discretize
    target_discrete = np.round(target_continuous / step_size) * step_size
    target_discrete = np.clip(target_discrete, -max_position, max_position)

    current_discrete = np.round(current_position / step_size) * step_size
    current_discrete = np.clip(current_discrete, -max_position, max_position)

    # Determine trade
    trade_size = target_discrete - current_discrete
    trade_required = abs(trade_size) >= step_size / 2

    # Compute limit price for the next discrete level
    if trade_size > 0:
        # Buying: compute price for target_discrete
        next_level = current_discrete + step_size
    elif trade_size < 0:
        # Selling: compute price for target_discrete
        next_level = current_discrete - step_size
    else:
        next_level = current_discrete

    next_level = np.clip(next_level, -max_position, max_position)
    limit_price = inverse_price_sigmoid(next_level / max_position, forecast, omega)

    return {
        'target_continuous': target_continuous,
        'target_discrete': target_discrete,
        'current_discrete': current_discrete,
        'trade_required': trade_required,
        'trade_size': trade_size,
        'limit_price': limit_price,
    }
