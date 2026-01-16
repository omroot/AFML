"""
Bet Sizing module for financial machine learning.

This module provides functions to convert ML model predictions into
position sizes, implementing the bet sizing strategies from AFML Chapter 10.

Key Components:
---------------

**Probability to Bet Size**
- `compute_bet_size_from_probability`: Convert single probability to bet size
- `get_signal_from_probabilities`: Convert array of probabilities to signals
- `probability_to_bet_size_binary`: Simplified binary classification helper

**Averaging Active Bets**
- `compute_average_active_signals`: Average overlapping signals to reduce turnover
- `compute_signal_with_decay`: Time-decay weighted averaging
- `resample_signals`: Resample to regular frequency

**Size Discretization**
- `discretize_signal`: Round signal to nearest step size
- `adaptive_discretization`: Vary step size based on confidence
- `should_trade`: Check if position change is needed

**Dynamic Position Sizing**
- `bet_size_sigmoid`: Sigmoid function for smooth position sizing
- `bet_size_power`: Power function alternative
- `compute_target_position`: Target position from price/forecast divergence
- `inverse_price_sigmoid`: Price at which position would be achieved
- `compute_limit_price`: Limit price for position updates

Key Concepts:
-------------

**Why Convert Probabilities?**
ML models output probabilities, but we need position sizes. The transformation
m = 2*Z[z] - 1 converts probabilities to [-1, 1] bet sizes, where:
- z-score tests if probability is significantly above random chance
- Z[.] is the standard normal CDF (maps to [0, 1])
- The final mapping gives [-1, 1] position sizes

**Why Average Active Bets?**
At any time, multiple trading signals may be active (overlapping holding
periods). Averaging these signals:
1. Reduces portfolio turnover
2. Smooths position changes
3. Lowers transaction costs

**Why Discretize?**
Small signal changes shouldn't trigger trades. Discretization:
- Rounds positions to discrete levels (e.g., 10%, 20%, etc.)
- Prevents overtrading from noise
- Makes positions more interpretable

**Dynamic Position Sizing**
For market makers with price forecasts, the sigmoid function:
    m[ω, x] = x / sqrt(ω + x²)
provides smooth position sizing based on price-forecast divergence.

Reference: AFML Chapter 10

Examples
--------
>>> from afml.bet_sizing import (
...     compute_bet_size_from_probability,
...     get_signal_from_probabilities,
...     discretize_signal,
...     compute_average_active_signals,
... )
>>>
>>> # Convert classifier probability to bet size
>>> prob = 0.75
>>> bet_size = compute_bet_size_from_probability(prob, num_classes=2)
>>> print(f"Bet size: {bet_size:.4f}")  # ~0.58
>>>
>>> # Discretize the signal
>>> discrete_bet = discretize_signal(bet_size, step_size=0.1)
>>> print(f"Discrete: {discrete_bet}")  # 0.6
"""

# Probability to bet size (Snippet 10.1)
from afml.bet_sizing.probability import (
    compute_bet_size_from_probability,
    get_signal_from_probabilities,
    probability_to_bet_size_binary,
    get_bet_size_sigmoid,
)

# Average active bets (Snippet 10.2)
from afml.bet_sizing.averaging import (
    compute_average_active_signals,
    get_signal_at_time,
    compute_signal_with_decay,
    resample_signals,
    compute_turnover,
)

# Size discretization (Snippet 10.3)
from afml.bet_sizing.discretization import (
    discretize_signal,
    get_discrete_levels,
    compute_required_trade,
    should_trade,
    adaptive_discretization,
    quantile_discretization,
    symmetric_discretization,
)

# Dynamic position sizing (Snippet 10.4)
from afml.bet_sizing.dynamic import (
    bet_size_sigmoid,
    bet_size_power,
    compute_target_position,
    inverse_price_sigmoid,
    compute_limit_price,
    compute_omega_from_price_range,
    get_position_schedule,
    get_limit_price_schedule,
    dynamic_position_update,
)

__all__ = [
    # Probability to bet size
    "compute_bet_size_from_probability",
    "get_signal_from_probabilities",
    "probability_to_bet_size_binary",
    "get_bet_size_sigmoid",
    # Average active bets
    "compute_average_active_signals",
    "get_signal_at_time",
    "compute_signal_with_decay",
    "resample_signals",
    "compute_turnover",
    # Size discretization
    "discretize_signal",
    "get_discrete_levels",
    "compute_required_trade",
    "should_trade",
    "adaptive_discretization",
    "quantile_discretization",
    "symmetric_discretization",
    # Dynamic position sizing
    "bet_size_sigmoid",
    "bet_size_power",
    "compute_target_position",
    "inverse_price_sigmoid",
    "compute_limit_price",
    "compute_omega_from_price_range",
    "get_position_schedule",
    "get_limit_price_schedule",
    "dynamic_position_update",
]
