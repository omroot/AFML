"""
Bet sizing from predicted probabilities.

This module provides functions to convert ML classifier probabilities
into bet sizes using statistical transformations.

Reference: AFML Chapter 10, Section 10.3, Snippet 10.1
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_bet_size_from_probability(
    probability: float,
    num_classes: int = 2,
    predicted_side: Optional[int] = None,
) -> float:
    """
    Compute bet size from predicted probability.

    Converts a classifier's predicted probability into a bet size
    using a z-score transformation and the Gaussian CDF.

    Parameters
    ----------
    probability : float
        The predicted probability of the chosen class, in range (0.5, 1].
        For binary classification, this is p[x=1].
        For multi-class, this is the maximum probability across classes.
    num_classes : int, default=2
        Number of possible classes/outcomes.
    predicted_side : int, optional
        The predicted side/label. For binary classification with labels
        {-1, 1} or {0, 1}, this determines the sign of the bet.
        If None, assumes positive side.

    Returns
    -------
    float
        Bet size in range [-1, 1], where:
        - -1 indicates a full short position
        - +1 indicates a full long position
        - 0 indicates no position

    Notes
    -----
    For binary classification (num_classes=2):
        z = (p - 0.5) / sqrt(p * (1 - p))
        m = 2 * Z[z] - 1

    For multi-class classification:
        z = (p - 1/|X|) / sqrt(p * (1 - p))
        m = side * (2 * Z[z] - 1)

    Where Z[.] is the CDF of the standard Gaussian.

    The intuition is that we test whether the probability is significantly
    different from random chance (0.5 for binary, 1/|X| for multi-class).

    References
    ----------
    AFML Chapter 10, Section 10.3

    Examples
    --------
    >>> # High confidence prediction
    >>> bet_size = compute_bet_size_from_probability(0.9, num_classes=2)
    >>> print(f"Bet size: {bet_size:.4f}")  # ~0.79

    >>> # Low confidence prediction
    >>> bet_size = compute_bet_size_from_probability(0.55, num_classes=2)
    >>> print(f"Bet size: {bet_size:.4f}")  # ~0.15
    """
    # Validate probability
    if probability <= 0 or probability > 1:
        raise ValueError(f"probability must be in (0, 1], got {probability}")

    # Null hypothesis probability (random chance)
    null_probability = 1.0 / num_classes

    # Handle edge case where probability equals null
    if probability <= null_probability:
        return 0.0

    # Compute z-score (test statistic)
    # z = (p - p0) / sqrt(p * (1 - p))
    z_score = (probability - null_probability) / np.sqrt(
        probability * (1 - probability)
    )

    # Convert to bet size using Gaussian CDF
    # m = 2 * Z[z] - 1, where Z is standard normal CDF
    bet_size = 2 * norm.cdf(z_score) - 1

    # Apply side if provided
    if predicted_side is not None:
        bet_size = np.sign(predicted_side) * bet_size

    return bet_size


def get_signal_from_probabilities(
    probabilities: Union[np.ndarray, pd.Series],
    predictions: Union[np.ndarray, pd.Series],
    num_classes: int = 2,
    events: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Convert classifier probabilities to bet signals.

    This is the main function for converting ML predictions to trading
    signals. It handles both binary and multi-class classification,
    and supports meta-labeling.

    Parameters
    ----------
    probabilities : array-like
        Predicted probabilities for each observation. For binary
        classification, this is the probability of class 1.
        For multi-class, this is the maximum probability.
    predictions : array-like
        Predicted labels/sides for each observation.
    num_classes : int, default=2
        Number of possible classes.
    events : pd.DataFrame, optional
        DataFrame with event information. If it contains a 'side' column,
        this is used for meta-labeling (the side is taken from the
        primary model, and the probability modulates the bet size).

    Returns
    -------
    pd.Series
        Bet signals in range [-1, 1] indexed by observation.

    Notes
    -----
    For meta-labeling:
    - The primary model provides the side (long/short)
    - The secondary model provides the probability of success
    - The bet size is: side * (2 * Z[z] - 1)

    For standard labeling:
    - The prediction provides both side and probability
    - The bet size incorporates the predicted label

    References
    ----------
    AFML Chapter 10, Snippet 10.1

    Examples
    --------
    >>> # Binary classification
    >>> probs = np.array([0.6, 0.8, 0.55, 0.9])
    >>> preds = np.array([1, 1, -1, 1])
    >>> signals = get_signal_from_probabilities(probs, preds)

    >>> # Meta-labeling with events DataFrame
    >>> events = pd.DataFrame({'side': [1, 1, -1, 1]})
    >>> signals = get_signal_from_probabilities(probs, preds, events=events)
    """
    # Convert to numpy arrays
    if isinstance(probabilities, pd.Series):
        index = probabilities.index
        probabilities = probabilities.values
    else:
        index = None

    if isinstance(predictions, pd.Series):
        if index is None:
            index = predictions.index
        predictions = predictions.values

    if len(probabilities) == 0:
        return pd.Series(dtype=float)

    # Compute z-scores for one-vs-rest approach
    # z = (p - 1/|X|) / sqrt(p * (1 - p))
    null_probability = 1.0 / num_classes

    # Clip probabilities to avoid numerical issues
    prob_clipped = np.clip(probabilities, null_probability + 1e-10, 1 - 1e-10)

    z_scores = (prob_clipped - null_probability) / np.sqrt(
        prob_clipped * (1 - prob_clipped)
    )

    # Compute bet size magnitude: 2 * Z[z] - 1
    bet_magnitude = 2 * norm.cdf(z_scores) - 1

    # Apply side (from predictions or meta-labeling)
    if events is not None and 'side' in events.columns:
        # Meta-labeling: use side from primary model
        if index is not None:
            sides = events.loc[index, 'side'].values
        else:
            sides = events['side'].values
        signals = sides * bet_magnitude
    else:
        # Standard labeling: use predicted labels as sides
        signals = np.sign(predictions) * bet_magnitude

    # Create output series
    if index is not None:
        return pd.Series(signals, index=index, name='signal')
    else:
        return pd.Series(signals, name='signal')


def probability_to_bet_size_binary(
    probability: Union[float, np.ndarray, pd.Series],
    side: Union[int, np.ndarray, pd.Series] = 1,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert binary classification probability to bet size.

    Simplified function for binary classification with explicit side.

    Parameters
    ----------
    probability : float or array-like
        Probability of positive outcome, in range [0.5, 1].
    side : int or array-like, default=1
        Side of the bet: 1 for long, -1 for short.

    Returns
    -------
    float or array-like
        Bet size(s) in range [-1, 1].

    Examples
    --------
    >>> # Single prediction
    >>> bet = probability_to_bet_size_binary(0.75, side=1)
    >>> print(f"Bet size: {bet:.4f}")

    >>> # Array of predictions
    >>> probs = np.array([0.6, 0.7, 0.8, 0.9])
    >>> sides = np.array([1, 1, -1, 1])
    >>> bets = probability_to_bet_size_binary(probs, sides)
    """
    # Handle scalar vs array
    is_scalar = np.isscalar(probability)

    if is_scalar:
        probability = np.array([probability])
        side = np.array([side]) if np.isscalar(side) else side

    # Clip probabilities
    prob_clipped = np.clip(probability, 0.5 + 1e-10, 1 - 1e-10)

    # Compute z-score
    z_scores = (prob_clipped - 0.5) / np.sqrt(prob_clipped * (1 - prob_clipped))

    # Compute bet size
    bet_sizes = side * (2 * norm.cdf(z_scores) - 1)

    if is_scalar:
        return float(bet_sizes[0])
    else:
        return bet_sizes


def get_bet_size_sigmoid(
    probability: Union[float, np.ndarray],
    num_classes: int = 2,
) -> Union[float, np.ndarray]:
    """
    Compute bet size using sigmoid-like transformation.

    Alternative to the Gaussian CDF approach that provides a smoother
    mapping from probability to bet size.

    Parameters
    ----------
    probability : float or array-like
        Predicted probability in range (0, 1).
    num_classes : int, default=2
        Number of classes.

    Returns
    -------
    float or array-like
        Bet size in range [0, 1] (magnitude only, apply side separately).

    Notes
    -----
    This uses: m = (p - 1/|X|) / (1 - 1/|X|)

    Which linearly maps [1/|X|, 1] to [0, 1].

    Examples
    --------
    >>> bet = get_bet_size_sigmoid(0.8, num_classes=2)
    >>> print(f"Bet size: {bet:.4f}")  # 0.6
    """
    null_prob = 1.0 / num_classes

    # Linear mapping from [null_prob, 1] to [0, 1]
    bet_size = (probability - null_prob) / (1 - null_prob)

    # Clip to valid range
    if isinstance(bet_size, np.ndarray):
        bet_size = np.clip(bet_size, 0, 1)
    else:
        bet_size = max(0, min(1, bet_size))

    return bet_size
