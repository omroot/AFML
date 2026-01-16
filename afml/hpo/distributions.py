"""
Custom probability distributions for hyperparameter optimization.

This module provides probability distributions that are useful for
hyperparameter tuning, particularly for parameters that don't respond
linearly to changes (e.g., C in SVC, gamma in RBF kernel).

Reference: AFML Chapter 9, Section 9.3.1, Snippet 9.4
"""

from typing import Optional, Union
import numpy as np


class LogUniformDistribution:
    """
    Log-uniform (reciprocal) distribution for hyperparameter sampling.

    A random variable X follows a log-uniform distribution between a and b
    if and only if log(X) ~ Uniform(log(a), log(b)).

    This distribution is useful for hyperparameters like C (SVC regularization)
    and gamma (RBF kernel), where the effect on model performance is often
    logarithmic rather than linear.

    Parameters
    ----------
    a : float
        Lower bound of the distribution (must be > 0).
    b : float
        Upper bound of the distribution (must be > a).

    Notes
    -----
    The cumulative distribution function (CDF) is:

        F(x) = (log(x) - log(a)) / (log(b) - log(a))  for a <= x <= b
             = 0                                        for x < a
             = 1                                        for x > b

    The probability density function (PDF) is:

        f(x) = 1 / (x * log(b/a))  for a <= x <= b
             = 0                    otherwise

    The CDF is invariant to the base of the logarithm since:
        log_c(x/a) / log_c(b/a) = log(x/a) / log(b/a)

    Why use log-uniform?
    - For U[0, 100], 99 percent of values are > 1, inefficient exploration
    - SVC can be as responsive to C: 0.01->1 as to C: 1->100
    - Log-uniform samples uniformly in log-space, better coverage

    References
    ----------
    AFML Chapter 9, Snippet 9.4: The logUniform_gen Class

    Examples
    --------
    >>> from afml.hpo.distributions import log_uniform
    >>> # Sample 1000 values between 1e-3 and 1e3
    >>> samples = log_uniform(a=1e-3, b=1e3).rvs(size=1000)
    >>> # Verify log-uniformity
    >>> import numpy as np
    >>> log_samples = np.log10(samples)
    >>> print(f"Log range: [{log_samples.min():.2f}, {log_samples.max():.2f}]")
    >>> # Should be roughly uniform in [-3, 3]
    """

    def __init__(
        self,
        a: float = 1e-3,
        b: float = 1e3,
        random_state: Optional[int] = None,
    ):
        if a <= 0:
            raise ValueError(f"Lower bound 'a' must be positive, got {a}")
        if b <= a:
            raise ValueError(
                f"Upper bound 'b' must be greater than 'a', got a={a}, b={b}"
            )

        self.a = a
        self.b = b
        self._rng = np.random.RandomState(random_state)

    def rvs(
        self,
        size: Optional[Union[int, tuple]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        """
        Generate random variates from the log-uniform distribution.

        Parameters
        ----------
        size : int or tuple, optional
            Number of samples to generate. If None, returns a single value.
        random_state : int or RandomState, optional
            Random state for reproducibility. Can be an int or a
            numpy.random.RandomState instance. If None, uses the instance's
            random state.

        Returns
        -------
        np.ndarray
            Random samples from the log-uniform distribution.

        Examples
        --------
        >>> dist = LogUniformDistribution(a=1e-2, b=1e2)
        >>> samples = dist.rvs(size=1000)
        >>> # Check that log(samples) is roughly uniform
        >>> import numpy as np
        >>> log_samples = np.log10(samples)
        >>> print(f"Mean of log10: {log_samples.mean():.2f}")  # Should be ~0
        """
        if random_state is not None:
            if isinstance(random_state, np.random.RandomState):
                rng = random_state
            else:
                rng = np.random.RandomState(random_state)
        else:
            rng = self._rng

        # Generate uniform samples in [0, 1]
        uniform_samples = rng.uniform(0, 1, size=size)

        # Transform to log-uniform: x = a * (b/a)^u
        log_uniform_samples = self.a * np.power(self.b / self.a, uniform_samples)

        return log_uniform_samples

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Cumulative distribution function.

        Parameters
        ----------
        x : np.ndarray
            Values at which to evaluate the CDF.

        Returns
        -------
        np.ndarray
            CDF values.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = (x >= self.a) & (x <= self.b)
        result[mask] = np.log(x[mask] / self.a) / np.log(self.b / self.a)
        result[x > self.b] = 1.0
        return result

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Percent point function (inverse of CDF).

        Parameters
        ----------
        q : np.ndarray
            Quantiles (between 0 and 1).

        Returns
        -------
        np.ndarray
            Values corresponding to the quantiles.
        """
        q = np.asarray(q)
        return self.a * np.power(self.b / self.a, q)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Probability density function.

        Parameters
        ----------
        x : np.ndarray
            Values at which to evaluate the PDF.

        Returns
        -------
        np.ndarray
            PDF values.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = (x >= self.a) & (x <= self.b)
        result[mask] = 1.0 / (x[mask] * np.log(self.b / self.a))
        return result

    def __repr__(self) -> str:
        return f"LogUniformDistribution(a={self.a}, b={self.b})"


def log_uniform(
    a: float = 1e-3,
    b: float = 1e3,
    random_state: Optional[int] = None,
) -> LogUniformDistribution:
    """
    Create a log-uniform distribution.

    Convenience function to create a LogUniformDistribution instance.

    Parameters
    ----------
    a : float, default=1e-3
        Lower bound of the distribution (must be > 0).
    b : float, default=1e3
        Upper bound of the distribution (must be > a).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    LogUniformDistribution
        A log-uniform distribution instance.

    Examples
    --------
    >>> from afml.hpo.distributions import log_uniform
    >>> # Create distribution for SVC's C parameter
    >>> C_dist = log_uniform(a=1e-2, b=1e2)
    >>> samples = C_dist.rvs(size=100)
    >>> print(f"Sample range: [{samples.min():.4f}, {samples.max():.4f}]")

    >>> # Use with RandomizedSearchCV
    >>> param_distributions = {
    ...     'C': log_uniform(a=1e-2, b=1e2),
    ...     'gamma': log_uniform(a=1e-2, b=1e2),
    ... }
    """
    return LogUniformDistribution(a=a, b=b, random_state=random_state)


class IntLogUniformDistribution:
    """
    Integer log-uniform distribution.

    Similar to LogUniformDistribution but returns integer values.
    Useful for hyperparameters that must be integers (e.g., n_estimators).

    Parameters
    ----------
    a : int
        Lower bound (must be >= 1).
    b : int
        Upper bound (must be > a).
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> dist = int_log_uniform(a=10, b=1000)
    >>> samples = dist.rvs(size=100)
    >>> print(f"All integers: {all(s == int(s) for s in samples)}")
    """

    def __init__(
        self,
        a: int = 10,
        b: int = 1000,
        random_state: Optional[int] = None,
    ):
        if a < 1:
            raise ValueError(f"Lower bound 'a' must be >= 1, got {a}")
        if b <= a:
            raise ValueError(
                f"Upper bound 'b' must be greater than 'a', got a={a}, b={b}"
            )

        self.a = int(a)
        self.b = int(b)
        self._log_uniform = LogUniformDistribution(
            a=float(a), b=float(b), random_state=random_state
        )

    def rvs(
        self,
        size: Optional[Union[int, tuple]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        """
        Generate random integer variates from the log-uniform distribution.

        Parameters
        ----------
        size : int or tuple, optional
            Number of samples to generate.
        random_state : int or RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        np.ndarray
            Random integer samples.
        """
        continuous_samples = self._log_uniform.rvs(size=size, random_state=random_state)
        return np.round(continuous_samples).astype(int)

    def __repr__(self) -> str:
        return f"IntLogUniformDistribution(a={self.a}, b={self.b})"


def int_log_uniform(
    a: int = 10,
    b: int = 1000,
    random_state: Optional[int] = None,
) -> IntLogUniformDistribution:
    """
    Create an integer log-uniform distribution.

    Parameters
    ----------
    a : int, default=10
        Lower bound (must be >= 1).
    b : int, default=1000
        Upper bound (must be > a).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    IntLogUniformDistribution
        An integer log-uniform distribution instance.

    Examples
    --------
    >>> dist = int_log_uniform(a=10, b=1000)
    >>> samples = dist.rvs(size=100)
    >>> print(f"All integers: {all(s == int(s) for s in samples)}")
    """
    return IntLogUniformDistribution(a=a, b=b, random_state=random_state)


def test_log_uniform_distribution():
    """
    Test the log-uniform distribution implementation.

    This function replicates the testing from AFML Snippet 9.4.

    Examples
    --------
    >>> test_log_uniform_distribution()
    """
    from scipy.stats import kstest
    import matplotlib.pyplot as plt

    # Create distribution
    a, b = 1e-3, 1e3
    size = 10000
    dist = log_uniform(a=a, b=b)

    # Generate samples
    samples = dist.rvs(size=size)

    # Test that log(samples) is uniformly distributed
    log_samples = np.log(samples)
    log_a, log_b = np.log(a), np.log(b)

    # Kolmogorov-Smirnov test for uniformity of log-samples
    ks_stat, p_value = kstest(
        log_samples, "uniform", args=(log_a, log_b - log_a)
    )

    print(f"Log-uniform distribution test:")
    print(f"  Sample size: {size}")
    print(f"  Range: [{a}, {b}]")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Log(samples) mean: {log_samples.mean():.4f} (expected: {(log_a + log_b)/2:.4f})")

    return ks_stat, p_value
