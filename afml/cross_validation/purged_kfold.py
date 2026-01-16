"""
Purged K-Fold cross-validation for financial data.

This module provides a K-Fold cross-validation class that handles
overlapping labels through purging and embargo, preventing information
leakage in financial applications.

Reference: AFML Chapter 7, Section 7.4.3
"""

from typing import Generator, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold


class PurgedKFold(_BaseKFold):
    """
    K-Fold cross-validation with purging and embargo for financial data.

    Extends scikit-learn's KFold to handle overlapping labels by:
    1. Purging training observations that overlap with test labels
    2. Applying an embargo period after the test set

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds.
    label_times : pd.Series
        Series where index is observation start time (t0) and values
        are observation end times (t1). This defines when each label's
        information period spans.
    embargo_pct : float, default=0.0
        Percentage of observations to embargo after each test set.
        A value of 0.01 means 1% embargo.

    Attributes
    ----------
    label_times : pd.Series
        The label times series.
    embargo_pct : float
        The embargo percentage.
    embargo_size : int
        Number of observations to embargo (computed from embargo_pct).

    Notes
    -----
    Unlike standard K-Fold CV, this implementation:
    - Does NOT shuffle (shuffle=False enforced)
    - Requires contiguous test sets (no training between test observations)
    - Removes overlapping observations from training
    - Applies embargo to handle serial correlation

    The test set is assumed to be contiguous - there should be no training
    observations between the first and last test observation.

    References
    ----------
    AFML Chapter 7, Snippet 7.3: Cross-Validation Class When Observations Overlap

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # Create label times (index=start, value=end)
    >>> label_times = pd.Series(
    ...     index=pd.date_range('2020-01-01', periods=1000, freq='D'),
    ...     data=pd.date_range('2020-01-10', periods=1000, freq='D')
    ... )
    >>>
    >>> # Create purged k-fold CV
    >>> cv = PurgedKFold(n_splits=5, label_times=label_times, embargo_pct=0.01)
    >>>
    >>> # Use with cross-validation
    >>> for train_idx, test_idx in cv.split(X):
    ...     model = RandomForestClassifier()
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 3,
        label_times: Optional[pd.Series] = None,
        embargo_pct: float = 0.0,
    ):
        if label_times is not None and not isinstance(label_times, pd.Series):
            raise ValueError("label_times must be a pandas Series")

        # Initialize parent with shuffle=False (required for financial data)
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)

        self.label_times = label_times
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray, optional
            Target variable (not used, for API compatibility).
        groups : np.ndarray, optional
            Group labels (not used, for API compatibility).

        Yields
        ------
        train_indices : np.ndarray
            Training set indices for this fold.
        test_indices : np.ndarray
            Test set indices for this fold.

        Raises
        ------
        ValueError
            If X and label_times have different lengths.
        """
        if self.label_times is None:
            raise ValueError("label_times must be provided")

        # Validate that X and label_times have same length
        if X.shape[0] != len(self.label_times):
            raise ValueError(
                f"X and label_times must have the same length. "
                f"Got X: {X.shape[0]}, label_times: {len(self.label_times)}"
            )

        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        # Compute embargo size
        embargo_size = int(num_samples * self.embargo_pct)

        # Generate fold boundaries
        fold_sizes = np.full(self.n_splits, num_samples // self.n_splits, dtype=int)
        fold_sizes[:num_samples % self.n_splits] += 1

        # Generate test indices for each fold
        test_starts = []
        current = 0
        for fold_size in fold_sizes:
            start = current
            end = current + fold_size
            test_starts.append((start, end))
            current = end

        # Generate train/test splits
        for test_start, test_end in test_starts:
            test_indices = indices[test_start:test_end]

            # Get test time range
            test_time_start = self.label_times.index[test_start]
            test_time_end = self.label_times.iloc[test_indices].max()

            # Find the maximum label end time index for embargo
            max_test_label_idx = self.label_times.index.searchsorted(
                self.label_times.iloc[test_indices].max()
            )

            # Get training indices:
            # 1. All indices before test that end before test starts
            # 2. All indices after test+embargo
            train_before = self.label_times[
                self.label_times <= test_time_start
            ].index
            train_before_indices = np.array([
                self.label_times.index.get_loc(t) for t in train_before
                if self.label_times.index.get_loc(t) < test_start
            ], dtype=np.int64)

            # Apply embargo after test
            embargo_end_idx = min(max_test_label_idx + embargo_size, num_samples)
            train_after_indices = indices[embargo_end_idx:].astype(np.int64)

            # Combine train indices
            if len(train_before_indices) == 0:
                train_indices = train_after_indices
            elif len(train_after_indices) == 0:
                train_indices = train_before_indices
            else:
                train_indices = np.concatenate([train_before_indices, train_after_indices])
            train_indices = np.sort(np.unique(train_indices)).astype(np.int64)

            yield train_indices, test_indices.astype(np.int64)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """
        Return the number of splitting iterations.

        Parameters
        ----------
        X : array-like, optional
            Always ignored, exists for compatibility.
        y : array-like, optional
            Always ignored, exists for compatibility.
        groups : array-like, optional
            Always ignored, exists for compatibility.

        Returns
        -------
        int
            Number of splits.
        """
        return self.n_splits


class PurgedWalkForwardCV:
    """
    Walk-forward cross-validation with purging and embargo.

    Unlike K-Fold CV where test sets can appear anywhere, walk-forward
    CV always trains on past data and tests on future data, respecting
    the temporal nature of financial data.

    Parameters
    ----------
    n_splits : int, default=5
        Number of train/test splits.
    train_pct : float, default=0.6
        Percentage of data to use for training in each split.
    label_times : pd.Series
        Series where index is observation start time and values are end times.
    embargo_pct : float, default=0.01
        Percentage of observations to embargo after training set.

    Notes
    -----
    Walk-forward CV structure:

    ```
    Split 1: [Train     ] [Test]
    Split 2:    [Train     ] [Test]
    Split 3:       [Train     ] [Test]
    ...
    ```

    Each split moves forward in time, simulating how the model would
    be deployed in practice.

    Examples
    --------
    >>> cv = PurgedWalkForwardCV(n_splits=5, train_pct=0.6, label_times=label_times)
    >>> for train_idx, test_idx in cv.split(X):
    ...     # train_idx always before test_idx (temporally)
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_pct: float = 0.6,
        label_times: Optional[pd.Series] = None,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.label_times = label_times
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate walk-forward train/test splits.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray, optional
            Target (not used).
        groups : np.ndarray, optional
            Groups (not used).

        Yields
        ------
        train_indices : np.ndarray
            Training indices.
        test_indices : np.ndarray
            Test indices.
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        # Compute sizes
        test_size = int(num_samples * (1 - self.train_pct) / self.n_splits)
        train_size = int(num_samples * self.train_pct)
        embargo_size = int(num_samples * self.embargo_pct)

        for i in range(self.n_splits):
            # Compute test boundaries
            test_end = num_samples - (self.n_splits - 1 - i) * test_size
            test_start = test_end - test_size

            # Compute train boundaries (with embargo)
            train_end = test_start - embargo_size
            train_start = max(0, train_end - train_size)

            if train_start >= train_end or test_start >= test_end:
                continue

            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]

            # Apply purging if label_times provided
            if self.label_times is not None:
                train_indices = self._purge_train_indices(
                    train_indices, test_indices
                )

            yield train_indices, test_indices

    def _purge_train_indices(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> np.ndarray:
        """Purge overlapping observations from training set."""
        if self.label_times is None:
            return train_indices

        # Get test time range
        test_start_time = self.label_times.index[test_indices[0]]

        # Remove training observations that extend into test period
        purged_mask = self.label_times.iloc[train_indices] < test_start_time
        return train_indices[purged_mask.values]

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits
