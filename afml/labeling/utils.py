"""
Utility functions for label processing and preparation.

This module provides helper functions for preparing labeled data for
machine learning, including handling class imbalance and data validation.

Reference: AFML Chapter 3, Section 3.9
"""

from typing import Optional
import pandas as pd
import numpy as np


def drop_rare_labels(
    events: pd.DataFrame,
    min_percentage: float = 0.05,
    label_column: str = "bin",
    min_classes: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Recursively drop observations with rare labels to address class imbalance.

    Some ML classifiers perform poorly when classes are heavily imbalanced.
    This function removes observations associated with extremely rare labels
    until all remaining labels appear at least min_percentage of the time
    (or until only min_classes remain).

    Parameters
    ----------
    events : pd.DataFrame
        A DataFrame containing a label column. Typically the output from
        get_labels() or get_labels_side_and_size().
    min_percentage : float, default=0.05
        Minimum percentage threshold. Labels appearing less than this
        fraction of total observations will be dropped.
    label_column : str, default='bin'
        Name of the column containing labels.
    min_classes : int, default=2
        Minimum number of classes to retain. The function stops dropping
        labels once this threshold is reached, regardless of min_percentage.
    verbose : bool, default=True
        If True, print information about dropped labels.

    Returns
    -------
    pd.DataFrame
        A DataFrame with rare labels removed.

    Notes
    -----
    The function operates recursively, dropping the rarest label first,
    then recalculating percentages and repeating until the stopping
    condition is met.

    This is particularly useful when:
    1. The '0' label is rare (few vertical barrier touches)
    2. Strong market trends create imbalance between +1 and -1 labels
    3. Meta-labeling produces heavily skewed binary labels

    Warnings
    --------
    Be cautious when using this function as it can significantly reduce
    your dataset size. Consider alternative approaches like:
    - Class weighting in your ML algorithm
    - Oversampling rare classes (SMOTE)
    - Using evaluation metrics robust to imbalance (F1, AUC)

    Examples
    --------
    >>> import pandas as pd
    >>> events = pd.DataFrame({
    ...     'ret': [0.01, -0.02, 0.005, 0.03, -0.01],
    ...     'bin': [1, -1, 0, 1, -1]
    ... })
    >>> # If '0' appears less than 5%, it will be dropped
    >>> balanced = drop_rare_labels(events, min_percentage=0.05)

    References
    ----------
    AFML Chapter 3, Snippet 3.8: Dropping Under-Populated Labels
    """
    if label_column not in events.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in events DataFrame. "
            f"Available columns: {list(events.columns)}"
        )

    result = events.copy()

    while True:
        # Calculate percentage of each label
        label_percentages = result[label_column].value_counts(normalize=True)

        # Check stopping conditions
        num_classes = label_percentages.shape[0]
        min_label_percentage = label_percentages.min()

        # Stop if minimum percentage threshold is met or we're at minimum classes
        if min_label_percentage > min_percentage or num_classes <= min_classes:
            break

        # Find and drop the rarest label
        rarest_label = label_percentages.idxmin()
        rarest_percentage = label_percentages.min()

        if verbose:
            print(
                f"Dropping label '{rarest_label}' "
                f"(appears in {rarest_percentage:.2%} of observations)"
            )

        result = result[result[label_column] != rarest_label]

    return result


def validate_events(
    events: pd.DataFrame,
    required_columns: Optional[list] = None,
) -> bool:
    """
    Validate that events DataFrame has required structure.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame to validate.
    required_columns : list, optional
        List of required column names. Defaults to ['t1'].

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If validation fails, with descriptive error message.
    """
    if required_columns is None:
        required_columns = ["t1"]

    # Check that events is a DataFrame
    if not isinstance(events, pd.DataFrame):
        raise ValueError(
            f"events must be a pandas DataFrame, got {type(events).__name__}"
        )

    # Check for required columns
    missing_columns = set(required_columns) - set(events.columns)
    if missing_columns:
        raise ValueError(
            f"events DataFrame missing required columns: {missing_columns}"
        )

    # Check that index is datetime-like
    if not isinstance(events.index, pd.DatetimeIndex):
        try:
            pd.DatetimeIndex(events.index)
        except Exception:
            raise ValueError(
                "events DataFrame index must be datetime-like"
            )

    return True


def compute_label_statistics(
    events: pd.DataFrame,
    label_column: str = "bin",
) -> pd.DataFrame:
    """
    Compute summary statistics for labels.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame with labels.
    label_column : str, default='bin'
        Name of the label column.

    Returns
    -------
    pd.DataFrame
        DataFrame with label statistics including count, percentage,
        and cumulative percentage.
    """
    if label_column not in events.columns:
        raise ValueError(f"Column '{label_column}' not found in events")

    counts = events[label_column].value_counts().sort_index()
    percentages = events[label_column].value_counts(normalize=True).sort_index()

    stats = pd.DataFrame({
        "count": counts,
        "percentage": percentages,
        "cumulative_percentage": percentages.cumsum(),
    })

    stats.index.name = "label"

    return stats


def balance_labels_by_undersampling(
    events: pd.DataFrame,
    label_column: str = "bin",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Balance labels by undersampling majority classes.

    An alternative to dropping rare labels - instead undersample the
    majority classes to match the minority class size.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame with labels.
    label_column : str, default='bin'
        Name of the label column.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with balanced labels through undersampling.

    Notes
    -----
    This approach preserves all minority class samples but discards
    majority class samples. Consider whether this is appropriate for
    your use case - you may lose valuable information.
    """
    if label_column not in events.columns:
        raise ValueError(f"Column '{label_column}' not found in events")

    # Find the size of the smallest class
    min_class_size = events[label_column].value_counts().min()

    # Sample from each class
    balanced_dfs = []
    for label in events[label_column].unique():
        label_subset = events[events[label_column] == label]
        sampled = label_subset.sample(
            n=min_class_size,
            random_state=random_state,
        )
        balanced_dfs.append(sampled)

    result = pd.concat(balanced_dfs)

    # Sort by index to maintain temporal order
    result = result.sort_index()

    return result
