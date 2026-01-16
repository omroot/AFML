"""
Enhanced Pipeline class for handling sample weights.

This module provides a custom Pipeline class that properly handles
sample_weight arguments, which is a limitation of sklearn's standard
Pipeline class.

Reference: AFML Chapter 9, Section 9.2, Snippet 9.2
"""

from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class SampleWeightPipeline(Pipeline):
    """
    Enhanced Pipeline that properly handles sample_weight.

    Scikit-learn's Pipeline.fit() method does not directly accept a
    sample_weight argument. Instead, it expects sample_weight to be
    passed via fit_params with the step name prefix. This class
    provides a workaround that allows passing sample_weight directly.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples that are chained.
        The last object must be an estimator.
    memory : str or object, default=None
        Used to cache the fitted transformers.
    verbose : bool, default=False
        If True, print progress messages.

    Notes
    -----
    The standard sklearn Pipeline requires sample_weight to be passed as:
        pipe.fit(X, y, clf__sample_weight=weights)

    This class allows:
        pipe.fit(X, y, sample_weight=weights)

    The sample_weight is automatically routed to the last step (estimator).

    References
    ----------
    AFML Chapter 9, Snippet 9.2: An Enhanced Pipeline Class

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from afml.hpo.pipeline import SampleWeightPipeline
    >>>
    >>> # Create pipeline
    >>> pipe = SampleWeightPipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', RandomForestClassifier())
    ... ])
    >>>
    >>> # Fit with sample weights (simplified API)
    >>> weights = compute_sample_weights(X, y)
    >>> pipe.fit(X, y, sample_weight=weights)
    """

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
        **fit_params: Any,
    ) -> "SampleWeightPipeline":
        """
        Fit the pipeline with optional sample weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, automatically routed to the
            final estimator step.
        **fit_params : dict
            Additional fit parameters to pass to the pipeline steps.

        Returns
        -------
        self : SampleWeightPipeline
            The fitted pipeline.

        Notes
        -----
        If sample_weight is provided, it is added to fit_params with
        the appropriate step name prefix (e.g., 'clf__sample_weight').
        """
        if sample_weight is not None:
            # Get the name of the last step (the estimator)
            last_step_name = self.steps[-1][0]

            # Add sample_weight to fit_params with the correct prefix
            fit_params[f"{last_step_name}__sample_weight"] = sample_weight

        return super().fit(X, y, **fit_params)


def create_pipeline_with_estimator(
    estimator: Any,
    steps: Optional[list] = None,
    estimator_name: str = "clf",
) -> SampleWeightPipeline:
    """
    Create a SampleWeightPipeline with a given estimator.

    Convenience function to create a pipeline with preprocessing steps
    and a final estimator.

    Parameters
    ----------
    estimator : estimator object
        The final estimator in the pipeline.
    steps : list, default=None
        List of (name, transformer) tuples for preprocessing.
        If None, creates a pipeline with just the estimator.
    estimator_name : str, default='clf'
        Name for the estimator step.

    Returns
    -------
    SampleWeightPipeline
        The constructed pipeline.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import SVC
    >>>
    >>> # Create pipeline with scaler and SVC
    >>> pipe = create_pipeline_with_estimator(
    ...     estimator=SVC(probability=True),
    ...     steps=[('scaler', StandardScaler())],
    ... )
    >>>
    >>> # Or just the estimator
    >>> pipe = create_pipeline_with_estimator(estimator=SVC())
    """
    if steps is None:
        steps = []

    # Add the estimator as the last step
    all_steps = list(steps) + [(estimator_name, estimator)]

    return SampleWeightPipeline(steps=all_steps)
