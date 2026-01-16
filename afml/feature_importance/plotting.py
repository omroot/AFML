"""
Visualization utilities for feature importance.

This module provides plotting functions to visualize feature importance
results from MDI, MDA, and SFI methods.

Reference: AFML Chapter 8, Section 8.6, Snippet 8.10
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(
    importance: pd.DataFrame,
    method: str = 'MDI',
    title: Optional[str] = None,
    oob_score: Optional[float] = None,
    oos_score: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
    top_n: Optional[int] = None,
    color_by_type: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature importance with error bars.

    Creates a horizontal bar chart showing feature importances with
    standard deviation error bars, sorted by importance.

    Parameters
    ----------
    importance : pd.DataFrame
        Feature importance DataFrame with 'mean' and 'std' columns.
    method : str, default='MDI'
        Method name for title ('MDI', 'MDA', or 'SFI').
    title : str, optional
        Custom title. If None, auto-generated from method.
    oob_score : float, optional
        Out-of-bag score to display in title.
    oos_score : float, optional
        Out-of-sample score to display in title.
    figsize : tuple, default=(10, 8)
        Figure size (width, height).
    top_n : int, optional
        Number of top features to show. If None, shows all.
    color_by_type : bool, default=False
        If True, color bars by feature type (I, R, N).
    save_path : str, optional
        Path to save the figure. If None, figure is displayed.

    Returns
    -------
    fig : matplotlib.Figure
        The generated figure.

    References
    ----------
    AFML Chapter 8, Snippet 8.10: Feature Importance Plotting Function

    Examples
    --------
    >>> importance = compute_mdi_importance(clf, feature_names)
    >>> fig = plot_feature_importance(
    ...     importance,
    ...     method='MDI',
    ...     oob_score=0.92,
    ...     oos_score=0.83
    ... )
    >>> plt.show()
    """
    # Sort by importance
    importance_sorted = importance.sort_values('mean', ascending=True)

    # Limit to top_n
    if top_n is not None:
        importance_sorted = importance_sorted.tail(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    if color_by_type:
        colors = []
        for name in importance_sorted.index:
            if name.startswith('I_'):
                colors.append('steelblue')
            elif name.startswith('R_'):
                colors.append('coral')
            elif name.startswith('N_'):
                colors.append('lightgray')
            else:
                colors.append('steelblue')
    else:
        colors = 'steelblue'

    # Plot horizontal bars with error bars
    y_pos = np.arange(len(importance_sorted))
    ax.barh(
        y_pos,
        importance_sorted['mean'],
        xerr=importance_sorted['std'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        error_kw={'ecolor': 'red', 'capsize': 3}
    )

    # Add reference line for MDI (uniform importance threshold)
    if method == 'MDI':
        uniform_threshold = 1.0 / len(importance)
        ax.axvline(
            x=uniform_threshold,
            color='red',
            linestyle='dotted',
            linewidth=1.5,
            label=f'Uniform (1/{len(importance)})'
        )
        ax.set_xlim(0, importance_sorted['mean'].max() * 1.1)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_sorted.index)
    ax.set_xlabel('Importance', fontsize=12)

    # Title
    if title is None:
        title_parts = [f'{method} Feature Importance']
        if oob_score is not None:
            title_parts.append(f'OOB={oob_score:.4f}')
        if oos_score is not None:
            title_parts.append(f'OOS={oos_score:.4f}')
        title = ' | '.join(title_parts)

    ax.set_title(title, fontsize=14)

    # Legend for color coding
    if color_by_type:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', label='Informative'),
            Patch(facecolor='coral', label='Redundant'),
            Patch(facecolor='lightgray', label='Noise'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    elif method == 'MDI':
        ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_importance_comparison(
    mdi_importance: pd.DataFrame,
    mda_importance: pd.DataFrame,
    sfi_importance: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 10),
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare feature importance across MDI, MDA, and SFI methods.

    Creates a side-by-side comparison of the three methods.

    Parameters
    ----------
    mdi_importance : pd.DataFrame
        MDI importance results.
    mda_importance : pd.DataFrame
        MDA importance results.
    sfi_importance : pd.DataFrame
        SFI importance results.
    figsize : tuple, default=(15, 10)
        Figure size.
    top_n : int, default=20
        Number of top features to show per method.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
        The comparison figure.

    Examples
    --------
    >>> fig = plot_importance_comparison(mdi_imp, mda_imp, sfi_imp)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    methods = ['MDI', 'MDA', 'SFI']
    importances = [mdi_importance, mda_importance, sfi_importance]

    for ax, method, imp in zip(axes, methods, importances):
        # Sort and limit
        imp_sorted = imp.sort_values('mean', ascending=True).tail(top_n)

        # Plot
        y_pos = np.arange(len(imp_sorted))
        ax.barh(
            y_pos,
            imp_sorted['mean'],
            xerr=imp_sorted['std'],
            color='steelblue',
            alpha=0.8,
            error_kw={'ecolor': 'red', 'capsize': 2}
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp_sorted.index, fontsize=8)
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')

    plt.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_importance_heatmap(
    importance_dict: dict,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmap of feature importance across methods.

    Parameters
    ----------
    importance_dict : dict
        Dictionary mapping method name to importance DataFrame.
    figsize : tuple, default=(12, 10)
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure

    Examples
    --------
    >>> importance_dict = {'MDI': mdi_imp, 'MDA': mda_imp, 'SFI': sfi_imp}
    >>> fig = plot_importance_heatmap(importance_dict)
    """
    # Create combined DataFrame
    combined = pd.DataFrame()
    for method, imp in importance_dict.items():
        combined[method] = imp['mean']

    # Sort by average importance
    combined['avg'] = combined.mean(axis=1)
    combined = combined.sort_values('avg', ascending=False)
    combined = combined.drop('avg', axis=1)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(combined.values, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(combined.columns)))
    ax.set_yticks(np.arange(len(combined.index)))
    ax.set_xticklabels(combined.columns)
    ax.set_yticklabels(combined.index, fontsize=8)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Importance', rotation=-90, va='bottom')

    ax.set_title('Feature Importance Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_importance_vs_pca(
    importance: pd.DataFrame,
    eigenvalues: pd.Series,
    method: str = 'MDI',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of feature importance vs PCA eigenvalues.

    Visualizes the relationship between supervised importance (MDI/MDA/SFI)
    and unsupervised PCA ranking. High correlation suggests pattern is
    not entirely overfit.

    Parameters
    ----------
    importance : pd.DataFrame
        Feature importance with 'mean' column.
    eigenvalues : pd.Series
        PCA eigenvalues for orthogonalized features.
    method : str, default='MDI'
        Name of importance method for title.
    figsize : tuple, default=(10, 8)
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure

    Examples
    --------
    >>> ortho_X, eigenvalues, _ = compute_orthogonal_features(X)
    >>> mdi_imp = compute_mdi_importance(clf_ortho, ortho_X.columns.tolist())
    >>> fig = plot_importance_vs_pca(mdi_imp, eigenvalues)
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau

    # Align data
    common_features = importance.index.intersection(eigenvalues.index)

    if len(common_features) == 0:
        raise ValueError("No common features between importance and eigenvalues")

    imp_values = importance.loc[common_features, 'mean'].values
    eig_values = eigenvalues.loc[common_features].values

    # Compute correlations
    pearson_r, _ = pearsonr(imp_values, eig_values)
    spearman_r, _ = spearmanr(imp_values, eig_values)
    kendall_tau, _ = kendalltau(imp_values, eig_values)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(eig_values, imp_values, alpha=0.6, edgecolors='black')

    # Add correlation line
    z = np.polyfit(eig_values, imp_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(eig_values.min(), eig_values.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Linear fit')

    # Labels
    ax.set_xlabel('Eigenvalue (PCA)', fontsize=12)
    ax.set_ylabel(f'{method} Importance', fontsize=12)
    ax.set_title(
        f'{method} Importance vs PCA Eigenvalue\n'
        f'Pearson: {pearson_r:.3f} | Spearman: {spearman_r:.3f} | Kendall: {kendall_tau:.3f}',
        fontsize=12
    )
    ax.legend()

    # Log scale if values span orders of magnitude
    if eig_values.max() / eig_values.min() > 100:
        ax.set_xscale('log')
    if imp_values.max() / imp_values.min() > 100:
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_importance_by_type(
    importance: pd.DataFrame,
    feature_names: list,
    method: str = 'MDI',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Box plot of feature importance grouped by feature type.

    Useful for synthetic datasets where feature types are known.

    Parameters
    ----------
    importance : pd.DataFrame
        Feature importance DataFrame.
    feature_names : list
        Feature names in standardized format (I_*, R_*, N_*).
    method : str, default='MDI'
        Method name for title.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure

    Examples
    --------
    >>> X, meta = generate_synthetic_dataset()
    >>> mdi_imp = compute_mdi_importance(clf, X.columns.tolist())
    >>> fig = plot_importance_by_type(mdi_imp, X.columns.tolist())
    """
    from afml.feature_importance.synthetic import get_feature_types

    # Get feature types
    feature_types = get_feature_types(feature_names)

    # Add type to importance
    importance_with_type = importance.copy()
    importance_with_type['type'] = importance_with_type.index.map(feature_types)

    # Rename types for display
    type_names = {'I': 'Informative', 'R': 'Redundant', 'N': 'Noise'}
    importance_with_type['type_name'] = importance_with_type['type'].map(type_names)

    # Create box plot
    fig, ax = plt.subplots(figsize=figsize)

    types = ['Informative', 'Redundant', 'Noise']
    colors = ['steelblue', 'coral', 'lightgray']

    box_data = [
        importance_with_type[importance_with_type['type_name'] == t]['mean'].values
        for t in types
    ]

    bp = ax.boxplot(box_data, labels=types, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title(f'{method} Importance by Feature Type', fontsize=14)

    # Add horizontal line at uniform importance
    uniform = 1.0 / len(feature_names)
    ax.axhline(y=uniform, color='red', linestyle='--', alpha=0.5,
               label=f'Uniform ({uniform:.4f})')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig
