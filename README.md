# AFML - Advances in Financial Machine Learning

A Python implementation of machine learning techniques for financial applications, based on the book *"Advances in Financial Machine Learning"* by Marcos López de Prado.

**Author:** Oualid Missaoui

---

## Overview

AFML is a comprehensive library that implements practical machine learning methodologies specifically designed for financial data. It addresses the unique challenges of financial ML, including non-IID data, overlapping labels, and information leakage that plague traditional approaches.

## Features

- **Triple-Barrier Labeling** - Generate robust training labels from price data
- **Sample Weights** - Handle non-IID nature of financial observations
- **Fractional Differentiation** - Balance stationarity and memory preservation
- **Purged Cross-Validation** - Prevent data leakage in model validation
- **Feature Importance** - MDI, MDA, and SFI methods with visualization
- **Ensemble Methods** - Financial-optimized bagging and random forests
- **Hyperparameter Optimization** - Grid/randomized search with purged CV
- **Bet Sizing** - Convert ML predictions to practical position sizes

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AFML.git
cd AFML
```

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
```

## Project Structure

```
AFML/
├── afml/                          # Main package
│   ├── labeling/                  # Triple-barrier labeling (Chapter 3)
│   ├── sample_weights/            # Sample weighting methods (Chapter 4)
│   ├── fdf/                       # Fractional differentiation (Chapter 5)
│   ├── ensemble_methods/          # Bagging & Random Forests (Chapter 6)
│   ├── cross_validation/          # Purged K-Fold & Walk-Forward (Chapter 7)
│   ├── feature_importance/        # MDI, MDA, SFI methods (Chapter 8)
│   ├── hpo/                       # Hyperparameter optimization (Chapter 9)
│   └── bet_sizing/                # Position sizing (Chapter 10)
├── notebooks/                     # Jupyter tutorials for each module
├── docs/                          # PDF documentation
└── data/                          # Sample financial data
```

## Quick Start

### Labeling with Triple-Barrier Method

```python
from afml.labeling import get_daily_volatility, add_vertical_barrier, get_events, get_labels

# Calculate dynamic volatility
volatility = get_daily_volatility(close_prices, span=100)

# Add vertical barriers (time limit)
t1 = add_vertical_barrier(close_prices.index, close_prices, num_days=10)

# Get events with triple barrier
events = get_events(
    close=close_prices,
    timestamps=timestamps,
    pt_sl=[1, 1],  # profit-take and stop-loss multipliers
    target=volatility,
    min_ret=0.01,
    num_threads=4,
    t1=t1
)

# Generate labels
labels = get_labels(events, close_prices)
```

### Purged Cross-Validation

```python
from afml.cross_validation import PurgedKFold, cv_score

# Create purged K-fold CV
cv = PurgedKFold(
    n_splits=5,
    t1=events['t1'],
    pct_embargo=0.01
)

# Score model with purging
scores = cv_score(classifier, X, y, cv=cv, sample_weight=weights)
```

### Feature Importance

```python
from afml.feature_importance import (
    compute_mdi_importance,
    compute_mda_importance,
    plot_feature_importance
)

# Mean Decrease Impurity (fast, in-sample)
mdi = compute_mdi_importance(forest, feature_names)

# Mean Decrease Accuracy (out-of-sample)
mda = compute_mda_importance(classifier, X, y, cv, sample_weight=weights)

# Visualize results
plot_feature_importance(mdi, title="MDI Feature Importance")
```

### Bet Sizing

```python
from afml.bet_sizing import (
    compute_bet_size_from_probability,
    compute_average_active_signals,
    discretize_signal
)

# Convert probability to bet size
bet_size = compute_bet_size_from_probability(probability, num_classes=2)

# Average overlapping signals to reduce turnover
avg_signal = compute_average_active_signals(signals, timestamps)

# Discretize to prevent overtrading
discrete_signal = discretize_signal(avg_signal, step_size=0.1)
```

## Modules

### Labeling (`afml.labeling`)
Implements the triple-barrier method for generating training labels from financial time series. Handles dynamic volatility calculation, event generation, and label extraction.

### Sample Weights (`afml.sample_weights`)
Addresses the non-IID nature of financial data through:
- Concurrent event analysis
- Sequential bootstrap sampling
- Return and uniqueness attribution
- Time decay weighting

### Fractional Differentiation (`afml.fdf`)
Solves the stationarity vs. memory dilemma:
- Standard and FFD fractional differentiation
- Automatic d-value optimization
- Stationarity testing (ADF)

### Cross-Validation (`afml.cross_validation`)
Financial-aware cross-validation preventing data leakage:
- **Purging**: Remove training data overlapping with test labels
- **Embargo**: Remove training data immediately after test period
- Purged K-Fold and Walk-Forward implementations

### Feature Importance (`afml.feature_importance`)
Three complementary methods for understanding model decisions:
- **MDI**: Mean Decrease Impurity (fast, tree-based)
- **MDA**: Mean Decrease Accuracy (out-of-sample, model-agnostic)
- **SFI**: Single Feature Importance (no substitution effects)

### Ensemble Methods (`afml.ensemble_methods`)
Financial-optimized ensemble learning:
- Bagging classifiers and regressors
- Random forest factories
- Bias-variance tradeoff analysis
- Scalable SVM bagging

### Hyperparameter Optimization (`afml.hpo`)
Tune models with financial-aware cross-validation:
- Grid search and randomized search
- Log-uniform parameter distributions
- Sample-weight aware pipelines

### Bet Sizing (`afml.bet_sizing`)
Convert ML outputs to trading positions:
- Probability to position size conversion
- Signal averaging to reduce turnover
- Discretization to prevent overtrading
- Dynamic position sizing functions

## Tutorials

The `notebooks/` directory contains Jupyter tutorials for each module:

- `labeling/` - Triple-barrier labeling examples
- `sample_weights/` - Sequential bootstrap and weighting
- `fdf/` - Fractional differentiation tutorials
- `cross_validation/` - Purged CV demonstrations
- `feature_importance/` - Importance analysis workflows
- `ensemble_methods/` - Ensemble learning examples
- `hpo/` - Hyperparameter tuning guides
- `bet_sizing/` - Position sizing examples
- `applications/trend_following/` - Complete strategy example

Launch Jupyter Lab:

```bash
jupyter lab
```

Or use the provided script:

```bash
./run.command
```

## Documentation

PDF documentation for each chapter is available in the `docs/` directory, providing theoretical background for each module's implementation.

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Jupyter (for notebooks)

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

## License

This project is for educational and research purposes.

---

**Author:** Oualid Missaoui
