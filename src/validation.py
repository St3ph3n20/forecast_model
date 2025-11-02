"""
Model Validation Module
========================

Validates baseline models on periods without price changes.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import random


def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error.

    Parameters
    ----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values

    Returns
    -------
    float
        MAPE as percentage (0-100)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return np.nan

    ape = np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100
    return np.mean(ape)


def find_stable_price_periods(df, price_changes_df, min_stable_days=50, buffer_days=30):
    """
    Find periods with stable pricing (no price changes).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'date' and 'price_per_unit' columns
    price_changes_df : pd.DataFrame
        DataFrame of known price changes
    min_stable_days : int, optional
        Minimum consecutive days of stable pricing (default: 50)
    buffer_days : int, optional
        Buffer days around price changes to exclude (default: 30)

    Returns
    -------
    list
        List of candidate dates for validation (midpoint of stable periods)
    """
    # Get price change dates
    change_dates = set(price_changes_df['date'])

    # Find stable periods
    stable_periods = []
    current_price = None
    period_start = None

    for idx, row in df.iterrows():
        # Skip if near a price change
        near_change = any(abs((row['date'] - cd).days) < buffer_days for cd in change_dates)

        if near_change:
            # Reset if we're in a period
            if period_start is not None:
                period_end = row['date'] - timedelta(days=1)
                period_days = (period_end - period_start).days
                if period_days >= min_stable_days:
                    stable_periods.append((period_start, period_end))
                period_start = None
                current_price = None
        else:
            # Check if price is same as before
            if current_price is None:
                current_price = row['price_per_unit']
                period_start = row['date']
            elif abs(row['price_per_unit'] - current_price) < 0.001:
                # Same price, continue period
                continue
            else:
                # Price changed, end period
                if period_start is not None:
                    period_end = row['date'] - timedelta(days=1)
                    period_days = (period_end - period_start).days
                    if period_days >= min_stable_days:
                        stable_periods.append((period_start, period_end))

                # Start new period
                current_price = row['price_per_unit']
                period_start = row['date']

    # Return midpoints of stable periods
    candidate_dates = []
    for start, end in stable_periods:
        midpoint = start + (end - start) / 2
        # Ensure we have enough lookback and forecast horizon
        if (midpoint - df['date'].min()).days >= 180 and \
           (df['date'].max() - midpoint).days >= 21:
            candidate_dates.append(midpoint)

    return candidate_dates


def validate_seasonal_model(df, price_changes_df, baseline_model_class,
                            n_test_periods=10, lookback_days=180, forecast_days=21,
                            random_seed=42):
    """
    Validate baseline model on periods without price changes.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    price_changes_df : pd.DataFrame
        DataFrame of price changes
    baseline_model_class : class
        Baseline model class to validate
    n_test_periods : int, optional
        Number of test periods (default: 10)
    lookback_days : int, optional
        Days for model fitting (default: 180)
    forecast_days : int, optional
        Days to predict ahead (default: 21)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)

    Returns
    -------
    dict
        Dictionary with validation results:
        - mean_mape: float
        - median_mape: float
        - individual_mapes: list
        - test_periods: list of dates
        - predictions: list of (actual, predicted) arrays
    """
    print(f"\nValidating {baseline_model_class.__name__}...")
    print(f"  Finding stable price periods...")

    # Find candidate periods
    candidates = find_stable_price_periods(df, price_changes_df)

    if len(candidates) < n_test_periods:
        print(f"  Warning: Only found {len(candidates)} suitable periods (need {n_test_periods})")
        n_test_periods = len(candidates)

    # Randomly sample test periods
    random.seed(random_seed)
    test_dates = random.sample(candidates, n_test_periods)
    test_dates = sorted(test_dates)

    print(f"  Testing on {n_test_periods} periods...")

    # Run validation
    mapes = []
    test_results = []

    for test_date in test_dates:
        try:
            # Fit model
            model = baseline_model_class()
            model.fit(df, test_date, lookback_days=lookback_days)

            # Predict next N days
            predictions = []
            actuals = []

            for day in range(forecast_days):
                pred_date = test_date + timedelta(days=day)

                # Get prediction
                pred = model.predict(pred_date)

                # Get actual
                actual_row = df[df['date'] == pred_date]
                if len(actual_row) == 0:
                    continue

                actual = actual_row['sales_qty'].values[0]

                predictions.append(pred)
                actuals.append(actual)

            # Calculate MAPE for this period
            if len(actuals) > 0:
                mape = calculate_mape(actuals, predictions)
                mapes.append(mape)

                test_results.append({
                    'date': test_date,
                    'mape': mape,
                    'r_squared': model.get_r_squared(),
                    'n_days': len(actuals),
                    'actuals': actuals,
                    'predictions': predictions
                })

                print(f"    {test_date.date()}: MAPE = {mape:.2f}%, R² = {model.get_r_squared():.3f}")
            else:
                print(f"    {test_date.date()}: Skipped (no data)")

        except Exception as e:
            print(f"    {test_date.date()}: Error - {e}")
            continue

    # Calculate summary statistics
    if len(mapes) > 0:
        results = {
            'mean_mape': np.mean(mapes),
            'median_mape': np.median(mapes),
            'std_mape': np.std(mapes),
            'min_mape': np.min(mapes),
            'max_mape': np.max(mapes),
            'n_periods': len(mapes),
            'individual_mapes': mapes,
            'test_results': test_results
        }

        print(f"\n  Validation Summary:")
        print(f"    Mean MAPE: {results['mean_mape']:.2f}%")
        print(f"    Median MAPE: {results['median_mape']:.2f}%")
        print(f"    Std Dev: {results['std_mape']:.2f}%")
        print(f"    Range: {results['min_mape']:.2f}% - {results['max_mape']:.2f}%")
        print(f"    Target: <10%")

        if results['mean_mape'] < 10.0:
            print(f"    Status: [PASS]")
        else:
            print(f"    Status: [FAIL]")

        return results
    else:
        print(f"\n  ERROR: No successful validation tests")
        return None
