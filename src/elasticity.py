"""
Elasticity Calculation Module
==============================

Calculates baseline-adjusted price elasticity using various baseline models.
"""

import numpy as np
import pandas as pd
from datetime import timedelta


def calculate_elasticity(df, price_change_date, baseline_model,
                         pre_days=28, post_days=21, lookback_days=180):
    """
    Calculate baseline-adjusted elasticity for a price change event.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'date', 'sales_qty', and 'price_per_unit' columns
    price_change_date : datetime
        Date of price change
    baseline_model : BaselineModel
        Fitted baseline model (LinearBaselineModel or SeasonalBaselineModel)
    pre_days : int, optional
        Days before price change for pre-period (default: 28)
    post_days : int, optional
        Days after price change for post-period (default: 21 per Phase 3 spec)
    lookback_days : int, optional
        Days for baseline lookback (default: 180)

    Returns
    -------
    dict or None
        Dictionary with elasticity and related metrics, or None if insufficient data

    Notes
    -----
    Elasticity formula:
    elasticity = ((post_volume - baseline) / baseline) / (price_change_pct / 100)

    Where:
    - post_volume: average daily volume in post-period
    - baseline: predicted volume from baseline model
    - price_change_pct: percentage price change
    """
    # Define periods
    pre_start = price_change_date - timedelta(days=pre_days)
    pre_end = price_change_date - timedelta(days=1)
    post_start = price_change_date
    post_end = price_change_date + timedelta(days=post_days - 1)

    # Extract periods
    pre_period = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)].copy()
    post_period = df[(df['date'] >= post_start) & (df['date'] <= post_end)].copy()

    # Check minimum data requirements
    if len(pre_period) < 20 or len(post_period) < 14:
        return None

    # Calculate pre and post metrics
    pre_volume = pre_period['sales_qty'].mean()
    pre_price = pre_period['price_per_unit'].mean()
    post_volume = post_period['sales_qty'].mean()
    post_price = post_period['price_per_unit'].mean()

    # Calculate price change
    price_change_pct = ((post_price - pre_price) / pre_price) * 100

    if abs(price_change_pct) < 0.01:  # Avoid division by near-zero
        return None

    # Fit baseline model
    try:
        baseline_model.fit(df, price_change_date, lookback_days=lookback_days)
    except ValueError as e:
        print(f"  Warning: Could not fit baseline for {price_change_date.date()}: {e}")
        return None

    # Predict baseline for post-period
    # Use average prediction across all post-period dates
    baseline_predictions = []
    for single_date in post_period['date']:
        try:
            pred = baseline_model.predict(single_date)
            baseline_predictions.append(pred)
        except Exception as e:
            print(f"  Warning: Could not predict for {single_date.date()}: {e}")
            continue

    if len(baseline_predictions) == 0:
        return None

    baseline_volume = np.mean(baseline_predictions)

    # Calculate elasticity
    volume_change_pct = ((post_volume - baseline_volume) / baseline_volume) * 100
    elasticity = volume_change_pct / price_change_pct

    # Return results
    result = {
        'date': price_change_date,
        'elasticity': elasticity,
        'pre_volume': pre_volume,
        'post_volume': post_volume,
        'baseline_volume': baseline_volume,
        'pre_price': pre_price,
        'post_price': post_price,
        'price_change_pct': price_change_pct,
        'volume_change_pct': volume_change_pct,
        'model_r_squared': baseline_model.get_r_squared(),
        'pre_days': len(pre_period),
        'post_days': len(post_period)
    }

    return result


def calculate_all_elasticities(df, price_changes_df, baseline_model_class,
                               pre_days=28, post_days=21, lookback_days=180):
    """
    Calculate elasticities for all price changes using a specific baseline model.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    price_changes_df : pd.DataFrame
        DataFrame of price changes (from identify_price_changes)
    baseline_model_class : class
        Class of baseline model to use (LinearBaselineModel or SeasonalBaselineModel)
    pre_days : int, optional
        Days before price change (default: 28)
    post_days : int, optional
        Days after price change (default: 21)
    lookback_days : int, optional
        Days for baseline (default: 180)

    Returns
    -------
    list
        List of elasticity result dictionaries
    """
    results = []

    print(f"\nCalculating elasticities using {baseline_model_class.__name__}...")

    for idx, pc in price_changes_df.iterrows():
        # Create new model instance for each price change
        model = baseline_model_class()

        # Calculate elasticity
        result = calculate_elasticity(
            df, pc['date'], model,
            pre_days=pre_days,
            post_days=post_days,
            lookback_days=lookback_days
        )

        if result is not None:
            results.append(result)
            print(f"  {pc['date'].date()}: E = {result['elasticity']:+.3f}, "
                  f"R² = {result['model_r_squared']:.3f}")
        else:
            print(f"  {pc['date'].date()}: Skipped (insufficient data)")

    print(f"\nSuccessfully calculated {len(results)} elasticities")

    return results


def elasticity_results_to_dataframe(results):
    """
    Convert list of elasticity results to DataFrame.

    Parameters
    ----------
    results : list
        List of result dictionaries from calculate_elasticity

    Returns
    -------
    pd.DataFrame
        DataFrame with all elasticity measurements
    """
    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Reorder columns for clarity
    cols = ['date', 'elasticity', 'price_change_pct',
            'pre_volume', 'post_volume', 'baseline_volume',
            'volume_change_pct', 'pre_price', 'post_price',
            'model_r_squared', 'pre_days', 'post_days']

    return df[cols]
