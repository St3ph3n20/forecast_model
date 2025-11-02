"""
Time-Decay Weighting Module
============================

Implements exponential time-decay weighting for elasticity aggregation.
"""

import numpy as np
import pandas as pd


def calculate_time_weight(days_ago, halflife=180):
    """
    Calculate exponential time-decay weight.

    Parameters
    ----------
    days_ago : float or array-like
        Number of days ago (can be negative for future dates)
    halflife : float, optional
        Halflife in days for exponential decay (default: 180 days per Phase 3 spec)

    Returns
    -------
    float or array-like
        Time-decay weight (0 to 1)

    Notes
    -----
    Formula: weight = exp(-ln(2) × days_ago / halflife)

    Phase 3 spec uses 180-day halflife (Phase 2 used 365 days)
    """
    return np.exp(-np.log(2) * days_ago / halflife)


def get_weighted_elasticity(elasticities, target_date, halflife=180):
    """
    Calculate time-weighted elasticity from historical measurements.

    Parameters
    ----------
    elasticities : list of dict
        List of elasticity measurements, each with 'date' and 'elasticity' keys
    target_date : datetime
        Date to calculate weighted elasticity for
    halflife : float, optional
        Halflife in days for exponential decay (default: 180 days)

    Returns
    -------
    float or None
        Weighted elasticity, or None if no data available

    Notes
    -----
    Weights are normalized to sum to 1.0 (100%)
    """
    if len(elasticities) == 0:
        return None

    weighted_sum = 0
    total_weight = 0

    for item in elasticities:
        days_ago = (target_date - item['date']).days
        weight = calculate_time_weight(days_ago, halflife=halflife)

        weighted_sum += item['elasticity'] * weight
        total_weight += weight

    if total_weight == 0:
        return None

    return weighted_sum / total_weight


def calculate_normalized_weights(dates, target_date, halflife=180):
    """
    Calculate normalized weights for a list of dates.

    Parameters
    ----------
    dates : array-like of datetime
        Dates to calculate weights for
    target_date : datetime
        Target date (reference point)
    halflife : float, optional
        Halflife in days (default: 180)

    Returns
    -------
    np.ndarray
        Normalized weights (sum to 1.0)

    Examples
    --------
    >>> dates = pd.date_range('2022-01-01', periods=5, freq='30D')
    >>> target = pd.Timestamp('2022-06-01')
    >>> weights = calculate_normalized_weights(dates, target, halflife=180)
    >>> weights.sum()
    1.0
    """
    days_ago = np.array([(target_date - d).days for d in dates])
    raw_weights = calculate_time_weight(days_ago, halflife=halflife)
    total_weight = raw_weights.sum()

    if total_weight == 0:
        return np.zeros(len(dates))

    return raw_weights / total_weight


def apply_time_weighting_to_elasticities(elasticity_df, date_col='date',
                                         elasticity_col='elasticity', halflife=180):
    """
    Apply time-decay weighting to a DataFrame of elasticities.

    Parameters
    ----------
    elasticity_df : pd.DataFrame
        DataFrame with elasticity measurements
    date_col : str, optional
        Name of date column (default: 'date')
    elasticity_col : str, optional
        Name of elasticity column (default: 'elasticity')
    halflife : float, optional
        Halflife in days (default: 180)

    Returns
    -------
    tuple
        (weighted_elasticity, weights_df) where:
        - weighted_elasticity: float, the time-weighted average
        - weights_df: DataFrame with normalized weights added

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2022-01-01', periods=5, freq='30D'),
    ...     'elasticity': [-0.5, -0.8, -1.0, -0.6, -0.4]
    ... })
    >>> weighted_elast, df_with_weights = apply_time_weighting_to_elasticities(df)
    """
    if len(elasticity_df) == 0:
        return None, elasticity_df

    df = elasticity_df.copy()

    # Use the most recent date as reference
    target_date = df[date_col].max()

    # Calculate days ago
    df['days_ago'] = (target_date - df[date_col]).dt.days

    # Calculate raw weights
    df['raw_weight'] = calculate_time_weight(df['days_ago'], halflife=halflife)

    # Normalize weights
    total_weight = df['raw_weight'].sum()
    df['normalized_weight'] = df['raw_weight'] / total_weight if total_weight > 0 else 0

    # Calculate weighted elasticity
    weighted_elasticity = (df[elasticity_col] * df['normalized_weight']).sum()

    return weighted_elasticity, df
