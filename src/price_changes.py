"""
Price Change Detection Module
==============================

Identifies significant price changes in sales data.
"""

import pandas as pd


def identify_price_changes(df, threshold_pct=1.0):
    """
    Identify significant price changes in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'price_per_unit' columns
    threshold_pct : float, optional
        Minimum percentage change to be considered significant (default: 1.0%)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date: date of price change
        - old_price: price before change
        - new_price: price after change
        - pct_change: percentage change

    Notes
    -----
    Based on Phase 3 spec: threshold = 1% (Phase 2 used 2%)
    """
    # Round prices to avoid floating point noise
    df = df.copy()
    df['price_rounded'] = df['price_per_unit'].round(4)
    df['price_change_flag'] = (df['price_rounded'] != df['price_rounded'].shift()).astype(int)

    # Identify price changes
    price_changes = []
    for i in range(1, len(df)):
        if df.loc[i, 'price_change_flag'] == 1:
            old_price = df.loc[i-1, 'price_per_unit']
            new_price = df.loc[i, 'price_per_unit']
            pct_change = ((new_price - old_price) / old_price) * 100

            # Only include changes above threshold
            if abs(pct_change) > threshold_pct:
                price_changes.append({
                    'date': df.loc[i, 'date'],
                    'old_price': old_price,
                    'new_price': new_price,
                    'pct_change': pct_change
                })

    # Convert to DataFrame
    price_changes_df = pd.DataFrame(price_changes)
    if len(price_changes_df) > 0:
        price_changes_df = price_changes_df.sort_values('date').reset_index(drop=True)

    print(f"\nPrice change detection:")
    print(f"  Threshold: >{threshold_pct}%")
    print(f"  Found: {len(price_changes_df)} significant price changes")

    if len(price_changes_df) > 0:
        print(f"\n  First 10 price changes:")
        for idx, row in price_changes_df.head(10).iterrows():
            print(f"    {row['date'].date()}: {row['old_price']:.4f} -> {row['new_price']:.4f} ({row['pct_change']:+.1f}%)")

    return price_changes_df


def get_price_change_context(df, price_change_date, pre_days=28, post_days=21, lookback_days=180):
    """
    Get context around a specific price change for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    price_change_date : datetime
        Date of price change
    pre_days : int, optional
        Days before price change for pre-period (default: 28)
    post_days : int, optional
        Days after price change for post-period (default: 21)
    lookback_days : int, optional
        Days before price change for baseline (default: 180)

    Returns
    -------
    dict
        Dictionary with 'pre_period', 'post_period', and 'lookback' DataFrames
    """
    from datetime import timedelta

    # Define periods
    pre_start = price_change_date - timedelta(days=pre_days)
    pre_end = price_change_date - timedelta(days=1)
    post_start = price_change_date
    post_end = price_change_date + timedelta(days=post_days-1)
    lookback_start = price_change_date - timedelta(days=lookback_days)

    # Extract periods
    pre_period = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)].copy()
    post_period = df[(df['date'] >= post_start) & (df['date'] <= post_end)].copy()
    lookback_period = df[(df['date'] >= lookback_start) & (df['date'] < price_change_date)].copy()

    return {
        'pre_period': pre_period,
        'post_period': post_period,
        'lookback': lookback_period,
        'lookback_start': lookback_start,
        'pre_start': pre_start,
        'pre_end': pre_end,
        'post_start': post_start,
        'post_end': post_end
    }
