"""
Data Loading and Cleaning Module
=================================

Handles loading and preprocessing of raw sales data.
"""

import pandas as pd
from pathlib import Path


def load_and_clean_data(csv_path, product_name='Milk Chocolate Digestives'):
    """
    Load and clean sales data from CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing sales data
    product_name : str, optional
        Product name to filter for (default: 'Milk Chocolate Digestives')

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        - date: datetime
        - sales_qty: float
        - price_per_unit: float
        - sales_gross: float
        - sales_net: float
        - product: str
        - article_id: str

    Notes
    -----
    - Removes 'Overall Result' rows
    - Parses dates from DD/MM/YYYY format
    - Removes commas from numeric fields
    - Calculates price_per_unit = sales_gross / sales_qty
    """
    # Load raw data
    df = pd.read_csv(csv_path)

    # Remove overall result rows
    df = df[df['Linked article'] != 'Overall Result'].copy()

    # Rename columns for clarity
    df.columns = ['article_id', 'product', 'date', 'sales_qty', 'sales_gross',
                  'sales_net', 'empty1', 'empty2']
    df = df.drop(['empty1', 'empty2'], axis=1)

    # Parse date
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    # Clean numeric fields
    df['sales_qty'] = df['sales_qty'].str.replace(',', '').astype(float)
    df['sales_gross'] = df['sales_gross'].str.replace(',', '').str.replace(' GBP', '').astype(float)
    df['sales_net'] = df['sales_net'].str.replace(',', '').str.replace(' GBP', '').astype(float)

    # Calculate price per unit
    df['price_per_unit'] = df['sales_gross'] / df['sales_qty']

    # Filter to specific product
    df_product = df[df['product'] == product_name].copy()
    df_product = df_product.sort_values('date').reset_index(drop=True)

    print(f"Data loaded successfully:")
    print(f"  Product: {product_name}")
    print(f"  Records: {len(df_product)}")
    print(f"  Date range: {df_product['date'].min().date()} to {df_product['date'].max().date()}")
    print(f"  Days: {(df_product['date'].max() - df_product['date'].min()).days + 1}")

    return df_product


def validate_data_quality(df):
    """
    Validate data quality and print warnings for any issues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate

    Returns
    -------
    bool
        True if data passes all quality checks
    """
    issues = []

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")

    # Check for negative prices
    if (df['price_per_unit'] <= 0).any():
        issues.append(f"Found {(df['price_per_unit'] <= 0).sum()} rows with non-positive prices")

    # Check for negative quantities
    if (df['sales_qty'] <= 0).any():
        issues.append(f"Found {(df['sales_qty'] <= 0).sum()} rows with non-positive quantities")

    # Check for date gaps
    date_diff = df['date'].diff()
    max_gap = date_diff.max().days
    if max_gap > 1:
        issues.append(f"Maximum date gap: {max_gap} days")

    # Print results
    if issues:
        print("\nData quality warnings:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\nData quality: All checks passed ✓")
        return True
