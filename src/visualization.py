"""
Visualization Module
====================

Generates comparison charts for Phase 2 vs Phase 3 analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_phase3_comparison_chart(comparison_df, output_path='outputs/phase3_comparison.png'):
    """
    Create 4-panel comparison chart for Phase 2 vs Phase 3.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with both phase2 and phase3 results
    output_path : str, optional
        Path to save figure

    Panels:
    1. Phase 2 vs Phase 3 Elasticities (scatter)
    2. R² Comparison (bar chart)
    3. Elasticity Changes (histogram)
    4. Weighted Elasticity Over Time (line chart)
    """
    fig = plt.figure(figsize=(18, 12))

    # Panel 1: Elasticity Scatter
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(comparison_df['phase2_elasticity'],
               comparison_df['phase3_elasticity'],
               alpha=0.6, s=100, edgecolor='black')

    # Add diagonal line (y=x)
    min_val = min(comparison_df['phase2_elasticity'].min(),
                  comparison_df['phase3_elasticity'].min())
    max_val = max(comparison_df['phase2_elasticity'].max(),
                  comparison_df['phase3_elasticity'].max())
    ax1.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='No Change Line')

    ax1.set_xlabel('Phase 2 Elasticity (Linear Baseline)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Phase 3 Elasticity (Seasonal Baseline)', fontweight='bold', fontsize=12)
    ax1.set_title('Elasticity Comparison: Phase 2 vs Phase 3', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: R² Comparison
    ax2 = plt.subplot(2, 2, 2)
    avg_r2_phase2 = comparison_df['phase2_r_squared'].mean()
    avg_r2_phase3 = comparison_df['phase3_r_squared'].mean()

    bars = ax2.bar(['Phase 2\n(Linear)', 'Phase 3\n(Seasonal)'],
                   [avg_r2_phase2 * 100, avg_r2_phase3 * 100],
                   color=['steelblue', 'green'],
                   alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.1f}%', ha='center', fontweight='bold', fontsize=12)

    ax2.set_ylabel('Average R² (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Model Fit Quality: R² Comparison', fontweight='bold', fontsize=14)
    ax2.set_ylim(0, max(avg_r2_phase2 * 100, avg_r2_phase3 * 100) * 1.2)
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3: Elasticity Change Distribution
    ax3 = plt.subplot(2, 2, 3)
    changes = comparison_df['phase3_elasticity'] - comparison_df['phase2_elasticity']

    ax3.hist(changes, bins=15, alpha=0.7, edgecolor='black', color='purple')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax3.axvline(changes.mean(), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {changes.mean():+.3f}')

    ax3.set_xlabel('Elasticity Change (Phase 3 - Phase 2)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax3.set_title('Distribution of Elasticity Changes', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Panel 4: Time Series
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(comparison_df['date'], comparison_df['phase2_elasticity'],
            'o-', label='Phase 2 (Linear)', linewidth=2, markersize=6)
    ax4.plot(comparison_df['date'], comparison_df['phase3_elasticity'],
            's-', label='Phase 3 (Seasonal)', linewidth=2, markersize=6)
    ax4.axhline(0, color='gray', linestyle=':', linewidth=1)

    ax4.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Elasticity', fontweight='bold', fontsize=12)
    ax4.set_title('Elasticity Over Time', fontweight='bold', fontsize=14)
    ax4.legend()
    ax4.grid(alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison chart saved: {output_path}")

    return fig


def create_seasonal_coefficients_chart(seasonal_coefs_df, output_path='outputs/seasonal_coefficients.png'):
    """
    Create bar chart showing seasonal coefficients.

    Parameters
    ----------
    seasonal_coefs_df : pd.DataFrame
        DataFrame from SeasonalBaselineModel.get_seasonal_coefficients()
    output_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Separate day-of-week and month effects
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_df = seasonal_coefs_df[seasonal_coefs_df['component'].isin(days)].copy()

    months = ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December']
    month_df = seasonal_coefs_df[seasonal_coefs_df['component'].isin(months)].copy()

    # Panel 1: Day-of-week effects
    colors = ['red' if x < 0 else 'green' for x in dow_df['effect_pct']]
    bars1 = ax1.bar(range(len(dow_df)), dow_df['effect_pct'],
                    color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    ax1.set_xticks(range(len(dow_df)))
    ax1.set_xticklabels(dow_df['component'], rotation=45, ha='right')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Effect (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Day-of-Week Effects', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, dow_df['effect_pct'])):
        ax1.text(bar.get_x() + bar.get_width()/2,
                val + (1 if val > 0 else -1),
                f'{val:+.1f}%', ha='center', fontweight='bold', fontsize=10)

    # Panel 2: Month effects
    colors2 = ['red' if x < 0 else 'green' for x in month_df['effect_pct']]
    bars2 = ax2.bar(range(len(month_df)), month_df['effect_pct'],
                    color=colors2, alpha=0.7, edgecolor='black', linewidth=2)

    ax2.set_xticks(range(len(month_df)))
    ax2.set_xticklabels(month_df['component'], rotation=45, ha='right')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Effect (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Month Effects', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, month_df['effect_pct'])):
        ax2.text(bar.get_x() + bar.get_width()/2,
                val + (0.5 if val > 0 else -0.5),
                f'{val:+.1f}%', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Seasonal coefficients chart saved: {output_path}")

    return fig


def create_validation_chart(validation_results, output_path='outputs/validation_results.png'):
    """
    Create validation results chart.

    Parameters
    ----------
    validation_results : dict
        Results from validate_seasonal_model()
    output_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: MAPE distribution
    mapes = validation_results['individual_mapes']
    ax1.hist(mapes, bins=10, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(validation_results['mean_mape'], color='red', linestyle='--',
               linewidth=2, label=f'Mean: {validation_results["mean_mape"]:.2f}%')
    ax1.axvline(10, color='green', linestyle='--', linewidth=2,
               label='Target: 10%')

    ax1.set_xlabel('MAPE (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax1.set_title('Validation MAPE Distribution', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: Actual vs Predicted (aggregated)
    all_actuals = []
    all_predictions = []

    for result in validation_results['test_results']:
        all_actuals.extend(result['actuals'])
        all_predictions.extend(result['predictions'])

    ax2.scatter(all_actuals, all_predictions, alpha=0.5, s=50, edgecolor='black')
    min_val = min(min(all_actuals), min(all_predictions))
    max_val = max(max(all_actuals), max(all_predictions))
    ax2.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='Perfect Prediction')

    ax2.set_xlabel('Actual Volume', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Predicted Volume', fontweight='bold', fontsize=12)
    ax2.set_title('Validation: Predicted vs Actual', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Validation chart saved: {output_path}")

    return fig
