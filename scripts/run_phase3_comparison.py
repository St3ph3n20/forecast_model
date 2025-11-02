"""
Phase 3 Elasticity Analysis: Phase 2 vs Phase 3 Comparison
===========================================================

This script runs the complete Phase 3 analysis, comparing:
- Phase 2: Linear baseline elasticity
- Phase 3: Seasonally-adjusted baseline elasticity

Outputs:
- elasticity_history.csv: All elasticity measurements with both methods
- phase3_comparison.png: 4-panel comparison visualization
- seasonal_coefficients.csv: Day-of-week and month effects
- phase3_summary.txt: Summary statistics
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_and_clean_data, validate_data_quality
from src.price_changes import identify_price_changes
from src.baseline_models import LinearBaselineModel, SeasonalBaselineModel
from src.elasticity import calculate_all_elasticities, elasticity_results_to_dataframe
from src.weighting import apply_time_weighting_to_elasticities
from src.validation import validate_seasonal_model
from src.visualization import (create_phase3_comparison_chart,
                               create_seasonal_coefficients_chart,
                               create_validation_chart)

import pandas as pd
import numpy as np
from datetime import datetime


def main():
    """Run complete Phase 3 analysis."""

    print("="*80)
    print("PHASE 3: SEASONALLY-ADJUSTED ELASTICITY ANALYSIS")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("[1] LOADING DATA")
    print("="*80)

    data_path = Path(__file__).parent.parent / "data" / "Biscuits FULL Dataset.csv"
    df = load_and_clean_data(data_path, product_name='Milk Chocolate Digestives')
    validate_data_quality(df)

    # ========================================================================
    # 2. IDENTIFY PRICE CHANGES
    # ========================================================================
    print("\n" + "="*80)
    print("[2] IDENTIFYING PRICE CHANGES")
    print("="*80)

    price_changes_df = identify_price_changes(df, threshold_pct=1.0)

    # ========================================================================
    # 3. CALCULATE PHASE 2 ELASTICITIES (Linear Baseline)
    # ========================================================================
    print("\n" + "="*80)
    print("[3] PHASE 2: LINEAR BASELINE ELASTICITIES")
    print("="*80)

    phase2_results = calculate_all_elasticities(
        df, price_changes_df, LinearBaselineModel,
        pre_days=28, post_days=21, lookback_days=180
    )

    phase2_df = elasticity_results_to_dataframe(phase2_results)

    # Apply time-decay weighting
    phase2_weighted, phase2_df_weighted = apply_time_weighting_to_elasticities(
        phase2_df, halflife=180
    )

    print(f"\nPhase 2 Summary:")
    print(f"  Elasticities calculated: {len(phase2_df)}")
    print(f"  Average R²: {phase2_df['model_r_squared'].mean():.3f} ({phase2_df['model_r_squared'].mean()*100:.1f}%)")
    print(f"  Weighted elasticity: {phase2_weighted:.3f}")

    # ========================================================================
    # 4. CALCULATE PHASE 3 ELASTICITIES (Seasonal Baseline)
    # ========================================================================
    print("\n" + "="*80)
    print("[4] PHASE 3: SEASONAL BASELINE ELASTICITIES")
    print("="*80)

    phase3_results = calculate_all_elasticities(
        df, price_changes_df, SeasonalBaselineModel,
        pre_days=28, post_days=21, lookback_days=180
    )

    phase3_df = elasticity_results_to_dataframe(phase3_results)

    # Apply time-decay weighting
    phase3_weighted, phase3_df_weighted = apply_time_weighting_to_elasticities(
        phase3_df, halflife=180
    )

    print(f"\nPhase 3 Summary:")
    print(f"  Elasticities calculated: {len(phase3_df)}")
    print(f"  Average R²: {phase3_df['model_r_squared'].mean():.3f} ({phase3_df['model_r_squared'].mean()*100:.1f}%)")
    print(f"  Weighted elasticity: {phase3_weighted:.3f}")

    # ========================================================================
    # 5. COMPARE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("[5] COMPARING PHASE 2 vs PHASE 3")
    print("="*80)

    # Merge results
    comparison_df = pd.DataFrame({
        'date': phase2_df['date'],
        'price_change_pct': phase2_df['price_change_pct'],
        'phase2_elasticity': phase2_df['elasticity'],
        'phase2_baseline': phase2_df['baseline_volume'],
        'phase2_r_squared': phase2_df['model_r_squared'],
        'phase3_elasticity': phase3_df['elasticity'],
        'phase3_baseline': phase3_df['baseline_volume'],
        'phase3_r_squared': phase3_df['model_r_squared'],
    })

    comparison_df['elasticity_change'] = (comparison_df['phase3_elasticity'] -
                                          comparison_df['phase2_elasticity'])
    comparison_df['abs_elasticity_change'] = comparison_df['elasticity_change'].abs()

    # Add normalized weights
    comparison_df = comparison_df.merge(
        phase3_df_weighted[['date', 'normalized_weight']],
        on='date',
        how='left'
    )

    # Print comparison
    print(f"\nWeighted Elasticity Comparison:")
    print(f"  Phase 2: {phase2_weighted:.3f}")
    print(f"  Phase 3: {phase3_weighted:.3f}")
    print(f"  Change: {phase3_weighted - phase2_weighted:+.3f}")
    print(f"  Change (%): {((phase3_weighted - phase2_weighted) / abs(phase2_weighted)) * 100:+.1f}%")

    print(f"\nR² Comparison:")
    print(f"  Phase 2 average: {comparison_df['phase2_r_squared'].mean()*100:.1f}%")
    print(f"  Phase 3 average: {comparison_df['phase3_r_squared'].mean()*100:.1f}%")
    print(f"  Improvement: +{(comparison_df['phase3_r_squared'].mean() - comparison_df['phase2_r_squared'].mean())*100:.1f} percentage points")

    print(f"\nElasticity Changes:")
    print(f"  Mean absolute change: {comparison_df['abs_elasticity_change'].mean():.3f}")
    print(f"  Elasticities changed by >0.2: {(comparison_df['abs_elasticity_change'] > 0.2).sum()} out of {len(comparison_df)}")

    # Largest changes
    print(f"\n  Top 5 largest changes:")
    top_changes = comparison_df.nlargest(5, 'abs_elasticity_change')
    for _, row in top_changes.iterrows():
        print(f"    {row['date'].date()}: {row['phase2_elasticity']:+.3f} → {row['phase3_elasticity']:+.3f} "
              f"(Δ = {row['elasticity_change']:+.3f})")

    # ========================================================================
    # 6. EXTRACT SEASONAL COEFFICIENTS
    # ========================================================================
    print("\n" + "="*80)
    print("[6] EXTRACTING SEASONAL COEFFICIENTS")
    print("="*80)

    # Get seasonal coefficients from one representative model
    # (We'll use the most recent price change)
    sample_model = SeasonalBaselineModel()
    sample_model.fit(df, price_changes_df.iloc[-1]['date'], lookback_days=180)
    seasonal_coefs = sample_model.get_seasonal_coefficients()

    print(f"\nSeasonal Effects Summary:")
    print(f"\nDay-of-Week Effects:")
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_coefs = seasonal_coefs[seasonal_coefs['component'].isin(dow_names)]
    for _, row in dow_coefs.iterrows():
        print(f"  {row['component']:10s}: {row['effect_pct']:+6.1f}%")

    print(f"\nMonth Effects (top 5 by magnitude):")
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_coefs = seasonal_coefs[seasonal_coefs['component'].isin(month_names)]
    top_months = month_coefs.reindex(month_coefs['effect_pct'].abs().sort_values(ascending=False).index).head(5)
    for _, row in top_months.iterrows():
        print(f"  {row['component']:10s}: {row['effect_pct']:+6.1f}%")

    # ========================================================================
    # 7. VALIDATE SEASONAL MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("[7] VALIDATING SEASONAL MODEL")
    print("="*80)

    validation_results = validate_seasonal_model(
        df, price_changes_df, SeasonalBaselineModel,
        n_test_periods=10, lookback_days=180, forecast_days=21
    )

    # ========================================================================
    # 8. GENERATE OUTPUTS
    # ========================================================================
    print("\n" + "="*80)
    print("[8] GENERATING OUTPUTS")
    print("="*80)

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save elasticity history
    elasticity_history_path = output_dir / "elasticity_history.csv"
    comparison_df.to_csv(elasticity_history_path, index=False)
    print(f"\n✓ Elasticity history saved: {elasticity_history_path}")

    # Save seasonal coefficients
    seasonal_coefs_path = output_dir / "seasonal_coefficients.csv"
    seasonal_coefs.to_csv(seasonal_coefs_path, index=False)
    print(f"✓ Seasonal coefficients saved: {seasonal_coefs_path}")

    # Generate visualizations
    create_phase3_comparison_chart(comparison_df, output_dir / "phase3_comparison.png")
    create_seasonal_coefficients_chart(seasonal_coefs, output_dir / "seasonal_coefficients.png")

    if validation_results:
        create_validation_chart(validation_results, output_dir / "validation_results.png")

    # Generate summary report
    summary_path = output_dir / "phase3_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("PHASE 3: SEASONAL ADJUSTMENT RESULTS\n")
        f.write("="*60 + "\n\n")

        f.write("MODEL QUALITY\n")
        f.write("-"*60 + "\n")
        f.write(f"Average Phase 2 R²: {comparison_df['phase2_r_squared'].mean()*100:.1f}%\n")
        f.write(f"Average Phase 3 R²: {comparison_df['phase3_r_squared'].mean()*100:.1f}%\n")
        f.write(f"Improvement: +{(comparison_df['phase3_r_squared'].mean() - comparison_df['phase2_r_squared'].mean())*100:.1f} percentage points\n\n")

        f.write("ELASTICITY COMPARISON\n")
        f.write("-"*60 + "\n")
        f.write(f"Phase 2 Weighted Elasticity: {phase2_weighted:.3f}\n")
        f.write(f"Phase 3 Weighted Elasticity: {phase3_weighted:.3f}\n")
        f.write(f"Difference: {phase3_weighted - phase2_weighted:+.3f}\n\n")

        f.write("CHANGES BY EVENT\n")
        f.write("-"*60 + "\n")
        f.write(f"Events with elasticity change >0.2: {(comparison_df['abs_elasticity_change'] > 0.2).sum()} out of {len(comparison_df)}\n")
        f.write(f"Mean absolute change: {comparison_df['abs_elasticity_change'].mean():.3f}\n\n")

        f.write("Largest changes:\n")
        for _, row in top_changes.iterrows():
            f.write(f"  {row['date'].date()}: {row['phase2_elasticity']:+.3f} → {row['phase3_elasticity']:+.3f} ({row['elasticity_change']:+.3f})\n")

        if validation_results:
            f.write("\n")
            f.write("VALIDATION RESULTS\n")
            f.write("-"*60 + "\n")
            f.write(f"Mean MAPE: {validation_results['mean_mape']:.2f}%\n")
            f.write(f"Median MAPE: {validation_results['median_mape']:.2f}%\n")
            f.write(f"Target: <10%\n")
            f.write(f"Status: {'PASS' if validation_results['mean_mape'] < 10 else 'FAIL'}\n\n")

        f.write("SEASONAL EFFECTS\n")
        f.write("-"*60 + "\n")
        f.write("Day-of-week effects:\n")
        for _, row in dow_coefs.iterrows():
            f.write(f"  {row['component']:10s}: {row['effect_pct']:+6.1f}%\n")
        f.write("\nTop 5 month effects:\n")
        for _, row in top_months.iterrows():
            f.write(f"  {row['component']:10s}: {row['effect_pct']:+6.1f}%\n")

        f.write("\n")
        f.write("SUCCESS CRITERIA\n")
        f.write("-"*60 + "\n")

        criteria_pass = []
        criteria_pass.append(comparison_df['phase3_r_squared'].mean() > 0.40)
        if validation_results:
            criteria_pass.append(validation_results['mean_mape'] < 10.0)
        else:
            criteria_pass.append(False)
        criteria_pass.append((comparison_df['abs_elasticity_change'] > 0.2).sum() >= 5)
        criteria_pass.append(abs(phase3_weighted - phase2_weighted) > 0.05)

        f.write(f"{'✓' if criteria_pass[0] else '✗'} Phase 3 R² > 40%: {comparison_df['phase3_r_squared'].mean()*100:.1f}%\n")
        f.write(f"{'✓' if criteria_pass[1] else '✗'} Validation MAPE < 10%: {validation_results['mean_mape'] if validation_results else 'N/A'}%\n")
        f.write(f"{'✓' if criteria_pass[2] else '✗'} At least 5 elasticities change by >0.2: {(comparison_df['abs_elasticity_change'] > 0.2).sum()}\n")
        f.write(f"{'✓' if criteria_pass[3] else '✗'} Weighted elasticity differs by >0.05: {abs(phase3_weighted - phase2_weighted):.3f}\n")
        f.write(f"\nOverall: {'ALL PASS' if all(criteria_pass) else 'SOME FAILURES'}\n")

    print(f"✓ Summary report saved: {summary_path}")

    # ========================================================================
    # 9. FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("[9] FINAL SUMMARY")
    print("="*80)

    print(f"\n✓ Phase 3 analysis complete!")
    print(f"\nSuccess Criteria:")
    print(f"  {'✓' if criteria_pass[0] else '✗'} Phase 3 R² > 40%")
    print(f"  {'✓' if criteria_pass[1] else '✗'} Validation MAPE < 10%")
    print(f"  {'✓' if criteria_pass[2] else '✗'} At least 5 elasticities change by >0.2")
    print(f"  {'✓' if criteria_pass[3] else '✗'} Weighted elasticity differs by >0.05")

    if all(criteria_pass):
        print(f"\n{'='*80}")
        print("SUCCESS: All criteria passed! ✓")
        print('='*80)
    else:
        print(f"\n{'='*80}")
        print("PARTIAL SUCCESS: Some criteria not met")
        print('='*80)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
