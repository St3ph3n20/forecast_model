"""
ELASTICITY FORECASTING BACKTEST
=================================
Compare two forecasting approaches:
1. Single Elasticity Method (most recent)
2. Time-Decayed Weighted Elasticity Method

For each historical price change, we:
- Pretend we're forecasting it using only PRIOR data
- Calculate what each method would have predicted
- Compare to actual results
- Measure accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from scipy import stats

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ELASTICITY FORECASTING BACKTEST: SINGLE vs TIME-DECAYED")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")

df = pd.read_csv('/mnt/project/Biscuits_FULL_Dataset.csv')
df = df[df['Linked article'] != 'Overall Result'].copy()
df.columns = ['article_id', 'product', 'date', 'sales_qty', 'sales_gross', 
              'sales_net', 'empty1', 'empty2']
df = df.drop(['empty1', 'empty2'], axis=1)

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['sales_qty'] = df['sales_qty'].str.replace(',', '').astype(float)
df['sales_gross'] = df['sales_gross'].str.replace(',', '').str.replace(' GBP', '').astype(float)
df['sales_net'] = df['sales_net'].str.replace(',', '').str.replace(' GBP', '').astype(float)
df['price_per_unit'] = df['sales_gross'] / df['sales_qty']

main_product = 'Milk Chocolate Digestives'
df_main = df[df['product'] == main_product].copy()
df_main = df_main.sort_values('date').reset_index(drop=True)

print(f"   Product: {main_product}")
print(f"   Records: {len(df_main)}")

# ============================================================================
# 2. IDENTIFY ALL PRICE CHANGES
# ============================================================================
print("\n[2] IDENTIFYING PRICE CHANGES...")

df_main['price_rounded'] = df_main['price_per_unit'].round(4)
df_main['price_change_flag'] = (df_main['price_rounded'] != df_main['price_rounded'].shift()).astype(int)

price_changes = []
for i in range(1, len(df_main)):
    if df_main.loc[i, 'price_change_flag'] == 1:
        old_price = df_main.loc[i-1, 'price_per_unit']
        new_price = df_main.loc[i, 'price_per_unit']
        pct_change = ((new_price - old_price) / old_price) * 100
        
        if abs(pct_change) > 2.0:  # Significant changes only
            price_changes.append({
                'date': df_main.loc[i, 'date'],
                'old_price': old_price,
                'new_price': new_price,
                'pct_change': pct_change
            })

price_changes_df = pd.DataFrame(price_changes)
price_changes_df = price_changes_df.sort_values('date').reset_index(drop=True)

print(f"   Found {len(price_changes_df)} significant price changes")
print("\n   First 10 price changes:")
for idx, row in price_changes_df.head(10).iterrows():
    print(f"   {row['date'].date()}: £{row['old_price']:.4f} → £{row['new_price']:.4f} ({row['pct_change']:+.1f}%)")

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================
print("\n[3] DEFINING HELPER FUNCTIONS...")

def calculate_baseline_adjusted_elasticity(df, price_change_date):
    """Calculate baseline-adjusted elasticity for a price change event"""
    
    pre_start = price_change_date - timedelta(days=28)
    pre_end = price_change_date - timedelta(days=1)
    post_start = price_change_date
    post_end = price_change_date + timedelta(days=27)
    
    pre_period = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)].copy()
    post_period = df[(df['date'] >= post_start) & (df['date'] <= post_end)].copy()
    
    if len(pre_period) < 20 or len(post_period) < 20:
        return None
    
    pre_volume = pre_period['sales_qty'].mean()
    pre_price = pre_period['price_per_unit'].mean()
    post_volume = post_period['sales_qty'].mean()
    post_price = post_period['price_per_unit'].mean()
    
    # Calculate baseline
    six_months_back = price_change_date - timedelta(days=180)
    historical = df[(df['date'] >= six_months_back) & 
                   (df['date'] < price_change_date)].copy()
    
    if len(historical) < 30:
        return None
    
    historical = historical.sort_values('date').reset_index(drop=True)
    n = len(historical)
    x = np.arange(n)
    y = historical['sales_qty'].values
    
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    baseline_trend = pre_volume + (slope * 14)
    
    # YoY
    yoy_date = price_change_date - timedelta(days=365)
    yoy_data = df[(df['date'] >= yoy_date - timedelta(days=28)) & 
                  (df['date'] <= yoy_date + timedelta(days=27))]
    
    if len(yoy_data) > 20:
        yoy_avg = yoy_data['sales_qty'].mean()
        yoy_change = (pre_volume - yoy_avg) / yoy_avg
        baseline_yoy = pre_volume * (1 + yoy_change)
        baseline = (baseline_trend * 0.6) + (baseline_yoy * 0.4)
    else:
        baseline = baseline_trend
    
    # Calculate elasticity
    price_change_pct = ((post_price - pre_price) / pre_price) * 100
    volume_change_pct = ((post_volume - baseline) / baseline) * 100
    
    if price_change_pct == 0:
        return None
    
    elasticity = volume_change_pct / price_change_pct
    
    return {
        'date': price_change_date,
        'elasticity': elasticity,
        'pre_volume': pre_volume,
        'post_volume': post_volume,
        'baseline': baseline,
        'pre_price': pre_price,
        'post_price': post_price,
        'price_change_pct': price_change_pct,
        'volume_change_pct': volume_change_pct,
        'daily_trend': slope
    }

def calculate_time_weight(days_ago, method='exponential'):
    """Calculate time decay weight"""
    if method == 'exponential':
        # Exponential decay with 1-year half-life
        return np.exp(-0.693 * days_ago / 365)
    elif method == 'linear':
        # Linear decay by year
        if days_ago <= 365:
            return 3.0
        elif days_ago <= 730:
            return 2.0
        else:
            return 1.0

def get_weighted_elasticity(historical_elasticities, target_date):
    """Calculate time-weighted elasticity"""
    if len(historical_elasticities) == 0:
        return None
    
    weighted_sum = 0
    total_weight = 0
    
    for item in historical_elasticities:
        days_ago = (target_date - item['date']).days
        weight = calculate_time_weight(days_ago, method='exponential')
        
        weighted_sum += item['elasticity'] * weight
        total_weight += weight
    
    return weighted_sum / total_weight

def forecast_volume(baseline, elasticity, price_change_pct):
    """Forecast volume using baseline and elasticity"""
    volume_change_pct = elasticity * price_change_pct
    predicted_volume = baseline * (1 + volume_change_pct / 100)
    return predicted_volume

print("   ✓ Functions defined")

# ============================================================================
# 4. RUN BACKTEST
# ============================================================================
print("\n[4] RUNNING BACKTEST...")

# Calculate elasticity for ALL price changes first
all_elasticities = []
print("\n   Calculating elasticities for all price changes...")

for idx, pc in price_changes_df.iterrows():
    result = calculate_baseline_adjusted_elasticity(df_main, pc['date'])
    if result is not None:
        all_elasticities.append(result)
        print(f"   {pc['date'].date()}: E = {result['elasticity']:.3f}")

print(f"\n   Successfully calculated {len(all_elasticities)} elasticities")

# Now run backtest - for each event, forecast using ONLY prior data
backtest_results = []

print("\n   Running backtest (forecasting each event using prior data)...")

for i in range(len(all_elasticities)):
    current_event = all_elasticities[i]
    
    # Get all PRIOR elasticity measurements
    prior_elasticities = [e for e in all_elasticities if e['date'] < current_event['date']]
    
    if len(prior_elasticities) == 0:
        print(f"   {current_event['date'].date()}: Skipping (no prior data)")
        continue
    
    # METHOD 1: Single Elasticity (most recent)
    most_recent_elasticity = prior_elasticities[-1]['elasticity']
    single_prediction = forecast_volume(
        current_event['baseline'],
        most_recent_elasticity,
        current_event['price_change_pct']
    )
    
    # METHOD 2: Time-Decayed Weighted Elasticity
    weighted_elasticity = get_weighted_elasticity(prior_elasticities, current_event['date'])
    weighted_prediction = forecast_volume(
        current_event['baseline'],
        weighted_elasticity,
        current_event['price_change_pct']
    )
    
    # Calculate errors
    actual = current_event['post_volume']
    single_error = single_prediction - actual
    weighted_error = weighted_prediction - actual
    single_error_pct = (single_error / actual) * 100
    weighted_error_pct = (weighted_error / actual) * 100
    
    backtest_results.append({
        'date': current_event['date'],
        'actual_volume': actual,
        'baseline': current_event['baseline'],
        'price_change_pct': current_event['price_change_pct'],
        'actual_elasticity': current_event['elasticity'],
        'n_prior_events': len(prior_elasticities),
        
        # Single method
        'single_elasticity': most_recent_elasticity,
        'single_prediction': single_prediction,
        'single_error': single_error,
        'single_error_pct': single_error_pct,
        'single_abs_error_pct': abs(single_error_pct),
        
        # Weighted method
        'weighted_elasticity': weighted_elasticity,
        'weighted_prediction': weighted_prediction,
        'weighted_error': weighted_error,
        'weighted_error_pct': weighted_error_pct,
        'weighted_abs_error_pct': abs(weighted_error_pct)
    })
    
    print(f"   {current_event['date'].date()}: Single={single_error_pct:+.1f}%, Weighted={weighted_error_pct:+.1f}%")

results_df = pd.DataFrame(backtest_results)
print(f"\n   ✓ Backtest complete: {len(results_df)} events analyzed")

# ============================================================================
# 5. CALCULATE PERFORMANCE METRICS
# ============================================================================
print("\n[5] CALCULATING PERFORMANCE METRICS...")

# Mean Absolute Error (MAE)
single_mae = results_df['single_abs_error_pct'].mean()
weighted_mae = results_df['weighted_abs_error_pct'].mean()

# Root Mean Squared Error (RMSE)
single_rmse = np.sqrt((results_df['single_error_pct']**2).mean())
weighted_rmse = np.sqrt((results_df['weighted_error_pct']**2).mean())

# Mean Error (bias)
single_bias = results_df['single_error_pct'].mean()
weighted_bias = results_df['weighted_error_pct'].mean()

# Median Absolute Error
single_median = results_df['single_abs_error_pct'].median()
weighted_median = results_df['weighted_abs_error_pct'].median()

# Win rate (% of time each method is more accurate)
single_wins = (results_df['single_abs_error_pct'] < results_df['weighted_abs_error_pct']).sum()
weighted_wins = (results_df['weighted_abs_error_pct'] < results_df['single_abs_error_pct']).sum()
ties = len(results_df) - single_wins - weighted_wins

# Statistical test
t_stat, p_value = stats.ttest_rel(results_df['single_abs_error_pct'], 
                                   results_df['weighted_abs_error_pct'])

print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

print(f"\n{'Metric':<30} {'Single Elasticity':>20} {'Time-Decayed':>20} {'Winner':>10}")
print("-"*80)
print(f"{'Mean Absolute Error (MAPE)':<30} {single_mae:>19.2f}% {weighted_mae:>19.2f}% {'←' if weighted_mae < single_mae else '→':>10}")
print(f"{'Root Mean Squared Error':<30} {single_rmse:>19.2f}% {weighted_rmse:>19.2f}% {'←' if weighted_rmse < single_rmse else '→':>10}")
print(f"{'Median Absolute Error':<30} {single_median:>19.2f}% {weighted_median:>19.2f}% {'←' if weighted_median < single_median else '→':>10}")
print(f"{'Mean Bias':<30} {single_bias:>19.2f}% {weighted_bias:>19.2f}% {'':>10}")
print(f"{'Win Rate':<30} {single_wins:>19} {weighted_wins:>19} {'←' if weighted_wins > single_wins else '→':>10}")

print(f"\n{'Statistical Significance':<30} t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print(f"{'':>30} ✓ Difference IS statistically significant")
else:
    print(f"{'':>30} ✗ Difference NOT statistically significant")

improvement = ((single_mae - weighted_mae) / single_mae) * 100
print(f"\n{'Improvement':<30} {improvement:>+19.1f}%")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n[6] CREATING VISUALIZATIONS...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Prediction vs Actual - Single Method
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(results_df['actual_volume'], results_df['single_prediction'], 
           alpha=0.6, s=100, edgecolor='black')
min_val = min(results_df['actual_volume'].min(), results_df['single_prediction'].min())
max_val = max(results_df['actual_volume'].max(), results_df['single_prediction'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Volume (units/day)', fontweight='bold')
ax1.set_ylabel('Predicted Volume (units/day)', fontweight='bold')
ax1.set_title(f'Single Elasticity Method\nMAPE: {single_mae:.2f}%', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Prediction vs Actual - Weighted Method
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(results_df['actual_volume'], results_df['weighted_prediction'], 
           alpha=0.6, s=100, edgecolor='black', color='green')
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Volume (units/day)', fontweight='bold')
ax2.set_ylabel('Predicted Volume (units/day)', fontweight='bold')
ax2.set_title(f'Time-Decayed Method\nMAPE: {weighted_mae:.2f}%', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Error Distribution
ax3 = plt.subplot(2, 3, 3)
ax3.hist(results_df['single_abs_error_pct'], bins=15, alpha=0.5, 
        label='Single', edgecolor='black', color='steelblue')
ax3.hist(results_df['weighted_abs_error_pct'], bins=15, alpha=0.5, 
        label='Time-Decayed', edgecolor='black', color='green')
ax3.axvline(single_mae, color='steelblue', linestyle='--', linewidth=2)
ax3.axvline(weighted_mae, color='green', linestyle='--', linewidth=2)
ax3.set_xlabel('Absolute Error (%)', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Error Distribution', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Error over time
ax4 = plt.subplot(2, 3, 4)
ax4.plot(results_df['date'], results_df['single_abs_error_pct'], 
        'o-', label='Single', linewidth=2, markersize=6)
ax4.plot(results_df['date'], results_df['weighted_abs_error_pct'], 
        'o-', label='Time-Decayed', linewidth=2, markersize=6)
ax4.set_xlabel('Date', fontweight='bold')
ax4.set_ylabel('Absolute Error (%)', fontweight='bold')
ax4.set_title('Forecast Accuracy Over Time', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

# Plot 5: Win/Loss comparison
ax5 = plt.subplot(2, 3, 5)
categories = ['Single\nWins', 'Ties', 'Time-Decayed\nWins']
values = [single_wins, ties, weighted_wins]
colors = ['steelblue', 'gray', 'green']
bars = ax5.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, val in zip(bars, values):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f'{val}', ha='center', fontweight='bold', fontsize=12)
ax5.set_ylabel('Number of Events', fontweight='bold')
ax5.set_title(f'Head-to-Head Comparison\n(Total: {len(results_df)} events)', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Error by number of prior events
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(results_df['n_prior_events'], results_df['single_abs_error_pct'], 
           alpha=0.5, s=80, label='Single', edgecolor='black')
ax6.scatter(results_df['n_prior_events'], results_df['weighted_abs_error_pct'], 
           alpha=0.5, s=80, label='Time-Decayed', edgecolor='black')
ax6.set_xlabel('Number of Prior Events', fontweight='bold')
ax6.set_ylabel('Absolute Error (%)', fontweight='bold')
ax6.set_title('Accuracy vs Data Availability', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('elasticity_backtest_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualization saved")
plt.show()

# ============================================================================
# 7. DETAILED RESULTS TABLE
# ============================================================================
print("\n[7] DETAILED RESULTS (Last 10 Events)...")

print("\n" + "="*80)
print(f"{'Date':<12} {'Actual':>10} {'Single':>10} {'S_Err%':>8} {'Weighted':>10} {'W_Err%':>8} {'Better':>8}")
print("-"*80)

for _, row in results_df.tail(10).iterrows():
    better = 'Weighted' if row['weighted_abs_error_pct'] < row['single_abs_error_pct'] else 'Single'
    print(f"{row['date'].date()} {row['actual_volume']:>10,.0f} "
          f"{row['single_prediction']:>10,.0f} {row['single_error_pct']:>+7.1f}% "
          f"{row['weighted_prediction']:>10,.0f} {row['weighted_error_pct']:>+7.1f}% "
          f"{better:>8}")

# ============================================================================
# 8. FINAL RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

if weighted_mae < single_mae and p_value < 0.05:
    winner = "TIME-DECAYED WEIGHTED ELASTICITY"
    print(f"\n✅ WINNER: {winner}")
    print(f"\n   The time-decayed method is:")
    print(f"   • {improvement:.1f}% more accurate than single elasticity")
    print(f"   • Statistically significant (p={p_value:.4f})")
    print(f"   • Won {weighted_wins}/{len(results_df)} head-to-head comparisons")
    print(f"\n   RECOMMENDATION: Adopt time-decayed method for production forecasting")
elif single_mae < weighted_mae and p_value < 0.05:
    winner = "SINGLE ELASTICITY (MOST RECENT)"
    print(f"\n✅ WINNER: {winner}")
    print(f"\n   The single elasticity method is:")
    print(f"   • {-improvement:.1f}% more accurate than time-decayed")
    print(f"   • Statistically significant (p={p_value:.4f})")
    print(f"   • Won {single_wins}/{len(results_df)} head-to-head comparisons")
    print(f"\n   RECOMMENDATION: Continue using single (most recent) elasticity")
else:
    print(f"\n⚖️ RESULT: NO CLEAR WINNER")
    print(f"\n   Difference not statistically significant (p={p_value:.4f})")
    print(f"   Both methods perform similarly:")
    print(f"   • Single MAPE: {single_mae:.2f}%")
    print(f"   • Weighted MAPE: {weighted_mae:.2f}%")
    print(f"   • Difference: {abs(improvement):.1f}%")
    print(f"\n   RECOMMENDATION: Use simpler method (single elasticity) unless")
    print(f"                   more data shows clear advantage for time-decayed")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

# Save results
results_df.to_csv('elasticity_backtest_results.csv', index=False)
print("\n✓ Results saved to: elasticity_backtest_results.csv")
print("✓ Visualization saved to: elasticity_backtest_comparison.png")