# Phase 3: Seasonally-Adjusted Elasticity Analysis

## Objective

Recalculate all historical price elasticities using seasonally-adjusted baselines (day-of-week + month effects) to isolate true price impact from calendar effects.

## Context

- **Current State**: Phase 2 uses simple linear trend for baseline → R² = 3.2%
- **Problem**: Day-of-week variance = 15.2% CV (Friday +20%, Sunday -18%)
- **Impact**: Elasticity contaminated by calendar effects
- **Solution**: Regression with dummy variables → R² = 45.6%

## Data

- **File**: `milk_choc_digestives_clean.csv` (1,378 days, 2022-01-03 to 2025-10-20)
- **Columns**: date, sales_qty, price_per_unit, sales_gross, sales_net
- **Price Changes**: 17 significant changes (>1%) identified in Phase 2

## Implementation

### Step 1: Build Seasonal Regression Model

```python
# For each historical price change, fit regression on 180-day lookback period:
# Volume = α + β₁×Days + β₂×Tue + β₃×Wed + ... + β₇×Sun +
#          β₈×Feb + β₉×Mar + ... + β₁₈×Dec

# Features:
- days_since_start: Linear trend
- dow_1 to dow_6: Tuesday through Sunday (Monday = reference)
- month_2 to month_12: February through December (January = reference)

# Fit using: sklearn.linear_model.LinearRegression
```

### Step 2: Calculate Seasonally-Adjusted Baselines

For each of 17 price changes:

1. Define lookback period: 180 days before price change
2. Fit regression model on lookback data
3. Use model to predict baseline for POST-period dates
4. Baseline accounts for:
   - Trend continuation
   - Specific days of week in post-period
   - Specific months in post-period

### Step 3: Recalculate Elasticities

```python
# For each price change:
pre_volume = mean(sales_qty in 28-day pre-period)
post_volume = mean(sales_qty in 21-day post-period)
baseline_volume = mean(model.predict(post-period dates))

# Phase 2 method (for comparison):
elasticity_phase2 = ((post_volume - baseline_simple) / baseline_simple) / price_change_pct

# Phase 3 method (NEW):
elasticity_phase3 = ((post_volume - baseline_seasonal) / baseline_seasonal) / price_change_pct
```

### Step 4: Apply Time-Decay Weighting

```python
# Same as Phase 2:
weight = exp(-ln(2) × days_ago / 180)
normalized_weight = weight / sum(weights)
weighted_elasticity_phase3 = sum(elasticity_phase3 × normalized_weight)
```

### Step 5: Validation

Test model accuracy on periods with NO price changes:

1. Select 10 random 21-day periods with stable pricing
2. Predict volume using seasonal model
3. Compare to actual
4. Calculate MAPE (Mean Absolute Percentage Error)

## Output Requirements

### 1. Elasticity History (CSV)

Columns:

- date
- price_change_pct
- pre_volume, post_volume
- baseline_phase2 (simple trend)
- baseline_phase3 (seasonal regression)
- elasticity_phase2
- elasticity_phase3
- elasticity_difference
- model_r_squared
- normalized_weight

### 2. Summary Statistics (TXT)

```
Phase 2 weighted elasticity: -0.113
Phase 3 weighted elasticity: [CALCULATE]
Change: [CALCULATE]

Average R² improvement: [baseline_simple R² → seasonal R²]
Validation MAPE: [CALCULATE]

Largest elasticity changes (top 5):
  [date] [phase2] → [phase3] ([difference])
```

### 3. Comparison Visualization (PNG)

4 panels:

- **Top Left**: Elasticity over time (Phase 2 vs Phase 3 scatter)
- **Top Right**: Elasticity difference distribution (histogram)
- **Bottom Left**: Baseline R² comparison (Phase 2 vs Phase 3 bar chart)
- **Bottom Right**: Validation - Predicted vs Actual on no-price-change periods

### 4. Seasonal Coefficients Table (CSV)

```
component,coefficient,pct_impact,std_error,p_value
intercept,[value],[N/A],[value],[value]
trend,[value],[N/A],[value],[value]
tuesday,[value],[pct],[value],[value]
wednesday,[value],[pct],[value],[value]
...
february,[value],[pct],[value],[value]
march,[value],[pct],[value],[value]
...
```

## Key Decisions

### Model Specification

- **Lookback**: 180 days (6 months before each price change)
- **Post-period**: 21 days after price change
- **Pre-period**: 28 days before price change
- **Decay halflife**: 180 days (unchanged from Phase 2)
- **Min observations**: 90 days in lookback (skip if insufficient)

### Dummy Encoding

- **Day-of-week**: Monday = reference (coefficient = 0)
- **Month**: January = reference (coefficient = 0)
- **No holiday flags in v1** (add later if needed)

### Quality Checks

- Flag if seasonal model R² < 0.20 (weak seasonality)
- Flag if any coefficient has |z-score| > 3 (outlier)
- Validate model assumptions: residuals should be ~normal

## Success Criteria

1. ✅ Phase 3 model R² > 40% (vs Phase 2 ~3%)
2. ✅ Validation MAPE < 10% on no-price-change periods
3. ✅ At least 5 elasticities change by >0.2 due to seasonal adjustment
4. ✅ Weighted elasticity changes meaningfully (>0.05 difference)

## Integration with Existing Work

- Use existing `elasticity_history.csv` from Phase 2 as starting point
- Maintain same 17 price change dates
- Keep same time-decay weighting methodology
- Add Phase 3 columns, don't replace Phase 2 (compare side-by-side)

## File Locations

- Input: `/home/claude/milk_choc_digestives_clean.csv`
- Input: `/home/claude/elasticity_history.csv` (Phase 2 results)
- Output: `/mnt/user-data/outputs/phase3_*.* `

## Notes

- Do NOT add holiday detection yet (keep simple first iteration)
- Do NOT use SARIMAX or complex time series models
- Focus on interpretability: clear coefficients for each day/month
- Prioritize correctness over speed
