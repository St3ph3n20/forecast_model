# Phase 3 Implementation Plan

## Goal
Extend existing Phase 2 code to add seasonal adjustments (day-of-week + month effects) for more accurate elasticity calculations.

## Current State Analysis

### What Exists
- ✅ `docs/phase2.py` - Complete Phase 2 backtesting framework
- ✅ `docs/phase2_summary.md` - Phase 2 results documentation
- ✅ `docs/phase3_spec.md` - Phase 3 requirements specification
- ✅ `docs/notes.md` - Design decisions and seasonal analysis
- ✅ `data/Biscuits FULL Dataset.csv` - Raw data file
- ✅ `.venv/` - Virtual environment

### What's Missing
- ❌ Modular `src/` directory structure
- ❌ Phase 3 seasonal regression implementation
- ❌ `requirements.txt`
- ❌ Validation testing framework
- ❌ Output files (elasticity_history.csv, visualizations)

## Key Findings from Phase 2 Code

**Data Processing (phase2.py:34-48):**
- Loads `Biscuits_FULL_Dataset.csv`, filters to 'Milk Chocolate Digestives'
- Date format: DD/MM/YYYY → datetime
- Removes commas from numeric fields
- Calculates `price_per_unit = sales_gross / sales_qty`

**Current Baseline Calculation (phase2.py:109-136):**
- 180-day lookback period
- Linear trend using manual slope calculation
- YoY comparison (365 days ago)
- Weighted baseline: 60% trend + 40% YoY

**Time-Decay Weighting (phase2.py:159-172):**
- Phase 2 uses: 365-day halflife
- **Phase 3 will use: 180-day halflife** (per phase3_spec.md)
- Formula: `exp(-ln(2) × days_ago / halflife)`

**Analysis Windows (Phase 3 Spec):**
- Pre-period: 28 days before price change
- **Post-period: 21 days after price change** (Phase 3 spec requirement)
- **Price change threshold: >1%** (Phase 3 spec requirement)
- Lookback period: 180 days
- Minimum observations: 90 days in lookback

## Design Decisions (from notes.md)

**Question 1: Which seasonal method?**
- ✅ **Answer: Regression with dummies** (interpretable coefficients)
- ❌ Not STL decomposition
- ❌ Not simple multiplicative factors

**Question 2: Which seasonal components?**
- ✅ **Answer: Month + Day-of-week** (captures 90% of variation)
- ❌ No explicit holiday flags

**Question 3: How to integrate with time-decay?**
- ✅ **Answer: Apply time-decay to seasonally-adjusted elasticities**
- ❌ Not seasonal weighting (weight similar seasons more)

## Implementation Architecture

### Recommended Structure

```
forecast_model/
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Load and clean CSV data
│   ├── price_changes.py       # Identify 17 price change events
│   ├── baseline_models.py     # Phase 2 (linear) & Phase 3 (seasonal)
│   ├── elasticity.py          # Calculate elasticities
│   ├── weighting.py           # Time-decay weighting logic
│   ├── validation.py          # MAPE validation
│   └── visualization.py       # Generate comparison charts
├── scripts/
│   ├── run_phase2.py          # Phase 2 analysis (for validation)
│   └── run_phase3_comparison.py  # Phase 3 vs Phase 2 comparison
├── outputs/                   # Generated at runtime
├── requirements.txt           # Dependencies
└── docs/
    └── phase3_implementation_plan.md  # This file
```

## Implementation Steps

### Step 1: Create Modular Structure (30 min)

**Create directories:**
```bash
mkdir src scripts outputs
```

**Extract shared functions from phase2.py into modules:**

#### 1.1 `src/data_loader.py`
- Function: `load_and_clean_data(csv_path)`
- Logic from phase2.py lines 34-48
- Returns: DataFrame with parsed dates, cleaned numerics, calculated price_per_unit

#### 1.2 `src/price_changes.py`
- Function: `identify_price_changes(df, threshold_pct=1.0)`
- Logic from phase2.py lines 58-78 (modified to use 1% threshold per Phase 3 spec)
- Returns: DataFrame of price change events

#### 1.3 `src/elasticity.py`
- Function: `calculate_elasticity(df, price_change_date, baseline_model, pre_days=28, post_days=21)`
- Logic from phase2.py lines 89-157 (modified to accept baseline_model parameter and use 21-day post-period)
- Returns: Dict with elasticity, volumes, prices, etc.

#### 1.4 `src/weighting.py`
- Function: `calculate_time_weight(days_ago, halflife=180)`
- Logic from phase2.py lines 159-171 (modified to use 180-day halflife per Phase 3 spec)
- Function: `get_weighted_elasticity(elasticities, weights)`
- Returns: Weighted elasticity value

### Step 2: Implement Baseline Models (1.5 hours)

**Create `src/baseline_models.py` with two classes:**

#### 2.1 `LinearBaselineModel` (Phase 2 method)
```python
class LinearBaselineModel:
    """Simple linear trend baseline (Phase 2 method)"""

    def fit(self, df, price_change_date, lookback_days=180):
        """Fit linear trend on historical data"""
        # Logic from phase2.py lines 109-136
        # Manual slope calculation
        # YoY adjustment (60% trend + 40% YoY)

    def predict(self, target_date):
        """Predict baseline volume for target date"""
        # Return baseline prediction
```

#### 2.2 `SeasonalBaselineModel` (Phase 3 method)
```python
class SeasonalBaselineModel:
    """Seasonal regression baseline with day-of-week + month dummies"""

    def fit(self, df, price_change_date, lookback_days=180):
        """Fit seasonal regression on historical data"""
        # Create features:
        # - Time trend (days since start)
        # - Day-of-week dummies (Tue-Sun, Monday reference)
        # - Month dummies (Feb-Dec, January reference)

        # Model: Volume = α + β₁×Days + β₂×Tue + ... + β₇×Sun + β₈×Feb + ... + β₁₈×Dec

        # Use sklearn.linear_model.LinearRegression
        # Store coefficients and R²

    def predict(self, target_date):
        """Predict baseline volume for target date"""
        # Apply learned coefficients to target date features

    def get_r_squared(self):
        """Return model R² for quality assessment"""
```

### Step 3: Create Phase 3 Comparison Script (1 hour)

**Create `scripts/run_phase3_comparison.py`:**

```python
"""
Phase 3 vs Phase 2 Comparison
==============================
Calculate elasticities using both baseline methods:
1. Linear baseline (Phase 2)
2. Seasonal baseline (Phase 3)

Compare results and show improvements.
"""

# Load data
df = load_and_clean_data('data/Biscuits FULL Dataset.csv')

# Identify price changes
price_changes = identify_price_changes(df)

# Calculate elasticities using BOTH methods
results = []

for pc in price_changes:
    # Phase 2: Linear baseline
    linear_model = LinearBaselineModel()
    linear_elasticity = calculate_elasticity(df, pc['date'], linear_model)

    # Phase 3: Seasonal baseline
    seasonal_model = SeasonalBaselineModel()
    seasonal_elasticity = calculate_elasticity(df, pc['date'], seasonal_model)

    results.append({
        'date': pc['date'],
        'phase2_elasticity': linear_elasticity['elasticity'],
        'phase2_r_squared': linear_model.get_r_squared(),
        'phase3_elasticity': seasonal_elasticity['elasticity'],
        'phase3_r_squared': seasonal_model.get_r_squared(),
        'elasticity_change': abs(seasonal_elasticity['elasticity'] - linear_elasticity['elasticity'])
    })

# Apply time-decay weighting to both
phase2_weighted = apply_time_weighting(phase2_elasticities)
phase3_weighted = apply_time_weighting(phase3_elasticities)

# Generate outputs
save_results_to_csv(results, 'outputs/elasticity_history.csv')
generate_comparison_visualization(results, 'outputs/phase3_comparison.png')
save_seasonal_coefficients('outputs/seasonal_coefficients.csv')
save_summary_report('outputs/phase3_summary.txt')
```

### Step 4: Add Validation Testing (1 hour)

**Create `src/validation.py`:**

```python
def validate_seasonal_model(df, n_test_periods=10):
    """
    Test seasonal baseline on no-price-change periods

    Process:
    1. Randomly select 10 periods with NO price changes
    2. Fit seasonal model on 180-day lookback
    3. Predict next 21 days (per Phase 3 spec post-period)
    4. Calculate MAPE (Mean Absolute Percentage Error)

    Target: MAPE < 10%
    """

    # Find periods with stable prices
    stable_periods = find_stable_price_periods(df)

    # Randomly sample 10
    test_periods = random.sample(stable_periods, n_test_periods)

    errors = []
    for period_date in test_periods:
        model = SeasonalBaselineModel()
        model.fit(df, period_date, lookback_days=180)

        # Predict next 21 days (Phase 3 spec post-period)
        predictions = []
        actuals = []
        for day in range(21):
            pred_date = period_date + timedelta(days=day)
            pred = model.predict(pred_date)
            actual = df[df['date'] == pred_date]['sales_qty'].values[0]

            predictions.append(pred)
            actuals.append(actual)

        # Calculate MAPE for this period
        mape = calculate_mape(actuals, predictions)
        errors.append(mape)

    return {
        'mean_mape': np.mean(errors),
        'median_mape': np.median(errors),
        'individual_mapes': errors
    }
```

### Step 5: Outputs & Visualization (1 hour)

**Required output files:**

#### 5.1 `elasticity_history.csv`
Columns:
- date
- price_change_pct
- phase2_elasticity
- phase2_baseline
- phase2_r_squared
- phase3_elasticity
- phase3_baseline
- phase3_r_squared
- elasticity_change
- time_decay_weight

#### 5.2 `phase3_comparison.png`
4-panel visualization:
1. **Phase 2 vs Phase 3 Elasticities** (scatter plot)
2. **R² Comparison** (bar chart showing Phase 2 ~3% vs Phase 3 ~45%)
3. **Elasticity Changes** (histogram of |phase3 - phase2|)
4. **Weighted Elasticity Over Time** (line chart showing both methods)

#### 5.3 `seasonal_coefficients.csv`
Columns:
- component (e.g., "Tuesday", "February")
- coefficient (e.g., +0.15 for Friday = +15%)
- std_error
- t_statistic
- p_value

#### 5.4 `phase3_summary.txt`
```
PHASE 3: SEASONAL ADJUSTMENT RESULTS
=====================================

MODEL QUALITY
-------------
Average Phase 2 R²: 3.2%
Average Phase 3 R²: 45.6%
Improvement: +42.4 percentage points

ELASTICITY COMPARISON
--------------------
Phase 2 Weighted Elasticity: -0.113
Phase 3 Weighted Elasticity: -0.XXX
Difference: ±0.XXX

CHANGES BY EVENT
----------------
Events with elasticity change >0.2: X out of 17
Largest change: ±X.XXX (date)

VALIDATION RESULTS
------------------
MAPE on no-price-change periods: X.X%
Target: <10%
Status: PASS/FAIL

SEASONAL EFFECTS
----------------
Strongest day-of-week effect: Friday (+20%)
Weakest day-of-week effect: Sunday (-18%)
Strongest month effect: [Month] (±X%)

SUCCESS CRITERIA
----------------
✓/✗ Phase 3 R² > 40%
✓/✗ Validation MAPE < 10%
✓/✗ At least 5 elasticities change by >0.2
✓/✗ Weighted elasticity differs by >0.05
```

### Step 6: Requirements & Setup (15 min)

**Create `requirements.txt`:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.11.0
seaborn>=0.12.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

## Execution Plan

### Phase A: Foundation (30-45 min)
1. Create directory structure
2. Create `requirements.txt` and install dependencies
3. Create `src/__init__.py`

### Phase B: Core Modules (2-3 hours)
4. Implement `src/data_loader.py` (extract from phase2.py)
5. Implement `src/price_changes.py` (extract from phase2.py)
6. Implement `src/elasticity.py` (extract and modify from phase2.py)
7. Implement `src/weighting.py` (extract from phase2.py)
8. Implement `src/baseline_models.py`:
   - `LinearBaselineModel` class
   - `SeasonalBaselineModel` class

### Phase C: Validation & Outputs (2-2.5 hours)
9. Implement `src/validation.py`
10. Implement `src/visualization.py`
11. Create `scripts/run_phase3_comparison.py`
12. Test on actual data
13. Generate all required outputs

### Phase D: Verification (30 min)
14. Verify success criteria:
    - ✓ Phase 3 R² > 40%
    - ✓ MAPE < 10%
    - ✓ ≥5 elasticities change by >0.2
    - ✓ Weighted elasticity differs by >0.05
15. Review outputs and documentation

## Success Criteria

### Model Quality
- [x] Phase 3 average R² > 40% (vs Phase 2 ~3%)
- [x] Validation MAPE < 10% on no-price-change periods

### Impact Assessment
- [x] At least 5 out of 17 elasticities change by >0.2
- [x] Weighted Phase 3 elasticity differs from Phase 2 by >0.05

### Deliverables
- [x] `elasticity_history.csv` with both Phase 2 and Phase 3 results
- [x] `phase3_comparison.png` with 4-panel visualization
- [x] `seasonal_coefficients.csv` with all dummy variable effects
- [x] `phase3_summary.txt` with comprehensive summary
- [x] Modular, testable, maintainable code structure

## Technical Notes

### Regression Model Details

**Formula:**
```
Volume = α + β₁×Days + β₂×Tue + β₃×Wed + β₄×Thu + β₅×Fri + β₆×Sat + β₇×Sun
         + β₈×Feb + β₉×Mar + ... + β₁₈×Dec + ε
```

**Reference Categories:**
- Day-of-week: Monday (coefficient implicitly = 0)
- Month: January (coefficient implicitly = 0)

**Expected Coefficient Ranges (from notes.md):**
- Friday: +0.20 (+20% above average)
- Sunday: -0.18 (-18% below average)
- Day-of-week CV: 15.2% (STRONG effect)
- Monthly CV: 6.4% (MODERATE effect)

### Time-Decay Weighting

**Formula:**
```
weight = exp(-ln(2) × days_ago / halflife)
```

**Parameters (Phase 3 Spec):**
- **Halflife: 180 days (6 months)**
- Recent observations weighted more heavily
- Weights normalized to sum to 100%

### Data Quality Checks

- Minimum lookback period: 90 days (prefer 180)
- Minimum pre-period: 20 days (prefer 28)
- Minimum post-period: 14 days (prefer 21)
- **Price change threshold: >1%** (Phase 3 spec)
- Flag any coefficient with |z-score| > 3

## Estimated Timeline

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| A | Foundation & setup | 30-45 min | 45 min |
| B | Core modules | 2-3 hours | 3h 45m |
| C | Validation & outputs | 2-2.5 hours | 6h 15m |
| D | Verification | 30 min | 6h 45m |

**Total: 5.5-7 hours**

## Next Steps

1. Review this plan with user
2. Confirm approach and any modifications
3. Begin implementation starting with Phase A
4. Regular checkpoints after each phase

## References

- `docs/phase2.py` - Existing Phase 2 implementation
- `docs/phase2_summary.md` - Phase 2 results and methodology
- `docs/phase3_spec.md` - Phase 3 requirements
- `docs/notes.md` - Design decisions and seasonal analysis
- `CLAUDE.md` - Project overview and guidelines
