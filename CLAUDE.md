# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated price elasticity forecasting system for retail/FMCG products. The system analyzes historical sales and pricing data to predict volume impacts of price changes.

**Current Product**: Milk Chocolate Digestives (example dataset)
**Analysis Period**: January 2022 - October 2025 (1,378 days)
**Key Innovation**: Time-weighted elasticity with seasonal adjustments

## Architecture

### Core Methodology (3-Phase Evolution)

1. **Phase 1**: Baseline-adjusted elasticity
   - Establishes baseline trends using 180-day historical lookback
   - Measures volume changes vs. baseline (not just vs. previous period)
   - Accounts for underlying trends to isolate price effects

2. **Phase 2**: Time-weighted elasticity with exponential decay (COMPLETED)
   - Applies exponential time-decay weighting (recent observations matter more)
   - Halflife: 180 days (6 months)
   - Weight formula: `exp(-ln(2) × days_ago / halflife)`
   - **Key Result**: Weighted elasticity = -0.113 (vs. -1.100 simple average)
   - Dramatically changes business recommendations

3. **Phase 3**: Seasonally-adjusted elasticity (SPECIFICATION READY)
   - Adds regression-based seasonal adjustments
   - Day-of-week effects (Friday +20%, Sunday -18%)
   - Month effects
   - Expected R² improvement: 3.2% → 45.6%
   - Model: `Volume = α + β₁×Days + β₂×Tue + ... + β₇×Sun + β₈×Feb + ... + β₁₈×Dec`

### Analysis Windows

- **Lookback period**: 180 days before price change
- **Pre-period**: 28 days before price change
- **Post-period**: 21 days after price change
- **Minimum price change threshold**: >1% (excludes noise)

### Elasticity Calculation

```
elasticity = ((post_volume - baseline_volume) / baseline_volume) / price_change_pct
weighted_elasticity = sum(elasticity × normalized_weight)
```

## Development Environment

- Python 3.12
- Virtual environment: `.venv/`
- Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)

### Expected Dependencies

The project will require:
- `pandas` - data manipulation
- `numpy` - numerical computations
- `scikit-learn` - regression models (LinearRegression)
- `matplotlib` - visualization
- `scipy` - statistical functions

## Project Structure

```
forecast_model/
├── .venv/              # Virtual environment (gitignored)
├── data/               # Dataset files
│   └── Biscuits FULL Dataset.csv
├── docs/               # Specifications and documentation
│   ├── phase2_summary.md
│   └── phase3_spec.md
├── src/                # Python source code (to be created)
├── tests/              # Test files (to be created)
├── outputs/            # Analysis outputs (gitignored)
├── .gitignore          # Git ignore rules
├── CLAUDE.md           # This file
└── requirements.txt    # Python dependencies (to be created)
```

### Key Files & Locations

- **Specifications**: `docs/phase2_summary.md`, `docs/phase3_spec.md`
- **Input Data**: `data/Biscuits FULL Dataset.csv`
- **Expected Outputs**: `outputs/` directory

### Output Files (Phase 2 Reference)

- `phase2_elasticity_analysis.png` - 4-panel visualization
- `elasticity_history.csv` - All elasticity measurements with weights
- `phase2_config.txt` - Configuration parameters
- `phase2_decayed_elasticity.py` - Analysis script

### Output Files (Phase 3 Requirements)

- `elasticity_history.csv` - Extended with Phase 3 columns
- `phase3_comparison.png` - 4-panel comparison visualization
- `seasonal_coefficients.csv` - Day-of-week and month effects
- `phase3_summary.txt` - Summary statistics

## Common Commands

### Environment Setup
```bash
# Install dependencies (create requirements.txt first)
pip install pandas numpy scikit-learn matplotlib scipy

# Or from requirements.txt
pip install -r requirements.txt
```

### Running Analysis
```bash
# Phase 2 (if script exists)
python phase2_decayed_elasticity.py

# Phase 3 (to be implemented)
python phase3_seasonal_elasticity.py
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_elasticity.py
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

## Development Workflow for Phase 3

When implementing Phase 3 seasonal adjustments:

1. **Load Data**: Read `data/Biscuits FULL Dataset.csv`
2. **Load Phase 2 Results**: Read existing `elasticity_history.csv` from `outputs/`
3. **Build Seasonal Model**: For each of 17 price changes:
   - Fit regression on 180-day lookback with day-of-week + month dummies
   - Use Monday and January as reference categories
   - Calculate model R² and validate coefficients
4. **Recalculate Elasticities**: Compare actual post-period vs. seasonal baseline
5. **Apply Time-Decay Weighting**: Same 180-day halflife as Phase 2
6. **Validate**: Test on 10 random no-price-change periods, calculate MAPE
7. **Output**: Generate all required CSVs, TXTs, and visualizations

## Key Design Principles

1. **Interpretability over complexity**: Use clear regression coefficients, not black-box models
2. **Baseline adjustment is critical**: Never measure elasticity as simple before/after comparison
3. **Recent data matters more**: Always apply time-decay weighting
4. **Validate thoroughly**: Test on periods without price changes to verify model accuracy
5. **Side-by-side comparison**: Keep Phase 2 and Phase 3 results for comparison

## Model Quality Checks

- Seasonal model R² should be >40%
- Validation MAPE should be <10%
- Flag if any coefficient has |z-score| > 3
- Ensure at least 90 days in lookback period
- Check residuals are approximately normal

## Success Criteria (Phase 3)

1. ✅ Phase 3 model R² > 40% (vs Phase 2 ~3%)
2. ✅ Validation MAPE < 10% on no-price-change periods
3. ✅ At least 5 elasticities change by >0.2 due to seasonal adjustment
4. ✅ Weighted elasticity changes meaningfully (>0.05 difference from Phase 2)

## Business Context

This system directly impacts pricing decisions with multi-million dollar revenue implications. Accurate elasticity estimates are critical:

- **Phase 2 Finding**: Market is much less price-sensitive than believed (-0.113 vs -1.100)
- **Revenue Impact**: £1.4M - £2.77M annual opportunity from better pricing decisions
- **Strategic Insight**: Brand strength has increased significantly over 3 years
- **Recommendation Shift**: From "don't raise prices" to "aggressive price increases viable"

## Notes

- Date format in CSV: YYYY-MM-DD
- Volume field: `sales_qty`
- Price field: `price_per_unit`
- All monetary values assumed to be in GBP (£)
- 17 historical price changes identified (May 2022 - August 2025)
