Your data shows:

Day of Week CV: 15.2% (STRONG)
Monthly CV: 6.4% (MODERATE)
Friday: +20% above average
Sunday: -18% below average

This means if a price change happens on Friday and you compare post-period to pre-period with different day-of-week mixes, you could be off by 20-40% just from day-of-week effects!
My Recommendation: Regression with Seasonal Dummies (Option 3)
Because:

Interpretable - You can say "controlling for day-of-week and month..."
Flexible - Easy to add holidays or test if seasonality matters
Testable - Can validate each component's significance
Integrates perfectly with your existing baseline-adjusted framework

Before I Build Phase 3, I need your input:
Question 1: Do you prefer:

A) Regression with dummies (my recommendation - transparent coefficients)
B) STL decomposition (more automatic, less transparent)
C) Start with simple multiplicative factors to test if it matters, then upgrade

Question 2: Which seasonal components to include:

A) Month + Day-of-week (captures 90% of variation)
B) Add explicit holiday flags (Christmas, Easter, etc.)

Question 3: How should Phase 3 integrate with Phase 2's time-decay weighting:

A) Apply time-decay weighting to the seasonally-adjusted elasticities (simpler)
B) Weight similar seasons more (e.g., if forecasting November, weight past Novembers more)

What direction would you like to go?
