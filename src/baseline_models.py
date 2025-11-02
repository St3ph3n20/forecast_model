"""
Baseline Models Module
======================

Implements baseline models for elasticity calculation:
- LinearBaselineModel: Phase 2 simple linear trend
- SeasonalBaselineModel: Phase 3 regression with seasonal dummies
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import timedelta


class LinearBaselineModel:
    """
    Phase 2: Simple linear trend baseline with optional YoY adjustment.

    This model fits a linear trend to historical data and optionally adjusts
    using year-over-year comparisons.

    Parameters
    ----------
    use_yoy : bool, optional
        Whether to include YoY adjustment (default: True)
    yoy_weight : float, optional
        Weight for YoY component in blend (default: 0.4)
    """

    def __init__(self, use_yoy=True, yoy_weight=0.4):
        self.use_yoy = use_yoy
        self.yoy_weight = yoy_weight
        self.slope = None
        self.intercept = None
        self.baseline_date = None
        self.baseline_volume = None
        self.yoy_baseline = None
        self.r_squared = None

    def fit(self, df, price_change_date, lookback_days=180):
        """
        Fit linear trend model on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with 'date' and 'sales_qty' columns
        price_change_date : datetime
            Date of price change (reference point)
        lookback_days : int, optional
            Days to look back for baseline (default: 180)

        Returns
        -------
        self
        """
        # Extract lookback period
        lookback_start = price_change_date - timedelta(days=lookback_days)
        historical = df[(df['date'] >= lookback_start) &
                       (df['date'] < price_change_date)].copy()

        if len(historical) < 30:
            raise ValueError(f"Insufficient historical data: {len(historical)} days (need >= 30)")

        # Fit linear trend using manual calculation (matching phase2.py)
        historical = historical.sort_values('date').reset_index(drop=True)
        n = len(historical)
        x = np.arange(n)
        y = historical['sales_qty'].values

        # Calculate slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        self.slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)

        # Calculate R² manually
        y_mean = y.mean()
        y_pred = self.slope * x + y.mean()
        ss_tot = np.sum((y - y_mean)**2)
        ss_res = np.sum((y - y_pred)**2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Store baseline info (pre-period average as reference)
        pre_start = price_change_date - timedelta(days=28)
        pre_end = price_change_date - timedelta(days=1)
        pre_period = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)]

        if len(pre_period) > 0:
            self.baseline_volume = pre_period['sales_qty'].mean()
        else:
            self.baseline_volume = y.mean()

        self.baseline_date = price_change_date

        # YoY adjustment (if enabled)
        if self.use_yoy:
            yoy_date = price_change_date - timedelta(days=365)
            yoy_data = df[(df['date'] >= yoy_date - timedelta(days=28)) &
                         (df['date'] <= yoy_date + timedelta(days=27))]

            if len(yoy_data) > 20:
                yoy_avg = yoy_data['sales_qty'].mean()
                yoy_change = (self.baseline_volume - yoy_avg) / yoy_avg
                self.yoy_baseline = self.baseline_volume * (1 + yoy_change)
            else:
                self.yoy_baseline = None

        return self

    def predict(self, target_date):
        """
        Predict baseline volume for a target date.

        Parameters
        ----------
        target_date : datetime
            Date to predict for

        Returns
        -------
        float
            Predicted baseline volume
        """
        if self.slope is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Calculate days from baseline
        days_from_baseline = (target_date - self.baseline_date).days

        # Trend-based baseline (project from pre-period average)
        # Assumes 14 days = midpoint of pre-period to midpoint of post-period
        baseline_trend = self.baseline_volume + (self.slope * (days_from_baseline + 14))

        # Blend with YoY if available
        if self.use_yoy and self.yoy_baseline is not None:
            baseline = (baseline_trend * (1 - self.yoy_weight)) + (self.yoy_baseline * self.yoy_weight)
        else:
            baseline = baseline_trend

        return baseline

    def get_r_squared(self):
        """Return model R²."""
        return self.r_squared if self.r_squared is not None else 0.0


class SeasonalBaselineModel:
    """
    Phase 3: Seasonal regression baseline with day-of-week and month dummies.

    Model formula:
    Volume = α + β₁×Days + β₂×Tue + ... + β₇×Sun + β₈×Feb + ... + β₁₈×Dec

    Reference categories: Monday (day-of-week), January (month)

    Parameters
    ----------
    None
    """

    def __init__(self):
        self.model = None
        self.r_squared = None
        self.coefficients = None
        self.feature_names = None
        self.baseline_date = None
        self.lookback_start = None

    def fit(self, df, price_change_date, lookback_days=180):
        """
        Fit seasonal regression model on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with 'date' and 'sales_qty' columns
        price_change_date : datetime
            Date of price change (reference point)
        lookback_days : int, optional
            Days to look back for baseline (default: 180)

        Returns
        -------
        self
        """
        # Extract lookback period
        lookback_start = price_change_date - timedelta(days=lookback_days)
        historical = df[(df['date'] >= lookback_start) &
                       (df['date'] < price_change_date)].copy()

        if len(historical) < 30:
            raise ValueError(f"Insufficient historical data: {len(historical)} days (need >= 30)")

        self.baseline_date = price_change_date
        self.lookback_start = lookback_start

        # Create features
        historical = historical.copy()
        historical['days_since_start'] = (historical['date'] - lookback_start).dt.days

        # Day-of-week dummies (Monday = 0 is reference, so we create dummies for Tue-Sun)
        historical['dayofweek'] = historical['date'].dt.dayofweek
        for day in range(1, 7):  # Tuesday (1) through Sunday (6)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            historical[f'dow_{day_names[day]}'] = (historical['dayofweek'] == day).astype(int)

        # Month dummies (January = 1 is reference, so we create dummies for Feb-Dec)
        historical['month'] = historical['date'].dt.month
        for month in range(2, 13):  # February (2) through December (12)
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            historical[f'month_{month_names[month]}'] = (historical['month'] == month).astype(int)

        # Prepare features
        feature_cols = ['days_since_start']
        feature_cols += [f'dow_{d}' for d in ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]
        feature_cols += [f'month_{m}' for m in ['Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

        X = historical[feature_cols].values
        y = historical['sales_qty'].values

        # Fit model
        self.model = LinearRegression()
        self.model.fit(X, y)

        # Calculate R²
        self.r_squared = self.model.score(X, y)

        # Store coefficients
        self.feature_names = feature_cols
        self.coefficients = {
            'intercept': self.model.intercept_,
            **{name: coef for name, coef in zip(feature_cols, self.model.coef_)}
        }

        return self

    def predict(self, target_date):
        """
        Predict baseline volume for a target date.

        Parameters
        ----------
        target_date : datetime or pd.Timestamp
            Date to predict for

        Returns
        -------
        float
            Predicted baseline volume
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to pandas Timestamp if needed
        if not isinstance(target_date, pd.Timestamp):
            target_date = pd.Timestamp(target_date)

        # Create features for target date
        features = {}
        features['days_since_start'] = (target_date - self.lookback_start).days

        # Day-of-week
        dayofweek = target_date.dayofweek
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for day in range(1, 7):
            features[f'dow_{day_names[day]}'] = 1 if dayofweek == day else 0

        # Month
        month = target_date.month
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for m in range(2, 13):
            features[f'month_{month_names[m]}'] = 1 if month == m else 0

        # Create feature vector in correct order
        X = np.array([[features[name] for name in self.feature_names]])

        # Predict
        prediction = self.model.predict(X)[0]

        return prediction

    def get_r_squared(self):
        """Return model R²."""
        return self.r_squared if self.r_squared is not None else 0.0

    def get_seasonal_coefficients(self):
        """
        Get seasonal coefficients as a formatted DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: component, coefficient, effect_pct
        """
        if self.coefficients is None:
            return None

        # Extract day-of-week effects
        dow_coefs = []
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        baseline_vol = self.coefficients['intercept']

        # Monday is reference (coef = 0)
        dow_coefs.append({
            'component': 'Monday',
            'coefficient': 0.0,
            'effect_pct': 0.0
        })

        for day in day_names[1:]:  # Tue-Sun
            key = f'dow_{day}'
            if key in self.coefficients:
                coef = self.coefficients[key]
                effect_pct = (coef / baseline_vol) * 100 if baseline_vol != 0 else 0
                dow_coefs.append({
                    'component': day,
                    'coefficient': coef,
                    'effect_pct': effect_pct
                })

        # Extract month effects
        month_coefs = []
        month_names_full = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
        month_abbrev = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # January is reference (coef = 0)
        month_coefs.append({
            'component': 'January',
            'coefficient': 0.0,
            'effect_pct': 0.0
        })

        for month_full, month_short in zip(month_names_full[1:], month_abbrev[1:]):
            key = f'month_{month_short}'
            if key in self.coefficients:
                coef = self.coefficients[key]
                effect_pct = (coef / baseline_vol) * 100 if baseline_vol != 0 else 0
                month_coefs.append({
                    'component': month_full,
                    'coefficient': coef,
                    'effect_pct': effect_pct
                })

        # Combine
        all_coefs = dow_coefs + month_coefs

        return pd.DataFrame(all_coefs)
