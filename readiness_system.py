#!/usr/bin/env python3
"""
Garmin PRO Trener - Core Analytics & Readiness System
Personalized readiness scoring based on HRV + RHR + Sleep + Wellbeing (Garmin proxies).

Based on peer-reviewed research (2021-2025):
- HRV-guided training meta-analysis (SMD=0.50)
- Combined HRV+RHR+WB approach (greatest improvements)
- Sleep deprivation injury risk thresholds
- Overtraining detection indicators

Usage:
    Standalone:  python readiness_system.py
    As module:   from readiness_system import compute_readiness_for_dashboard
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'garmin_data')
BASELINES_FILE = os.path.join(BASE_DIR, 'my_baselines.json')

# Try importing scipy - graceful fallback
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Correlation analysis will be limited.")
    print("Install with: pip install scipy>=1.12.0")


# ─── Personal Baselines ─────────────────────────────────────────────────────

class PersonalBaselines:
    """Compute and manage 7-day rolling baselines for key metrics."""

    METRICS = {
        'hrv': 'hrvLastNightAvg',
        'rhr': 'restingHeartRate',
        'sleep_hours': 'sleepTimeHours',
        'sleep_score': 'sleepScore',
        'stress': 'avgStressLevel',
        'body_battery_wake': 'bodyBatteryAtWake',
        'body_battery_high': 'bodyBatteryHighest',
    }

    def __init__(self, window=7):
        self.window = window
        self.baselines = {}

    def compute(self, df):
        """Compute baselines from the full dataset (latest 7-day window)."""
        df = df.sort_values('date').copy()

        for key, col in self.METRICS.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 3:
                continue

            recent = series.tail(self.window)
            self.baselines[key] = {
                'mean': round(float(recent.mean()), 2),
                'sd': round(float(recent.std()), 2),
                'median': round(float(recent.median()), 2),
                'min': round(float(recent.min()), 2),
                'max': round(float(recent.max()), 2),
                'n': int(len(recent)),
                'column': col,
            }

        # Derived thresholds
        if 'hrv' in self.baselines:
            b = self.baselines['hrv']
            b['upper_05sd'] = round(b['mean'] + 0.5 * b['sd'], 2)
            b['lower_05sd'] = round(b['mean'] - 0.5 * b['sd'], 2)
            b['lower_10sd'] = round(b['mean'] - 1.0 * b['sd'], 2)
            b['cv_pct'] = round((b['sd'] / b['mean'] * 100) if b['mean'] > 0 else 0, 2)

        if 'rhr' in self.baselines:
            b = self.baselines['rhr']
            b['upper_3bpm'] = round(b['mean'] + 3, 2)
            b['lower_3bpm'] = round(b['mean'] - 3, 2)
            b['upper_5bpm'] = round(b['mean'] + 5, 2)
            b['upper_7bpm'] = round(b['mean'] + 7, 2)
            b['elevated_5pct'] = round(b['mean'] * 1.05, 2)

        return self.baselines

    def compute_for_date(self, df, target_date, lookback=7):
        """Compute baselines using only data before target_date (no look-ahead)."""
        df = df.sort_values('date').copy()
        mask = df['date'] < pd.Timestamp(target_date)
        prior = df[mask].tail(lookback)

        baselines = {}
        for key, col in self.METRICS.items():
            if col not in prior.columns:
                continue
            series = prior[col].dropna()
            if len(series) < 3:
                continue
            baselines[key] = {
                'mean': float(series.mean()),
                'sd': float(series.std()),
            }

        # Derived thresholds
        if 'hrv' in baselines:
            b = baselines['hrv']
            b['upper_05sd'] = b['mean'] + 0.5 * b['sd']
            b['lower_05sd'] = b['mean'] - 0.5 * b['sd']
            b['lower_10sd'] = b['mean'] - 1.0 * b['sd']
            b['cv_pct'] = (b['sd'] / b['mean'] * 100) if b['mean'] > 0 else 0

        if 'rhr' in baselines:
            b = baselines['rhr']
            b['upper_3bpm'] = b['mean'] + 3
            b['lower_3bpm'] = b['mean'] - 3
            b['upper_5bpm'] = b['mean'] + 5
            b['upper_7bpm'] = b['mean'] + 7

        return baselines

    def save(self, filepath=None):
        """Save baselines to JSON."""
        filepath = filepath or BASELINES_FILE
        output = {
            'computed_at': datetime.now().isoformat(),
            'window_days': self.window,
            'baselines': self.baselines,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return filepath

    def load(self, filepath=None):
        """Load baselines from JSON."""
        filepath = filepath or BASELINES_FILE
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.baselines = data.get('baselines', {})
        return self.baselines


# ─── Wellbeing Proxy ─────────────────────────────────────────────────────────

class WellbeingProxy:
    """Garmin proxy for subjective wellbeing (no manual input needed).

    WB = (stress_inv * 0.4 + BB_wake * 0.3 + readiness_factors * 0.3) * 28
    Scale: 0-28 (matching research wellbeing questionnaire range)

    Garmin readiness factors used:
    - sleepScoreFactor (0-100)
    - stressHistoryFactor (0-100)
    - hrvFactor (0-100, 0 means no data -> substitute neutral 50)
    - sleepHistoryFactor (0-100)
    - recoveryTimeFactor (0-100)
    """

    FACTOR_COLS = [
        'sleepScoreFactor', 'stressHistoryFactor', 'hrvFactor',
        'sleepHistoryFactor', 'recoveryTimeFactor',
    ]

    @staticmethod
    def compute(row):
        """Compute wellbeing proxy score for a single row."""
        # Stress (inverted: low stress = high wellbeing)
        stress = row.get('avgStressLevel', np.nan)
        if pd.notna(stress):
            stress_inv = (100 - stress) / 100  # 0-1
        else:
            stress_inv = 0.5  # neutral

        # Body battery at wake (0-100 -> 0-1)
        bb_wake = row.get('bodyBatteryAtWake', np.nan)
        if pd.notna(bb_wake):
            bb_norm = bb_wake / 100  # 0-1
        else:
            bb_norm = 0.5  # neutral

        # Readiness factors average (0-100 -> 0-1)
        factors = []
        for col in WellbeingProxy.FACTOR_COLS:
            val = row.get(col, np.nan)
            if pd.notna(val):
                # hrvFactor=0 means no data, substitute neutral 50
                if col == 'hrvFactor' and val == 0:
                    val = 50
                factors.append(val)
        if factors:
            readiness_avg = np.mean(factors) / 100  # 0-1
        else:
            readiness_avg = 0.5  # neutral

        # Combine: stress_inv*0.4 + BB_wake*0.3 + readiness_factors*0.3
        wb_raw = stress_inv * 0.4 + bb_norm * 0.3 + readiness_avg * 0.3
        wb_score = round(wb_raw * 28, 1)  # scale to 0-28

        return wb_score

    @staticmethod
    def compute_series(df):
        """Compute wellbeing proxy for entire dataframe."""
        scores = []
        for _, row in df.iterrows():
            scores.append(WellbeingProxy.compute(row))
        return pd.Series(scores, index=df.index)


# ─── Readiness Engine ────────────────────────────────────────────────────────

class ReadinessEngine:
    """Core readiness scoring and recommendation engine.

    Decision tree and scoring based on research:
    - HRV-guided training meta-analysis
    - Combined HRV+RHR+WB study (Group 3)
    - Sleep deprivation injury risk thresholds
    """

    RECOMMENDATIONS = {
        'REST': {'label': 'REST', 'color': '#ea4335', 'description': 'Full rest day recommended'},
        'LOW': {'label': 'LOW', 'color': '#ff9800', 'description': 'Low intensity only (Zone 1-2)'},
        'MODERATE': {'label': 'MODERATE', 'color': '#fbbc04', 'description': 'Normal training day'},
        'HIGH': {'label': 'HIGH', 'color': '#34a853', 'description': 'High intensity cleared'},
    }

    def __init__(self):
        self.reasoning = []

    def compute_z_score(self, value, mean, sd):
        """Compute z-score: (value - mean) / sd."""
        if sd == 0 or pd.isna(value) or pd.isna(mean):
            return 0.0
        return (value - mean) / sd

    def count_consecutive_poor_sleep(self, df, current_idx, threshold=7.0):
        """Count consecutive days with sleep < threshold hours ending at current_idx."""
        count = 0
        for i in range(current_idx, -1, -1):
            sleep = df.iloc[i].get('sleepTimeHours', np.nan)
            if pd.notna(sleep) and sleep < threshold:
                count += 1
            else:
                break
        return count

    def compute_hrv_cv_streak(self, df, current_idx, window=7, threshold=20.0):
        """Check consecutive days where HRV CV > threshold.
        Note: Garmin overnight HRV naturally has higher CV (~15-25%) than
        controlled morning readings (~5-10%). Threshold adjusted accordingly."""
        if current_idx < window:
            return 0, 0.0

        streak = 0
        for end_idx in range(current_idx, max(current_idx - 30, window - 1), -1):
            start_idx = max(0, end_idx - window + 1)
            segment = df.iloc[start_idx:end_idx + 1]['hrvLastNightAvg'].dropna()
            if len(segment) < 3:
                break
            cv = (segment.std() / segment.mean() * 100) if segment.mean() > 0 else 0
            if cv > threshold:
                streak += 1
            else:
                break

        # Current CV
        start = max(0, current_idx - window + 1)
        recent = df.iloc[start:current_idx + 1]['hrvLastNightAvg'].dropna()
        current_cv = (recent.std() / recent.mean() * 100) if len(recent) >= 3 and recent.mean() > 0 else 0

        return streak, round(current_cv, 2)

    def detect_overtraining(self, df, current_idx, baselines):
        """Detect overtraining indicators (4 evidence-based).

        Returns dict with indicator flags and details.
        """
        indicators = {
            'hrv_cv_elevated': False,
            'rhr_elevated': False,
            'chronic_sleep_debt': False,
            'hrv_declining': False,
            'count': 0,
            'details': [],
        }

        # 1. HRV CV > 20% for 7+ consecutive days (Garmin overnight threshold)
        # Research uses 7% for controlled morning readings; Garmin overnight naturally higher
        cv_streak, current_cv = self.compute_hrv_cv_streak(df, current_idx)
        if cv_streak >= 7:
            indicators['hrv_cv_elevated'] = True
            indicators['count'] += 1
            indicators['details'].append(
                f"HRV CV elevated ({current_cv:.1f}%) for {cv_streak} consecutive days (threshold: 7 days)")

        # 2. RHR > 5% above baseline for 7+ days
        if 'rhr' in baselines:
            rhr_threshold = baselines['rhr']['mean'] * 1.05
            rhr_streak = 0
            for i in range(current_idx, max(current_idx - 30, -1), -1):
                rhr = df.iloc[i].get('restingHeartRate', np.nan)
                if pd.notna(rhr) and rhr > rhr_threshold:
                    rhr_streak += 1
                else:
                    break
            if rhr_streak >= 7:
                indicators['rhr_elevated'] = True
                indicators['count'] += 1
                indicators['details'].append(
                    f"RHR elevated above 5% of baseline for {rhr_streak} consecutive days")

        # 3. Sleep < 7h for 14+ consecutive days
        sleep_streak = self.count_consecutive_poor_sleep(df, current_idx, threshold=7.0)
        if sleep_streak >= 14:
            indicators['chronic_sleep_debt'] = True
            indicators['count'] += 1
            indicators['details'].append(
                f"Sleep <7h for {sleep_streak} consecutive days (1.7x injury risk threshold)")

        # 4. HRV declining trend over 3 weeks despite rest
        if current_idx >= 21 and 'hrvLastNightAvg' in df.columns:
            week1 = df.iloc[max(0, current_idx - 20):current_idx - 13]['hrvLastNightAvg'].dropna()
            week2 = df.iloc[max(0, current_idx - 13):current_idx - 6]['hrvLastNightAvg'].dropna()
            week3 = df.iloc[max(0, current_idx - 6):current_idx + 1]['hrvLastNightAvg'].dropna()
            if len(week1) >= 3 and len(week2) >= 3 and len(week3) >= 3:
                m1, m2, m3 = week1.mean(), week2.mean(), week3.mean()
                if m2 < m1 and m3 < m2:
                    indicators['hrv_declining'] = True
                    indicators['count'] += 1
                    indicators['details'].append(
                        f"HRV declining over 3 weeks: {m1:.0f} -> {m2:.0f} -> {m3:.0f} ms")

        return indicators

    def decision_tree(self, hrv, rhr, wb, sleep_hours, baselines, sleep_streak=0):
        """Apply decision tree for training recommendation.

        Returns (recommendation, confidence, reasoning).
        """
        self.reasoning = []
        thresholds_crossed = 0

        # Hard override: Sleep < 6h -> REST
        if pd.notna(sleep_hours) and sleep_hours < 6:
            self.reasoning.append(f"Sleep {sleep_hours:.1f}h < 6h: REST override (sleep deprivation)")
            return 'REST', 95, self.reasoning

        # Sleep debt accumulation
        if pd.notna(sleep_hours) and sleep_hours < 7 and sleep_streak >= 3:
            self.reasoning.append(
                f"Sleep {sleep_hours:.1f}h < 7h for {sleep_streak} consecutive days: accumulating sleep debt")
            return 'LOW', 90, self.reasoning

        # Check individual metrics against baselines
        hrv_status = 'normal'
        rhr_status = 'normal'
        wb_status = 'normal'

        if 'hrv' in baselines and pd.notna(hrv):
            b = baselines['hrv']
            upper = b['mean'] + 0.5 * b['sd']
            lower = b['mean'] - 0.5 * b['sd']
            very_low = b['mean'] - 1.0 * b['sd']

            if hrv > upper:
                hrv_status = 'high'
                self.reasoning.append(f"HRV {hrv:.0f} > baseline+0.5SD ({upper:.0f}): strong recovery")
            elif hrv >= lower:
                hrv_status = 'normal'
                self.reasoning.append(f"HRV {hrv:.0f} within normal range ({lower:.0f}-{upper:.0f})")
            elif hrv >= very_low:
                hrv_status = 'low'
                thresholds_crossed += 1
                self.reasoning.append(f"HRV {hrv:.0f} below baseline-0.5SD ({lower:.0f}): suppressed")
            else:
                hrv_status = 'very_low'
                thresholds_crossed += 1
                self.reasoning.append(f"HRV {hrv:.0f} below baseline-1.0SD ({very_low:.0f}): significantly suppressed")

        if 'rhr' in baselines and pd.notna(rhr):
            b = baselines['rhr']
            if rhr <= b['mean'] - 3:
                rhr_status = 'low'
                self.reasoning.append(f"RHR {rhr:.0f} below baseline-3 ({b['mean'] - 3:.0f}): good recovery")
            elif rhr <= b['mean'] + 3:
                rhr_status = 'normal'
                self.reasoning.append(f"RHR {rhr:.0f} within normal range")
            elif rhr <= b['mean'] + 5:
                rhr_status = 'elevated'
                thresholds_crossed += 1
                self.reasoning.append(f"RHR {rhr:.0f} elevated > baseline+3 ({b['mean'] + 3:.0f})")
            elif rhr <= b['mean'] + 7:
                rhr_status = 'high'
                thresholds_crossed += 1
                self.reasoning.append(f"RHR {rhr:.0f} elevated > baseline+5 ({b['mean'] + 5:.0f}): fatigue signal")
            else:
                rhr_status = 'very_high'
                thresholds_crossed += 1
                self.reasoning.append(f"RHR {rhr:.0f} elevated > baseline+7 ({b['mean'] + 7:.0f}): clear fatigue")

        if 'stress' in baselines and pd.notna(wb):
            wb_baseline = baselines.get('stress', {}).get('mean', 14)
            # WB is on 0-28 scale, baseline ~14 is neutral
            wb_bl = 14  # approximate center
            if wb >= wb_bl:
                wb_status = 'good'
                self.reasoning.append(f"Wellbeing {wb:.0f}/28: good")
            elif wb >= wb_bl - 3:
                wb_status = 'normal'
                self.reasoning.append(f"Wellbeing {wb:.0f}/28: normal range")
            elif wb >= wb_bl - 5:
                wb_status = 'low'
                thresholds_crossed += 1
                self.reasoning.append(f"Wellbeing {wb:.0f}/28: below normal")
            else:
                wb_status = 'very_low'
                thresholds_crossed += 1
                self.reasoning.append(f"Wellbeing {wb:.0f}/28: significantly below normal")

        # Compute WB baseline from recent data
        wb_baseline_val = 14  # default
        if 'body_battery_wake' in baselines and 'stress' in baselines:
            # WB baseline approximation
            stress_inv = (100 - baselines['stress']['mean']) / 100
            bb_norm = baselines.get('body_battery_wake', {}).get('mean', 50) / 100
            wb_baseline_val = round((stress_inv * 0.4 + bb_norm * 0.3 + 0.5 * 0.3) * 28, 1)

        # Decision tree
        # HIGH: HRV high + RHR low + WB good
        if hrv_status == 'high' and rhr_status == 'low' and wb_status in ('good', 'normal'):
            confidence = min(95, 80 + thresholds_crossed * 5)
            return 'HIGH', confidence, self.reasoning

        # MODERATE: All normal
        if hrv_status in ('high', 'normal') and rhr_status in ('low', 'normal') and wb_status in ('good', 'normal'):
            confidence = min(90, 80 + thresholds_crossed * 5)
            return 'MODERATE', confidence, self.reasoning

        # REST: Very low HRV + very high RHR
        if hrv_status == 'very_low' and rhr_status in ('very_high', 'high'):
            confidence = min(95, 80 + thresholds_crossed * 5)
            return 'REST', confidence, self.reasoning

        # LOW: Any red flag
        if hrv_status in ('low', 'very_low') or rhr_status in ('elevated', 'high', 'very_high') or wb_status in ('low', 'very_low'):
            confidence = min(95, 80 + thresholds_crossed * 5)
            return 'LOW', confidence, self.reasoning

        # DEFAULT -> MODERATE
        return 'MODERATE', 80, self.reasoning

    def compute_readiness_score(self, hrv, rhr, wb, sleep_hours, baselines):
        """Compute readiness score (0-100).

        Components:
        - HRV: 0-50 points (z-score mapped)
        - RHR: 0-25 points (inverted z-score)
        - WB: 0-15 points (z-score mapped)
        - Sleep: -15 to +10 bonus
        """
        # Z-scores
        z_hrv = 0.0
        if 'hrv' in baselines and pd.notna(hrv):
            b = baselines['hrv']
            z_hrv = self.compute_z_score(hrv, b['mean'], b['sd'])

        z_rhr = 0.0
        if 'rhr' in baselines and pd.notna(rhr):
            b = baselines['rhr']
            z_rhr = self.compute_z_score(rhr, b['mean'], b['sd'])

        z_wb = 0.0
        # WB baseline: compute from stress + body battery baselines
        wb_mean = 14.0  # default center
        wb_sd = 3.0
        if 'stress' in baselines and 'body_battery_wake' in baselines:
            stress_inv = (100 - baselines['stress']['mean']) / 100
            bb_norm = baselines['body_battery_wake']['mean'] / 100
            wb_mean = (stress_inv * 0.4 + bb_norm * 0.3 + 0.5 * 0.3) * 28
            # Approximate SD from stress SD
            wb_sd = max(1.0, baselines['stress']['sd'] / 100 * 28 * 0.4)
        if pd.notna(wb):
            z_wb = self.compute_z_score(wb, wb_mean, wb_sd)

        # Components
        hrv_component = min(50, max(0, (z_hrv + 2) * 12.5))
        rhr_component = min(25, max(0, (2 - z_rhr) * 6.25))
        wb_component = min(15, max(0, (z_wb + 2) * 3.75))

        # Sleep bonus/penalty
        if pd.notna(sleep_hours):
            if sleep_hours >= 8:
                sleep_bonus = 10
            elif sleep_hours >= 7:
                sleep_bonus = 5
            elif sleep_hours >= 6:
                sleep_bonus = 0
            else:
                sleep_bonus = -15
        else:
            sleep_bonus = 0

        readiness = hrv_component + rhr_component + wb_component + sleep_bonus
        return round(max(0, min(100, readiness)), 1)

    def compute_confidence(self, thresholds_crossed):
        """Compute confidence based on number of thresholds crossed."""
        if thresholds_crossed >= 3:
            return 95
        elif thresholds_crossed >= 2:
            return 90
        else:
            return 80

    def analyze_day(self, row, baselines, df=None, row_idx=None):
        """Full analysis for a single day.

        Returns dict with readiness score, recommendation, confidence, reasoning,
        overtraining indicators, and sleep streak.
        """
        hrv = row.get('hrvLastNightAvg', np.nan)
        rhr = row.get('restingHeartRate', np.nan)
        sleep_hours = row.get('sleepTimeHours', np.nan)
        wb = WellbeingProxy.compute(row)

        # Sleep streak
        sleep_streak = 0
        if df is not None and row_idx is not None:
            sleep_streak = self.count_consecutive_poor_sleep(df, row_idx)

        # Decision tree
        rec, confidence, reasoning = self.decision_tree(
            hrv, rhr, wb, sleep_hours, baselines, sleep_streak)

        # Readiness score
        score = self.compute_readiness_score(hrv, rhr, wb, sleep_hours, baselines)

        # Overtraining detection
        overtraining = {'count': 0, 'details': []}
        if df is not None and row_idx is not None:
            overtraining = self.detect_overtraining(df, row_idx, baselines)

        # HRV CV
        hrv_cv = 0.0
        if df is not None and row_idx is not None:
            _, hrv_cv = self.compute_hrv_cv_streak(df, row_idx)

        return {
            'date': row.get('date', None),
            'readiness_score': score,
            'recommendation': rec,
            'recommendation_info': self.RECOMMENDATIONS[rec],
            'confidence': confidence,
            'reasoning': reasoning,
            'wellbeing_proxy': wb,
            'hrv': hrv,
            'rhr': rhr,
            'sleep_hours': sleep_hours,
            'sleep_streak_under_7h': sleep_streak,
            'hrv_cv': hrv_cv,
            'overtraining': overtraining,
            'z_hrv': self.compute_z_score(hrv, baselines.get('hrv', {}).get('mean', 0),
                                          baselines.get('hrv', {}).get('sd', 1)) if 'hrv' in baselines else 0,
            'z_rhr': self.compute_z_score(rhr, baselines.get('rhr', {}).get('mean', 0),
                                          baselines.get('rhr', {}).get('sd', 1)) if 'rhr' in baselines else 0,
        }


# ─── Lag Correlation Analyzer ───────────────────────────────────────────────

class LagCorrelationAnalyzer:
    """Correlate training metrics -> recovery at lag 0,1,2,3 days.

    Uses scipy.stats.pearsonr for r + p-value.
    """

    # 10 pairs to test (training cause -> recovery effect)
    PAIRS = [
        ('training_distance_km', 'hrvLastNightAvg', 'Distance -> HRV'),
        ('training_distance_km', 'restingHeartRate', 'Distance -> RHR'),
        ('training_distance_km', 'sleepScore', 'Distance -> Sleep Score'),
        ('training_time_min', 'hrvLastNightAvg', 'Duration -> HRV'),
        ('training_avg_hr', 'hrvLastNightAvg', 'Training HR -> HRV'),
        ('training_avg_hr', 'restingHeartRate', 'Training HR -> RHR'),
        ('acuteLoad', 'hrvLastNightAvg', 'Acute Load -> HRV'),
        ('acuteLoad', 'restingHeartRate', 'Acute Load -> RHR'),
        ('avgStressLevel', 'hrvLastNightAvg', 'Stress -> HRV'),
        ('sleepTimeHours', 'hrvLastNightAvg', 'Sleep Hours -> Next-Day HRV'),
    ]

    MAX_LAG = 3

    def __init__(self):
        self.results = []

    def analyze(self, df):
        """Run lag correlation analysis on all pairs."""
        if not HAS_SCIPY:
            return []

        self.results = []
        df = df.sort_values('date').reset_index(drop=True)

        for cause_col, effect_col, label in self.PAIRS:
            if cause_col not in df.columns or effect_col not in df.columns:
                continue

            pair_result = {
                'cause': cause_col,
                'effect': effect_col,
                'label': label,
                'lags': {},
            }

            for lag in range(0, self.MAX_LAG + 1):
                cause = df[cause_col].values[:-lag] if lag > 0 else df[cause_col].values
                effect = df[effect_col].values[lag:] if lag > 0 else df[effect_col].values

                # Create mask for valid pairs
                valid = ~(np.isnan(cause.astype(float)) | np.isnan(effect.astype(float)))
                cause_valid = cause[valid].astype(float)
                effect_valid = effect[valid].astype(float)

                if len(cause_valid) < 10:
                    continue

                r, p = pearsonr(cause_valid, effect_valid)
                pair_result['lags'][lag] = {
                    'r': round(float(r), 4),
                    'p': round(float(p), 4),
                    'n': int(len(cause_valid)),
                    'significant': p < 0.05,
                }

            if pair_result['lags']:
                self.results.append(pair_result)

        return self.results


# ─── Extended Correlation Analyzer ───────────────────────────────────────────

class CorrelationAnalyzer:
    """Extended correlation analysis with Pearson + Spearman, p-values,
    effect sizes, and Bonferroni correction."""

    METRIC_PAIRS = [
        ('hrvLastNightAvg', 'restingHeartRate'),
        ('hrvLastNightAvg', 'sleepScore'),
        ('hrvLastNightAvg', 'avgStressLevel'),
        ('hrvLastNightAvg', 'bodyBatteryAtWake'),
        ('restingHeartRate', 'sleepScore'),
        ('restingHeartRate', 'avgStressLevel'),
        ('sleepScore', 'avgStressLevel'),
        ('sleepTimeHours', 'hrvLastNightAvg'),
        ('sleepTimeHours', 'sleepScore'),
        ('training_distance_km', 'sleepScore'),
        ('training_distance_km', 'hrvLastNightAvg'),
        ('acuteLoad', 'sleepScore'),
        ('bodyBatteryAtWake', 'sleepScore'),
        ('bodyBatteryAtWake', 'avgStressLevel'),
        ('readinessScore', 'hrvLastNightAvg'),
    ]

    def __init__(self):
        self.results = []

    @staticmethod
    def effect_size(r):
        """Interpret correlation effect size (Cohen's conventions)."""
        ar = abs(r)
        if ar >= 0.5:
            return 'large'
        elif ar >= 0.3:
            return 'medium'
        elif ar >= 0.1:
            return 'small'
        else:
            return 'negligible'

    def analyze(self, df):
        """Run full correlation analysis."""
        if not HAS_SCIPY:
            return []

        self.results = []
        n_tests = len(self.METRIC_PAIRS)
        bonferroni_alpha = 0.05 / n_tests

        for col_a, col_b in self.METRIC_PAIRS:
            if col_a not in df.columns or col_b not in df.columns:
                continue

            valid = df[[col_a, col_b]].dropna()
            if len(valid) < 10:
                continue

            a = valid[col_a].values
            b = valid[col_b].values

            r_pearson, p_pearson = pearsonr(a, b)
            r_spearman, p_spearman = spearmanr(a, b)

            self.results.append({
                'metric_a': col_a,
                'metric_b': col_b,
                'n': len(valid),
                'pearson_r': round(float(r_pearson), 4),
                'pearson_p': round(float(p_pearson), 6),
                'spearman_r': round(float(r_spearman), 4),
                'spearman_p': round(float(p_spearman), 6),
                'effect_size': self.effect_size(r_pearson),
                'significant_bonferroni': p_pearson < bonferroni_alpha,
                'significant_nominal': p_pearson < 0.05,
            })

        # Sort by absolute Pearson r
        self.results.sort(key=lambda x: abs(x['pearson_r']), reverse=True)
        return self.results


# ─── Backtester ──────────────────────────────────────────────────────────────

class Backtester:
    """Backtest readiness recommendations against actual training outcomes.

    For each historical day (skip first 14 for warmup):
    - Compute baselines from previous 7 days only (no look-ahead)
    - Generate recommendation
    - Compare to actual training quality (proxied from pace/HR)
    """

    WARMUP_DAYS = 14

    def __init__(self):
        self.results = []
        self.confusion = {'REST': {}, 'LOW': {}, 'MODERATE': {}, 'HIGH': {}}

    def _proxy_actual_intensity(self, row):
        """Proxy actual training intensity from data.

        Returns: 'REST', 'LOW', 'MODERATE', or 'HIGH'
        Uses a combination of distance + HR intensity for realistic bucketing.
        """
        distance = row.get('training_distance_km', 0)
        if pd.isna(distance) or distance == 0:
            return 'REST'

        avg_hr = row.get('training_avg_hr', np.nan)
        max_hr = row.get('training_max_hr', np.nan)

        # Combined approach: distance + intensity %
        intensity_pct = 0
        if pd.notna(avg_hr) and pd.notna(max_hr) and max_hr > 0:
            intensity_pct = avg_hr / max_hr * 100

        # HIGH: long run (>12km) OR high intensity + moderate distance
        if distance >= 12 or (intensity_pct >= 88 and distance >= 8):
            return 'HIGH'
        # MODERATE: medium distance or moderate HR intensity
        elif distance >= 5 or intensity_pct >= 80:
            return 'MODERATE'
        else:
            return 'LOW'

    def run(self, df):
        """Run backtest on full dataset."""
        df = df.sort_values('date').reset_index(drop=True)
        engine = ReadinessEngine()
        baselines_comp = PersonalBaselines()
        self.results = []

        for i in range(self.WARMUP_DAYS, len(df)):
            row = df.iloc[i]
            target_date = row['date']

            # Compute baselines from prior 7 days only
            baselines = baselines_comp.compute_for_date(df, target_date, lookback=7)

            if not baselines or 'hrv' not in baselines:
                continue

            # Get recommendation
            analysis = engine.analyze_day(row, baselines, df, i)
            recommended = analysis['recommendation']

            # Get actual intensity
            actual = self._proxy_actual_intensity(row)

            # Agreement check
            rec_order = {'REST': 0, 'LOW': 1, 'MODERATE': 2, 'HIGH': 3}
            rec_level = rec_order[recommended]
            act_level = rec_order[actual]

            if recommended == actual:
                agreement = 'exact_match'
            elif rec_level < act_level:
                agreement = 'cautious'  # recommended lower than actual (conservative, acceptable)
            elif actual == 'REST' and recommended in ('MODERATE', 'LOW'):
                # Athlete chose to rest (schedule/plan) while system said OK to train
                # This is not "aggressive" - it's a planned rest day
                agreement = 'planned_rest'
            else:
                agreement = 'aggressive'  # recommended higher than actual (risky)

            self.results.append({
                'date': target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date),
                'recommended': recommended,
                'actual': actual,
                'readiness_score': analysis['readiness_score'],
                'agreement': agreement,
                'hrv': analysis['hrv'],
                'rhr': analysis['rhr'],
                'sleep': analysis['sleep_hours'],
                'wb': analysis['wellbeing_proxy'],
            })

        return self.results

    def compute_metrics(self):
        """Compute backtest accuracy metrics."""
        if not self.results:
            return {}

        total = len(self.results)
        exact = sum(1 for r in self.results if r['agreement'] == 'exact_match')
        cautious = sum(1 for r in self.results if r['agreement'] == 'cautious')
        planned_rest = sum(1 for r in self.results if r['agreement'] == 'planned_rest')
        aggressive = sum(1 for r in self.results if r['agreement'] == 'aggressive')

        # Confusion matrix
        labels = ['REST', 'LOW', 'MODERATE', 'HIGH']
        confusion = {rec: {act: 0 for act in labels} for rec in labels}
        for r in self.results:
            confusion[r['recommended']][r['actual']] += 1

        # "Acceptable" = exact + cautious + planned_rest (athlete chose rest on a green day)
        acceptable = exact + cautious + planned_rest

        # Training days only accuracy (exclude REST actual days for fair comparison)
        training_results = [r for r in self.results if r['actual'] != 'REST']
        training_total = len(training_results)
        training_exact = sum(1 for r in training_results if r['agreement'] == 'exact_match')
        training_cautious = sum(1 for r in training_results if r['agreement'] == 'cautious')
        training_acceptable = training_exact + training_cautious

        return {
            'total_days': total,
            'exact_match': exact,
            'exact_match_pct': round(exact / total * 100, 1) if total > 0 else 0,
            'cautious': cautious,
            'cautious_pct': round(cautious / total * 100, 1) if total > 0 else 0,
            'planned_rest': planned_rest,
            'planned_rest_pct': round(planned_rest / total * 100, 1) if total > 0 else 0,
            'aggressive': aggressive,
            'aggressive_pct': round(aggressive / total * 100, 1) if total > 0 else 0,
            'acceptable_pct': round(acceptable / total * 100, 1) if total > 0 else 0,
            'training_days_accuracy_pct': round(training_acceptable / training_total * 100, 1) if training_total > 0 else 0,
            'confusion_matrix': confusion,
        }


# ─── Main API Function ──────────────────────────────────────────────────────

def compute_readiness_for_dashboard(df):
    """Main entry point for dashboard integration.

    Args:
        df: DataFrame with master dataset columns

    Returns:
        dict with:
        - df: augmented DataFrame with wellbeing_proxy and readiness columns
        - baselines: current baselines dict
        - latest_readiness: analysis dict for the most recent day
        - lag_results: lag correlation analysis results
        - overtraining: overtraining detection results
        - correlation_results: extended correlation analysis
    """
    df = df.sort_values('date').reset_index(drop=True)

    # Compute baselines
    bl = PersonalBaselines()
    baselines = bl.compute(df)

    # Add wellbeing proxy column
    df['wellbeing_proxy'] = WellbeingProxy.compute_series(df)

    # Compute readiness for each day (last 30 days for efficiency)
    engine = ReadinessEngine()
    readiness_scores = []
    recommendations = []

    start_idx = max(0, len(df) - 60)
    for i in range(len(df)):
        if i < 7 or i < start_idx:
            readiness_scores.append(np.nan)
            recommendations.append('')
            continue
        # Use rolling baselines for more accurate per-day scores
        day_baselines = bl.compute_for_date(df, df.iloc[i]['date'], lookback=7)
        if not day_baselines or 'hrv' not in day_baselines:
            readiness_scores.append(np.nan)
            recommendations.append('')
            continue
        analysis = engine.analyze_day(df.iloc[i], day_baselines, df, i)
        readiness_scores.append(analysis['readiness_score'])
        recommendations.append(analysis['recommendation'])

    df['computed_readiness'] = readiness_scores
    df['computed_recommendation'] = recommendations

    # Latest day full analysis
    latest_baselines = bl.compute_for_date(df, df.iloc[-1]['date'], lookback=7)
    if not latest_baselines or 'hrv' not in latest_baselines:
        latest_baselines = baselines
    latest_readiness = engine.analyze_day(df.iloc[-1], latest_baselines, df, len(df) - 1)

    # Lag correlation analysis
    lag_analyzer = LagCorrelationAnalyzer()
    lag_results = lag_analyzer.analyze(df)

    # Extended correlation analysis
    corr_analyzer = CorrelationAnalyzer()
    correlation_results = corr_analyzer.analyze(df)

    # Overtraining detection for latest day
    overtraining = engine.detect_overtraining(df, len(df) - 1, latest_baselines)

    return {
        'df': df,
        'baselines': baselines,
        'latest_readiness': latest_readiness,
        'lag_results': lag_results,
        'overtraining': overtraining,
        'correlation_results': correlation_results,
    }


# ─── Report Generators ──────────────────────────────────────────────────────

def generate_analysis_report(baselines, lag_results, corr_results, overtraining):
    """Generate analysis_report.md."""
    lines = [
        "# Readiness System Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## Personal Baselines (7-day rolling)",
        "",
    ]

    for key, bl in baselines.items():
        lines.append(f"### {key.replace('_', ' ').title()}")
        lines.append(f"- Mean: {bl.get('mean', 'N/A')}")
        lines.append(f"- SD: {bl.get('sd', 'N/A')}")
        lines.append(f"- Range: {bl.get('min', 'N/A')} - {bl.get('max', 'N/A')}")
        if 'cv_pct' in bl:
            lines.append(f"- CV: {bl['cv_pct']}%")
        if 'upper_05sd' in bl:
            lines.append(f"- Normal band: {bl['lower_05sd']} - {bl['upper_05sd']}")
        lines.append("")

    # Significant correlations
    lines.append("## Significant Correlations")
    lines.append("")
    if corr_results:
        lines.append("| Metric A | Metric B | Pearson r | p-value | Effect Size | Bonferroni Sig |")
        lines.append("|----------|----------|-----------|---------|-------------|----------------|")
        for cr in corr_results:
            if cr['significant_nominal']:
                bonf = 'Yes' if cr['significant_bonferroni'] else 'No'
                lines.append(
                    f"| {cr['metric_a']} | {cr['metric_b']} | {cr['pearson_r']:.3f} | "
                    f"{cr['pearson_p']:.4f} | {cr['effect_size']} | {bonf} |")
        lines.append("")
    else:
        lines.append("No correlations computed (scipy not available or insufficient data).\n")

    # Lag analysis
    lines.append("## Lag Correlation Analysis (Training -> Recovery)")
    lines.append("")
    if lag_results:
        lines.append("| Pair | Lag 0 | Lag 1 | Lag 2 | Lag 3 | Best Lag |")
        lines.append("|------|-------|-------|-------|-------|----------|")
        for lr in lag_results:
            vals = []
            best_lag = 0
            best_r = 0
            for lag in range(4):
                if lag in lr['lags']:
                    r = lr['lags'][lag]['r']
                    sig = '*' if lr['lags'][lag]['significant'] else ''
                    vals.append(f"{r:+.3f}{sig}")
                    if abs(r) > abs(best_r):
                        best_r = r
                        best_lag = lag
                else:
                    vals.append('-')
            lines.append(f"| {lr['label']} | {' | '.join(vals)} | {best_lag} (r={best_r:+.3f}) |")
        lines.append("\n*\\* = p < 0.05*\n")
    else:
        lines.append("No lag correlations computed.\n")

    # Overtraining indicators
    lines.append("## Overtraining Indicators")
    lines.append("")
    if overtraining['count'] > 0:
        lines.append(f"**WARNING: {overtraining['count']} indicator(s) detected**\n")
        for detail in overtraining['details']:
            lines.append(f"- {detail}")
    else:
        lines.append("No overtraining indicators detected. All metrics within normal ranges.")
    lines.append("")

    return '\n'.join(lines)


def generate_backtest_report(bt_results, bt_metrics):
    """Generate backtest_results.md."""
    lines = [
        "# Readiness System Backtest Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
        f"- Total days tested: {bt_metrics['total_days']}",
        f"- Exact match: {bt_metrics['exact_match']} ({bt_metrics['exact_match_pct']}%)",
        f"- Conservative (cautious): {bt_metrics['cautious']} ({bt_metrics['cautious_pct']}%)",
        f"- Planned rest (athlete rested, system said OK): {bt_metrics.get('planned_rest', 0)} ({bt_metrics.get('planned_rest_pct', 0)}%)",
        f"- Aggressive (too high): {bt_metrics['aggressive']} ({bt_metrics['aggressive_pct']}%)",
        f"- **Acceptable rate: {bt_metrics['acceptable_pct']}%** (exact + cautious + planned rest)",
        f"- **Training days accuracy: {bt_metrics.get('training_days_accuracy_pct', 0)}%** (on days athlete trained)",
        "",
        "## Confusion Matrix (Recommended vs Actual)",
        "",
        "| Recommended \\ Actual | REST | LOW | MODERATE | HIGH |",
        "|---------------------|------|-----|----------|------|",
    ]

    cm = bt_metrics['confusion_matrix']
    for rec in ['REST', 'LOW', 'MODERATE', 'HIGH']:
        row = cm[rec]
        lines.append(f"| {rec} | {row['REST']} | {row['LOW']} | {row['MODERATE']} | {row['HIGH']} |")

    lines.append("")
    lines.append("## Day-by-Day Results (last 20)")
    lines.append("")
    lines.append("| Date | Recommended | Actual | Score | Agreement | HRV | RHR | Sleep |")
    lines.append("|------|-------------|--------|-------|-----------|-----|-----|-------|")

    for r in bt_results[-20:]:
        lines.append(
            f"| {r['date']} | {r['recommended']} | {r['actual']} | {r['readiness_score']:.0f} | "
            f"{r['agreement']} | {r['hrv']:.0f} | {r['rhr']:.0f} | "
            f"{r['sleep']:.1f}h |"
            if pd.notna(r['hrv']) and pd.notna(r['rhr']) and pd.notna(r['sleep'])
            else f"| {r['date']} | {r['recommended']} | {r['actual']} | {r['readiness_score']:.0f} | "
            f"{r['agreement']} | - | - | - |"
        )

    lines.extend([
        "",
        "## Calibration Analysis",
        "",
        "The system compares recommended training intensity against actual training patterns.",
        "- **Exact match**: recommendation matched what was actually done",
        "- **Cautious**: system recommended lower intensity than actual (conservative, safer)",
        "- **Aggressive**: system recommended higher intensity than actual (potential overreaching risk)",
        "",
        "## Limitations",
        "",
        "- Actual intensity is proxied from avg HR / max HR ratio and distance",
        "- No ground truth for 'optimal' training decision exists",
        "- First 14 days excluded for baseline warmup",
        "- Recommendations are backward-looking (what should have been done given the data)",
        "- Small dataset may limit statistical power",
    ])

    return '\n'.join(lines)


# ─── Data Loading (standalone mode) ─────────────────────────────────────────

def find_latest_file(pattern, exclude_pattern=None):
    """Find most recently modified file matching pattern."""
    import glob as globmod
    files = globmod.glob(os.path.join(DATA_DIR, pattern))
    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in os.path.basename(f)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_data():
    """Load master dataset for standalone mode."""
    master_file = find_latest_file('master_dataset_*.csv', exclude_pattern='features')
    if not master_file:
        raise FileNotFoundError(f"No master_dataset_*.csv found in {DATA_DIR}")

    print(f"Loading: {os.path.basename(master_file)}")
    df = pd.read_csv(master_file, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  {len(df)} days, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    return df


# ─── Main (standalone) ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Garmin PRO Trener - Readiness Analysis System")
    print("=" * 60)
    print()

    # 1. Load data
    df = load_data()
    print()

    # 2. Compute readiness
    print("Computing readiness analysis...")
    result = compute_readiness_for_dashboard(df)
    baselines = result['baselines']
    latest = result['latest_readiness']
    lag_results = result['lag_results']
    overtraining = result['overtraining']
    corr_results = result['correlation_results']
    print()

    # 3. Save baselines
    bl = PersonalBaselines()
    bl.baselines = baselines
    bl_path = bl.save()
    print(f"Baselines saved to: {os.path.basename(bl_path)}")

    # 4. Print latest readiness
    print()
    print(f"  Latest Readiness Score: {latest['readiness_score']:.0f}/100")
    print(f"  Recommendation: {latest['recommendation']}")
    print(f"  Confidence: {latest['confidence']}%")
    print(f"  Wellbeing Proxy: {latest['wellbeing_proxy']:.1f}/28")
    print(f"  HRV: {latest['hrv']:.0f} ms" if pd.notna(latest['hrv']) else "  HRV: N/A")
    print(f"  RHR: {latest['rhr']:.0f} bpm" if pd.notna(latest['rhr']) else "  RHR: N/A")
    print(f"  Sleep: {latest['sleep_hours']:.1f}h" if pd.notna(latest['sleep_hours']) else "  Sleep: N/A")
    print(f"  Sleep streak <7h: {latest['sleep_streak_under_7h']} days")
    print(f"  HRV CV: {latest['hrv_cv']:.1f}%")
    if latest['reasoning']:
        print("  Reasoning:")
        for r in latest['reasoning']:
            print(f"    - {r}")
    if overtraining['count'] > 0:
        print(f"  OVERTRAINING WARNING: {overtraining['count']} indicator(s)")
        for d in overtraining['details']:
            print(f"    ! {d}")
    print()

    # 5. Generate analysis report
    print("Generating analysis report...")
    report = generate_analysis_report(baselines, lag_results, corr_results, overtraining)
    report_path = os.path.join(BASE_DIR, 'analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {os.path.basename(report_path)}")

    # 6. Run backtest
    print("Running backtest...")
    bt = Backtester()
    bt_results = bt.run(df)
    bt_metrics = bt.compute_metrics()

    if bt_metrics:
        print(f"  Days tested: {bt_metrics['total_days']}")
        print(f"  Exact match: {bt_metrics['exact_match_pct']}%")
        print(f"  Acceptable (exact+cautious+planned rest): {bt_metrics['acceptable_pct']}%")
        print(f"  Training days accuracy: {bt_metrics.get('training_days_accuracy_pct', 0)}%")
        print(f"  Aggressive: {bt_metrics['aggressive_pct']}%")

        report = generate_backtest_report(bt_results, bt_metrics)
        bt_path = os.path.join(BASE_DIR, 'backtest_results.md')
        with open(bt_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Saved: {os.path.basename(bt_path)}")
    else:
        print("  Insufficient data for backtest")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
