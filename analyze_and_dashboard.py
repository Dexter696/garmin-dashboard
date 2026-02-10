#!/usr/bin/env python3
"""
Garmin PRO Trener - Analysis & Interactive Dashboard
Reads master dataset + activities, engineers features, runs correlation analysis,
and generates a standalone interactive HTML dashboard using Plotly.
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import date, datetime, timedelta

warnings.filterwarnings('ignore')

# Import training plan (optional - graceful fallback if not available)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from training_plan import (
        MILESTONES, DAILY_PLAN, PLAN_START, RACE_DATE, TARGET_PACE, TARGET_TIME,
        PHASES, STATUS_COLORS, STATUS_LABELS,
        get_current_week, get_week_start_end, get_current_phase, get_phase_info,
        get_days_to_race, evaluate_metric,
    )
    HAS_TRAINING_PLAN = True
except ImportError:
    HAS_TRAINING_PLAN = False

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'garmin_data')
OUTPUT_HTML = os.path.join(BASE_DIR, 'dashboard.html')
OUTPUT_FEATURES = os.path.join(DATA_DIR, 'master_dataset_features.csv')

COLORS = {
    'primary': '#1a73e8',
    'secondary': '#ea4335',
    'success': '#34a853',
    'warning': '#fbbc04',
    'info': '#4285f4',
    'purple': '#9c27b0',
    'teal': '#009688',
    'orange': '#ff9800',
    'deep_sleep': '#1a237e',
    'light_sleep': '#5c6bc0',
    'rem_sleep': '#7c4dff',
    'awake_sleep': '#ff5252',
    'bg_dark': '#1e1e2e',
    'bg_card': '#2a2a3e',
    'bg_surface': '#313145',
    'text_primary': '#e0e0e0',
    'text_secondary': '#a0a0b0',
    'grid': '#3a3a4e',
}

# ─── Data Loading ────────────────────────────────────────────────────────────

def find_latest_file(pattern, exclude_pattern=None):
    """Find the most recently modified file matching a glob pattern."""
    files = glob.glob(os.path.join(DATA_DIR, pattern))
    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in os.path.basename(f)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_data():
    """Load master dataset and activities CSV."""
    master_file = find_latest_file('master_dataset_*.csv', exclude_pattern='features')
    activities_file = find_latest_file('activities_*.csv')

    if not master_file:
        raise FileNotFoundError(f"No master_dataset_*.csv found in {DATA_DIR}")
    if not activities_file:
        raise FileNotFoundError(f"No activities_*.csv found in {DATA_DIR}")

    print(f"Loading master dataset: {os.path.basename(master_file)}")
    print(f"Loading activities: {os.path.basename(activities_file)}")

    df = pd.read_csv(master_file, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    act = pd.read_csv(activities_file, parse_dates=['datum'])
    act = act.sort_values('datum').reset_index(drop=True)

    print(f"Master dataset: {len(df)} days, {len(df.columns)} columns")
    print(f"Activities: {len(act)} activities, {len(act.columns)} columns")

    return df, act

# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df, act):
    """Add derived features: rolling averages, recovery score, trends, flags."""

    # Rolling averages (7-day)
    rolling_cols = ['avgStressLevel', 'restingHeartRate', 'sleepScore',
                    'hrvLastNightAvg', 'bodyBatteryHighest', 'bodyBatteryLowest',
                    'training_distance_km', 'training_time_min']
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_7d_avg'] = df[col].rolling(7, min_periods=2).mean()

    # Rolling averages (14-day)
    for col in ['sleepScore', 'avgStressLevel', 'hrvLastNightAvg']:
        if col in df.columns:
            df[f'{col}_14d_avg'] = df[col].rolling(14, min_periods=3).mean()

    # Composite recovery score (0-100 scale)
    # Based on: sleep score, HRV, low stress, high body battery
    sleep_norm = df['sleepScore'].fillna(50) / 100
    hrv_norm = df['hrvLastNightAvg'].fillna(df['hrvLastNightAvg'].median()) / df['hrvLastNightAvg'].max() if df['hrvLastNightAvg'].max() > 0 else 0
    stress_norm = 1 - (df['avgStressLevel'].fillna(50) / 100)
    bb_norm = df['bodyBatteryHighest'].fillna(50) / 100
    df['recovery_score'] = ((sleep_norm * 0.3 + hrv_norm * 0.3 + stress_norm * 0.2 + bb_norm * 0.2) * 100).round(1)

    # Training load classification
    df['is_training_day'] = df['training_distance_km'].notna() & (df['training_distance_km'] > 0)
    df['is_rest_day'] = ~df['is_training_day']

    # Training intensity (based on avg HR relative to max HR)
    df['training_intensity'] = np.where(
        df['training_avg_hr'].notna() & df['training_max_hr'].notna() & (df['training_max_hr'] > 0),
        (df['training_avg_hr'] / df['training_max_hr'] * 100).round(1),
        np.nan
    )

    # Body battery delta
    df['body_battery_delta'] = df['bodyBatteryCharged'] - df['bodyBatteryDrained']

    # Day of week
    df['day_of_week'] = df['date'].dt.day_name()
    df['day_num'] = df['date'].dt.dayofweek

    # Sleep efficiency (time asleep vs time in bed)
    df['sleep_efficiency'] = np.where(
        df['sleepTimeSeconds'].notna() & df['awakeSleepSeconds'].notna() & (df['sleepTimeSeconds'] > 0),
        ((df['sleepTimeSeconds'] - df['awakeSleepSeconds']) / df['sleepTimeSeconds'] * 100).round(1),
        np.nan
    )

    # Deep sleep percentage
    df['deep_sleep_pct'] = np.where(
        df['sleepTimeSeconds'].notna() & df['deepSleepSeconds'].notna() & (df['sleepTimeSeconds'] > 0),
        (df['deepSleepSeconds'] / df['sleepTimeSeconds'] * 100).round(1),
        np.nan
    )

    # Next-day performance flag (did training happen the next day?)
    df['trained_next_day'] = df['is_training_day'].shift(-1)

    # Stress trend (change from previous day)
    df['stress_change'] = df['avgStressLevel'].diff()

    # HRV trend
    df['hrv_change'] = df['hrvLastNightAvg'].diff()

    # Cumulative training distance
    df['cumulative_distance_km'] = df['training_distance_km'].fillna(0).cumsum()

    # Cumulative training time (hours)
    df['cumulative_training_hours'] = (df['training_time_min'].fillna(0).cumsum() / 60).round(2)

    # Acute Training Load rolling 7-day sum
    df['acute_load_7d'] = df['training_distance_km'].fillna(0).rolling(7, min_periods=1).sum()

    # Chronic Training Load rolling 28-day average
    df['chronic_load_28d'] = df['training_distance_km'].fillna(0).rolling(28, min_periods=7).mean()

    # Training Stress Balance
    df['training_stress_balance'] = np.where(
        df['chronic_load_28d'] > 0,
        (df['acute_load_7d'] / 7 - df['chronic_load_28d']).round(2),
        np.nan
    )

    # Add running-specific metrics from activities
    running = act[act['typ'] == 'running'].copy()
    running_by_date = running.groupby('datum').agg({
        'tempo_min_km': 'mean',
        'avg_cadence': 'mean',
        'avg_stride_length': 'mean',
        'avg_ground_contact_time': 'mean',
        'training_load': 'sum',
        'elevation_gain': 'sum',
    }).reset_index()
    running_by_date.columns = ['date', 'running_pace', 'running_cadence', 'running_stride',
                                'running_gct', 'activity_training_load', 'running_elevation_gain']

    df = df.merge(running_by_date, on='date', how='left')

    return df

# ─── Correlation Analysis ────────────────────────────────────────────────────

def run_correlation_analysis(df):
    """Compute correlation matrix for key metrics."""
    key_cols = [
        'training_distance_km', 'training_time_min', 'training_avg_hr',
        'avgStressLevel', 'restingHeartRate', 'sleepScore', 'sleepTimeHours',
        'deep_sleep_pct', 'hrvLastNightAvg', 'bodyBatteryHighest', 'bodyBatteryLowest',
        'bodyBatteryAtWake', 'recovery_score', 'readinessScore', 'vo2MaxPrecise',
        'enduranceScore', 'acuteLoad', 'running_pace', 'steps',
    ]
    available = [c for c in key_cols if c in df.columns]
    corr_df = df[available].corr()
    return corr_df, available

def get_top_correlations(corr_df, n=15):
    """Extract top n strongest correlations (absolute value), excluding self-correlations."""
    pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_df.iloc[i, j]
            if pd.notna(val):
                pairs.append((cols[i], cols[j], val))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:n]

# ─── Training Insights ───────────────────────────────────────────────────────

def generate_insights(df, act):
    """Generate text-based training insights."""
    insights = []

    # Best / worst training days
    training_days = df[df['is_training_day']].copy()
    if len(training_days) > 0:
        best = training_days.loc[training_days['training_distance_km'].idxmax()]
        insights.append(f"<b>Longest training</b>: {best['training_distance_km']:.1f} km on {best['date'].strftime('%Y-%m-%d')} ({best['training_time_min']:.0f} min)")

        if 'recovery_score' in training_days.columns and training_days['recovery_score'].notna().any():
            best_rec = training_days.loc[training_days['recovery_score'].idxmax()]
            insights.append(f"<b>Best recovery before training</b>: score {best_rec['recovery_score']:.0f} on {best_rec['date'].strftime('%Y-%m-%d')}")

    # Sleep before good runs
    running_acts = act[act['typ'] == 'running'].copy()
    if len(running_acts) > 0:
        best_pace_runs = running_acts.nsmallest(5, 'tempo_min_km')
        best_dates = best_pace_runs['datum'].dt.normalize()
        sleep_before = []
        for d in best_dates:
            prev_day = d - pd.Timedelta(days=1)
            prev_sleep = df[df['date'] == prev_day]['sleepScore']
            if len(prev_sleep) > 0 and pd.notna(prev_sleep.values[0]):
                sleep_before.append(prev_sleep.values[0])
        if sleep_before:
            insights.append(f"<b>Avg sleep score before 5 fastest runs</b>: {np.mean(sleep_before):.0f}/100")

    # Fastest run
    if len(running_acts) > 0:
        fastest = running_acts.loc[running_acts['tempo_min_km'].idxmin()]
        insights.append(f"<b>Fastest pace</b>: {fastest['tempo_min_km']:.2f} min/km on {fastest['datum'].strftime('%Y-%m-%d')} ({fastest['vzdalenost_km']:.1f} km)")

        longest = running_acts.loc[running_acts['vzdalenost_km'].idxmax()]
        insights.append(f"<b>Longest run</b>: {longest['vzdalenost_km']:.1f} km on {longest['datum'].strftime('%Y-%m-%d')} (pace {longest['tempo_min_km']:.2f} min/km)")

    # Stress impact
    if 'avgStressLevel' in df.columns and 'running_pace' in df.columns:
        run_days = df[df['running_pace'].notna()].copy()
        if len(run_days) > 5:
            low_stress = run_days[run_days['avgStressLevel'] <= run_days['avgStressLevel'].median()]
            high_stress = run_days[run_days['avgStressLevel'] > run_days['avgStressLevel'].median()]
            if len(low_stress) > 0 and len(high_stress) > 0:
                low_pace = low_stress['running_pace'].mean()
                high_pace = high_stress['running_pace'].mean()
                diff = high_pace - low_pace
                if abs(diff) > 0.01:
                    direction = "slower" if diff > 0 else "faster"
                    insights.append(f"<b>Stress impact on pace</b>: High-stress days are {abs(diff):.2f} min/km {direction}")

    # Recovery patterns
    if 'recovery_score' in df.columns:
        hard_days = training_days[training_days['training_distance_km'] > training_days['training_distance_km'].quantile(0.75)] if len(training_days) > 3 else pd.DataFrame()
        if len(hard_days) > 0:
            next_day_recovery = []
            for _, row in hard_days.iterrows():
                next_date = row['date'] + pd.Timedelta(days=1)
                next_rec = df[df['date'] == next_date]['recovery_score']
                if len(next_rec) > 0 and pd.notna(next_rec.values[0]):
                    next_day_recovery.append(next_rec.values[0])
            if next_day_recovery:
                insights.append(f"<b>Avg recovery after hard sessions</b>: {np.mean(next_day_recovery):.0f}/100 (based on {len(next_day_recovery)} sessions)")

    # VO2 Max trend
    vo2_data = df[df['vo2MaxPrecise'].notna()]
    if len(vo2_data) >= 2:
        first_vo2 = vo2_data.iloc[0]['vo2MaxPrecise']
        last_vo2 = vo2_data.iloc[-1]['vo2MaxPrecise']
        change = last_vo2 - first_vo2
        direction = "improved" if change > 0 else "decreased"
        insights.append(f"<b>VO2 Max trend</b>: {direction} by {abs(change):.1f} ({first_vo2:.1f} → {last_vo2:.1f})")

    # Training consistency
    total_days = len(df)
    training_day_count = df['is_training_day'].sum()
    insights.append(f"<b>Training consistency</b>: {training_day_count}/{total_days} days ({training_day_count/total_days*100:.0f}%)")

    # Total distance & time
    total_km = df['training_distance_km'].sum()
    total_hours = df['training_time_min'].sum() / 60
    insights.append(f"<b>Total distance</b>: {total_km:.1f} km in {total_hours:.1f} hours across {int(training_day_count)} sessions")

    # Average metrics
    avg_sleep = df['sleepScore'].mean()
    avg_stress = df['avgStressLevel'].mean()
    avg_rhr = df['restingHeartRate'].mean()
    insights.append(f"<b>Average sleep score</b>: {avg_sleep:.0f}/100 | <b>Avg stress</b>: {avg_stress:.0f} | <b>Avg RHR</b>: {avg_rhr:.0f} bpm")

    return insights

# ─── Dashboard Building ──────────────────────────────────────────────────────

def make_card_html(title, value, subtitle="", color=COLORS['primary']):
    """Generate HTML for an overview card."""
    return f"""
    <div style="background:{COLORS['bg_card']};border-radius:12px;padding:20px 24px;min-width:160px;
                border-left:4px solid {color};flex:1;">
        <div style="color:{COLORS['text_secondary']};font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">{title}</div>
        <div style="color:{COLORS['text_primary']};font-size:28px;font-weight:700;">{value}</div>
        <div style="color:{COLORS['text_secondary']};font-size:12px;margin-top:4px;">{subtitle}</div>
    </div>"""

def build_overview_cards(df, act):
    """Build HTML for overview cards section."""
    total_km = df['training_distance_km'].sum()
    total_hours = df['training_time_min'].sum() / 60
    total_activities = int(df['is_training_day'].sum())

    avg_sleep = df['sleepScore'].mean()
    avg_stress = df['avgStressLevel'].mean()

    vo2_latest = df[df['vo2MaxPrecise'].notna()]['vo2MaxPrecise'].iloc[-1] if df['vo2MaxPrecise'].notna().any() else 'N/A'
    readiness_latest = df[df['readinessScore'].notna()]['readinessScore'].iloc[-1] if df['readinessScore'].notna().any() else 'N/A'
    endurance_latest = df[df['enduranceScore'].notna()]['enduranceScore'].iloc[-1] if df['enduranceScore'].notna().any() else 'N/A'
    hill_latest = df[df['hillScore'].notna()]['hillScore'].iloc[-1] if df['hillScore'].notna().any() else 'N/A'

    date_range = f"{df['date'].min().strftime('%d %b %Y')} – {df['date'].max().strftime('%d %b %Y')}"

    cards_row1 = f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;">
        {make_card_html("Total Distance", f"{total_km:.1f} km", f"{total_activities} sessions", COLORS['primary'])}
        {make_card_html("Total Time", f"{total_hours:.1f} hrs", date_range, COLORS['info'])}
        {make_card_html("Activities", f"{len(act)}", f"{act['typ'].nunique()} types", COLORS['success'])}
        {make_card_html("Avg Sleep Score", f"{avg_sleep:.0f}", "out of 100", COLORS['purple'])}
    </div>"""

    cards_row2 = f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:30px;">
        {make_card_html("Avg Stress", f"{avg_stress:.0f}", "lower is better", COLORS['warning'])}
        {make_card_html("Current VO2 Max", f"{vo2_latest}", "ml/kg/min", COLORS['secondary'])}
        {make_card_html("Readiness", f"{readiness_latest:.0f}" if isinstance(readiness_latest, float) else readiness_latest, "latest score", COLORS['teal'])}
        {make_card_html("Endurance / Hill", f"{endurance_latest:.0f} / {hill_latest:.0f}" if isinstance(endurance_latest, float) else f"{endurance_latest} / {hill_latest}", "score", COLORS['orange'])}
    </div>"""

    return cards_row1 + cards_row2

def build_training_timeline(df):
    """Section 2: Training timeline with distance, sleep, stress, readiness."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Training Distance & Wellness Metrics', 'Body Battery Range'),
    )

    # Distance bars
    training = df[df['is_training_day']].copy()
    fig.add_trace(go.Bar(
        x=training['date'], y=training['training_distance_km'],
        name='Distance (km)', marker_color=COLORS['primary'], opacity=0.8,
        hovertemplate='%{x|%d %b}<br>Distance: %{y:.1f} km<extra></extra>',
    ), row=1, col=1)

    # Sleep score line
    sleep_data = df[df['sleepScore'].notna()]
    fig.add_trace(go.Scatter(
        x=sleep_data['date'], y=sleep_data['sleepScore'],
        name='Sleep Score', mode='lines+markers', line=dict(color=COLORS['purple'], width=2),
        marker=dict(size=4), yaxis='y2',
        hovertemplate='%{x|%d %b}<br>Sleep: %{y:.0f}<extra></extra>',
    ), row=1, col=1)

    # Stress line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['avgStressLevel'],
        name='Avg Stress', mode='lines', line=dict(color=COLORS['warning'], width=2, dash='dot'),
        yaxis='y2',
        hovertemplate='%{x|%d %b}<br>Stress: %{y:.0f}<extra></extra>',
    ), row=1, col=1)

    # Readiness line
    readiness_data = df[df['readinessScore'].notna()]
    fig.add_trace(go.Scatter(
        x=readiness_data['date'], y=readiness_data['readinessScore'],
        name='Readiness', mode='lines+markers', line=dict(color=COLORS['success'], width=2),
        marker=dict(size=4), yaxis='y2',
        hovertemplate='%{x|%d %b}<br>Readiness: %{y:.0f}<extra></extra>',
    ), row=1, col=1)

    # Body battery range (high/low)
    bb_data = df[df['bodyBatteryHighest'].notna()].copy()
    fig.add_trace(go.Scatter(
        x=bb_data['date'], y=bb_data['bodyBatteryHighest'],
        name='Body Battery High', mode='lines', line=dict(color=COLORS['success'], width=1),
        fill=None,
        hovertemplate='%{x|%d %b}<br>BB High: %{y}<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bb_data['date'], y=bb_data['bodyBatteryLowest'],
        name='Body Battery Low', mode='lines', line=dict(color=COLORS['secondary'], width=1),
        fill='tonexty', fillcolor='rgba(52, 168, 83, 0.2)',
        hovertemplate='%{x|%d %b}<br>BB Low: %{y}<extra></extra>',
    ), row=2, col=1)

    # Body battery at wake
    bw_data = df[df['bodyBatteryAtWake'].notna()]
    fig.add_trace(go.Scatter(
        x=bw_data['date'], y=bw_data['bodyBatteryAtWake'],
        name='BB at Wake', mode='markers', marker=dict(color=COLORS['teal'], size=6, symbol='diamond'),
        hovertemplate='%{x|%d %b}<br>BB at Wake: %{y}<extra></extra>',
    ), row=2, col=1)

    fig.update_layout(
        height=550, template='plotly_dark',
        paper_bgcolor=COLORS['bg_dark'], plot_bgcolor=COLORS['bg_card'],
        font=dict(color=COLORS['text_primary']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
        margin=dict(l=50, r=50, t=60, b=30),
        yaxis=dict(title='Distance (km)', gridcolor=COLORS['grid']),
        yaxis2=dict(title='Score (0-100)', overlaying='y', side='right', range=[0, 105], gridcolor=COLORS['grid']),
        yaxis3=dict(title='Body Battery', gridcolor=COLORS['grid'], range=[0, 105]),
        xaxis2=dict(gridcolor=COLORS['grid']),
    )
    # Assign sleep/stress/readiness to secondary y-axis
    fig.data[1].update(yaxis='y2')
    fig.data[2].update(yaxis='y2')
    fig.data[3].update(yaxis='y2')

    return fig

def build_performance_trends(df):
    """Section 3: VO2 Max, Endurance, Hill, Running Pace, Training Load trends."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('VO2 Max Trend', 'Endurance & Hill Score', 'Running Pace Trend', 'Training Load (Acute)'),
        vertical_spacing=0.15, horizontal_spacing=0.1,
    )

    # VO2 Max
    vo2 = df[df['vo2MaxPrecise'].notna()]
    fig.add_trace(go.Scatter(
        x=vo2['date'], y=vo2['vo2MaxPrecise'],
        name='VO2 Max', mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2.5), marker=dict(size=6),
        hovertemplate='%{x|%d %b}<br>VO2 Max: %{y:.1f}<extra></extra>',
    ), row=1, col=1)
    if 'vo2MaxPrecise_7d_avg' in df.columns:
        vo2_avg = df[df['vo2MaxPrecise'].notna()]  # same filter to align
        # Compute a proper trend via rolling on non-null
        vo2_series = df.set_index('date')['vo2MaxPrecise'].dropna()
        if len(vo2_series) >= 3:
            vo2_trend = vo2_series.rolling(5, min_periods=2).mean()
            fig.add_trace(go.Scatter(
                x=vo2_trend.index, y=vo2_trend.values,
                name='VO2 Trend', mode='lines',
                line=dict(color=COLORS['info'], width=2, dash='dash'),
                hovertemplate='%{x|%d %b}<br>VO2 Trend: %{y:.1f}<extra></extra>',
            ), row=1, col=1)

    # Endurance & Hill Score
    end = df[df['enduranceScore'].notna()]
    fig.add_trace(go.Scatter(
        x=end['date'], y=end['enduranceScore'],
        name='Endurance', mode='lines+markers',
        line=dict(color=COLORS['success'], width=2), marker=dict(size=4),
        hovertemplate='%{x|%d %b}<br>Endurance: %{y:.0f}<extra></extra>',
    ), row=1, col=2)
    hill = df[df['hillScore'].notna()]
    fig.add_trace(go.Scatter(
        x=hill['date'], y=hill['hillScore'],
        name='Hill Score', mode='lines+markers',
        line=dict(color=COLORS['orange'], width=2), marker=dict(size=4),
        hovertemplate='%{x|%d %b}<br>Hill: %{y:.0f}<extra></extra>',
    ), row=1, col=2)

    # Running pace trend
    has_pace = 'running_pace' in df.columns and df['running_pace'].notna().any()
    pace = df[df['running_pace'].notna()] if has_pace else pd.DataFrame()
    if len(pace) > 0:
        fig.add_trace(go.Scatter(
            x=pace['date'], y=pace['running_pace'],
            name='Pace (min/km)', mode='lines+markers',
            line=dict(color=COLORS['secondary'], width=2), marker=dict(size=6),
            hovertemplate='%{x|%d %b}<br>Pace: %{y:.2f} min/km<extra></extra>',
        ), row=2, col=1)
        # Trend line
        if len(pace) >= 3:
            pace_series = df.set_index('date')['running_pace'].dropna()
            pace_trend = pace_series.rolling(5, min_periods=2).mean()
            fig.add_trace(go.Scatter(
                x=pace_trend.index, y=pace_trend.values,
                name='Pace Trend', mode='lines',
                line=dict(color=COLORS['info'], width=2, dash='dash'),
                hovertemplate='%{x|%d %b}<br>Trend: %{y:.2f} min/km<extra></extra>',
            ), row=2, col=1)

    # Training load (acute from readiness data)
    load = df[df['acuteLoad'].notna()]
    fig.add_trace(go.Scatter(
        x=load['date'], y=load['acuteLoad'],
        name='Acute Load', mode='lines+markers',
        line=dict(color=COLORS['purple'], width=2), marker=dict(size=4),
        hovertemplate='%{x|%d %b}<br>Acute Load: %{y:.0f}<extra></extra>',
    ), row=2, col=2)

    fig.update_layout(
        height=550, template='plotly_dark',
        paper_bgcolor=COLORS['bg_dark'], plot_bgcolor=COLORS['bg_card'],
        font=dict(color=COLORS['text_primary']),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5, font=dict(size=10)),
        margin=dict(l=50, r=50, t=70, b=30),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS['grid'], row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=COLORS['grid'], row=(i-1)//2+1, col=(i-1)%2+1)

    # Invert y-axis for pace (lower is better)
    fig.update_yaxes(autorange='reversed', title_text='min/km (lower=faster)', row=2, col=1)

    return fig

def build_recovery_wellness(df):
    """Section 4: Sleep breakdown, HRV, Recovery time, Resting HR."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sleep Breakdown (hours)', 'HRV Trend & Status', 'Recovery Time (hours)', 'Resting Heart Rate'),
        vertical_spacing=0.15, horizontal_spacing=0.1,
    )

    sleep = df[df['sleepTimeSeconds'].notna()].copy()

    # Sleep breakdown stacked area
    if len(sleep) > 0:
        fig.add_trace(go.Scatter(
            x=sleep['date'], y=sleep['deepSleepSeconds'] / 3600,
            name='Deep Sleep', mode='lines', stackgroup='sleep',
            line=dict(width=0), fillcolor=COLORS['deep_sleep'],
            hovertemplate='%{x|%d %b}<br>Deep: %{y:.1f}h<extra></extra>',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sleep['date'], y=sleep['lightSleepSeconds'] / 3600,
            name='Light Sleep', mode='lines', stackgroup='sleep',
            line=dict(width=0), fillcolor=COLORS['light_sleep'],
            hovertemplate='%{x|%d %b}<br>Light: %{y:.1f}h<extra></extra>',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sleep['date'], y=sleep['remSleepSeconds'] / 3600,
            name='REM Sleep', mode='lines', stackgroup='sleep',
            line=dict(width=0), fillcolor=COLORS['rem_sleep'],
            hovertemplate='%{x|%d %b}<br>REM: %{y:.1f}h<extra></extra>',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sleep['date'], y=sleep['awakeSleepSeconds'] / 3600,
            name='Awake', mode='lines', stackgroup='sleep',
            line=dict(width=0), fillcolor=COLORS['awake_sleep'],
            hovertemplate='%{x|%d %b}<br>Awake: %{y:.1f}h<extra></extra>',
        ), row=1, col=1)

    # HRV trend
    hrv = df[df['hrvLastNightAvg'].notna()]
    fig.add_trace(go.Scatter(
        x=hrv['date'], y=hrv['hrvLastNightAvg'],
        name='HRV Avg', mode='lines+markers',
        line=dict(color=COLORS['teal'], width=2), marker=dict(size=4),
        hovertemplate='%{x|%d %b}<br>HRV: %{y:.0f} ms<extra></extra>',
    ), row=1, col=2)
    if 'hrvLastNightAvg_7d_avg' in df.columns:
        hrv_avg = df[df['hrvLastNightAvg_7d_avg'].notna()]
        fig.add_trace(go.Scatter(
            x=hrv_avg['date'], y=hrv_avg['hrvLastNightAvg_7d_avg'],
            name='HRV 7d Avg', mode='lines',
            line=dict(color=COLORS['info'], width=2, dash='dash'),
            hovertemplate='%{x|%d %b}<br>7d Avg: %{y:.0f} ms<extra></extra>',
        ), row=1, col=2)

    # HRV status markers
    hrv_status = df[df['hrvStatus'].notna() & (df['hrvStatus'] != '')].copy()
    if len(hrv_status) > 0:
        status_colors = {'BALANCED': COLORS['success'], 'UNBALANCED': COLORS['warning'],
                         'LOW': COLORS['secondary'], 'NONE': COLORS['text_secondary']}
        for status, color in status_colors.items():
            s = hrv_status[hrv_status['hrvStatus'] == status]
            if len(s) > 0:
                fig.add_trace(go.Scatter(
                    x=s['date'], y=s['hrvLastNightAvg'],
                    name=f'HRV {status.title()}', mode='markers',
                    marker=dict(color=color, size=8, symbol='circle'),
                    hovertemplate='%{x|%d %b}<br>HRV: %{y:.0f} ms<br>Status: ' + status + '<extra></extra>',
                ), row=1, col=2)

    # Recovery time
    rec = df[df['recoveryTimeMin'].notna()]
    fig.add_trace(go.Scatter(
        x=rec['date'], y=rec['recoveryTimeMin'] / 60,
        name='Recovery Time', mode='lines+markers',
        line=dict(color=COLORS['orange'], width=2), marker=dict(size=4),
        hovertemplate='%{x|%d %b}<br>Recovery: %{y:.1f} hours<extra></extra>',
    ), row=2, col=1)

    # Resting HR
    rhr = df[df['restingHeartRate'].notna()]
    fig.add_trace(go.Scatter(
        x=rhr['date'], y=rhr['restingHeartRate'],
        name='Resting HR', mode='lines+markers',
        line=dict(color=COLORS['secondary'], width=2), marker=dict(size=4),
        hovertemplate='%{x|%d %b}<br>RHR: %{y:.0f} bpm<extra></extra>',
    ), row=2, col=2)
    if 'restingHeartRate_7d_avg' in df.columns:
        rhr_avg = df[df['restingHeartRate_7d_avg'].notna()]
        fig.add_trace(go.Scatter(
            x=rhr_avg['date'], y=rhr_avg['restingHeartRate_7d_avg'],
            name='RHR 7d Avg', mode='lines',
            line=dict(color=COLORS['info'], width=2, dash='dash'),
            hovertemplate='%{x|%d %b}<br>7d Avg: %{y:.0f} bpm<extra></extra>',
        ), row=2, col=2)

    fig.update_layout(
        height=550, template='plotly_dark',
        paper_bgcolor=COLORS['bg_dark'], plot_bgcolor=COLORS['bg_card'],
        font=dict(color=COLORS['text_primary']),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5, font=dict(size=10)),
        margin=dict(l=50, r=50, t=70, b=30),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS['grid'], row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=COLORS['grid'], row=(i-1)//2+1, col=(i-1)%2+1)

    return fig

def build_correlation_heatmap(corr_df):
    """Section 5: Correlation heatmap."""
    # Friendly labels
    label_map = {
        'training_distance_km': 'Distance (km)',
        'training_time_min': 'Training Time',
        'training_avg_hr': 'Training Avg HR',
        'avgStressLevel': 'Avg Stress',
        'restingHeartRate': 'Resting HR',
        'sleepScore': 'Sleep Score',
        'sleepTimeHours': 'Sleep Hours',
        'deep_sleep_pct': 'Deep Sleep %',
        'hrvLastNightAvg': 'HRV Avg',
        'bodyBatteryHighest': 'BB Highest',
        'bodyBatteryLowest': 'BB Lowest',
        'bodyBatteryAtWake': 'BB at Wake',
        'recovery_score': 'Recovery Score',
        'readinessScore': 'Readiness',
        'vo2MaxPrecise': 'VO2 Max',
        'enduranceScore': 'Endurance',
        'acuteLoad': 'Acute Load',
        'running_pace': 'Running Pace',
        'steps': 'Steps',
    }
    labels = [label_map.get(c, c) for c in corr_df.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=labels, y=labels,
        colorscale='RdBu_r', zmin=-1, zmax=1,
        text=np.round(corr_df.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=9),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        height=600, template='plotly_dark',
        paper_bgcolor=COLORS['bg_dark'], plot_bgcolor=COLORS['bg_card'],
        font=dict(color=COLORS['text_primary']),
        margin=dict(l=120, r=50, t=30, b=120),
        xaxis=dict(tickangle=45),
    )
    return fig

def build_activities_breakdown(act):
    """Bonus: Activities breakdown by type."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Activities by Type', 'Distance by Activity Type'),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
    )

    type_counts = act['typ'].value_counts()
    fig.add_trace(go.Pie(
        labels=type_counts.index, values=type_counts.values,
        hole=0.4, textinfo='label+value',
        marker=dict(colors=px.colors.qualitative.Set2),
    ), row=1, col=1)

    type_dist = act.groupby('typ')['vzdalenost_km'].sum().sort_values(ascending=True)
    fig.add_trace(go.Bar(
        x=type_dist.values, y=type_dist.index,
        orientation='h', marker_color=COLORS['primary'],
        text=[f'{v:.1f} km' for v in type_dist.values],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f} km<extra></extra>',
    ), row=1, col=2)

    fig.update_layout(
        height=350, template='plotly_dark',
        paper_bgcolor=COLORS['bg_dark'], plot_bgcolor=COLORS['bg_card'],
        font=dict(color=COLORS['text_primary']),
        margin=dict(l=50, r=80, t=50, b=30),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=COLORS['grid'], row=1, col=2)
    fig.update_yaxes(gridcolor=COLORS['grid'], row=1, col=2)

    return fig

def format_correlation_insights(top_corrs):
    """Format top correlations as HTML list with interpretation."""
    interpretations = {
        ('sleepScore', 'hrvLastNightAvg'): 'Better sleep correlates with higher HRV',
        ('sleepScore', 'readinessScore'): 'Sleep quality drives training readiness',
        ('avgStressLevel', 'sleepScore'): 'Stress negatively impacts sleep quality',
        ('avgStressLevel', 'hrvLastNightAvg'): 'Higher stress associated with lower HRV',
        ('restingHeartRate', 'hrvLastNightAvg'): 'Lower RHR associated with higher HRV (fitness)',
        ('bodyBatteryHighest', 'sleepScore'): 'Good sleep charges body battery higher',
        ('recovery_score', 'readinessScore'): 'Recovery score reflects readiness well',
        ('training_distance_km', 'acuteLoad'): 'More distance increases acute training load',
        ('running_pace', 'training_avg_hr'): 'Faster pace requires higher heart rate effort',
    }

    html = '<div style="display:grid;gap:8px;">'
    for a, b, val in top_corrs:
        strength = 'Strong' if abs(val) > 0.7 else ('Moderate' if abs(val) > 0.4 else 'Weak')
        direction = 'positive' if val > 0 else 'negative'
        color = COLORS['success'] if val > 0.4 else (COLORS['secondary'] if val < -0.4 else COLORS['text_secondary'])

        key = (a, b) if (a, b) in interpretations else ((b, a) if (b, a) in interpretations else None)
        interp = interpretations.get(key, '')

        label_map = {
            'training_distance_km': 'Distance', 'training_time_min': 'Training Time',
            'training_avg_hr': 'Avg HR', 'avgStressLevel': 'Stress',
            'restingHeartRate': 'RHR', 'sleepScore': 'Sleep Score',
            'sleepTimeHours': 'Sleep Hours', 'deep_sleep_pct': 'Deep Sleep %',
            'hrvLastNightAvg': 'HRV', 'bodyBatteryHighest': 'BB High',
            'bodyBatteryLowest': 'BB Low', 'bodyBatteryAtWake': 'BB Wake',
            'recovery_score': 'Recovery', 'readinessScore': 'Readiness',
            'vo2MaxPrecise': 'VO2 Max', 'enduranceScore': 'Endurance',
            'acuteLoad': 'Acute Load', 'running_pace': 'Pace', 'steps': 'Steps',
        }
        a_label = label_map.get(a, a)
        b_label = label_map.get(b, b)

        html += f"""
        <div style="background:{COLORS['bg_surface']};padding:10px 14px;border-radius:8px;display:flex;align-items:center;gap:12px;">
            <div style="min-width:55px;text-align:center;font-size:18px;font-weight:700;color:{color};">{val:+.2f}</div>
            <div>
                <div style="color:{COLORS['text_primary']};font-size:13px;"><b>{a_label}</b> ↔ <b>{b_label}</b> ({strength} {direction})</div>
                {'<div style="color:' + COLORS['text_secondary'] + ';font-size:11px;margin-top:2px;">' + interp + '</div>' if interp else ''}
            </div>
        </div>"""
    html += '</div>'
    return html

def _compute_week_data(df, act, week_num, current_week):
    """Compute all metrics + daily activity data for a single training week.
    Returns a dict with metrics, evaluations, and day-by-day activity info."""
    import json as _json

    milestone = MILESTONES.get(week_num, {})
    week_start, week_end = get_week_start_end(week_num)
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(week_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    week_act = act[(act['datum'] >= ws) & (act['datum'] <= we)].copy()
    week_runs = week_act[week_act['typ'] == 'running'].copy()
    week_df = df[(df['date'] >= ws) & (df['date'] <= we)].copy()

    # --- Actual metrics ---
    actual_km = round(week_runs['vzdalenost_km'].sum(), 1) if len(week_runs) > 0 else 0
    actual_runs = int(len(week_runs))
    actual_long = round(float(week_runs['vzdalenost_km'].max()), 1) if len(week_runs) > 0 else 0

    easy_r = week_runs[(week_runs['tempo_min_km'] > 5.5) & (week_runs['prumer_tep'] > 0)]
    actual_easy_hr = round(float(easy_r['prumer_tep'].mean()), 0) if len(easy_r) > 0 else None

    cad_r = week_runs[week_runs['avg_cadence'].notna() & (week_runs['avg_cadence'] > 0)]
    actual_cadence = round(float(cad_r['avg_cadence'].mean()), 0) if len(cad_r) > 0 else None

    actual_vo2 = None
    if 'vo2MaxPrecise' in week_df.columns:
        v = week_df['vo2MaxPrecise'].dropna()
        actual_vo2 = round(float(v.iloc[-1]), 1) if len(v) > 0 else None

    actual_readiness = None
    if 'readinessScore' in week_df.columns:
        r = week_df['readinessScore'].dropna()
        actual_readiness = round(float(r.mean()), 0) if len(r) > 0 else None

    actual_hrv = None
    if 'hrvLastNightAvg' in week_df.columns:
        h = week_df['hrvLastNightAvg'].dropna()
        actual_hrv = round(float(h.mean()), 0) if len(h) > 0 else None

    actual_sleep = None
    if 'sleepScore' in week_df.columns:
        s = week_df['sleepScore'].dropna()
        actual_sleep = round(float(s.mean()), 0) if len(s) > 0 else None

    actual_rhr = None
    if 'restingHeartRate' in week_df.columns:
        rr = week_df['restingHeartRate'].dropna()
        actual_rhr = round(float(rr.mean()), 0) if len(rr) > 0 else None

    # % easy
    pct_easy = None
    if len(week_runs) > 0:
        hr_runs = week_runs[week_runs['prumer_tep'] > 0]
        if len(hr_runs) > 0:
            pct_easy = round(len(hr_runs[hr_runs['prumer_tep'] < 155]) / len(hr_runs) * 100, 0)

    # --- Evaluate 7 metrics ---
    km_min = milestone.get('target_weekly_km_min', 0)
    km_max = milestone.get('target_weekly_km_max', 0)

    evals = []  # list of (name, actual_str, target_str, status)

    st, _ = evaluate_metric(actual_km, (km_min, km_max), "range")
    evals.append(("Weekly km", f"{actual_km:.1f}", f"{km_min}-{km_max}", st))

    st, _ = evaluate_metric(actual_runs, milestone.get('target_runs_per_week', 0), "higher_is_better")
    evals.append(("Runs", str(actual_runs), str(milestone.get('target_runs_per_week', '-')), st))

    st, _ = evaluate_metric(actual_long, milestone.get('target_long_run_km', 0), "higher_is_better")
    evals.append(("Long Run", f"{actual_long:.1f} km", f"{milestone.get('target_long_run_km', '-')} km", st))

    if actual_easy_hr and milestone.get('expected_easy_hr'):
        st, _ = evaluate_metric(actual_easy_hr, milestone['expected_easy_hr'], "lower_is_better")
        evals.append(("Easy HR", f"{actual_easy_hr:.0f} bpm", f"< {milestone['expected_easy_hr']}", st))
    else:
        evals.append(("Easy HR", "-", f"< {milestone.get('expected_easy_hr', '-')}", "no_data"))

    if actual_vo2:
        st, _ = evaluate_metric(actual_vo2, milestone.get('expected_vo2max', 0), "higher_is_better")
        evals.append(("VO2max", f"{actual_vo2:.1f}", f"{milestone.get('expected_vo2max', '-')}", st))
    else:
        evals.append(("VO2max", "-", f"{milestone.get('expected_vo2max', '-')}", "no_data"))

    if actual_cadence:
        st, _ = evaluate_metric(actual_cadence, milestone.get('expected_cadence', 0), "higher_is_better")
        evals.append(("Cadence", f"{actual_cadence:.0f} spm", f"{milestone.get('expected_cadence', '-')}", st))
    else:
        evals.append(("Cadence", "-", f"{milestone.get('expected_cadence', '-')}", "no_data"))

    if actual_readiness:
        st, _ = evaluate_metric(actual_readiness, milestone.get('expected_readiness_avg', 0), "higher_is_better")
        evals.append(("Readiness", f"{actual_readiness:.0f}", f"{milestone.get('expected_readiness_avg', '-')}", st))
    else:
        evals.append(("Readiness", "-", f"{milestone.get('expected_readiness_avg', '-')}", "no_data"))

    # --- Day-by-day activity classification (Mon=0 .. Sun=6) ---
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days = []
    for i in range(7):
        day_date = week_start + timedelta(days=i)
        d_ts = pd.Timestamp(day_date)
        d_act = week_act[(week_act['datum'] >= d_ts) & (week_act['datum'] < d_ts + pd.Timedelta(days=1))]
        d_runs = d_act[d_act['typ'] == 'running']

        if len(d_runs) > 0:
            run_km = float(d_runs['vzdalenost_km'].sum())
            run_hr = float(d_runs['prumer_tep'].mean()) if d_runs['prumer_tep'].mean() > 0 else 0
            # Classify: long run if it's the week's longest and > 8km
            is_long = (run_km >= actual_long - 0.1) and actual_long >= 8 and run_km >= 8
            if is_long:
                day_type = "long"
                day_color = "#ea4335"  # red
            elif run_hr > 0 and run_hr < 155:
                day_type = "easy"
                day_color = "#34a853"  # green
            else:
                day_type = "other"
                day_color = "#fbbc04"  # yellow
            day_info = f"{run_km:.1f} km"
        elif len(d_act) > 0:
            # Non-running activity
            day_type = "other"
            day_color = "#fbbc04"  # yellow
            types = d_act['typ'].unique()
            day_info = ', '.join(str(t) for t in types)
        else:
            day_type = "rest"
            day_color = "#4285f4"  # blue
            day_info = "Rest"

        days.append({
            "label": day_labels[i],
            "date": day_date.strftime("%d %b"),
            "type": day_type,
            "color": day_color,
            "info": day_info,
        })

    # Overall week status for the timeline dot
    real_statuses = [s for _, _, _, s in evals if s != 'no_data']
    if not real_statuses or week_num > current_week:
        dot_status = "no_data"
    elif 'behind' in real_statuses:
        dot_status = "behind"
    elif 'warning' in real_statuses:
        dot_status = "warning"
    elif 'ahead' in real_statuses:
        dot_status = "ahead"
    else:
        dot_status = "on_track"

    # Build planned days (from DAILY_PLAN) with actual dates
    planned_days_raw = DAILY_PLAN.get(week_num, [])
    planned_days = []
    for i, pd_entry in enumerate(planned_days_raw):
        day_date = week_start + timedelta(days=i)
        planned_days.append({
            "label": day_labels[i],
            "date": day_date.strftime("%d %b"),
            "type": pd_entry["type"],
            "color": pd_entry["color"],
            "info": pd_entry["info"],
        })

    return {
        "week": week_num,
        "phase": milestone.get('phase', ''),
        "key_workout": milestone.get('key_workout', ''),
        "focus_notes": milestone.get('focus_notes', ''),
        "dot_status": dot_status,
        "evals": [{"name": n, "actual": a, "target": t, "status": s} for n, a, t, s in evals],
        "days": days,
        "planned_days": planned_days,
        "weekly_km": actual_km,
    }


def build_training_plan_progress(df, act):
    """Build the Training Plan Progress section for the top of the dashboard.
    Returns HTML string with race countdown, clickable timeline, 7 metric cards,
    7-day activity row, trend charts, and coaching notes."""
    if not HAS_TRAINING_PLAN:
        return ""
    import json as _json

    today = date.today()
    current_week = get_current_week(today)
    if current_week < 1 or current_week > 16:
        days_to_race = get_days_to_race(today)
        if days_to_race > 0:
            return f"""
            <div class="section">
                <div class="section-title">Training Plan Progress <span class="badge">16-Week Plan</span></div>
                <div style="background:{COLORS['bg_card']};border-radius:12px;padding:24px;text-align:center;">
                    <div style="font-size:16px;color:{COLORS['text_secondary']};">Training plan starts on {PLAN_START.strftime('%d %b %Y')}</div>
                    <div style="font-size:28px;font-weight:700;margin:10px 0;">{days_to_race} days to race</div>
                </div>
            </div>"""
        return ""

    days_to_race = get_days_to_race(today)
    phase = get_current_phase(current_week)
    phase_info = get_phase_info(phase) if phase else None
    phase_color = phase_info['color'] if phase_info else COLORS['text_secondary']
    milestone = MILESTONES.get(current_week, {})

    # --- Precompute ALL 16 weeks ---
    all_weeks_data = {}
    for w in range(1, 17):
        all_weeks_data[w] = _compute_week_data(df, act, w, current_week)

    # Serialize to JSON for JavaScript
    weeks_json = _json.dumps(all_weeks_data, ensure_ascii=False)

    status_colors_json = _json.dumps(STATUS_COLORS)
    status_labels_json = _json.dumps(STATUS_LABELS)

    # --- Build 16-week timeline dots (static HTML, made clickable via JS) ---
    timeline_html = '<div id="tp-timeline" style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end;justify-content:center;padding:16px 0;">'
    for w in range(1, 17):
        wd = all_weeks_data[w]
        dot_color = STATUS_COLORS.get(wd['dot_status'], '#666')
        is_current = w == current_week
        sz = 28 if is_current else 22
        border = "2px solid #fff" if is_current else "1px solid #3a3a4e"

        timeline_html += f"""
        <div style="text-align:center;cursor:pointer;" title="Week {w}: {wd['phase']}" onclick="selectWeek({w})">
            <div id="tp-dot-{w}" style="width:{sz}px;height:{sz}px;border-radius:50%;background:{dot_color};border:{border};
                        display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff;
                        transition:transform 0.15s,box-shadow 0.15s;"
                 onmouseover="this.style.transform='scale(1.2)';this.style.boxShadow='0 0 8px {dot_color}'"
                 onmouseout="this.style.transform='scale(1)';this.style.boxShadow='none'">{w}</div>
        </div>"""
    timeline_html += '</div>'

    # Phase legend
    phase_legend = '<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:4px;">'
    for pname, pinfo in PHASES.items():
        phase_legend += f'<span style="font-size:11px;color:{COLORS["text_secondary"]};"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{pinfo["color"]};vertical-align:middle;margin-right:4px;"></span>{pname}</span>'
    phase_legend += '</div>'

    # Day type legend
    day_legend = f"""
    <div style="display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-top:8px;font-size:11px;color:{COLORS['text_secondary']};">
        <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#34a853;vertical-align:middle;margin-right:3px;"></span>Easy Run</span>
        <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#ea4335;vertical-align:middle;margin-right:3px;"></span>Long Run</span>
        <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#fbbc04;vertical-align:middle;margin-right:3px;"></span>Other Activity</span>
        <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#4285f4;vertical-align:middle;margin-right:3px;"></span>Rest Day</span>
    </div>"""

    # --- Build trend chart (Plotly) ---
    trend_weeks = list(range(1, 17))
    actual_km_list = []
    expected_km_list = []
    for w in trend_weeks:
        ms = MILESTONES[w]
        expected_km_list.append((ms['target_weekly_km_min'] + ms['target_weekly_km_max']) / 2)
        wd = all_weeks_data[w]
        actual_km_list.append(wd['weekly_km'] if wd['weekly_km'] > 0 and w <= current_week else None)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_weeks, y=actual_km_list,
        name='Actual km', mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2.5), marker=dict(size=7),
        connectgaps=False,
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_weeks, y=expected_km_list,
        name='Target km', mode='lines',
        line=dict(color=COLORS['success'], width=2, dash='dash'),
    ))
    fig_trend.add_vline(x=current_week, line_dash="dot", line_color="#fff", opacity=0.5)
    fig_trend.update_layout(
        height=280, template='plotly_dark',
        paper_bgcolor=COLORS['bg_card'], plot_bgcolor=COLORS['bg_card'],
        font=dict(color=COLORS['text_primary'], size=11),
        legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center', font=dict(size=10)),
        margin=dict(l=50, r=30, t=30, b=40),
        xaxis=dict(title='Week', dtick=1, gridcolor=COLORS['grid']),
        yaxis=dict(title='km', gridcolor=COLORS['grid']),
    )
    trend_html = fig_trend.to_html(full_html=False, include_plotlyjs=False)

    # --- Assemble section ---
    section = f"""
    <div class="section" style="background:linear-gradient(180deg, rgba(26,35,126,0.15), transparent);border-radius:16px;padding:24px;margin-bottom:40px;">
        <div class="section-title" style="border-bottom-color:{phase_color};">
            Training Plan Progress
            <span class="badge" style="background:{phase_color};color:#fff;" id="tp-phase-badge">{phase}</span>
            <span class="badge" id="tp-week-badge">Week {current_week} of 16</span>
        </div>

        <!-- Race countdown -->
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px;">
            <div style="background:{COLORS['bg_card']};border-radius:10px;padding:14px 22px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:{COLORS['primary']};">{days_to_race}</div>
                <div style="font-size:11px;color:{COLORS['text_secondary']};text-transform:uppercase;">Days to Race</div>
            </div>
            <div style="background:{COLORS['bg_card']};border-radius:10px;padding:14px 22px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:{phase_color};" id="tp-week-num">{current_week}/16</div>
                <div style="font-size:11px;color:{COLORS['text_secondary']};text-transform:uppercase;">Week</div>
            </div>
            <div style="background:{COLORS['bg_card']};border-radius:10px;padding:14px 22px;flex:1;min-width:200px;" id="tp-key-workout-box">
                <div style="font-size:14px;font-weight:600;color:{COLORS['text_primary']};" id="tp-key-workout">{milestone.get('key_workout', '-')}</div>
                <div style="font-size:11px;color:{COLORS['text_secondary']};margin-top:4px;">Key workout</div>
            </div>
        </div>

        <!-- 7 Milestone progress cards (updated by JS) -->
        <div id="tp-cards" style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;"></div>

        <!-- 7-day activity rows: Actual + Planned (updated by JS) -->
        <div style="margin-bottom:20px;">
            <div style="color:{COLORS['text_secondary']};font-size:12px;margin-bottom:6px;" id="tp-days-label">Daily View</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <div style="color:{COLORS['text_secondary']};font-size:11px;font-weight:600;min-width:60px;text-transform:uppercase;letter-spacing:1px;">Actual</div>
                <div id="tp-days-actual" style="display:flex;gap:8px;flex-wrap:wrap;flex:1;"></div>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="color:{COLORS['text_secondary']};font-size:11px;font-weight:600;min-width:60px;text-transform:uppercase;letter-spacing:1px;">Plan</div>
                <div id="tp-days-plan" style="display:flex;gap:8px;flex-wrap:wrap;flex:1;"></div>
            </div>
            {day_legend}
        </div>

        <!-- 16-week timeline -->
        <div style="margin-bottom:16px;">
            <div style="color:{COLORS['text_secondary']};font-size:12px;margin-bottom:4px;">16-Week Timeline (click a week)</div>
            {timeline_html}
            {phase_legend}
        </div>

        <!-- Trend chart + Coaching -->
        <div style="display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-top:16px;">
            <div>
                <div style="color:{COLORS['text_secondary']};font-size:12px;margin-bottom:4px;">Weekly Distance: Actual vs Target</div>
                {trend_html}
            </div>
            <div>
                <div style="color:{COLORS['text_secondary']};font-size:12px;margin-bottom:6px;">Coaching Notes</div>
                <div id="tp-coaching"></div>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        var WEEKS = {weeks_json};
        var SC = {status_colors_json};
        var SL = {status_labels_json};
        var currentWeek = {current_week};
        var selectedWeek = currentWeek;

        function renderCards(w) {{
            var data = WEEKS[w];
            if (!data) return;
            var html = '';
            data.evals.forEach(function(ev) {{
                var color = SC[ev.status] || '#666';
                var label = SL[ev.status] || 'N/A';
                html += '<div style="background:#313145;border-radius:10px;padding:12px 16px;border-left:4px solid ' + color + ';flex:1;min-width:130px;">'
                    + '<div style="display:flex;justify-content:space-between;align-items:center;">'
                    + '<span style="color:#a0a0b0;font-size:11px;text-transform:uppercase;">' + ev.name + '</span>'
                    + '<span style="background:' + color + ';color:#fff;padding:1px 8px;border-radius:8px;font-size:10px;font-weight:600;">' + label + '</span>'
                    + '</div>'
                    + '<div style="color:#e0e0e0;font-size:20px;font-weight:700;margin:4px 0;">' + ev.actual + '</div>'
                    + '<div style="color:#a0a0b0;font-size:11px;">Target: ' + ev.target + '</div>'
                    + '</div>';
            }});
            document.getElementById('tp-cards').innerHTML = html;
        }}

        function renderDayRow(containerId, daysList) {{
            var html = '';
            daysList.forEach(function(d) {{
                html += '<div style="background:' + d.color + '18;border:2px solid ' + d.color + ';border-radius:10px;padding:10px 12px;flex:1;min-width:100px;text-align:center;">'
                    + '<div style="font-size:12px;font-weight:700;color:' + d.color + ';">' + d.label + '</div>'
                    + '<div style="font-size:10px;color:#a0a0b0;margin:2px 0;">' + d.date + '</div>'
                    + '<div style="font-size:12px;font-weight:600;color:#e0e0e0;margin-top:4px;">' + d.info + '</div>'
                    + '</div>';
            }});
            document.getElementById(containerId).innerHTML = html;
        }}

        function renderDays(w) {{
            var data = WEEKS[w];
            if (!data) return;
            renderDayRow('tp-days-actual', data.days);
            renderDayRow('tp-days-plan', data.planned_days || []);
            document.getElementById('tp-days-label').innerText = 'Daily View - Week ' + w;
        }}

        function renderCoaching(w) {{
            var data = WEEKS[w];
            if (!data) return;
            var notes = [];
            var good = [], bad = [], warn = [];
            data.evals.forEach(function(ev) {{
                if (ev.status === 'on_track' || ev.status === 'ahead') good.push(ev.name);
                else if (ev.status === 'behind') bad.push(ev.name);
                else if (ev.status === 'warning') warn.push(ev.name);
            }});
            if (good.length) notes.push('<b>On track:</b> ' + good.join(', '));
            if (bad.length) notes.push('<b>Needs work:</b> ' + bad.join(', '));
            if (warn.length) notes.push('<b>Watch:</b> ' + warn.join(', '));
            notes.push('<b>Key workout:</b> ' + data.key_workout);
            notes.push('<b>Coach:</b> ' + data.focus_notes);
            var html = '';
            notes.forEach(function(n) {{
                html += '<div style="background:#313145;padding:8px 14px;border-radius:8px;color:#e0e0e0;font-size:12px;line-height:1.5;margin-bottom:6px;">' + n + '</div>';
            }});
            document.getElementById('tp-coaching').innerHTML = html;
        }}

        function updateHeader(w) {{
            var data = WEEKS[w];
            if (!data) return;
            document.getElementById('tp-week-badge').innerText = 'Week ' + w + ' of 16';
            document.getElementById('tp-week-num').innerText = w + '/16';
            document.getElementById('tp-key-workout').innerText = data.key_workout || '-';
            document.getElementById('tp-phase-badge').innerText = data.phase;
        }}

        function highlightDot(w) {{
            for (var i = 1; i <= 16; i++) {{
                var dot = document.getElementById('tp-dot-' + i);
                if (!dot) continue;
                if (i === w) {{
                    dot.style.border = '2px solid #fff';
                    dot.style.width = '28px';
                    dot.style.height = '28px';
                }} else {{
                    dot.style.border = '1px solid #3a3a4e';
                    dot.style.width = '22px';
                    dot.style.height = '22px';
                }}
            }}
        }}

        window.selectWeek = function(w) {{
            selectedWeek = w;
            renderCards(w);
            renderDays(w);
            renderCoaching(w);
            updateHeader(w);
            highlightDot(w);
        }};

        // Initial render
        selectWeek(currentWeek);
    }})();
    </script>"""

    return section


def build_dashboard(df, act, corr_df, top_corrs, insights):
    """Assemble the full HTML dashboard."""

    # Build all figures
    fig_timeline = build_training_timeline(df)
    fig_performance = build_performance_trends(df)
    fig_recovery = build_recovery_wellness(df)
    fig_heatmap = build_correlation_heatmap(corr_df)
    fig_activities = build_activities_breakdown(act)

    # Convert to HTML divs
    timeline_html = fig_timeline.to_html(full_html=False, include_plotlyjs=False)
    performance_html = fig_performance.to_html(full_html=False, include_plotlyjs=False)
    recovery_html = fig_recovery.to_html(full_html=False, include_plotlyjs=False)
    heatmap_html = fig_heatmap.to_html(full_html=False, include_plotlyjs=False)
    activities_html = fig_activities.to_html(full_html=False, include_plotlyjs=False)

    # Build training plan progress section
    progress_html = build_training_plan_progress(df, act)

    # Build overview cards
    cards_html = build_overview_cards(df, act)

    # Build correlation insights
    corr_insights_html = format_correlation_insights(top_corrs)

    # Build training insights
    insights_html = '<div style="display:grid;gap:8px;">'
    for insight in insights:
        insights_html += f'<div style="background:{COLORS["bg_surface"]};padding:10px 14px;border-radius:8px;color:{COLORS["text_primary"]};font-size:13px;line-height:1.5;">{insight}</div>'
    insights_html += '</div>'

    # Date info
    date_from = df['date'].min().strftime('%d %b %Y')
    date_to = df['date'].max().strftime('%d %b %Y')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Garmin PRO Trener Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: {COLORS['bg_dark']};
            color: {COLORS['text_primary']};
            padding: 0;
        }}
        .dashboard-header {{
            background: linear-gradient(135deg, #1a237e, #0d47a1);
            padding: 32px 40px;
            border-bottom: 3px solid {COLORS['primary']};
        }}
        .dashboard-header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .dashboard-header p {{
            color: {COLORS['text_secondary']};
            font-size: 14px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid {COLORS['grid']};
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-title .badge {{
            font-size: 11px;
            padding: 3px 10px;
            border-radius: 12px;
            background: {COLORS['bg_surface']};
            color: {COLORS['text_secondary']};
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
            .container {{ padding: 20px; }}
        }}
        .plotly-graph-div {{ border-radius: 12px; overflow: hidden; }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>Garmin PRO Trener Dashboard</h1>
        <p>{date_from} – {date_to} &nbsp;|&nbsp; {len(df)} days &nbsp;|&nbsp; {len(act)} activities</p>
    </div>
    <div class="container">

        {progress_html}

        <div class="section">
            <div class="section-title">Overview <span class="badge">Summary</span></div>
            {cards_html}
        </div>

        <div class="section">
            <div class="section-title">Training Timeline <span class="badge">Daily</span></div>
            {timeline_html}
        </div>

        <div class="section">
            <div class="section-title">Performance Trends <span class="badge">Trends</span></div>
            {performance_html}
        </div>

        <div class="section">
            <div class="section-title">Recovery & Wellness <span class="badge">Health</span></div>
            {recovery_html}
        </div>

        <div class="section">
            <div class="section-title">Activities Breakdown <span class="badge">Activities</span></div>
            {activities_html}
        </div>

        <div class="section">
            <div class="section-title">Correlation Analysis <span class="badge">Statistics</span></div>
            <div class="two-col">
                <div>
                    <div style="color:{COLORS['text_secondary']};font-size:13px;margin-bottom:10px;">Correlation Matrix (key metrics)</div>
                    {heatmap_html}
                </div>
                <div>
                    <div style="color:{COLORS['text_secondary']};font-size:13px;margin-bottom:10px;">Top Correlations</div>
                    {corr_insights_html}
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Training Insights <span class="badge">Analysis</span></div>
            {insights_html}
        </div>

        <div style="text-align:center;padding:30px 0;color:{COLORS['text_secondary']};font-size:12px;">
            Generated by Garmin PRO Trener &nbsp;|&nbsp; Data: {date_from} – {date_to}
        </div>
    </div>
</body>
</html>"""

    return html

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Garmin PRO Trener - Analysis & Dashboard Generator")
    print("=" * 60)
    print()

    # 1. Load data
    df, act = load_data()
    print()

    # 2. Feature engineering
    print("Engineering features...")
    df = engineer_features(df, act)
    print(f"  Added features: {len(df.columns)} total columns")

    # 3. Save enriched dataset
    df.to_csv(OUTPUT_FEATURES, index=False)
    print(f"  Saved enriched dataset to: {os.path.basename(OUTPUT_FEATURES)}")
    print()

    # 4. Correlation analysis
    print("Running correlation analysis...")
    corr_df, corr_cols = run_correlation_analysis(df)
    top_corrs = get_top_correlations(corr_df, n=15)
    print(f"  Analyzed {len(corr_cols)} metrics")
    print(f"  Top correlation: {top_corrs[0][0]} <-> {top_corrs[0][1]} = {top_corrs[0][2]:.3f}")
    print()

    # 5. Generate insights
    print("Generating training insights...")
    insights = generate_insights(df, act)
    print(f"  Generated {len(insights)} insights")
    print()

    # 6. Build dashboard
    print("Building interactive dashboard...")
    html = build_dashboard(df, act, corr_df, top_corrs, insights)

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Dashboard saved to: {OUTPUT_HTML}")

    file_size_mb = os.path.getsize(OUTPUT_HTML) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print()
    print("Done! Open dashboard.html in your browser to view the interactive dashboard.")

if __name__ == '__main__':
    main()
