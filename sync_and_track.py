#!/usr/bin/env python3
"""
Garmin PRO Trener - One-Click Sync & Progress Tracker
Pulls fresh Garmin data, regenerates dashboard, and tracks 16-week progress.

Usage:
    python sync_and_track.py           # Full sync + progress check
    python sync_and_track.py --skip-sync   # Skip data sync, just recalculate progress
    python sync_and_track.py --week 3      # Show progress for a specific week
"""

import os
import sys
import glob
import csv
import argparse
import subprocess
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project dir to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from training_plan import (
    MILESTONES, PLAN_START, RACE_DATE, TARGET_PACE, TARGET_TIME, RACE_DISTANCE,
    PHASES, STATUS_COLORS, STATUS_LABELS,
    get_current_week, get_week_start_end, get_current_phase, get_phase_info,
    get_days_to_race, evaluate_metric,
)

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(BASE_DIR, 'garmin_data')
PROGRESS_LOG = os.path.join(BASE_DIR, 'progress_log.csv')
PROGRESS_HTML = os.path.join(BASE_DIR, 'progress_dashboard.html')

# ─── Data Sync ──────────────────────────────────────────────────────────────

def run_sync():
    """Run the full data sync pipeline: activities -> health -> dashboard."""
    print("=" * 60)
    print("  STEP 1: Syncing Garmin Data")
    print("=" * 60)

    scripts = [
        ('garmin_fixed.py', 'Downloading activities...'),
        ('garmin_health_fixed.py', 'Downloading health data...'),
        ('analyze_and_dashboard.py', 'Generating base dashboard...'),
    ]

    for script, msg in scripts:
        script_path = os.path.join(BASE_DIR, script)
        if not os.path.exists(script_path):
            print(f"  WARNING: {script} not found, skipping")
            continue

        print(f"\n  >> {msg}")
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
            )
            if result.returncode == 0:
                # Print last few lines of output for status
                lines = result.stdout.strip().split('\n')
                for line in lines[-3:]:
                    print(f"     {line}")
                print(f"  OK: {script} completed")
            else:
                print(f"  ERROR: {script} failed (exit code {result.returncode})")
                if result.stderr:
                    for line in result.stderr.strip().split('\n')[-5:]:
                        print(f"     {line}")
        except subprocess.TimeoutExpired:
            print(f"  ERROR: {script} timed out (10 min)")
        except Exception as e:
            print(f"  ERROR: {script} - {e}")

    print()

# ─── Data Loading ───────────────────────────────────────────────────────────

def find_latest_file(pattern, exclude_pattern=None):
    """Find the most recently modified file matching a glob pattern."""
    files = glob.glob(os.path.join(DATA_DIR, pattern))
    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in os.path.basename(f)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_data():
    """Load master dataset and activities."""
    master_file = find_latest_file('master_dataset_*.csv', exclude_pattern='features')
    activities_file = find_latest_file('activities_*.csv')

    if not master_file:
        print(f"  ERROR: No master_dataset_*.csv found in {DATA_DIR}")
        return None, None
    if not activities_file:
        print(f"  ERROR: No activities_*.csv found in {DATA_DIR}")
        return None, None

    print(f"  Loading: {os.path.basename(master_file)}")
    print(f"  Loading: {os.path.basename(activities_file)}")

    df = pd.read_csv(master_file, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    act = pd.read_csv(activities_file, parse_dates=['datum'])
    act = act.sort_values('datum').reset_index(drop=True)

    return df, act

# ─── Weekly Metrics Computation ─────────────────────────────────────────────

def compute_weekly_metrics(df, act, week_num):
    """Compute actual metrics for a specific training week."""
    week_start, week_end = get_week_start_end(week_num)

    # Convert to datetime for comparison
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(week_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # Filter activities for this week (running only for most metrics)
    week_act = act[(act['datum'] >= ws) & (act['datum'] <= we)].copy()
    week_runs = week_act[week_act['typ'] == 'running'].copy()

    # Filter master data for this week
    week_df = df[(df['date'] >= ws) & (df['date'] <= we)].copy()

    metrics = {
        'week': week_num,
        'week_start': week_start.strftime('%Y-%m-%d'),
        'week_end': week_end.strftime('%Y-%m-%d'),
        'phase': get_current_phase(week_num),
        'computed_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    # Weekly distance (all activities)
    metrics['weekly_km'] = round(week_runs['vzdalenost_km'].sum(), 1) if len(week_runs) > 0 else 0

    # Number of runs
    metrics['runs_count'] = len(week_runs)

    # Longest run
    metrics['longest_run_km'] = round(week_runs['vzdalenost_km'].max(), 1) if len(week_runs) > 0 else 0

    # Average easy-run HR (runs with pace > 5:30, i.e., slower/easy)
    easy_runs = week_runs[(week_runs['tempo_min_km'] > 5.5) & (week_runs['prumer_tep'] > 0)]
    metrics['avg_easy_hr'] = round(easy_runs['prumer_tep'].mean(), 0) if len(easy_runs) > 0 else None

    # Average cadence
    cadence_runs = week_runs[week_runs['avg_cadence'].notna() & (week_runs['avg_cadence'] > 0)]
    metrics['avg_cadence'] = round(cadence_runs['avg_cadence'].mean(), 0) if len(cadence_runs) > 0 else None

    # % easy runs (avg HR < 155)
    if len(week_runs) > 0:
        hr_runs = week_runs[week_runs['prumer_tep'] > 0]
        if len(hr_runs) > 0:
            easy_count = len(hr_runs[hr_runs['prumer_tep'] < 155])
            metrics['pct_easy_runs'] = round(easy_count / len(hr_runs) * 100, 0)
        else:
            metrics['pct_easy_runs'] = None
    else:
        metrics['pct_easy_runs'] = None

    # VO2max (latest reading this week)
    if 'vo2MaxPrecise' in week_df.columns:
        vo2_vals = week_df['vo2MaxPrecise'].dropna()
        metrics['vo2max'] = round(vo2_vals.iloc[-1], 1) if len(vo2_vals) > 0 else None
    else:
        metrics['vo2max'] = None

    # Average readiness score
    if 'readinessScore' in week_df.columns:
        readiness_vals = week_df['readinessScore'].dropna()
        metrics['avg_readiness'] = round(readiness_vals.mean(), 0) if len(readiness_vals) > 0 else None
    else:
        metrics['avg_readiness'] = None

    # Average HRV
    if 'hrvLastNightAvg' in week_df.columns:
        hrv_vals = week_df['hrvLastNightAvg'].dropna()
        metrics['avg_hrv'] = round(hrv_vals.mean(), 0) if len(hrv_vals) > 0 else None
    else:
        metrics['avg_hrv'] = None

    # Average sleep score
    if 'sleepScore' in week_df.columns:
        sleep_vals = week_df['sleepScore'].dropna()
        metrics['avg_sleep_score'] = round(sleep_vals.mean(), 0) if len(sleep_vals) > 0 else None
    else:
        metrics['avg_sleep_score'] = None

    # Resting HR (weekly average)
    if 'restingHeartRate' in week_df.columns:
        rhr_vals = week_df['restingHeartRate'].dropna()
        metrics['avg_resting_hr'] = round(rhr_vals.mean(), 0) if len(rhr_vals) > 0 else None
    else:
        metrics['avg_resting_hr'] = None

    return metrics


def evaluate_week(metrics, week_num):
    """Compare actual metrics against milestone targets. Returns list of (metric_name, actual, target, status, pct)."""
    milestone = MILESTONES.get(week_num)
    if not milestone:
        return []

    evaluations = []

    # Weekly km (range target)
    km_min = milestone['target_weekly_km_min']
    km_max = milestone['target_weekly_km_max']
    status, pct = evaluate_metric(metrics.get('weekly_km'), (km_min, km_max), "range")
    evaluations.append({
        'metric': 'Weekly Distance',
        'actual': f"{metrics.get('weekly_km', 0):.1f} km",
        'target': f"{km_min}-{km_max} km",
        'status': status,
        'pct': pct,
    })

    # Runs per week
    status, pct = evaluate_metric(metrics.get('runs_count'), milestone['target_runs_per_week'], "higher_is_better")
    evaluations.append({
        'metric': 'Runs This Week',
        'actual': str(metrics.get('runs_count', 0)),
        'target': str(milestone['target_runs_per_week']),
        'status': status,
        'pct': pct,
    })

    # Longest run
    status, pct = evaluate_metric(metrics.get('longest_run_km'), milestone['target_long_run_km'], "higher_is_better")
    evaluations.append({
        'metric': 'Longest Run',
        'actual': f"{metrics.get('longest_run_km', 0):.1f} km",
        'target': f"{milestone['target_long_run_km']} km",
        'status': status,
        'pct': pct,
    })

    # Easy HR (lower is better)
    if milestone['expected_easy_hr'] and metrics.get('avg_easy_hr'):
        status, pct = evaluate_metric(metrics['avg_easy_hr'], milestone['expected_easy_hr'], "lower_is_better")
        evaluations.append({
            'metric': 'Avg Easy HR',
            'actual': f"{metrics['avg_easy_hr']:.0f} bpm",
            'target': f"< {milestone['expected_easy_hr']} bpm",
            'status': status,
            'pct': pct,
        })

    # VO2max
    if metrics.get('vo2max'):
        status, pct = evaluate_metric(metrics['vo2max'], milestone['expected_vo2max'], "higher_is_better")
        evaluations.append({
            'metric': 'VO2max',
            'actual': f"{metrics['vo2max']:.1f}",
            'target': f"{milestone['expected_vo2max']:.1f}",
            'status': status,
            'pct': pct,
        })

    # Readiness
    if metrics.get('avg_readiness'):
        status, pct = evaluate_metric(metrics['avg_readiness'], milestone['expected_readiness_avg'], "higher_is_better")
        evaluations.append({
            'metric': 'Avg Readiness',
            'actual': f"{metrics['avg_readiness']:.0f}",
            'target': f"{milestone['expected_readiness_avg']}",
            'status': status,
            'pct': pct,
        })

    # Cadence
    if metrics.get('avg_cadence'):
        status, pct = evaluate_metric(metrics['avg_cadence'], milestone['expected_cadence'], "higher_is_better")
        evaluations.append({
            'metric': 'Avg Cadence',
            'actual': f"{metrics['avg_cadence']:.0f} spm",
            'target': f"{milestone['expected_cadence']} spm",
            'status': status,
            'pct': pct,
        })

    return evaluations

# ─── Progress Log ───────────────────────────────────────────────────────────

def append_progress_log(metrics):
    """Append weekly metrics to progress_log.csv (creates file if needed)."""
    file_exists = os.path.exists(PROGRESS_LOG)

    fieldnames = [
        'week', 'week_start', 'week_end', 'phase', 'computed_at',
        'weekly_km', 'runs_count', 'longest_run_km', 'avg_easy_hr',
        'avg_cadence', 'pct_easy_runs', 'vo2max', 'avg_readiness',
        'avg_hrv', 'avg_sleep_score', 'avg_resting_hr',
    ]

    # Check if this week already has an entry today
    if file_exists:
        existing = pd.read_csv(PROGRESS_LOG)
        today_str = date.today().strftime('%Y-%m-%d')
        same_week_today = existing[
            (existing['week'] == metrics['week']) &
            (existing['computed_at'].str.startswith(today_str))
        ]
        if len(same_week_today) > 0:
            print(f"  Progress log already has entry for week {metrics['week']} today, updating...")
            # Remove old entry for this week+today
            existing = existing[~(
                (existing['week'] == metrics['week']) &
                (existing['computed_at'].str.startswith(today_str))
            )]
            existing.to_csv(PROGRESS_LOG, index=False)

    with open(PROGRESS_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists or os.path.getsize(PROGRESS_LOG) == 0:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"  Saved to progress_log.csv")

# ─── Coaching Notes Auto-Generation ────────────────────────────────────────

def generate_coaching_notes(metrics, evaluations, week_num):
    """Generate auto coaching notes based on actual vs target comparison."""
    notes = []
    milestone = MILESTONES.get(week_num, {})

    # What's going well
    good = [e for e in evaluations if e['status'] in ('on_track', 'ahead')]
    if good:
        good_names = ', '.join(e['metric'] for e in good)
        notes.append(f"Going well: {good_names}")

    # What needs attention
    behind = [e for e in evaluations if e['status'] == 'behind']
    if behind:
        for e in behind:
            notes.append(f"Needs attention: {e['metric']} is at {e['actual']} (target: {e['target']})")

    warning = [e for e in evaluations if e['status'] == 'warning']
    if warning:
        for e in warning:
            notes.append(f"Watch: {e['metric']} slightly below target ({e['actual']} vs {e['target']})")

    # Phase-specific advice
    phase = milestone.get('phase', '')
    if 'Aerobic Reset' in phase:
        notes.append("Phase focus: Keep ALL runs easy. No temptation to push hard. Build the base.")
    elif 'Build' in phase:
        notes.append("Phase focus: One quality session per week max. Easy days must be truly easy.")
    elif 'Peak' in phase:
        notes.append("Phase focus: Big volume weeks. Prioritize sleep and nutrition for recovery.")
    elif 'Taper' in phase:
        notes.append("Phase focus: Trust your fitness. Volume is down but intensity stays. Stay sharp, stay fresh.")

    # Key workout reminder
    if milestone.get('key_workout'):
        notes.append(f"Key workout: {milestone['key_workout']}")

    # Focus notes from plan
    if milestone.get('focus_notes'):
        notes.append(f"Coach says: {milestone['focus_notes']}")

    return notes

# ─── Progress Dashboard HTML ───────────────────────────────────────────────

def build_progress_dashboard(all_weekly_metrics, all_evaluations, current_week):
    """Build a standalone progress tracking HTML dashboard."""

    today = date.today()
    days_to_race = get_days_to_race(today)
    phase = get_current_phase(current_week)
    phase_info = get_phase_info(phase) if phase else None
    phase_color = phase_info['color'] if phase_info else '#666'

    # Current week's data
    current_metrics = all_weekly_metrics.get(current_week, {})
    current_evals = all_evaluations.get(current_week, [])
    coaching_notes = generate_coaching_notes(current_metrics, current_evals, current_week)

    # Build milestone cards HTML
    cards_html = ""
    for ev in current_evals:
        color = STATUS_COLORS.get(ev['status'], '#666')
        label = STATUS_LABELS.get(ev['status'], 'N/A')
        cards_html += f"""
        <div style="background:#2a2a3e;border-radius:12px;padding:16px 20px;border-left:4px solid {color};flex:1;min-width:180px;">
            <div style="color:#a0a0b0;font-size:11px;text-transform:uppercase;letter-spacing:1px;">{ev['metric']}</div>
            <div style="color:#e0e0e0;font-size:24px;font-weight:700;margin:6px 0;">{ev['actual']}</div>
            <div style="color:#a0a0b0;font-size:12px;">Target: {ev['target']}</div>
            <div style="margin-top:8px;">
                <span style="background:{color};color:#fff;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:600;">{label}</span>
            </div>
        </div>"""

    # Build 16-week timeline HTML
    timeline_dots = ""
    for w in range(1, 17):
        w_metrics = all_weekly_metrics.get(w, {})
        w_evals = all_evaluations.get(w, [])
        w_start, w_end = get_week_start_end(w)

        # Overall week status: worst status of all metrics
        if w_evals:
            statuses = [e['status'] for e in w_evals]
            if 'behind' in statuses:
                dot_color = STATUS_COLORS['behind']
            elif 'warning' in statuses:
                dot_color = STATUS_COLORS['warning']
            elif 'ahead' in statuses:
                dot_color = STATUS_COLORS['ahead']
            else:
                dot_color = STATUS_COLORS['on_track']
        else:
            dot_color = STATUS_COLORS['no_data']

        is_current = w == current_week
        dot_size = 32 if is_current else 24
        border = "3px solid #fff" if is_current else "2px solid #3a3a4e"
        w_phase = MILESTONES[w]['phase']

        timeline_dots += f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px;" title="Week {w}: {w_phase} ({w_start} - {w_end})">
            <div style="width:{dot_size}px;height:{dot_size}px;border-radius:50%;background:{dot_color};
                        border:{border};display:flex;align-items:center;justify-content:center;
                        font-size:10px;font-weight:700;color:#fff;cursor:pointer;">{w}</div>
            <div style="font-size:9px;color:#a0a0b0;">{w_phase[:3]}</div>
        </div>"""

    # Build trend data for charts (actual vs expected)
    weeks_with_data = sorted(all_weekly_metrics.keys())
    trend_weeks = []
    trend_actual_km = []
    trend_expected_km_mid = []
    trend_actual_vo2 = []
    trend_expected_vo2 = []
    trend_actual_hr = []
    trend_expected_hr = []
    trend_actual_cadence = []
    trend_expected_cadence = []

    for w in range(1, 17):
        m = all_weekly_metrics.get(w, {})
        ms = MILESTONES[w]
        trend_weeks.append(w)
        trend_actual_km.append(m.get('weekly_km', None))
        trend_expected_km_mid.append((ms['target_weekly_km_min'] + ms['target_weekly_km_max']) / 2)
        trend_actual_vo2.append(m.get('vo2max', None))
        trend_expected_vo2.append(ms['expected_vo2max'])
        trend_actual_hr.append(m.get('avg_easy_hr', None))
        trend_expected_hr.append(ms['expected_easy_hr'])
        trend_actual_cadence.append(m.get('avg_cadence', None))
        trend_expected_cadence.append(ms['expected_cadence'])

    # Convert to JSON-safe strings
    def to_js_array(arr):
        return '[' + ','.join('null' if v is None else str(v) for v in arr) + ']'

    # Training distribution pie (this week)
    easy_pct = current_metrics.get('pct_easy_runs', 0) or 0
    hard_pct = 100 - easy_pct if easy_pct else 0

    # Coaching notes HTML
    notes_html = ""
    for note in coaching_notes:
        icon = ""
        if note.startswith("Going well"):
            icon = "style='border-left:3px solid #34a853;'"
        elif note.startswith("Needs attention"):
            icon = "style='border-left:3px solid #ea4335;'"
        elif note.startswith("Watch"):
            icon = "style='border-left:3px solid #fbbc04;'"
        elif note.startswith("Phase focus"):
            icon = "style='border-left:3px solid #4285f4;'"
        elif note.startswith("Key workout"):
            icon = "style='border-left:3px solid #9c27b0;'"
        else:
            icon = "style='border-left:3px solid #666;'"

        notes_html += f"""
        <div {icon} style="background:#313145;padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;
                          color:#e0e0e0;font-size:13px;line-height:1.5;border-left-width:3px;border-left-style:solid;">
            {note}
        </div>"""

    # Week-by-week log table
    log_rows = ""
    for w in weeks_with_data:
        m = all_weekly_metrics[w]
        ms = MILESTONES.get(w, {})
        row_bg = "#2a2a3e" if w % 2 == 0 else "#252538"
        current_marker = " font-weight:700;color:#4285f4;" if w == current_week else ""
        log_rows += f"""
        <tr style="background:{row_bg};{current_marker}">
            <td style="padding:8px 12px;">{w}</td>
            <td style="padding:8px 12px;">{m.get('phase', '-')}</td>
            <td style="padding:8px 12px;">{m.get('weekly_km', 0):.1f} / {ms.get('target_weekly_km_min', '-')}-{ms.get('target_weekly_km_max', '-')}</td>
            <td style="padding:8px 12px;">{m.get('runs_count', 0)} / {ms.get('target_runs_per_week', '-')}</td>
            <td style="padding:8px 12px;">{m.get('longest_run_km', 0):.1f} / {ms.get('target_long_run_km', '-')}</td>
            <td style="padding:8px 12px;">{m.get('avg_easy_hr', '-') if m.get('avg_easy_hr') else '-'}</td>
            <td style="padding:8px 12px;">{m.get('vo2max', '-') if m.get('vo2max') else '-'}</td>
            <td style="padding:8px 12px;">{m.get('avg_cadence', '-') if m.get('avg_cadence') else '-'}</td>
            <td style="padding:8px 12px;">{m.get('avg_readiness', '-') if m.get('avg_readiness') else '-'}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Garmin PRO Trener - Progress Tracker</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e2e;
            color: #e0e0e0;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e, #0d47a1);
            padding: 28px 40px;
            border-bottom: 3px solid {phase_color};
        }}
        .header h1 {{ font-size: 26px; font-weight: 700; }}
        .header .subtitle {{ color: #a0a0b0; font-size: 14px; margin-top: 4px; }}
        .countdown {{
            display: flex; gap: 24px; margin-top: 14px; flex-wrap: wrap;
        }}
        .countdown-item {{
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .countdown-item .num {{
            font-size: 28px; font-weight: 700; color: #fff;
        }}
        .countdown-item .label {{
            font-size: 11px; color: #a0a0b0; text-transform: uppercase; letter-spacing: 1px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 40px; }}
        .section {{ margin-bottom: 36px; }}
        .section-title {{
            font-size: 18px; font-weight: 600; margin-bottom: 14px;
            padding-bottom: 8px; border-bottom: 2px solid #3a3a4e;
        }}
        .cards {{ display: flex; gap: 14px; flex-wrap: wrap; }}
        .timeline {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-end; justify-content: center; padding: 20px 0; }}
        .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 900px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .container {{ padding: 20px; }}
            .cards {{ flex-direction: column; }}
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #313145; padding: 10px 12px; text-align: left; font-weight: 600; color: #a0a0b0;
             text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }}
        td {{ border-bottom: 1px solid #3a3a4e; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Half Marathon Progress Tracker</h1>
        <div class="subtitle">Target: {TARGET_TIME} ({TARGET_PACE} min/km) on {RACE_DATE.strftime('%d %b %Y')}</div>
        <div class="countdown">
            <div class="countdown-item">
                <div class="num">{days_to_race}</div>
                <div class="label">Days to Race</div>
            </div>
            <div class="countdown-item">
                <div class="num">{current_week} / 16</div>
                <div class="label">Current Week</div>
            </div>
            <div class="countdown-item" style="border-left:3px solid {phase_color};">
                <div class="num" style="font-size:20px;">{phase or 'Pre-plan'}</div>
                <div class="label">Current Phase</div>
            </div>
            <div class="countdown-item">
                <div class="num" style="font-size:16px;">{MILESTONES.get(current_week, {}).get('key_workout', '-')}</div>
                <div class="label">Key Workout</div>
            </div>
        </div>
    </div>

    <div class="container">

        <!-- This Week's Targets -->
        <div class="section">
            <div class="section-title">This Week's Targets (Week {current_week})</div>
            <div class="cards">
                {cards_html}
            </div>
        </div>

        <!-- 16-Week Timeline -->
        <div class="section">
            <div class="section-title">16-Week Timeline</div>
            <div class="timeline">
                {timeline_dots}
            </div>
            <div style="display:flex;gap:16px;justify-content:center;margin-top:10px;font-size:11px;">
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{STATUS_COLORS['on_track']};vertical-align:middle;"></span> On Track</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{STATUS_COLORS['ahead']};vertical-align:middle;"></span> Ahead</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{STATUS_COLORS['warning']};vertical-align:middle;"></span> Warning</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{STATUS_COLORS['behind']};vertical-align:middle;"></span> Behind</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{STATUS_COLORS['no_data']};vertical-align:middle;"></span> No Data</span>
            </div>
        </div>

        <!-- Trend Charts -->
        <div class="section">
            <div class="section-title">Trend Charts (Actual vs Expected)</div>
            <div class="chart-grid">
                <div id="chart-km" style="background:#2a2a3e;border-radius:12px;padding:10px;"></div>
                <div id="chart-vo2" style="background:#2a2a3e;border-radius:12px;padding:10px;"></div>
                <div id="chart-hr" style="background:#2a2a3e;border-radius:12px;padding:10px;"></div>
                <div id="chart-cadence" style="background:#2a2a3e;border-radius:12px;padding:10px;"></div>
            </div>
        </div>

        <!-- Training Distribution -->
        <div class="section">
            <div class="section-title">Training Distribution (This Week)</div>
            <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
                <div id="chart-pie" style="background:#2a2a3e;border-radius:12px;padding:10px;width:350px;height:300px;"></div>
                <div style="background:#2a2a3e;border-radius:12px;padding:20px;flex:1;min-width:250px;">
                    <div style="font-size:14px;font-weight:600;margin-bottom:12px;">Week {current_week} Summary</div>
                    <div style="font-size:13px;line-height:2;color:#a0a0b0;">
                        Total distance: <b style="color:#e0e0e0;">{current_metrics.get('weekly_km', 0):.1f} km</b><br>
                        Runs: <b style="color:#e0e0e0;">{current_metrics.get('runs_count', 0)}</b><br>
                        Longest run: <b style="color:#e0e0e0;">{current_metrics.get('longest_run_km', 0):.1f} km</b><br>
                        Easy runs: <b style="color:#e0e0e0;">{easy_pct:.0f}%</b><br>
                        Avg HRV: <b style="color:#e0e0e0;">{current_metrics.get('avg_hrv', '-')}</b><br>
                        Avg Sleep: <b style="color:#e0e0e0;">{current_metrics.get('avg_sleep_score', '-')}</b><br>
                        Resting HR: <b style="color:#e0e0e0;">{current_metrics.get('avg_resting_hr', '-')}</b>
                    </div>
                </div>
            </div>
        </div>

        <!-- Coaching Notes -->
        <div class="section">
            <div class="section-title">Coaching Notes</div>
            {notes_html}
        </div>

        <!-- Week-by-Week Log -->
        <div class="section">
            <div class="section-title">Week-by-Week Log (Actual / Target)</div>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Week</th><th>Phase</th><th>Distance (km)</th><th>Runs</th>
                            <th>Long Run (km)</th><th>Easy HR</th><th>VO2max</th><th>Cadence</th><th>Readiness</th>
                        </tr>
                    </thead>
                    <tbody>
                        {log_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div style="text-align:center;padding:30px 0;color:#a0a0b0;font-size:12px;">
            Generated by Garmin PRO Trener | {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>

    <script>
        var layout_base = {{
            template: 'plotly_dark',
            paper_bgcolor: '#2a2a3e',
            plot_bgcolor: '#2a2a3e',
            font: {{ color: '#e0e0e0', size: 11 }},
            margin: {{ l: 50, r: 30, t: 40, b: 40 }},
            legend: {{ orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', font: {{ size: 10 }} }},
            xaxis: {{ title: 'Week', dtick: 1, gridcolor: '#3a3a4e' }},
            yaxis: {{ gridcolor: '#3a3a4e' }},
        }};

        var weeks = {to_js_array(trend_weeks)};

        // Weekly km chart
        Plotly.newPlot('chart-km', [
            {{ x: weeks, y: {to_js_array(trend_actual_km)}, name: 'Actual km', mode: 'lines+markers',
               line: {{ color: '#1a73e8', width: 2.5 }}, marker: {{ size: 7 }}, connectgaps: false }},
            {{ x: weeks, y: {to_js_array(trend_expected_km_mid)}, name: 'Target km', mode: 'lines',
               line: {{ color: '#34a853', width: 2, dash: 'dash' }} }}
        ], Object.assign({{}}, layout_base, {{ title: 'Weekly Distance (km)', yaxis: {{ title: 'km', gridcolor: '#3a3a4e' }} }}, {{ height: 320 }}));

        // VO2max chart
        Plotly.newPlot('chart-vo2', [
            {{ x: weeks, y: {to_js_array(trend_actual_vo2)}, name: 'Actual VO2max', mode: 'lines+markers',
               line: {{ color: '#ea4335', width: 2.5 }}, marker: {{ size: 7 }}, connectgaps: false }},
            {{ x: weeks, y: {to_js_array(trend_expected_vo2)}, name: 'Target VO2max', mode: 'lines',
               line: {{ color: '#34a853', width: 2, dash: 'dash' }} }}
        ], Object.assign({{}}, layout_base, {{ title: 'VO2max Trend', yaxis: {{ title: 'ml/kg/min', gridcolor: '#3a3a4e' }} }}, {{ height: 320 }}));

        // Easy HR chart (inverted - lower is better)
        Plotly.newPlot('chart-hr', [
            {{ x: weeks, y: {to_js_array(trend_actual_hr)}, name: 'Actual Easy HR', mode: 'lines+markers',
               line: {{ color: '#ff9800', width: 2.5 }}, marker: {{ size: 7 }}, connectgaps: false }},
            {{ x: weeks, y: {to_js_array(trend_expected_hr)}, name: 'Target Easy HR', mode: 'lines',
               line: {{ color: '#34a853', width: 2, dash: 'dash' }} }}
        ], Object.assign({{}}, layout_base, {{ title: 'Easy-Run Heart Rate (lower = better)',
            yaxis: {{ title: 'bpm', gridcolor: '#3a3a4e', autorange: 'reversed' }} }}, {{ height: 320 }}));

        // Cadence chart
        Plotly.newPlot('chart-cadence', [
            {{ x: weeks, y: {to_js_array(trend_actual_cadence)}, name: 'Actual Cadence', mode: 'lines+markers',
               line: {{ color: '#9c27b0', width: 2.5 }}, marker: {{ size: 7 }}, connectgaps: false }},
            {{ x: weeks, y: {to_js_array(trend_expected_cadence)}, name: 'Target Cadence', mode: 'lines',
               line: {{ color: '#34a853', width: 2, dash: 'dash' }} }}
        ], Object.assign({{}}, layout_base, {{ title: 'Running Cadence (spm)', yaxis: {{ title: 'spm', gridcolor: '#3a3a4e' }} }}, {{ height: 320 }}));

        // Pie chart
        Plotly.newPlot('chart-pie', [{{
            values: [{easy_pct}, {hard_pct}],
            labels: ['Easy (<155 bpm)', 'Moderate/Hard (>155 bpm)'],
            type: 'pie', hole: 0.45,
            marker: {{ colors: ['#34a853', '#ea4335'] }},
            textinfo: 'label+percent',
            textfont: {{ size: 11 }},
        }}], {{
            paper_bgcolor: '#2a2a3e', font: {{ color: '#e0e0e0', size: 11 }},
            margin: {{ l: 20, r: 20, t: 20, b: 20 }}, showlegend: false, height: 280,
        }});
    </script>
</body>
</html>"""

    return html

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Garmin PRO Trener - Sync & Progress Tracker')
    parser.add_argument('--skip-sync', action='store_true', help='Skip data sync, just compute progress')
    parser.add_argument('--week', type=int, help='Show progress for a specific week')
    args = parser.parse_args()

    print("=" * 60)
    print("  Garmin PRO Trener - Sync & Progress Tracker")
    print("=" * 60)
    print()

    today = date.today()
    current_week = get_current_week(today)
    target_week = args.week if args.week else current_week

    print(f"  Today: {today}")
    print(f"  Current week: {current_week} of 16")
    print(f"  Phase: {get_current_phase(current_week)}")
    print(f"  Days to race: {get_days_to_race(today)}")
    print()

    # Step 1: Sync data (unless skipped)
    if not args.skip_sync:
        run_sync()
    else:
        print("  Skipping data sync (--skip-sync)")
        print()

    # Step 2: Load data
    print("=" * 60)
    print("  STEP 2: Loading Data")
    print("=" * 60)

    df, act = load_data()
    if df is None or act is None:
        print("\n  Cannot proceed without data. Run data sync first.")
        return

    print(f"  Master dataset: {len(df)} days, Activities: {len(act)} records")
    print()

    # Step 3: Compute metrics for all weeks that have data
    print("=" * 60)
    print("  STEP 3: Computing Weekly Metrics")
    print("=" * 60)

    all_weekly_metrics = {}
    all_evaluations = {}

    # Compute for all weeks up to current (or specified)
    max_week = min(target_week, 16)
    for w in range(1, max_week + 1):
        week_start, week_end = get_week_start_end(w)
        # Only compute if we have any data in this range
        ws = pd.Timestamp(week_start)
        we = pd.Timestamp(week_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

        has_activities = len(act[(act['datum'] >= ws) & (act['datum'] <= we)]) > 0
        has_health = len(df[(df['date'] >= ws) & (df['date'] <= we)]) > 0

        if has_activities or has_health:
            metrics = compute_weekly_metrics(df, act, w)
            evals = evaluate_week(metrics, w)
            all_weekly_metrics[w] = metrics
            all_evaluations[w] = evals

            # Print summary
            statuses = [e['status'] for e in evals]
            status_str = ', '.join(f"{s}:{statuses.count(s)}" for s in set(statuses))
            print(f"  Week {w:2d} ({week_start}): {metrics.get('weekly_km', 0):.1f} km, "
                  f"{metrics.get('runs_count', 0)} runs | {status_str}")

    if not all_weekly_metrics:
        print("  No data found for any training week yet.")
        print("  Make sure you have activities within the plan dates.")
        return

    print()

    # Step 4: Save progress log
    print("=" * 60)
    print("  STEP 4: Saving Progress Log")
    print("=" * 60)

    if target_week in all_weekly_metrics:
        append_progress_log(all_weekly_metrics[target_week])
    print()

    # Step 5: Generate progress dashboard
    print("=" * 60)
    print("  STEP 5: Generating Progress Dashboard")
    print("=" * 60)

    html = build_progress_dashboard(all_weekly_metrics, all_evaluations, target_week)
    with open(PROGRESS_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    file_size_kb = os.path.getsize(PROGRESS_HTML) / 1024
    print(f"  Dashboard saved: {PROGRESS_HTML}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print()

    # Step 6: Print coaching summary
    print("=" * 60)
    print(f"  COACHING SUMMARY - Week {target_week}")
    print("=" * 60)

    if target_week in all_weekly_metrics:
        evals = all_evaluations.get(target_week, [])
        coaching = generate_coaching_notes(all_weekly_metrics[target_week], evals, target_week)
        for note in coaching:
            print(f"  - {note}")
    print()

    print("=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"\n  Open progress_dashboard.html in your browser to view progress.")
    print(f"  Open dashboard.html for the full training analytics dashboard.")


if __name__ == '__main__':
    main()
