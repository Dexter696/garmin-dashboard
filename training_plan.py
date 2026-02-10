#!/usr/bin/env python3
"""
Garmin PRO Trener - 16-Week Half Marathon Training Plan
Target: Half Marathon at 5:00/km pace (1:45:00) on 2026-05-31

Defines weekly milestones, phases, and helper functions for progress tracking.
"""

from datetime import datetime, date, timedelta

# ─── Race Configuration ─────────────────────────────────────────────────────

RACE_DATE = date(2026, 5, 31)
PLAN_START = date(2026, 2, 9)  # Monday of week 1
TARGET_PACE = 5.00  # min/km
TARGET_TIME = "1:45:00"
RACE_DISTANCE = 21.1  # km

# ─── Phase Definitions ──────────────────────────────────────────────────────

PHASES = {
    "Aerobic Reset": {"weeks": [1, 2, 3, 4], "color": "#4CAF50", "description": "80% easy, build base, no hard sessions"},
    "Build": {"weeks": [5, 6, 7, 8, 9], "color": "#FF9800", "description": "1 quality session/week, build to 38 km/week"},
    "Peak": {"weeks": [10, 11, 12, 13], "color": "#F44336", "description": "Peak at 45 km, long runs to 20 km"},
    "Taper": {"weeks": [14, 15, 16], "color": "#2196F3", "description": "Volume down 50%, maintain intensity, race"},
}

# ─── 16-Week Milestones ────────────────────────────────────────────────────

MILESTONES = {
    1: {
        "phase": "Aerobic Reset",
        "target_weekly_km_min": 22,
        "target_weekly_km_max": 25,
        "target_runs_per_week": 4,
        "target_long_run_km": 10,
        "expected_easy_hr": 158,
        "expected_vo2max": 51.0,
        "expected_readiness_avg": 45,
        "expected_cadence": 155,
        "key_workout": "All easy runs + strides",
        "focus_notes": "All easy, strides only. Focus on consistent easy running to rebuild aerobic base. Keep HR below 158.",
    },
    2: {
        "phase": "Aerobic Reset",
        "target_weekly_km_min": 24,
        "target_weekly_km_max": 27,
        "target_runs_per_week": 4,
        "target_long_run_km": 11,
        "expected_easy_hr": 157,
        "expected_vo2max": 51.0,
        "expected_readiness_avg": 47,
        "expected_cadence": 156,
        "key_workout": "Easy runs + strength 2x",
        "focus_notes": "All easy, strength 2x/week. Add bodyweight strength sessions to complement running.",
    },
    3: {
        "phase": "Aerobic Reset",
        "target_weekly_km_min": 26,
        "target_weekly_km_max": 30,
        "target_runs_per_week": 4,
        "target_long_run_km": 12,
        "expected_easy_hr": 156,
        "expected_vo2max": 51.0,
        "expected_readiness_avg": 48,
        "expected_cadence": 157,
        "key_workout": "Easy runs + cadence drills",
        "focus_notes": "All easy, cadence drills. Gradually increase cadence through focused drills and short strides.",
    },
    4: {
        "phase": "Aerobic Reset",
        "target_weekly_km_min": 20,
        "target_weekly_km_max": 23,
        "target_runs_per_week": 3,
        "target_long_run_km": 10,
        "expected_easy_hr": 155,
        "expected_vo2max": 51.5,
        "expected_readiness_avg": 50,
        "expected_cadence": 158,
        "key_workout": "Recovery week",
        "focus_notes": "Recovery week. Reduce volume by ~25%. Let body absorb 3 weeks of base building.",
    },
    5: {
        "phase": "Build",
        "target_weekly_km_min": 28,
        "target_weekly_km_max": 32,
        "target_runs_per_week": 4,
        "target_long_run_km": 13,
        "expected_easy_hr": 155,
        "expected_vo2max": 51.5,
        "expected_readiness_avg": 50,
        "expected_cadence": 159,
        "key_workout": "1st tempo session (20 min at 5:15/km)",
        "focus_notes": "First tempo session! 20 min at 5:15/km after thorough warmup. Keep other runs easy.",
    },
    6: {
        "phase": "Build",
        "target_weekly_km_min": 30,
        "target_weekly_km_max": 34,
        "target_runs_per_week": 4,
        "target_long_run_km": 14,
        "expected_easy_hr": 154,
        "expected_vo2max": 52.0,
        "expected_readiness_avg": 51,
        "expected_cadence": 160,
        "key_workout": "1st intervals (5x1000m at 4:45/km)",
        "focus_notes": "First interval session! 5x1000m at 4:45/km with 90s jog recovery. Easy tempo on another day.",
    },
    7: {
        "phase": "Build",
        "target_weekly_km_min": 32,
        "target_weekly_km_max": 36,
        "target_runs_per_week": 4,
        "target_long_run_km": 15,
        "expected_easy_hr": 153,
        "expected_vo2max": 52.0,
        "expected_readiness_avg": 52,
        "expected_cadence": 161,
        "key_workout": "Tempo progression (25 min at 5:10/km)",
        "focus_notes": "Tempo progression. Extend to 25 min at 5:10/km. Long run increases to 15 km.",
    },
    8: {
        "phase": "Build",
        "target_weekly_km_min": 25,
        "target_weekly_km_max": 28,
        "target_runs_per_week": 3,
        "target_long_run_km": 12,
        "expected_easy_hr": 153,
        "expected_vo2max": 52.0,
        "expected_readiness_avg": 53,
        "expected_cadence": 162,
        "key_workout": "Recovery week",
        "focus_notes": "Recovery week. Reduce volume. One easy tempo only. Let body absorb build phase gains.",
    },
    9: {
        "phase": "Build",
        "target_weekly_km_min": 34,
        "target_weekly_km_max": 38,
        "target_runs_per_week": 4,
        "target_long_run_km": 16,
        "expected_easy_hr": 152,
        "expected_vo2max": 52.5,
        "expected_readiness_avg": 53,
        "expected_cadence": 163,
        "key_workout": "Race-pace intro (3 km at 5:00/km in long run)",
        "focus_notes": "Race-pace intro! Include 3 km at 5:00/km inside your long run. Feel the target pace.",
    },
    10: {
        "phase": "Peak",
        "target_weekly_km_min": 36,
        "target_weekly_km_max": 40,
        "target_runs_per_week": 4,
        "target_long_run_km": 18,
        "expected_easy_hr": 151,
        "expected_vo2max": 52.5,
        "expected_readiness_avg": 52,
        "expected_cadence": 164,
        "key_workout": "Long tempo sections (2x15 min at 5:05/km)",
        "focus_notes": "Long tempo sections. 2x15 min at 5:05/km in a separate session. Long run 18 km with race-pace finish.",
    },
    11: {
        "phase": "Peak",
        "target_weekly_km_min": 38,
        "target_weekly_km_max": 42,
        "target_runs_per_week": 4,
        "target_long_run_km": 19,
        "expected_easy_hr": 150,
        "expected_vo2max": 53.0,
        "expected_readiness_avg": 51,
        "expected_cadence": 165,
        "key_workout": "6x1000m at 4:40/km",
        "focus_notes": "Key interval session: 6x1000m at 4:40/km. Long run 19 km. This is a big week - manage fatigue.",
    },
    12: {
        "phase": "Peak",
        "target_weekly_km_min": 40,
        "target_weekly_km_max": 45,
        "target_runs_per_week": 5,
        "target_long_run_km": 20,
        "expected_easy_hr": 150,
        "expected_vo2max": 53.0,
        "expected_readiness_avg": 50,
        "expected_cadence": 166,
        "key_workout": "20 km long run!",
        "focus_notes": "Peak week! 20 km long run with last 5 km at race pace. This is the hardest week - trust the training.",
    },
    13: {
        "phase": "Peak",
        "target_weekly_km_min": 30,
        "target_weekly_km_max": 33,
        "target_runs_per_week": 4,
        "target_long_run_km": 14,
        "expected_easy_hr": 149,
        "expected_vo2max": 53.0,
        "expected_readiness_avg": 53,
        "expected_cadence": 167,
        "key_workout": "Recovery - absorb fitness",
        "focus_notes": "Recovery week. Let body absorb peak phase. Easy running only with some strides. Eat and sleep well.",
    },
    14: {
        "phase": "Taper",
        "target_weekly_km_min": 30,
        "target_weekly_km_max": 34,
        "target_runs_per_week": 4,
        "target_long_run_km": 16,
        "expected_easy_hr": 149,
        "expected_vo2max": 53.0,
        "expected_readiness_avg": 55,
        "expected_cadence": 168,
        "key_workout": "Last hard session (4x1000m at 4:35/km)",
        "focus_notes": "Last hard session! 4x1000m at 4:35/km. After this, no more hard workouts. Trust the fitness.",
    },
    15: {
        "phase": "Taper",
        "target_weekly_km_min": 22,
        "target_weekly_km_max": 25,
        "target_runs_per_week": 3,
        "target_long_run_km": 12,
        "expected_easy_hr": 148,
        "expected_vo2max": 53.0,
        "expected_readiness_avg": 58,
        "expected_cadence": 168,
        "key_workout": "Short sharp intervals (6x400m)",
        "focus_notes": "Short sharp intervals to maintain turnover. 6x400m at fast pace. Everything else very easy and short.",
    },
    16: {
        "phase": "Taper",
        "target_weekly_km_min": 12,
        "target_weekly_km_max": 15,
        "target_runs_per_week": 2,
        "target_long_run_km": 21.1,  # RACE!
        "expected_easy_hr": None,  # Race week - no easy HR target
        "expected_vo2max": 53.0,
        "expected_readiness_avg": 60,
        "expected_cadence": 168,
        "key_workout": "RACE DAY!",
        "focus_notes": "RACE WEEK! 2-3 short easy shakeout runs before race. Hydrate, sleep, carb-load. Trust your training!",
    },
}

# ─── Day-by-Day Training Plan (Mon=0 .. Sun=6) ─────────────────────────────
# type: "rest", "easy", "long", "tempo", "intervals", "race"
# color: green=easy, red=long, yellow=quality, blue=rest

DAILY_PLAN = {
    # ── Phase 1: Aerobic Reset ──────────────────────────────────────────────
    1: [  # 25 km, 4 runs, all easy + strides
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Strength"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km + strides"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 10 km (easy)"},
    ],
    2: [  # 27 km, 4 runs, easy + strength
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Strength"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Strength"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km + strides"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 11 km (easy)"},
    ],
    3: [  # 30 km, 4 runs, cadence drills
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km + drills"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Strength"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 12 km (easy)"},
    ],
    4: [  # 20 km, 3 runs, RECOVERY WEEK
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "long",  "color": "#ea4335", "info": "Long 10 km (easy)"},
    ],
    # ── Phase 2: Build + Quality ────────────────────────────────────────────
    5: [  # 31 km, 4 runs, 1st tempo
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Strength"},
        {"type": "tempo", "color": "#fbbc04", "info": "Tempo 7 km (20' @ 5:15)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 13 km"},
    ],
    6: [  # 33 km, 4 runs, 1st intervals
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Strength"},
        {"type": "intervals", "color": "#fbbc04", "info": "Intervals 8 km (5x1000m @ 4:45)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 14 km"},
    ],
    7: [  # 34 km, 4 runs, tempo progression
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Strength"},
        {"type": "tempo", "color": "#fbbc04", "info": "Tempo 8 km (25' @ 5:10)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 15 km"},
    ],
    8: [  # 25 km, 3 runs, RECOVERY WEEK
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 7 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "long",  "color": "#ea4335", "info": "Long 12 km (easy)"},
    ],
    9: [  # 36 km, 4 runs, race-pace intro
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 7 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Strength"},
        {"type": "tempo", "color": "#fbbc04", "info": "Race-pace 8 km (3 km @ 5:00)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 16 km (last 3 km @ 5:00)"},
    ],
    # ── Phase 3: Race-Specific Peak ─────────────────────────────────────────
    10: [  # 38 km, 4 runs, long tempo sections
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 7 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Strength"},
        {"type": "tempo", "color": "#fbbc04", "info": "Tempo 8 km (2x15' @ 5:05)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 18 km (race-pace finish)"},
    ],
    11: [  # 40 km, 4 runs, hard interval session
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 7 km"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "intervals", "color": "#fbbc04", "info": "Intervals 9 km (6x1000m @ 4:40)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "long",  "color": "#ea4335", "info": "Long 19 km"},
    ],
    12: [  # 45 km, 5 runs, PEAK WEEK - 20 km long run!
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 7 km"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "tempo", "color": "#fbbc04", "info": "Tempo 8 km (30' @ 5:05)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 20 km (last 5 km @ 5:00)"},
    ],
    13: [  # 31 km, 4 runs, RECOVERY WEEK
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km + strides"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 14 km (easy)"},
    ],
    # ── Phase 4: Taper ──────────────────────────────────────────────────────
    14: [  # 34 km, 4 runs, last hard session
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 6 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "intervals", "color": "#fbbc04", "info": "Intervals 7 km (4x1000m @ 4:35)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "long",  "color": "#ea4335", "info": "Long 16 km"},
    ],
    15: [  # 23 km, 3 runs, short sharp
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 5 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "intervals", "color": "#fbbc04", "info": "Intervals 6 km (6x400m fast)"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "long",  "color": "#ea4335", "info": "Long 12 km (easy)"},
    ],
    16: [  # RACE WEEK
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Easy 4 km + strides"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "easy",  "color": "#34a853", "info": "Shakeout 3 km"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest"},
        {"type": "rest",  "color": "#4285f4", "info": "Rest / Carb-load"},
        {"type": "race",  "color": "#ff6d00", "info": "RACE 21.1 km!"},
    ],
}

# ─── Helper Functions ───────────────────────────────────────────────────────

def get_current_week(today=None):
    """Get the current training week number (1-16). Returns 0 if before plan, 17+ if after."""
    if today is None:
        today = date.today()
    if isinstance(today, datetime):
        today = today.date()
    delta = (today - PLAN_START).days
    if delta < 0:
        return 0
    return (delta // 7) + 1


def get_week_start_end(week_num):
    """Get the start (Monday) and end (Sunday) dates for a given week number."""
    start = PLAN_START + timedelta(weeks=week_num - 1)
    end = start + timedelta(days=6)
    return start, end


def get_current_phase(week_num=None):
    """Get the training phase for a given week number."""
    if week_num is None:
        week_num = get_current_week()
    if week_num < 1 or week_num > 16:
        return None
    return MILESTONES[week_num]["phase"]


def get_phase_info(phase_name):
    """Get phase metadata (color, weeks, description)."""
    return PHASES.get(phase_name)


def get_milestone(week_num):
    """Get milestone targets for a specific week."""
    return MILESTONES.get(week_num)


def get_days_to_race(today=None):
    """Get number of days until race day."""
    if today is None:
        today = date.today()
    if isinstance(today, datetime):
        today = today.date()
    return (RACE_DATE - today).days


def get_weeks_remaining(today=None):
    """Get number of full weeks remaining until race."""
    days = get_days_to_race(today)
    return max(0, days // 7)


def evaluate_metric(actual, target, metric_type="higher_is_better"):
    """
    Evaluate a metric against its target.

    metric_type:
        "higher_is_better" - e.g., weekly km, runs, cadence, VO2max, readiness
        "lower_is_better"  - e.g., easy HR, resting HR
        "range"            - actual should be within (target_min, target_max)

    Returns: (status, pct_of_target)
        status: "on_track", "ahead", "warning", "behind"
    """
    if actual is None or target is None:
        return "no_data", 0

    if metric_type == "lower_is_better":
        # For HR: being below target is good
        if actual <= target:
            return "on_track", 100
        elif actual <= target * 1.05:
            return "warning", round(target / actual * 100, 1)
        else:
            return "behind", round(target / actual * 100, 1)

    elif metric_type == "range":
        # target is a tuple (min, max)
        target_min, target_max = target
        mid = (target_min + target_max) / 2
        if target_min <= actual <= target_max:
            return "on_track", 100
        elif actual > target_max:
            if actual <= target_max * 1.15:
                return "ahead", round(actual / mid * 100, 1)
            return "ahead", round(actual / mid * 100, 1)
        elif actual >= target_min * 0.7:
            return "warning", round(actual / mid * 100, 1)
        else:
            return "behind", round(actual / mid * 100, 1)

    else:  # higher_is_better
        pct = round(actual / target * 100, 1) if target > 0 else 0
        if pct >= 110:
            return "ahead", pct
        elif pct >= 90:
            return "on_track", pct
        elif pct >= 70:
            return "warning", pct
        else:
            return "behind", pct


STATUS_COLORS = {
    "on_track": "#34a853",   # green
    "ahead": "#4285f4",      # blue
    "warning": "#fbbc04",    # yellow
    "behind": "#ea4335",     # red
    "no_data": "#666666",    # grey
}

STATUS_LABELS = {
    "on_track": "On Track",
    "ahead": "Ahead",
    "warning": "Warning",
    "behind": "Behind",
    "no_data": "No Data",
}


if __name__ == "__main__":
    print("=" * 60)
    print("  Garmin PRO Trener - 16-Week Training Plan")
    print("=" * 60)
    print()
    print(f"  Race: Half Marathon on {RACE_DATE}")
    print(f"  Target: {TARGET_TIME} ({TARGET_PACE} min/km)")
    print(f"  Plan start: {PLAN_START}")
    print(f"  Current week: {get_current_week()}")
    print(f"  Current phase: {get_current_phase()}")
    print(f"  Days to race: {get_days_to_race()}")
    print()

    for week in range(1, 17):
        m = MILESTONES[week]
        start, end = get_week_start_end(week)
        marker = " <-- YOU ARE HERE" if week == get_current_week() else ""
        print(f"  Week {week:2d} ({start} - {end}) | {m['phase']:15s} | "
              f"{m['target_weekly_km_min']}-{m['target_weekly_km_max']} km | "
              f"Long: {m['target_long_run_km']} km | {m['key_workout']}{marker}")
