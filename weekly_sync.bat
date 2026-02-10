@echo off
title Garmin PRO Trener - Weekly Sync
echo ============================================================
echo   Garmin PRO Trener - Weekly Sync ^& Progress Tracker
echo ============================================================
echo.

cd /d "%~dp0"

echo Running full sync pipeline...
echo.

python sync_and_track.py

echo.
echo ============================================================
echo   Sync complete! Opening dashboard...
echo ============================================================
echo.

if exist "progress_dashboard.html" (
    start "" "progress_dashboard.html"
)

echo Press any key to close...
pause >nul
