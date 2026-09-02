@echo off
title RollCall AI - Face Recognition Attendance System
echo ===================================================
echo     RollCall AI - Face Attendance System
echo ===================================================
echo.
echo Select Run Mode:
echo   [1] Launch Streamlit Web App (Dashboard, Enroll, Web Roll Call)
echo   [2] Launch Continuous Real-time Desktop Kiosk Mode (OpenCV 30 FPS)
echo   [3] Run System Test Suite
echo   [4] Seed Demo Data
echo.
set /p choice="Enter option (1/2/3/4) [Default 1]: "

if "%choice%"=="2" (
    echo Starting Kiosk Mode...
    python main.py kiosk
) else if "%choice%"=="3" (
    echo Running Tests...
    python main.py test
    pause
) else if "%choice%"=="4" (
    echo Seeding Demo Data...
    python main.py seed
    pause
) else (
    echo Starting Web App...
    python main.py web
)
