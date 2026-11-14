@echo off
title Air Canvas - OpenCV
echo ========================================================
echo   Launching Air Canvas - OpenCV
echo   Pinch thumb & index to draw | Relax your hand!
echo ========================================================
python main.py
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
)

