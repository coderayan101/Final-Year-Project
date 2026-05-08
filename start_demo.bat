@echo off
REM ============================================================
REM Multimodal Crop Disease Predictor - Demo Launcher
REM Double-click this file to start the full demo in 2 windows.
REM
REM Prerequisites:
REM   1. Phone hotspot ON ("OnePlus Nord CE4" / 12345678)
REM   2. Laptop connected to that hotspot
REM   3. ESP32 powered on (via power bank or USB)
REM ============================================================

echo.
echo ===========================================================
echo   Multimodal Crop Disease Predictor - Demo Launcher
echo ===========================================================
echo.
echo BEFORE CONTINUING, make sure:
echo   [1] Your phone hotspot is ON
echo   [2] Your laptop is connected to the hotspot
echo   [3] ESP32 is powered on (red LED visible)
echo.
echo Press any key when ready, or close this window to abort.
pause >nul

REM Move into project folder
cd /d "%~dp0"

echo.
echo [1/2] Starting Flask server in a new window...
start "Flask Server (Backend)" cmd /k "venv\Scripts\python.exe -m server.flask_app"

echo Waiting 45 seconds for Flask to load models...
timeout /t 45 /nobreak

echo.
echo [2/2] Starting Streamlit UI in a new window...
start "Streamlit UI (Frontend)" cmd /k "venv\Scripts\streamlit run ui/streamlit_app.py"

echo.
echo ===========================================================
echo   Demo started!
echo   Two terminal windows have opened:
echo     - Flask Server (keeps the backend running)
echo     - Streamlit UI (keeps the web app running)
echo
echo   Your browser will auto-open at http://localhost:8501
echo
echo   To STOP the demo: close both terminal windows.
echo ===========================================================
echo.
echo This launcher window can be closed now.
pause
