@echo off
setlocal

cd /d "%~dp0"
title DLU Dashboard

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Khong tim thay .venv
    pause
    exit /b 1
)

echo Dang don process Streamlit cu...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*-m streamlit run dashboard.py*' -or $_.CommandLine -like '*streamlit run dashboard.py*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>nul

echo Dang mo dashboard tai http://127.0.0.1:8511
start "DLU Dashboard" cmd /k ".venv\Scripts\activate.bat && python -m streamlit run dashboard.py"

timeout /t 3 >nul
start "" "http://127.0.0.1:8511"

echo Dashboard da duoc khoi dong trong cua so rieng.
pause
