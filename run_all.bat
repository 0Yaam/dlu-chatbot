@echo off
setlocal

cd /d "%~dp0"
title DLU-Chatbot Launcher

echo ==========================================
echo   DLU-Chatbot - Run All
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Khong tim thay virtual environment tai .venv
    echo Hay tao moi truong ao truoc:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [ERROR] Khong tim thay file .env
    echo Hay tao file .env tu .env.example truoc khi chay.
    echo.
    pause
    exit /b 1
)

echo [1/3] Kiem tra dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Cai dat dependencies that bai.
    pause
    exit /b 1
)

echo.
echo Dang don cac process cu...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*uvicorn app.main:app*' -or $_.CommandLine -like '*streamlit run dashboard.py*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>nul

echo.
echo [2/3] Mo FastAPI backend trong cua so moi...
start "DLU Backend" cmd /k ".venv\Scripts\activate.bat && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [3/3] Mo Streamlit dashboard trong cua so moi...
start "DLU Dashboard" cmd /k ".venv\Scripts\activate.bat && python -m streamlit run dashboard.py"

echo.
echo Da khoi dong backend va dashboard.
echo - Backend:   http://127.0.0.1:8000/health
echo - Dashboard: http://127.0.0.1:8511
echo.
echo Ban co the dong cua so nay, cac cua so da mo van tiep tuc chay.
echo.
pause
