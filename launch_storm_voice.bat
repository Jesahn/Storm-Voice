@echo off
TITLE Storm-Voice Core Launcher (Storm-Bot Engine)

echo ====================================================================
echo                   STORM-VOICE AI PLATFORM LAUNCHER                   
echo ====================================================================
echo [Storm Isolation] Locking all cache ^& temporary storage to D:\ Drive...

set "BASE_DIR=%~dp0"
set "TMPDIR=%BASE_DIR%tmp"
set "TEMP=%BASE_DIR%tmp"
set "TMP=%BASE_DIR%tmp"
set "PIP_CACHE_DIR=%BASE_DIR%.cache\pip"
set "HF_HOME=%BASE_DIR%.cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%BASE_DIR%.cache\huggingface"
set "TORCH_HOME=%BASE_DIR%.cache\torch"

if not exist "%BASE_DIR%tmp" mkdir "%BASE_DIR%tmp"
if not exist "%BASE_DIR%.cache" mkdir "%BASE_DIR%.cache"

echo [Storm System] Checking for lingering processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo [Storm System] Freeing port 8000 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

echo [Storm System] Starting Storm-Voice server on http://localhost:8000 ...
echo [Storm System] Connecting to LM Studio Gemma 4 E2B at http://localhost:1234 ...
echo.

"%BASE_DIR%.venv\Scripts\python.exe" -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

pause
