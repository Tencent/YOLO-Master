@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
python scripts\latest_main_smoke.py --repo-root "%~dp0..\.." %*
exit /b %ERRORLEVEL%
