@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
python -m pytest tests -q
if errorlevel 1 exit /b %ERRORLEVEL%
python -m ruff check src tests scripts
exit /b %ERRORLEVEL%
