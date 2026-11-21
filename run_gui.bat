@echo off
setlocal
cd /d %~dp0

if exist winsper_env\Scripts\python.exe (
  winsper_env\Scripts\python.exe gui.py
) else (
  echo Virtualenv not found: winsper_env
  echo Create it first: python -m venv winsper_env
  exit /b 1
)

