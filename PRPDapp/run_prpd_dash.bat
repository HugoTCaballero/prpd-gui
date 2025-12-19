@echo off
setlocal

set "PS1=%~dp0run_prpd_dash.ps1"

if not exist "%PS1%" (
  echo [ERROR] No encuentro "%PS1%".
  pause
  exit /b 1
)

REM Si arrastras un XML, va en %1; si no, el .ps1 preguntara
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" "%~1"

endlocal
