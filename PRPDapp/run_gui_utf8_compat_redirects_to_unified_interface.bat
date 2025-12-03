@echo off
setlocal
REM Propósito: compatibilidad. Redirige a la GUI unificada (PRPDapp.main) vía main_utf8.
REM Úsalo solo si alguien abre esta variante; la interfaz recomendada es run_gui.bat
pushd "%~dp0.."
set PYTHONUTF8=1
python -m PRPDapp.main_utf8 %*
popd
pause

