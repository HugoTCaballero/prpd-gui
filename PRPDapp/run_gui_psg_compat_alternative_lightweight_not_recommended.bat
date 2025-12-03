@echo off
setlocal
REM Propósito: compatibilidad. Interfaz alternativa ligera (PySimpleGUI).
REM No recomendada; la interfaz oficial es PRPDapp.main (run_gui.bat)
pushd "%~dp0.."
set PYTHONUTF8=1
python -m PRPDapp.psg_main %*
popd
pause

