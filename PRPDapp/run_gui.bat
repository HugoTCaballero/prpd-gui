@echo off
setlocal
REM Ir al directorio raiz del repo (padre de PRPDapp)
pushd "%~dp0.."
set PYTHONUTF8=1
REM Ejecutar como modulo para que las importaciones de paquete funcionen
python -m PRPDapp.main %*
popd
pause
