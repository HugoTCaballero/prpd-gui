@echo off
setlocal

REM === Rutas base ===
set "SCRIPT=%~dp0PRPD_Dash.py"
set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%SCRIPT%" (
  echo [ERROR] No encuentro "%SCRIPT%".
  echo Guarda PRPD_Dash.py en la misma carpeta que este .BAT.
  pause
  exit /b 1
)

REM === Tomar XML: por drag&drop o por teclado ===
if "%~1"=="" (
  echo Arrastra y suelta tu XML sobre este .BAT o escribe la ruta completa:
  set /p XML=Ruta del XML:
) else (
  set "XML=%~1"
)
REM Quitar comillas si las pegaron
set "XML=%XML:"=%"

if not exist "%XML%" (
  echo [ERROR] No encuentro "%XML%".
  pause
  exit /b 1
)

REM === Comprobar Python del sistema (para crear venv) ===
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] No se encontro "python" en PATH. Instala Python 3.10+ desde python.org.
  pause
  exit /b 1
)

REM === Crear venv si hace falta ===
if not exist "%PYTHON_EXE%" (
  echo Creando entorno virtual en "%VENV_DIR%"...
  python -m venv "%VENV_DIR%"
)

REM === Instalar/actualizar dependencias ===
echo Verificando dependencias en el venv \(dash, plotly, numpy, kaleido\)...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel >nul
"%PYTHON_EXE%" -m pip install dash plotly numpy kaleido

REM === Lanzar el navegador y la app ===
start "" http://127.0.0.1:8050/
echo Iniciando PRPD Dash \(venv\) con: "%XML%"
"%PYTHON_EXE%" "%SCRIPT%" "%XML%"

echo.
echo [FIN] Cierra esta ventana si ya terminaste.
pause

endlocal
