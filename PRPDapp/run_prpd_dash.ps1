param(
  [string]$XmlPath
)

$ErrorActionPreference = "Stop"

# 1) Resolver ruta del script y PRPD
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Prpd = Join-Path $ScriptDir "PRPD_Dash.py"
if (-not (Test-Path $Prpd)) {
  Write-Host "[ERROR] No encuentro PRPD_Dash.py en $ScriptDir" -ForegroundColor Red
  Read-Host "Presiona Enter para salir"
  exit 1
}

# 2) Pedir XML si no vino como argumento
if (-not $XmlPath -or $XmlPath -eq "") {
  $XmlPath = Read-Host "Arrastra y suelta tu XML aquí o escribe su ruta completa"
}
# Limpiar comillas
$XmlPath = $XmlPath.Trim('"')

# 3) Normalizar a ruta absoluta
try {
  $XmlFull = (Resolve-Path -LiteralPath $XmlPath).Path
} catch {
  Write-Host "[ERROR] No encuentro '$XmlPath'." -ForegroundColor Red
  Read-Host "Presiona Enter para salir"
  exit 1
}

# 4) Comprobar Python del sistema
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
  Write-Host "[ERROR] No se encontró 'python' en PATH. Instálalo desde python.org" -ForegroundColor Red
  Read-Host "Presiona Enter para salir"
  exit 1
}

# 5) Crear/usar venv local
$VenvDir = Join-Path $ScriptDir ".venv"
$PyExe   = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $PyExe)) {
  Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
  & python -m venv $VenvDir
}

# 6) Instalar dependencias en el venv
Write-Host "Verificando dependencias en el venv (dash, plotly, numpy, kaleido)..." -ForegroundColor Cyan
& $PyExe -m pip install --upgrade pip | Out-Null
foreach ($pkg in @("dash","plotly","numpy","kaleido")) {
  $has = & $PyExe -m pip show $pkg 2>$null
  if (-not $has) { & $PyExe -m pip install $pkg }
}

# 7) Abrir navegador
Start-Process "http://127.0.0.1:8050/"

# 8) Ejecutar la app
Write-Host "Iniciando PRPD Dash (venv) con: $XmlFull" -ForegroundColor Green
& $PyExe $Prpd $XmlFull

Write-Host "`n[FIN] Cierra esta ventana si ya terminaste." -ForegroundColor Yellow
Read-Host "Presiona Enter para salir"
