# tests/run_profiles.ps1
param(
  [string]$XML_DIR = ".",
  [string]$OUT_DIR = "out",
  [string]$PYEXE   = "python"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OUT_DIR | Out-Null

# XML esperados en la raíz (ajusta si cambia)
$Xmls = @(
  "1superficial.xml",
  "1superficial2coronas.xml",
  "2cavidades.xml",
  "corona.xml",
  "doblecavidad.xml"
) | ForEach-Object { Join-Path $XML_DIR $_ }

$profiles = @(
  @{ Name="strict"; Args="--pair-max-phase-deg 5 --pair-min-weight-ratio 0.9 --pair-miss-penalty 0.8 --sub-min-pct 0.20 --summary-mode auto --pair-hard-thresholds" },
  @{ Name="lax";    Args="--pair-max-phase-deg 90 --pair-min-weight-ratio 0.1 --pair-miss-penalty 0.0 --sub-min-pct 0.02 --summary-mode auto" }
)

Write-Host "Python:" -NoNewline; & $PYEXE -V
Write-Host "Exe:" (& where.exe $PYEXE) -ForegroundColor DarkGray

foreach($x in $Xmls){
  if(!(Test-Path $x)){ throw "No existe $x en $XML_DIR" }
  $base = Split-Path $x -Leaf
  foreach($p in $profiles){
    $outPrefix = Join-Path $OUT_DIR ("{0}_{1}" -f [IO.Path]::GetFileNameWithoutExtension($base), $p.Name)
    $cmd = "$PYEXE bloque5.py `"$x`" --sensor auto --tipo-tx seco --subclusters $($p.Args) --out-prefix `"$outPrefix`""
    Write-Host "`n==> $cmd" -ForegroundColor Cyan
    & cmd /c $cmd
    if($LASTEXITCODE -ne 0){ throw "Fallo con $base ($($p.Name))" }
  }
}

function CountPairs($prefix){
  $f = "$prefix`_paired_sources.csv"
  if(Test-Path $f){ (Import-Csv $f).Count } else { 0 }
}

Write-Host "`n=== RESUMEN DE PARES (M) ===" -ForegroundColor Yellow
foreach($x in $Xmls){
  $stem = [IO.Path]::GetFileNameWithoutExtension((Split-Path $x -Leaf))
  $strict = Join-Path $OUT_DIR "${stem}_strict"
  $lax    = Join-Path $OUT_DIR "${stem}_lax"
  "{0,-25} M_strict={1} | M_lax={2}" -f $stem, (CountPairs $strict), (CountPairs $lax)
}
