param(
    [string]$Xml = "",      # Ruta a un archivo XML específico
    [string]$Folder = "",   # Carpeta con varios XML a procesar en batch
    [int]$KManual = 3,        # Valor de k a utilizar en modo manual (B3/B5)
    [string]$PhaseAlign = "auto"  # Alineación de fase: "auto" o un número
)

# Cambiar al directorio del script para que las rutas relativas funcionen
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# Asegurar que la carpeta de salida existe
if (-not (Test-Path "out")) {
    New-Item -ItemType Directory -Path "out" | Out-Null
}

# Construir lista de XMLs a procesar
$xmlList = @()
if ($Folder) {
    if (-not (Test-Path $Folder)) {
        Write-Host "Carpeta no encontrada: $Folder" -ForegroundColor Red
        exit
    }
    $xmlList = Get-ChildItem -Path $Folder -Filter *.xml | Sort-Object Name
    if (-not $xmlList) {
        Write-Host "No se encontraron archivos XML en la carpeta." -ForegroundColor Yellow
        exit
    }
} elseif ($Xml) {
    if (-not (Test-Path $Xml)) {
        Write-Host "Archivo XML no encontrado: $Xml" -ForegroundColor Red
        exit
    }
    $xmlList = @(Get-Item $Xml)
} else {
    Write-Host "Debe proporcionar -Xml <archivo> o -Folder <carpeta>" -ForegroundColor Red
    exit
}

# Determinar corrimiento de fase flag
if ($PhaseAlign -eq "auto") {
    $phaseFlag = @("--phase-align", "auto")
} else {
    $phaseFlag = @("--phase-align", $PhaseAlign)
}

# Procesar cada XML
$count = 0
foreach ($xmlItem in $xmlList) {
    $count++
    $xmlPath = $xmlItem.FullName
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($xmlPath)
    $isBatch = $Folder -ne ""
    if ($isBatch) {
        # Crear subcarpeta batch con índice
        $batchDir = Join-Path "out" "batch"
        if (-not (Test-Path $batchDir)) { New-Item -ItemType Directory -Path $batchDir | Out-Null }
        $subDir = Join-Path $batchDir ("{0:000}_" -f $count) + $stem
        if (-not (Test-Path $subDir)) { New-Item -ItemType Directory -Path $subDir | Out-Null }
        $basePrefix = Join-Path $subDir $stem
    } else {
        $basePrefix = Join-Path "out" $stem
    }
    # 1. Bloque 3 natural (HDBSCAN) con Plotly 3D
    $cmdB3Nat = @(
        "python", "bloque3.py", $xmlPath,
        "--mode", "natural",
        "--palette", "paper",
        "--alpha-base", "0.25",
        "--alpha-clusters", "0.85",
        "--sub-min-pct", "0.02",
        "--out-prefix", $basePrefix
    ) + $phaseFlag + @("--plotly-3d")
    & $cmdB3Nat
    # 2. Bloque 3 aligned (K manual)
    $cmdB3Al = @(
        "python", "bloque3.py", $xmlPath,
        "--mode", "aligned",
        "--k-use", $KManual,
        "--palette", "paper",
        "--alpha-base", "0.25",
        "--alpha-clusters", "0.85",
        "--sub-min-pct", "0.02",
        "--out-prefix", $basePrefix
    ) + $phaseFlag + @("--plotly-3d")
    & $cmdB3Al
    # 3. Bloque 4 (curvas y k_auto)
    $cmdB4 = @(
        "python", "bloque4.py",
        "--xml", $xmlPath,
        "--recluster-after-align",
        "--out-prefix", $basePrefix
    ) + $phaseFlag
    & $cmdB4
    # 4. Bloque 5 (tipificación y emparejado)
    $cmdB5 = @(
        "python", "bloque5.py", $xmlPath,
        "--sensor", "auto",
        "--tipo-tx", "seco",
        "--k-use", $KManual,
        "--allow-otras", "true",
        "--otras-min-score", "0.12",
        "--otras-cap", "0.25",
        "--subclusters",
        "--sub-min-pct", "0.02",
        "--palette", "paper",
        "--alpha-base", "0.25",
        "--alpha-clusters", "0.85",
        "--pair-max-phase-deg", "25",
        "--pair-max-y-ks", "0.25",
        "--pair-min-weight-ratio", "0.4",
        "--pair-miss-penalty", "0.15",
        "--out-prefix", $basePrefix,
        "--recluster-after-align"
    ) + $phaseFlag
    & $cmdB5
    # 5. Bloque 6 (reporte final) – no pasar --k-use
    # Bloque 6 (reporte final).  No se pasa --k-use; se pasa --k-manual para
    # mostrar la comparativa cuando se especifica KManual.  Ajustamos el
    # orden de los argumentos para claridad.
    $cmdB6 = @(
        "python", "bloque6.py",
        "--xml", $xmlPath,
        "--sensor", "auto",
        "--tipo-tx", "seco",
        "--allow-otras", "true",
        "--otras-min-score", "0.12",
        "--otras-cap", "0.25",
        "--subclusters",
        "--sub-min-pct", "0.02",
        "--embed-assets",
        "--no-pdf",
        "--out-prefix", $basePrefix
    ) + $phaseFlag
    # Añadir k_manual si positivo
    if ($KManual -gt 0) {
        $cmdB6 += @("--k-manual", $KManual)
    }
    $cmdB6 += @("--recluster-after-align")
    & $cmdB6
}