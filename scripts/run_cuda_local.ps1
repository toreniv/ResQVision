$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputsDir = Join-Path $ProjectRoot "outputs"
$FrontendDataDir = Join-Path $ProjectRoot "frontend\public\data"
$CudaSource = Join-Path $ProjectRoot "resqvision.cu"
$ExePath = Join-Path $OutputsDir "resqvision.exe"
$Converter = Join-Path $ProjectRoot "scripts\csv_to_json.py"
$RequiredCsv = @(
    "benchmark_results.csv",
    "risk_ranking.csv",
    "attention_stats.csv"
)
$RequiredJson = @(
    "benchmark_results.json",
    "risk_ranking.json",
    "attention_stats.json"
)

New-Item -ItemType Directory -Path $OutputsDir -Force | Out-Null
New-Item -ItemType Directory -Path $FrontendDataDir -Force | Out-Null

if (-not (Test-Path $CudaSource)) {
    throw "CUDA source not found: $CudaSource"
}

Write-Host "[INFO] Compiling CUDA source with nvcc..."
& nvcc -O2 $CudaSource -o $ExePath
if ($LASTEXITCODE -ne 0) {
    throw "nvcc compilation failed."
}

Write-Host "[INFO] Running local CUDA pipeline..."
Push-Location $OutputsDir
try {
    & $ExePath
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA executable failed."
    }
} finally {
    Pop-Location
}

foreach ($FileName in $RequiredCsv) {
    $CsvPath = Join-Path $OutputsDir $FileName
    if (-not (Test-Path $CsvPath)) {
        throw "Expected CSV was not generated: $CsvPath"
    }
}

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

Write-Host "[INFO] Converting CSV outputs to JSON..."
Push-Location $ProjectRoot
try {
    & $Python $Converter
    if ($LASTEXITCODE -ne 0) {
        throw "CSV to JSON conversion failed."
    }
} finally {
    Pop-Location
}

foreach ($FileName in $RequiredJson) {
    $JsonPath = Join-Path $FrontendDataDir $FileName
    if (-not (Test-Path $JsonPath)) {
        throw "Expected JSON was not generated: $JsonPath"
    }
}

Write-Host "[OK] Local CUDA data pipeline complete. JSON files are in frontend/public/data."
