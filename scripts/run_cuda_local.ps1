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
& nvcc -O2 -arch=sm_89 $CudaSource -o $ExePath
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
$PathPython = Get-Command python -ErrorAction SilentlyContinue
$PathPy = Get-Command py -ErrorAction SilentlyContinue
$PlatformIoPython = Join-Path "C:\Users\$env:USERNAME" ".platformio\python3\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} elseif ($PathPython) {
    $Python = $PathPython.Source
} elseif ($PathPy) {
    $Python = $PathPy.Source
} elseif (Test-Path $PlatformIoPython) {
    $Python = $PlatformIoPython
} else {
    throw @"
Python 3.10+ is required to convert CUDA CSV outputs to JSON.

Install Python and enable "Add python.exe to PATH".
Then rerun .\setup.ps1 and .\scripts\run_cuda_local.ps1.
"@
}

Write-Host "[INFO] Using Python executable: $Python"

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
