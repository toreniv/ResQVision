$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CheckCuda = Join-Path $ProjectRoot "scripts\check_cuda.ps1"
$RunLocal = Join-Path $ProjectRoot "scripts\run_cuda_local.ps1"
$ImportColab = Join-Path $ProjectRoot "scripts\import_colab_outputs.ps1"
$FuseYolo = Join-Path $ProjectRoot "scripts\fuse_yolo_to_tactical.py"
$DataDir = Join-Path $ProjectRoot "frontend\public\data"

Write-Host "ResQVision data pipeline"
Write-Host "------------------------"

& $CheckCuda
$CudaAvailable = ($LASTEXITCODE -eq 0)
$PipelineExitCode = 1
$CompletedThroughFallback = $false

if ($CudaAvailable) {
    Write-Host "[INFO] CUDA is available. Running local CUDA pipeline."
    & $RunLocal
    $PipelineExitCode = $LASTEXITCODE

    if ($PipelineExitCode -eq 0) {
        Write-Host "[OK] Data pipeline completed successfully through local CUDA."
    } else {
        Write-Host "[WARN] Local CUDA pipeline failed. Trying Colab ZIP import."
        & $ImportColab
        $PipelineExitCode = $LASTEXITCODE
        if ($PipelineExitCode -eq 0) {
            $CompletedThroughFallback = $true
        }
    }
} else {
    Write-Host "[INFO] CUDA is unavailable. Falling back to Colab ZIP import."
    & $ImportColab
    $PipelineExitCode = $LASTEXITCODE
    if ($PipelineExitCode -eq 0) {
        $CompletedThroughFallback = $true
    }
}

if ($CompletedThroughFallback) {
    Write-Host "[OK] Data pipeline completed successfully through Colab fallback."
}

$ExitBeforeFusion = $PipelineExitCode
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

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
    $Python = $null
}

Write-Host ""
Write-Host "[INFO] Running YOLO tactical fusion."

if ($Python) {
    Write-Host "[INFO] Using Python executable: $Python"
    & $Python $FuseYolo
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Tactical fusion artifact refreshed."
    } else {
        Write-Host "[WARN] Tactical fusion failed. The frontend will keep using risk ranking or fallback data."
    }
} else {
    Write-Host "[WARN] Python was not found. Skipping tactical fusion."
    Write-Host "[WARN] The frontend will keep using risk ranking or fallback data."
}
$PipelineExitCode = $ExitBeforeFusion

Write-Host ""
Write-Host "Next step:"
Write-Host "  cd frontend"
Write-Host "  npm run dev"

if ($PipelineExitCode -ne 0) {
    Write-Host ""
    Write-Host "[WARN] Data pipeline did not complete. The frontend will still use existing JSON or built-in fallback data."
}

exit $PipelineExitCode
