$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CheckCuda = Join-Path $ProjectRoot "scripts\check_cuda.ps1"
$RunLocal = Join-Path $ProjectRoot "scripts\run_cuda_local.ps1"
$ImportColab = Join-Path $ProjectRoot "scripts\import_colab_outputs.ps1"
$FuseYolo = Join-Path $ProjectRoot "scripts\fuse_yolo_to_tactical.py"

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
Write-Host ""
Write-Host "[INFO] Running YOLO tactical fusion."
python $FuseYolo
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Tactical fusion artifact refreshed."
} else {
    Write-Host "[WARN] Tactical fusion failed. The frontend will keep using risk ranking or fallback data."
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
