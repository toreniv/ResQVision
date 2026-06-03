$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CheckCuda = Join-Path $ProjectRoot "scripts\check_cuda.ps1"
$RunLocal = Join-Path $ProjectRoot "scripts\run_cuda_local.ps1"
$ImportColab = Join-Path $ProjectRoot "scripts\import_colab_outputs.ps1"

Write-Host "ResQVision data pipeline"
Write-Host "------------------------"

& $CheckCuda
$CudaAvailable = ($LASTEXITCODE -eq 0)

if ($CudaAvailable) {
    Write-Host "[INFO] CUDA is available. Running local CUDA pipeline."
    & $RunLocal
    $PipelineExitCode = $LASTEXITCODE
} else {
    Write-Host "[INFO] CUDA is unavailable. Falling back to Colab ZIP import."
    & $ImportColab
    $PipelineExitCode = $LASTEXITCODE
}

Write-Host ""
Write-Host "Next step:"
Write-Host "  cd frontend"
Write-Host "  npm run dev"

if ($PipelineExitCode -ne 0) {
    Write-Host ""
    Write-Host "[WARN] Data pipeline did not complete. The frontend will still use existing JSON or built-in fallback data."
}

exit $PipelineExitCode
