$ErrorActionPreference = "SilentlyContinue"

$gpuDetected = $false
$nvccDetected = $false

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    & $nvidiaSmi.Source -L *> $null
    $gpuDetected = ($LASTEXITCODE -eq 0)
}

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    & $nvcc.Source --version *> $null
    $nvccDetected = ($LASTEXITCODE -eq 0)
}

if ($gpuDetected) {
    Write-Host "[OK] NVIDIA GPU detected"
} else {
    Write-Host "[WARN] NVIDIA GPU not detected"
}

if ($nvccDetected) {
    Write-Host "[OK] nvcc detected"
} else {
    Write-Host "[WARN] nvcc not detected"
}

if ($gpuDetected -and $nvccDetected) {
    exit 0
}

exit 1
