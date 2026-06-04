# ResQVision - Windows bootstrap script
# Run from project root: .\setup.ps1
# Idempotent: safe to run multiple times.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== ResQVision Setup ===" -ForegroundColor Cyan
Write-Host ""

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pythonArgs = @()

if (-not $pythonCommand) {
    $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
    $pythonArgs = @("-3")
}

if (-not $pythonCommand) {
    throw "Python 3 was not found. Install Python 3 and enable 'Add python.exe to PATH', then rerun .\setup.ps1."
}

# ---------------------------------------------------------------------------
# 1. Python virtual environment
# ---------------------------------------------------------------------------
$venvDir = "venv"

if (Test-Path "$venvDir\Scripts\Activate.ps1") {
    Write-Host "[OK] Virtual environment already exists - skipping creation." -ForegroundColor Green
} else {
    Write-Host "[1/5] Creating Python virtual environment in .\venv ..." -ForegroundColor Yellow
    & $pythonCommand.Source @pythonArgs -m venv $venvDir
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 2. Activate venv
# ---------------------------------------------------------------------------
Write-Host "[2/5] Activating virtual environment ..." -ForegroundColor Yellow
& ".\$venvDir\Scripts\Activate.ps1"
Write-Host "[OK] Virtual environment active." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 3. Install Python dependencies
# ---------------------------------------------------------------------------
Write-Host "[3/5] Installing Python dependencies from requirements.txt ..." -ForegroundColor Yellow
& ".\$venvDir\Scripts\python.exe" -m pip install -r requirements.txt --quiet
Write-Host "[OK] Python dependencies installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4 & 5. Frontend - npm install
# ---------------------------------------------------------------------------
Write-Host "[4/5] Installing frontend Node dependencies ..." -ForegroundColor Yellow
Push-Location frontend
npm install --silent
Pop-Location
Write-Host "[OK] Node dependencies installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/5] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  npm run dev          (from frontend/)"
Write-Host "  python scripts\yolo_detect.py --image scripts\sample_input.png"
Write-Host ""
