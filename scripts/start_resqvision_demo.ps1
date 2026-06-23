param(
    [ValidateSet("Auto", "LocalCuda", "Colab")]
    [string]$Mode = "Auto",
    [switch]$StartYolo,
    [switch]$NoYolo,
    [switch]$OpenBrowser,
    [switch]$NoOpenBrowser,
    [switch]$SkipCuda,
    [switch]$SkipInstall,
    [switch]$UseColabImport
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogsDir = Join-Path $ProjectRoot "logs"
$ProcessFile = Join-Path $LogsDir "resqvision_processes.json"
$FrontendUrl = "http://localhost:5173"
$YoloHealthUrl = "http://127.0.0.1:8000/api/health"
$FrontendDataDir = Join-Path $ProjectRoot "frontend\public\data"
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot "venv\Scripts\pip.exe"
$DownloadsZip = Join-Path $env:USERPROFILE "Downloads\resqvision_cuda_outputs.zip"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "== $Message =="
}

function Write-Ok($Message) {
    Write-Host "[OK] $Message"
}

function Write-Warn($Message) {
    Write-Host "[WARN] $Message"
}

function Test-HttpOk($Url) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 300
    } catch {
        return $false
    }
}

function Wait-Http($Url, $Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpOk $Url) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Test-PortListening($Port) {
    $netstat = & cmd /c "netstat -ano | findstr :$Port" 2>$null
    return ($netstat -match "LISTENING")
}

function Save-ProcessState($YoloPid, $FrontendPid) {
    $state = [ordered]@{
        yolo_pid = $YoloPid
        frontend_pid = $FrontendPid
        frontend_url = $FrontendUrl
        yolo_health_url = $YoloHealthUrl
        updated_at = (Get-Date).ToString("s")
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath $ProcessFile -Encoding UTF8
}

function Get-PythonCommand {
    if (Test-Path $VenvPython) {
        return $VenvPython
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }
    return $null
}

function Ensure-ProjectRoot {
    if (-not (Test-Path (Join-Path $ProjectRoot "resqvision.cu"))) {
        throw "resqvision.cu not found. Run this from the ResQVision project root."
    }
    if (-not (Test-Path (Join-Path $ProjectRoot "frontend"))) {
        throw "frontend/ not found. Run this from the ResQVision project root."
    }
    if (-not (Test-Path (Join-Path $ProjectRoot "scripts"))) {
        throw "scripts/ not found. Run this from the ResQVision project root."
    }
}

function Ensure-Environment {
    if ($SkipInstall) {
        Write-Warn "Skipping dependency installation checks."
        return
    }

    if (-not (Test-Path $VenvPython)) {
        $python = Get-PythonCommand
        if (-not $python) {
            throw "Python was not found. Install Python 3.10+ and rerun the launcher."
        }
        Write-Host "[INFO] Creating venv..."
        & $python -m venv (Join-Path $ProjectRoot "venv")
    }

    if (Test-Path (Join-Path $ProjectRoot "requirements.txt")) {
        $requirements = Join-Path $ProjectRoot "requirements.txt"
        $stamp = Join-Path $LogsDir "requirements.sha256"
        $currentHash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
        $previousHash = if (Test-Path $stamp) { Get-Content -LiteralPath $stamp -Raw } else { "" }
        if ($currentHash -ne $previousHash.Trim()) {
            Write-Host "[INFO] Installing Python requirements..."
            & $VenvPython -m pip install -r $requirements
            if ($LASTEXITCODE -eq 0) {
                $currentHash | Set-Content -LiteralPath $stamp -Encoding ASCII
            }
        } else {
            Write-Ok "Python requirements already installed for current requirements.txt."
        }
    }

    $frontendNodeModules = Join-Path $ProjectRoot "frontend\node_modules"
    if (-not (Test-Path $frontendNodeModules)) {
        Write-Host "[INFO] Installing frontend dependencies..."
        Push-Location (Join-Path $ProjectRoot "frontend")
        try {
            & cmd /c npm install
        } finally {
            Pop-Location
        }
    } else {
        Write-Ok "Frontend dependencies already installed."
    }
}

function Ensure-CudaData {
    if ($SkipCuda) {
        Write-Warn "Skipping CUDA data generation/import."
        return
    }

    $checkCuda = Join-Path $ProjectRoot "scripts\check_cuda.ps1"
    $runCuda = Join-Path $ProjectRoot "scripts\run_cuda_local.ps1"
    $importColab = Join-Path $ProjectRoot "scripts\import_colab_outputs.ps1"
    $localCudaOk = $false

    if (-not $UseColabImport -and $Mode -ne "Colab") {
        Write-Host "[INFO] Checking local CUDA availability..."
        & powershell -ExecutionPolicy Bypass -File $checkCuda
        $localCudaOk = ($LASTEXITCODE -eq 0)
    }

    if ($localCudaOk -and $Mode -ne "Colab") {
        Write-Host "[INFO] Local CUDA available. Running local CUDA pipeline..."
        & powershell -ExecutionPolicy Bypass -File $runCuda
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Local CUDA pipeline failed. Falling back to Colab import if available."
        } else {
            Write-Ok "Local CUDA pipeline complete."
            return
        }
    }

    if (Test-Path $DownloadsZip) {
        Write-Host "[INFO] Importing Colab export from $DownloadsZip"
        & powershell -ExecutionPolicy Bypass -File $importColab
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Colab CUDA outputs imported."
        } else {
            Write-Warn "Colab import failed. Existing frontend JSON files were left untouched."
        }
    } else {
        Write-Host "[WARN] Colab ZIP not found at $DownloadsZip" -ForegroundColor Yellow
        Write-Host "[INFO] To get CUDA benchmark data:"
        Write-Host "[INFO]   1. Open ResQVision_Colab_Workflow.ipynb in Google Colab"
        Write-Host "[INFO]   2. Runtime > Change runtime type > GPU"
        Write-Host "[INFO]   3. Run All"
        Write-Host "[INFO]   4. Download resqvision_cuda_outputs.zip when prompted"
        Write-Host "[INFO]   5. Place it in your Downloads folder and re-run this launcher"
        Write-Host "[WARN] Proceeding with existing local data if available." -ForegroundColor Yellow
        Write-Host "       If no local data exists, the dashboard may show empty charts."
    }
}

function Validate-Data {
    $validator = Join-Path $ProjectRoot "scripts\validate_dashboard_data.py"
    if (Test-Path $validator) {
        & $VenvPython $validator
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Dashboard data validation found required issues. The dashboard may use fallback data."
        }
    }
}

function Start-YoloBackend {
    if ($NoYolo) {
        Write-Warn "YOLO backend disabled by -NoYolo."
        return $null
    }

    $serverScript = Join-Path $ProjectRoot "scripts\yolo_server.py"
    if (-not (Test-Path $serverScript)) {
        Write-Warn "scripts/yolo_server.py not found. YOLO backend offline."
        return $null
    }

    if (Test-HttpOk $YoloHealthUrl) {
        Write-Ok "YOLO backend: ONLINE (existing server reused)"
        return $null
    }

    $shouldTryYolo = $StartYolo -or (-not $NoYolo)
    if (-not $shouldTryYolo) {
        return $null
    }

    if (-not (Test-Path $VenvPython)) {
        Write-Warn "YOLO backend: OFFLINE - venv Python not found"
        return $null
    }

    $modelCandidates = @(
        (Join-Path $ProjectRoot "models\drone_tactical_best.pt"),
        (Join-Path $ProjectRoot "yolov8s.pt"),
        (Join-Path $ProjectRoot "yolov8n.pt")
    )
    $modelExists = $false
    foreach ($candidate in $modelCandidates) {
        if (Test-Path $candidate) {
            $modelExists = $true
            break
        }
    }
    if (-not $modelExists) {
        Write-Warn "YOLO backend: OFFLINE - no YOLO model weights found"
        return $null
    }

    Write-Host "[INFO] Starting YOLO backend..."
    $serverWrapper = Join-Path $ProjectRoot "scripts\yolo_server_logged.py"
    $serverEntry = if (Test-Path $serverWrapper) { "scripts\yolo_server_logged.py" } else { "scripts\yolo_server.py" }
    $cmd = "/k `"`"$VenvPython`" `"$serverEntry`"`""
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden

    if (Wait-Http $YoloHealthUrl 20) {
        Write-Ok "YOLO backend: ONLINE"
        return $process.Id
    }

    Write-Warn "YOLO backend: OFFLINE - dashboard will use human-reviewed fallback"
    return $process.Id
}

function Run-TacticalFusion {
    $fusion = Join-Path $ProjectRoot "scripts\fuse_yolo_to_tactical.py"
    if (-not (Test-Path $fusion)) {
        Write-Warn "Tactical fusion: skipped or fallback active"
        return
    }
    try {
        & $VenvPython $fusion
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Tactical fusion complete or fallback generated."
        } else {
            Write-Warn "Tactical fusion: skipped or fallback active"
        }
    } catch {
        Write-Warn "Tactical fusion: skipped or fallback active"
    }
}

function Start-Frontend {
    if (Test-HttpOk $FrontendUrl) {
        Write-Ok "Frontend: ONLINE (existing Vite server reused)"
        return $null
    }

    Write-Host "[INFO] Starting React/Vite dashboard..."
    $logPath = Join-Path $LogsDir "frontend.log"
    $frontendDir = Join-Path $ProjectRoot "frontend"
    $cmd = "/c npm run dev ^>^> `"$logPath`" 2^>^&1"
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $frontendDir -PassThru -WindowStyle Hidden

    if (Wait-Http $FrontendUrl 30) {
        Write-Ok "Frontend: ONLINE at $FrontendUrl"
    } else {
        Write-Warn "Frontend did not respond within 30 seconds. Check logs\frontend.log"
    }
    return $process.Id
}

New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
Set-Location $ProjectRoot

Write-Host ""
Write-Host "ResQVision Automated Demo Launcher"
Write-Host "=================================="

Ensure-ProjectRoot

Write-Step "Environment"
Ensure-Environment

Write-Step "CUDA data"
Ensure-CudaData
Validate-Data

Write-Step "Optional YOLO backend"
$yoloPid = Start-YoloBackend

Write-Step "Tactical fusion"
Run-TacticalFusion

Write-Step "Frontend dashboard"
$frontendPid = Start-Frontend
Save-ProcessState $yoloPid $frontendPid

$shouldOpenBrowser = -not $NoOpenBrowser
if ($OpenBrowser) { $shouldOpenBrowser = $true }
if ($shouldOpenBrowser) {
    Start-Process $FrontendUrl | Out-Null
}

Write-Step "Summary"
& powershell -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "scripts\status_resqvision_demo.ps1")
Write-Host "[OK] Demo launcher finished. Browser URL: $FrontendUrl"
exit 0
