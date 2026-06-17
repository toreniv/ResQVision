$ErrorActionPreference = "SilentlyContinue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProcessFile = Join-Path $ProjectRoot "logs\resqvision_processes.json"

Write-Host "Stopping ResQVision demo processes..."

function Stop-ProcessTree($ProcessId, $Name) {
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) {
        Write-Host "[INFO] $Name PID $ProcessId is not running"
        return
    }

    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId"
    foreach ($child in $children) {
        Stop-ProcessTree ([int]$child.ProcessId) "$Name child"
    }

    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "[OK] Stopped $Name PID $ProcessId"
}

if (-not (Test-Path $ProcessFile)) {
    Write-Host "[INFO] No process file found at $ProcessFile"
    exit 0
}

try {
    $state = Get-Content -LiteralPath $ProcessFile -Raw | ConvertFrom-Json
} catch {
    Write-Host "[WARN] Could not read process file. Nothing stopped."
    exit 1
}

foreach ($entry in @(
    @{ Name = "YOLO backend"; Id = $state.yolo_pid },
    @{ Name = "Frontend"; Id = $state.frontend_pid }
)) {
    if ($entry.Id) {
        Stop-ProcessTree ([int]$entry.Id) $entry.Name
    }
}

Remove-Item -LiteralPath $ProcessFile -Force -ErrorAction SilentlyContinue
Write-Host "[OK] ResQVision demo stop complete."
exit 0
