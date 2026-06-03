$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ZipPath = Join-Path $env:USERPROFILE "Downloads\resqvision_cuda_outputs.zip"
$TempDir = Join-Path $ProjectRoot "temp_exports"
$FrontendDataDir = Join-Path $ProjectRoot "frontend\public\data"
$RequiredJson = @(
    "benchmark_results.json",
    "risk_ranking.json",
    "attention_stats.json"
)

if (-not (Test-Path $ZipPath)) {
    Write-Host "[ACTION REQUIRED] Run the Colab notebook and download resqvision_cuda_outputs.zip"
    Write-Host "[INFO] Existing frontend JSON files were left untouched."
    exit 1
}

New-Item -ItemType Directory -Path $FrontendDataDir -Force | Out-Null

if (Test-Path $TempDir) {
    Remove-Item -LiteralPath $TempDir -Recurse -Force
}

New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

try {
    Write-Host "[INFO] Extracting Colab export..."
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $TempDir -Force

    foreach ($FileName in $RequiredJson) {
        $Matches = Get-ChildItem -LiteralPath $TempDir -Recurse -Filter $FileName
        if (-not $Matches) {
            throw "Required JSON not found in Colab ZIP: $FileName"
        }

        Copy-Item -LiteralPath $Matches[0].FullName -Destination $FrontendDataDir -Force
        Write-Host "[OK] Imported $FileName"
    }
} catch {
    Write-Host "[WARN] Colab import failed: $($_.Exception.Message)"
    exit 1
} finally {
    if (Test-Path $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force
    }
}

Write-Host "[OK] Colab data import complete. JSON files are in frontend/public/data."
exit 0
