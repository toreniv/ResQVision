$ErrorActionPreference = "SilentlyContinue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DataDir = Join-Path $ProjectRoot "frontend\public\data"
$FrontendUrl = "http://localhost:5173"
$YoloHealthUrl = "http://127.0.0.1:8000/api/health"

function Test-HttpOk($Url) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 300
    } catch {
        return $false
    }
}

function Print-Status($Name, $Ok, $Detail = "") {
    $status = if ($Ok) { "yes" } else { "no" }
    if ($Detail) {
        Write-Host ("{0,-34} {1,-6} {2}" -f $Name, $status, $Detail)
    } else {
        Write-Host ("{0,-34} {1}" -f $Name, $status)
    }
}

$benchmarkPath = Join-Path $DataDir "benchmark_results.json"
$riskPath = Join-Path $DataDir "risk_ranking.json"
$attentionPath = Join-Path $DataDir "attention_stats.json"
$fusionPath = Join-Path $DataDir "tactical_fusion.json"
$humanReviewPath = Join-Path $DataDir "human_review_detections.json"
$previewPath = Join-Path $DataDir "human_review_preview.jpg"

$cudaDataExists = (Test-Path $benchmarkPath) -and (Test-Path $riskPath)
$yoloOnline = Test-HttpOk $YoloHealthUrl
$frontendOnline = Test-HttpOk $FrontendUrl

Write-Host ""
Write-Host "ResQVision Demo Status"
Write-Host "======================"
Print-Status "CUDA data exists" $cudaDataExists
Print-Status "benchmark_results.json exists" (Test-Path $benchmarkPath)
Print-Status "risk_ranking.json exists" (Test-Path $riskPath)
Print-Status "attention_stats.json exists" (Test-Path $attentionPath)
Write-Host ("{0,-34} {1}" -f "YOLO backend", $(if ($yoloOnline) { "online" } else { "offline" }))
Write-Host ("{0,-34} {1}" -f "Frontend", $(if ($frontendOnline) { "online" } else { "offline" }))
Print-Status "Tactical fusion" (Test-Path $fusionPath) "frontend falls back to risk_ranking.json if missing"
Print-Status "Human-reviewed detections" (Test-Path $humanReviewPath)
Print-Status "Human-reviewed preview" (Test-Path $previewPath)
Write-Host ("{0,-34} {1}" -f "Recommended browser URL", $FrontendUrl)
Write-Host ""

exit 0
