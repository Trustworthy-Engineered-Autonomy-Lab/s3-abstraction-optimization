param(
    [double]$HoursPerModel = 3.0,
    [switch]$Fresh,
    [switch]$ResumeLegacy60
)

$ErrorActionPreference = "Stop"

if ($HoursPerModel -lt 0) {
    throw "HoursPerModel must be nonnegative."
}
if ($Fresh -and $ResumeLegacy60) {
    throw "Fresh and ResumeLegacy60 are mutually exclusive."
}

$syntheticDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $syntheticDir "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runner = Join-Path $syntheticDir "run_synthetic.py"
$artifactDir = Join-Path $syntheticDir "artifacts"
$timeLimitSec = [math]::Round($HoursPerModel * 3600.0, 3)

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

$common = @(
    "--time-limit-sec", "$timeLimitSec",
    "--checkpoint-interval-sec", "900",
    "--counterexample-backend", "graph"
)

$args60 = @(
    $runner,
    "--grid-size", "60",
    "--checkpoint", (Join-Path $artifactDir "synthetic_cegar_60x60.pkl"),
    "--max-total-states", "4900"
) + $common

if ($Fresh) {
    $args60 += "--fresh"
}
elseif ($ResumeLegacy60) {
    $args60 += @(
        "--resume-from",
        (Join-Path $artifactDir "legacy\synthetic_60x60.pkl")
    )
}

Write-Host "[BATCH] Starting 60x60 synthetic run."
& $python @args60
if ($LASTEXITCODE -ne 0) {
    throw "60x60 run failed with exit code $LASTEXITCODE."
}

$args90 = @(
    $runner,
    "--grid-size", "90",
    "--checkpoint", (Join-Path $artifactDir "synthetic_cegar_90x90.pkl"),
    "--max-total-states", "10000"
) + $common

if ($Fresh) {
    $args90 += "--fresh"
}

Write-Host "[BATCH] Starting 90x90 synthetic run."
& $python @args90
if ($LASTEXITCODE -ne 0) {
    throw "90x90 run failed with exit code $LASTEXITCODE."
}

Write-Host "[BATCH] Both synthetic runs completed successfully."
