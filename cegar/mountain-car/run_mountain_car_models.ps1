param(
    [double]$HoursPerModel = 3.0,
    [switch]$Fresh,
    [switch]$ResumeExisting,
    [switch]$ResumeLegacy
)

$ErrorActionPreference = "Stop"

if ($HoursPerModel -lt 0) {
    throw "HoursPerModel must be nonnegative."
}
$selectedModes = @($Fresh, $ResumeExisting, $ResumeLegacy) |
    Where-Object { $_ }
if ($selectedModes.Count -gt 1) {
    throw "Fresh, ResumeExisting, and ResumeLegacy are mutually exclusive."
}

$mountainCarDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $mountainCarDir "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runner = Join-Path $mountainCarDir "run_mountain_car.py"
$artifactDir = Join-Path $mountainCarDir "artifacts"
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
    "--checkpoint", (Join-Path $artifactDir "mountain_car_cegar_60x60.pkl"),
    "--max-total-states", "4900"
) + $common

$args90 = @(
    $runner,
    "--grid-size", "90",
    "--checkpoint", (Join-Path $artifactDir "mountain_car_cegar_90x90.pkl"),
    "--max-total-states", "10000"
) + $common

if ($ResumeExisting) {
    $args60 += "--resume-existing"
    $args90 += "--resume-existing"
}
elseif ($ResumeLegacy) {
    $args60 += @(
        "--resume-from",
        (Join-Path $artifactDir "legacy\mountaincar_60x60.pkl")
    )
    $args90 += @(
        "--resume-from",
        (Join-Path $artifactDir "legacy\mountain_90x90.pkl")
    )
}
else {
    # Fresh is the safe default. This prevents an ordinary rerun from
    # silently loading an already state-capped output and doing zero CEGAR
    # iterations.
    $args60 += "--fresh"
    $args90 += "--fresh"
}

Write-Host "[BATCH] Starting 60x60 Mountain Car run."
& $python @args60
if ($LASTEXITCODE -ne 0) {
    throw "60x60 run failed with exit code $LASTEXITCODE."
}

Write-Host "[BATCH] Starting 90x90 Mountain Car run."
& $python @args90
if ($LASTEXITCODE -ne 0) {
    throw "90x90 run failed with exit code $LASTEXITCODE."
}

Write-Host "[BATCH] Both Mountain Car runs completed successfully."
