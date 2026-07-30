param(
    [double]$HoursPerModel = 3.0,
    [switch]$Fresh,
    [switch]$ResumeLegacy40
)

$ErrorActionPreference = "Stop"

if ($HoursPerModel -lt 0) {
    throw "HoursPerModel must be nonnegative."
}
if ($Fresh -and $ResumeLegacy40) {
    throw "Fresh and ResumeLegacy40 are mutually exclusive."
}

$unicycleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $unicycleDir "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runner = Join-Path $unicycleDir "refine_whole_space_pi.py"
$artifactDir = Join-Path $unicycleDir "artifacts"
$timeLimitSec = [math]::Round($HoursPerModel * 3600.0, 3)

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

$common = @(
    "--time-limit-sec", "$timeLimitSec",
    "--checkpoint-interval-sec", "900",
    "--counterexample-backend", "graph"
)

$args40 = @(
    $runner,
    "--grid-size", "40",
    "--checkpoint", (Join-Path $artifactDir "unicycle_cegar_40x40x40.pkl"),
    "--max-total-states", "125000"
) + $common

if ($Fresh) {
    $args40 += "--fresh"
}
elseif ($ResumeLegacy40) {
    $args40 += @(
        "--resume-from",
        (Join-Path $artifactDir "legacy\unicycle_refinement_model.pkl")
    )
}

Write-Host "[BATCH] Starting 40x40x40 run."
& $python @args40
if ($LASTEXITCODE -ne 0) {
    throw "40x40x40 run failed with exit code $LASTEXITCODE."
}

$args90 = @(
    $runner,
    "--grid-size", "90",
    "--checkpoint", (Join-Path $artifactDir "unicycle_cegar_90x90x90.pkl"),
    "--max-total-states", "1000000"
) + $common

if ($Fresh) {
    $args90 += "--fresh"
}

Write-Host "[BATCH] Starting 90x90x90 run."
& $python @args90
if ($LASTEXITCODE -ne 0) {
    throw "90x90x90 run failed with exit code $LASTEXITCODE."
}

Write-Host "[BATCH] Both runs completed successfully."
