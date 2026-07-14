param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("train", "heldout")]
    [string]$Split,

    [Parameter(Mandatory = $true)]
    [int]$FirstSeed,

    [Parameter(Mandatory = $true)]
    [int]$LastSeed
)

$ErrorActionPreference = "Stop"
if ($LastSeed -lt $FirstSeed) {
    throw "LastSeed must be greater than or equal to FirstSeed"
}

$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
$experiment = Join-Path $repo "scripts\compare_forecast_encoders_rl.py"
$outDir = Join-Path $repo "output\rl_forecast\replan_candidate_context\shards"
$cache = Join-Path $outDir "candidate_${Split}_${FirstSeed}_${LastSeed}.npz"
$seeds = $FirstSeed..$LastSeed | ForEach-Object { "$_" }
$arguments = @(
    $experiment,
    "generate-demos",
    "--demo-cache", $cache,
    "--demo-seeds"
) + $seeds + @(
    "--episode-hours", "720"
)

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Write-Output "[$(Get-Date -Format o)] starting split=$Split seeds=$FirstSeed..$LastSeed"
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "candidate demonstration collection failed with exit code $LASTEXITCODE"
}
Write-Output "[$(Get-Date -Format o)] completed split=$Split seeds=$FirstSeed..$LastSeed"
