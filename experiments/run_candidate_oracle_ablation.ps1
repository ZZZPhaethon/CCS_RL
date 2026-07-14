param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("phase_control", "oracle_candidate")]
    [string]$Condition
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
$experiment = Join-Path $repo "scripts\compare_forecast_encoders_rl.py"
$trainCache = Join-Path $repo "output\rl_forecast\replan_candidate_context\candidate_train_0_99_v4.npz"
$heldoutCache = Join-Path $repo "output\rl_forecast\replan_candidate_context\candidate_heldout_121_140_v4.npz"
$outDir = Join-Path $repo "output\rl_forecast\replan_candidate_context\$Condition"
$variant = if ($Condition -eq "oracle_candidate") {
    "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate"
} else {
    "fixed_scale_tcn_mode_destination_replan_phase"
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

foreach ($seed in 0..4) {
    Write-Output "[$(Get-Date -Format o)] starting condition=$Condition seed=$seed"
    & $python $experiment train `
        --variant $variant `
        --demo-cache $trainCache `
        --heldout-demo-cache $heldoutCache `
        --bc-objective decision_only `
        --bc-only `
        --imitation-only `
        --bc-epochs 50 `
        --bc-batch-size 256 `
        --model-seed $seed `
        --device cuda `
        --out-dir $outDir `
        --verbose 0
    if ($LASTEXITCODE -ne 0) {
        throw "training failed for condition=$Condition seed=$seed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format o)] completed condition=$Condition seed=$seed"
}
