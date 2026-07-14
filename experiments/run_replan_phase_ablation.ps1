param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 3, 5)]
    [int]$ReplanWeight
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
$experiment = Join-Path $repo "scripts\compare_forecast_encoders_rl.py"
$trainCache = Join-Path $repo "output\rl_forecast\corrected_forecast_cache\destination_mask_train_0_99_v4.npz"
$heldoutCache = Join-Path $repo "output\rl_forecast\corrected_forecast_cache\destination_mask_heldout_121_140_v4.npz"
$outDir = Join-Path $repo "output\rl_forecast\replan_phase_ablation\phase_w$ReplanWeight"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

foreach ($seed in 0..4) {
    Write-Output "[$(Get-Date -Format o)] starting weight=$ReplanWeight seed=$seed"
    & $python $experiment train `
        --variant fixed_scale_tcn_mode_destination_replan_phase `
        --demo-cache $trainCache `
        --heldout-demo-cache $heldoutCache `
        --bc-objective decision_only `
        --bc-only `
        --bc-epochs 50 `
        --bc-batch-size 256 `
        --replan-action-weight $ReplanWeight `
        --model-seed $seed `
        --eval-seeds 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 `
        --device cuda `
        --out-dir $outDir `
        --verbose 0
    if ($LASTEXITCODE -ne 0) {
        throw "training failed for weight=$ReplanWeight seed=$seed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format o)] completed weight=$ReplanWeight seed=$seed"
}
