$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
$script = Join-Path $repo "scripts\train_learned_plan_context_bc.py"
$trainCache = Join-Path $repo "output\rl_forecast\replan_candidate_context\candidate_train_0_99_v4.npz"
$heldoutCache = Join-Path $repo "output\rl_forecast\replan_candidate_context\candidate_heldout_121_140_v4.npz"
$selectorDir = Join-Path $repo "output\rl_forecast\replan_candidate_context\selector_50ep"
$outDir = Join-Path $repo "output\rl_forecast\replan_candidate_context\learned_plan_context"
$env:PYTHONPATH = Join-Path $repo "src"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

foreach ($seed in 0..4) {
    $probabilities = Join-Path $selectorDir "candidate_selector_seed${seed}_probabilities.npz"
    Write-Output "[$(Get-Date -Format o)] starting learned_plan_context seed=$seed"
    & $python $script `
        --train-cache $trainCache `
        --heldout-cache $heldoutCache `
        --selector-probabilities $probabilities `
        --model-seed $seed `
        --epochs 50 `
        --device cuda `
        --out-dir $outDir
    if ($LASTEXITCODE -ne 0) {
        throw "learned plan-context BC failed for seed=$seed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format o)] completed learned_plan_context seed=$seed"
}
