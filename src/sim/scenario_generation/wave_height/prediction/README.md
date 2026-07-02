# Wave-Height Prediction Outputs

Default wave-height artifacts are written under:

```text
output/wave_height/
```

Typical files:

```text
phase1_route_wave_2010_2014.csv
wave_height_gru.pt
wave_height_gru_history.csv
wave_height_gru_test_predictions.csv
wave_height_phase1_record0_zoom.png
```

Baseline evaluation:

```powershell
python -m sim.scenario_generation.wave_height.prediction.train_baselines `
  --lookback-hours 168 `
  --horizon-hours 168
```

GRU training:

```powershell
python -m sim.scenario_generation.wave_height.prediction.train_gru `
  --lookback-hours 168 `
  --horizon-hours 168 `
  --epochs 30 `
  --batch-size 128 `
  --hidden-size 128 `
  --device cuda
```

LSTM training with early stopping:

```powershell
python -m sim.scenario_generation.wave_height.prediction.train_lstm `
  --lookback-hours 168 `
  --horizon-hours 168 `
  --max-epochs 100 `
  --patience 8 `
  --batch-size 128 `
  --hidden-size 128 `
  --num-layers 2 `
  --dropout 0.2 `
  --weight-decay 1e-4 `
  --device cuda
```

The LSTM command writes the best validation model, not the final epoch model.
