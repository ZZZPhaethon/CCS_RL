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

Rolling prediction for MPC/scenario generation:

```powershell
python -m sim.scenario_generation.wave_height.prediction.predict_lstm `
  --model output\wave_height\wave_height_lstm_168h.pt `
  --csv output\wave_height\phase1_route_wave_2010_2014.csv `
  --target-year 2014 `
  --replan-every-hours 24 `
  --device cuda `
  --output output\wave_height\wave_height_lstm_2014_rolling24_predictions.csv
```

Use the rolling LSTM forecast as the environment scenario generator:

```python
from sim.environment.factories import make_phase1_env
from sim.scenario_generation import ScenarioConfig
from sim.scenario_generation.wave_height import LSTMWaveHeightScenarioGenerator

env = make_phase1_env()
scenario_generator = LSTMWaveHeightScenarioGenerator.from_env(
    env,
    "output/wave_height/wave_height_lstm_2014_rolling24_predictions.csv",
    config=ScenarioConfig(episode_hours=168),
)
env.scenario_generator = scenario_generator
```

If ``fixed_start_global_record`` is omitted, each sampled scenario chooses one
available rolling forecast start deterministically from the episode seed. To pin
one MPC window, pass a ``global_record`` that appears at ``horizon_index=0`` in
the rolling prediction CSV.

Leg-level weather for non-fixed routes:

```powershell
python -m sim.scenario_generation.wave_height.export_leg_dataset `
  --wave-dir "D:\wave Height" `
  --output output\wave_height\phase1_leg_wave_2010_2014.csv
```

This writes weather by ``leg_id`` such as ``brevik->oygarden_terminal`` or
``brevik->yara_sluiskil``. When this file exists at the default output path,
``build_phase1_env()`` uses the five-year seasonal mean speed factor as the
default leg-aware scenario generator:

```python
from sim.environment.factories import build_phase1_env

env = build_phase1_env()
env.reset(seed=1)
```

The simulator and rolling MILP now prefer ``Scenario.leg_speed_factor`` for the
current ``origin->destination`` leg and fall back to ``vessel_speed_factor`` when
a leg-specific series is absent. ``LegWaveClimatologyScenarioGenerator`` sets
``vessel_speed_factor`` to 1.0 by default, so the weather slowdown comes from
the leg-level CSV rather than the old random vessel-level weather.
