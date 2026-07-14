# Mode-Aware Weather Observation and Paired RL Experiment Design

## Objective

Reduce redundant weather features while preserving all weather information that is meaningful for each weather source, then compare three-vessel RL performance under weather-window start rates of 0.3 and 1.0 events per week.

The experiment keeps all non-weather disturbances, scenario parameters, training hyperparameters, and evaluation seeds fixed. The only paired treatment variable is `weather_window_rate_per_week`.

## Scope

This change covers:

- the weather portion of `CCSEnv` observations;
- propagation of the weather observation layout from the selected weather mode;
- propagation and reporting of `weather_window_rate_per_week` through the RL and HPC entry points;
- local tests, an HPC smoke test, paired training, and controller comparison.

It does not change the physical weather generator, the meaning of the existing 24 h or 168 h summaries, action spaces, rewards, or non-weather disturbances.

## Observation Layouts

The two annual clock features, `hour_of_year_sin` and `hour_of_year_cos`, are removed from every RL observation layout. `hour_of_week` remains part of the 38-dimensional base observation because it represents the weekly operating cycle.

### No weather observation

When `include_weather_obs` is false, the observation remains the existing 38-dimensional base observation for the three-vessel scenario.

### Global window weather

When `include_weather_obs` is true and `weather_mode="window"`, the environment uses a compact global-weather layout:

- 38 base features;
- one global current speed factor;
- global 24 h speed-factor mean and minimum;
- global 168 h speed-factor mean and minimum;
- one current normalized travel-time estimate for each vessel-destination pair.

For three vessels and four destinations, this is `38 + 5 + 12 = 55` features.

The global summaries appear once because the window generator gives every vessel and route the same weather-speed trajectory. Travel times remain per vessel and destination because route distances and current origins differ.

### Route-specific weather

When `include_weather_obs` is true and the weather mode is `leg_wave_climatology`, `wave_height_netcdf`, or `lstm_forecast`, the environment retains route-specific weather features:

- 38 base features;
- for every vessel-destination pair: current speed factor, 24 h mean, 24 h minimum, 168 h mean, 168 h minimum, and current normalized travel time.

For three vessels and four destinations, this is `38 + 3 * 4 * 6 = 110` features.

## Configuration and Data Flow

The environment configuration gains an explicit weather-observation layout with two valid values: `global` and `leg`. The Phase 1 environment factory chooses the layout from `weather_mode`:

- `window` -> `global`;
- all supported route-specific weather modes -> `leg`.

The observation schema is therefore known before the first reset and remains fixed throughout training, as required by Gymnasium.

`weather_window_rate_per_week` is exposed through:

1. `make_native_env`;
2. `experiments/compare_reward_modes_bc.py` CLI;
3. `hpc/submit_reward_modes_bc.sh` as `WEATHER_WINDOW_RATE_PER_WEEK`;
4. model/result tags and Markdown reports.

The project-wide `ScenarioConfig` default remains independent of the paired experiment. Each experimental job passes its rate explicitly so unrelated simulations are not silently changed.

## Validation and Error Handling

- Reject unknown weather-observation layouts rather than silently selecting one.
- Assert that every observation length equals `feature_names` and the Gym observation-space shape.
- Verify that annual clock features are absent from both global and route-specific layouts.
- Verify that global window summaries occur once, not once per route.
- Verify that route-specific modes retain independent per-leg features.
- Verify that the new weather-window rate reaches `ScenarioConfig`, appears in output tags/reports, and is present in the HPC command.

Existing 38-, 112-, and other incompatible checkpoints are not resumed into the new 55- or 110-dimensional policies.

## Paired HPC Experiment

Run two new weather-aware, three-vessel jobs with identical settings except for the explicit weather-window start rate:

| Parameter | Control | Treatment |
|---|---:|---:|
| `weather_window_rate_per_week` | 0.3 | 1.0 |
| observation layout | global, 55 dim | global, 55 dim |
| scenario | `northern_lights_phase1_3vessels` | same |
| episode length | 720 h | same |
| reward mode | `vent_first` | same |
| BC teacher | greedy | same |
| BC episodes / epochs | 100 / 20 | same |
| non-WAIT weight | 20 | same |
| kickstart coefficient | 1.0 | same |
| PPO timesteps | 100,000 | same |
| evaluation seeds | 101-105 | same |

Before formal training, run local tests and a short Borg environment/smoke job that prints the observation shape, weather rate, CUDA state, and a short BC/PPO rollout.

Each formal job reports idle, greedy, BC stochastic/deterministic, and PPO stochastic/deterministic metrics. The final comparison includes storage rate, loss rate, stored and vented tonnes, operating cost, vent penalty, total actual cost, and cost per stored tonne.

## Success Criteria

The work is complete when:

1. local tests pass for 38-, 55-, and 110-dimensional layouts;
2. no RL observation contains annual clock features;
3. both explicit weather rates are recorded in reproducible output names and reports;
4. the Borg smoke job completes without shape, dependency, or CUDA errors;
5. both paired training jobs complete successfully;
6. results are retrieved and summarized in one side-by-side comparison with greedy and BC baselines.
