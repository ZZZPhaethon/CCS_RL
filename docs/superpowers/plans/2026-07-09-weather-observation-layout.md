# Mode-Aware Weather Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace redundant window-weather observations with a compact global layout, remove annual clock features, expose an explicit weather-window start rate, and run paired three-vessel RL experiments at 0.3 and 1.0 events per week.

**Architecture:** `CCSEnvConfig` owns a fixed per-environment weather observation layout (`global` or `leg`). `build_phase1_env` derives that layout from `weather_mode` before Gymnasium reads the observation shape. The RL/HPC entry points pass `weather_window_rate_per_week` explicitly into `ScenarioConfig`, while preserving the user's current project-wide default and unrelated working-tree changes.

**Tech Stack:** Python 3.11, unittest/pytest, Gymnasium, sb3-contrib MaskablePPO, Bash/SLURM on Imperial Borg.

## Global Constraints

- Do not modify or revert the user's existing changes in `src/sim/scenario_generation/generator.py`, `tests/test_scenario.py`, MILP files, logs, or unrelated tests.
- Remove `hour_of_year_sin` and `hour_of_year_cos` from every RL observation layout.
- Three-vessel observation sizes must be 38 without weather, 55 for global window weather, and 110 for route-specific weather.
- Window weather rate must be passed explicitly for both paired jobs; do not rely on the global default.
- Formal jobs use `vent_first`, greedy BC with 100 episodes / 20 epochs, non-WAIT weight 20, kickstart coefficient 1.0, 100,000 PPO timesteps, and evaluation seeds 101-105.
- Existing incompatible checkpoints must not be resumed.

---

### Task 1: Add mode-aware weather observation layouts

**Files:**
- Modify: `src/sim/environment/env.py:31-122,218-257,617-710`
- Test: `tests/test_env.py:1-82`

**Interfaces:**
- Consumes: `CCSEnvConfig(include_weather_obs: bool, weather_observation_layout: str)`.
- Produces: `CCSEnv.feature_names`, `CCSEnv._observation()`, and `CCSEnv._global_weather_observation()` with stable matching lengths.

- [ ] **Step 1: Write failing layout tests**

Replace the existing annual-clock assertions and add compact-layout coverage:

```python
def test_global_weather_observation_is_compact(self):
    base = _env()
    env = _env(include_weather_obs=True, weather_observation_layout="global")
    obs = env.reset(seed=0)
    names = env.feature_names
    destination_count = len(env.terminal_ids) + len(env.emitter_ids)

    self.assertEqual(len(obs), base.observation_size + 5 + len(env.vessel_ids) * destination_count)
    self.assertNotIn("hour_of_year_sin", names)
    self.assertNotIn("hour_of_year_cos", names)
    self.assertEqual(names.count("weather.speed_24h_mean"), 1)
    self.assertIn("vessel_a.to_source_b.travel_hours_now", names)

def test_leg_weather_observation_keeps_candidate_leg_forecasts(self):
    env = _env(include_weather_obs=True, weather_observation_layout="leg")
    obs = env.reset(seed=0)
    names = env.feature_names

    self.assertEqual(len(obs), env.observation_size)
    self.assertNotIn("hour_of_year_sin", names)
    self.assertNotIn("hour_of_year_cos", names)
    self.assertIn("vessel_a.to_source_b.leg_speed_24h_mean", names)
    self.assertIn("vessel_a.to_source_b.leg_speed_168h_min", names)

def test_unknown_weather_observation_layout_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "weather_observation_layout"):
        _env(include_weather_obs=True, weather_observation_layout="unknown")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_env.py::EnvSpaceTests -q`

Expected: failures because `weather_observation_layout` and compact global feature names do not exist, and annual clock features are still present.

- [ ] **Step 3: Implement the minimal layouts**

In `CCSEnvConfig`, add:

```python
include_weather_obs: bool = False
weather_observation_layout: str = "leg"
```

In `CCSEnv.__init__`, validate:

```python
if self.config.weather_observation_layout not in {"global", "leg"}:
    raise ValueError(
        "weather_observation_layout must be 'global' or 'leg', "
        f"got {self.config.weather_observation_layout!r}."
    )
```

Replace the weather portion of `feature_names` with:

```python
if self.config.include_weather_obs:
    if self.config.weather_observation_layout == "global":
        names += [
            "weather.speed_now",
            "weather.speed_24h_mean",
            "weather.speed_24h_min",
            "weather.speed_168h_mean",
            "weather.speed_168h_min",
        ]
        for vid in self.vessel_ids:
            names += [
                f"{vid}.{label}.travel_hours_now"
                for label, _destination_id in self._weather_destination_slots()
            ]
    else:
        for vid in self.vessel_ids:
            for label, _destination_id in self._weather_destination_slots():
                names += [
                    f"{vid}.{label}.leg_speed_now",
                    f"{vid}.{label}.leg_speed_24h_mean",
                    f"{vid}.{label}.leg_speed_24h_min",
                    f"{vid}.{label}.leg_speed_168h_mean",
                    f"{vid}.{label}.leg_speed_168h_min",
                    f"{vid}.{label}.travel_hours_now",
                ]
```

Replace the weather portion of `_observation` with:

```python
if self.config.include_weather_obs:
    if self.config.weather_observation_layout == "global":
        obs += self._global_weather_observation()
    else:
        for vid in self.vessel_ids:
            obs += self._weather_observation_for_vessel(vid)
```

Add:

```python
def _global_weather_observation(self) -> list[float]:
    vessel_id = self.vessel_ids[0]
    now = self._weather_speed_at("", vessel_id, 0)
    mean24, min24 = self._weather_speed_forecast("", vessel_id, 24)
    mean168, min168 = self._weather_speed_forecast("", vessel_id, 168)
    values = [now, mean24, min24, mean168, min168]
    for vid in self.vessel_ids:
        route = self._routes[vid]
        origin_id = self._weather_reference_origin(vid)
        for _label, destination_id in self._weather_destination_slots():
            values.append(
                self._normalized_travel_hours(origin_id, destination_id, route, now)
            )
    return values
```

Remove the now-unused `import math`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_env.py::EnvSpaceTests -q`

Expected: all `EnvSpaceTests` pass.

- [ ] **Step 5: Commit the observation implementation**

```bash
git add src/sim/environment/env.py tests/test_env.py
git commit -m "Add mode-aware weather observations"
```

---

### Task 2: Select the layout from the weather mode

**Files:**
- Modify: `src/sim/environment/factories.py:11-83`
- Modify: `tests/test_env_scenarios.py:113-205`

**Interfaces:**
- Consumes: `build_phase1_env(..., weather_mode: WeatherMode, config: CCSEnvConfig)`.
- Produces: `env.config.weather_observation_layout == "global"` for `window`, otherwise `"leg"`.

- [ ] **Step 1: Write failing factory-selection tests**

```python
def test_phase1_window_weather_uses_global_observation_layout(self):
    env = build_phase1_env(
        scenario="northern_lights_phase1_3vessels",
        config=CCSEnvConfig(episode_hours=3, include_weather_obs=True),
        scenario_config=ScenarioConfig(episode_hours=3),
        weather_mode="window",
    )
    self.assertEqual(env.config.weather_observation_layout, "global")
    self.assertEqual(env.observation_size, 55)

def test_phase1_leg_weather_uses_leg_observation_layout(self):
    # Add this assertion to the existing temporary leg-wave CSV test.
    self.assertEqual(env.config.weather_observation_layout, "leg")
    self.assertEqual(env.observation_size, 110)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_env_scenarios.py -q`

Expected: window environment still has the default `leg` layout and size 110.

- [ ] **Step 3: Implement mode selection without mutating caller config**

Import `replace` and construct the effective config:

```python
from dataclasses import replace

weather_observation_layout = "global" if weather_mode == "window" else "leg"
env_config = replace(
    config or CCSEnvConfig(),
    weather_observation_layout=weather_observation_layout,
)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_env.py tests/test_env_scenarios.py tests/test_gym_env.py -q`

Expected: all selected tests pass and three-vessel sizes are 38/55/110 as configured.

- [ ] **Step 5: Commit the factory behavior**

```bash
git add src/sim/environment/factories.py tests/test_env_scenarios.py
git commit -m "Select weather observation layout by source"
```

---

### Task 3: Propagate and report the weather-window rate

**Files:**
- Modify: `src/sim/train.py:24-100`
- Modify: `scripts/compare_reward_modes_bc.py:65-72,77-112,224-365`
- Modify: `tests/test_compare_reward_modes_bc.py`

**Interfaces:**
- Consumes: `weather_window_rate_per_week: float` from CLI.
- Produces: explicit `ScenarioConfig.weather_window_rate_per_week`, `_weatherrate{value}` artifact tags, and report metadata.

- [ ] **Step 1: Write failing propagation and naming tests**

```python
def test_weather_rate_tag_marks_output_artifacts(self):
    args = SimpleNamespace(weather_window_rate_per_week=0.3)
    self.assertEqual(compare.weather_rate_tag(args), "_weatherrate0.3")

def test_weather_window_rate_arg_is_parsed(self):
    with patch(
        "sys.argv",
        ["compare_reward_modes_bc.py", "--weather-window-rate-per-week", "0.3"],
    ):
        args = compare.parse_args()
    self.assertEqual(args.weather_window_rate_per_week, 0.3)

# Extend test_make_env_passes_weather_obs_to_native_env:
self.assertEqual(
    make_native_env.call_args.kwargs["weather_window_rate_per_week"],
    1.0,
)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_compare_reward_modes_bc.py -q`

Expected: missing CLI option, tag helper, and `make_native_env` keyword.

- [ ] **Step 3: Implement rate propagation**

Add to `make_native_env` and `ScenarioConfig` construction:

```python
weather_window_rate_per_week: float = 1.0,

scenario_config=ScenarioConfig(
    # existing fields...
    weather_window_rate_per_week=weather_window_rate_per_week,
)
```

Add to the comparison script:

```python
def weather_rate_tag(args) -> str:
    return f"_weatherrate{args.weather_window_rate_per_week:g}"

parser.add_argument("--weather-window-rate-per-week", type=float, default=1.0)
```

Pass it in `make_env`, append `weather_rate_tag(args)` to both model and report stems, and add this report line:

```python
f"weather_window_rate_per_week={args.weather_window_rate_per_week:g}",
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_compare_reward_modes_bc.py tests/test_train.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the experiment parameter**

```bash
git add src/sim/train.py scripts/compare_reward_modes_bc.py tests/test_compare_reward_modes_bc.py
git commit -m "Expose weather window rate to RL experiments"
```

---

### Task 4: Wire the parameter into the Borg submission script

**Files:**
- Modify: `hpc/submit_reward_modes_bc.sh:17-90`
- Modify: `tests/test_project_structure.py:58-73`

**Interfaces:**
- Consumes: exported `WEATHER_WINDOW_RATE_PER_WEEK`.
- Produces: logged CLI argument `--weather-window-rate-per-week`.

- [ ] **Step 1: Write the failing script-structure test**

```python
def test_reward_modes_hpc_script_passes_weather_window_rate(self):
    script = (ROOT / "hpc" / "submit_reward_modes_bc.sh").read_text(encoding="utf-8")
    self.assertIn(
        'WEATHER_WINDOW_RATE_PER_WEEK="${WEATHER_WINDOW_RATE_PER_WEEK:-1.0}"',
        script,
    )
    self.assertIn('echo "Weather window rate per week: $WEATHER_WINDOW_RATE_PER_WEEK"', script)
    self.assertIn('--weather-window-rate-per-week "$WEATHER_WINDOW_RATE_PER_WEEK"', script)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_project_structure.py::ProjectStructureTests::test_reward_modes_hpc_script_passes_weather_window_rate -q`

Expected: assertion failure because the variable and CLI argument are absent.

- [ ] **Step 3: Implement the Bash wiring**

Add:

```bash
WEATHER_WINDOW_RATE_PER_WEEK="${WEATHER_WINDOW_RATE_PER_WEEK:-1.0}"
echo "Weather window rate per week: $WEATHER_WINDOW_RATE_PER_WEEK"
```

and pass:

```bash
--weather-window-rate-per-week "$WEATHER_WINDOW_RATE_PER_WEEK" \
```

- [ ] **Step 4: Run focused and full local verification**

Run:

```bash
python -m pytest tests/test_project_structure.py tests/test_env.py tests/test_env_scenarios.py tests/test_gym_env.py tests/test_compare_reward_modes_bc.py tests/test_train.py -q
```

Expected: all selected tests pass.

Then run:

```bash
python -m pytest -q
```

Expected: the full suite passes; if unrelated dirty-worktree tests fail, record them separately and verify all task-owned tests still pass.

- [ ] **Step 5: Commit the HPC entry point**

```bash
git add hpc/submit_reward_modes_bc.sh tests/test_project_structure.py
git commit -m "Pass weather window rate in Borg jobs"
```

---

### Task 5: Deploy, smoke-test, and run the paired Borg jobs

**Files:**
- Sync: `src/sim/environment/env.py`
- Sync: `src/sim/environment/factories.py`
- Sync: `src/sim/train.py`
- Sync: `scripts/compare_reward_modes_bc.py`
- Sync: `hpc/submit_reward_modes_bc.sh`
- Retrieve: `output/rl_ppo/*weatherrate0.3*`
- Retrieve: `output/rl_ppo/*weatherrate1*`

**Interfaces:**
- Consumes: tested local source and Borg conda environment `mas-ccus`.
- Produces: smoke log, two trained models, two CSV/Markdown reports, and paired comparison metrics.

- [ ] **Step 1: Check Borg GPU/queue and remote directories**

```bash
ssh hx721@borg-login.ese.ic.ac.uk "squeue -u hx721 -o '%.18i %.9P %.40j %.2t %.10M %R'"
ssh hx721@borg-login.ese.ic.ac.uk "ls -ld /scratch_root/hx721/CCS_RLLLM /scratch_root/hx721/CCS_RLLLM/src/sim/environment"
```

Expected: project directories exist; record current queue state before submitting.

- [ ] **Step 2: Sync only required files**

Create missing remote directories with `mkdir -p`, then use `scp` for exactly the five files listed above. Do not sync `.git`, outputs, checkpoints, logs, or unrelated dirty files.

- [ ] **Step 3: Submit and verify a short smoke job**

```bash
ssh hx721@borg-login.ese.ic.ac.uk "cd /scratch_root/hx721/CCS_RLLLM && sbatch --qos=short --time=00:30:00 --mem=32G --cpus-per-task=4 --export=ALL,WEATHER_WINDOW_RATE_PER_WEEK=1.0,BC_EPISODES=2,BC_EPOCHS=1,NONWAIT_WEIGHT=20,KICKSTART_COEF=1.0,TIMESTEPS=2048,REWARD_MODES=vent_first hpc/submit_reward_modes_bc.sh"
```

Expected: a job ID. Monitor `squeue`, `sacct`, and matching logs until completion. Verify CUDA is available, observation shape is 55, rate is 1.0, and the job exits 0.

- [ ] **Step 4: Submit paired formal jobs**

Submit two jobs with identical exports except the explicit rate:

```text
WEATHER_WINDOW_RATE_PER_WEEK=0.3
WEATHER_WINDOW_RATE_PER_WEEK=1.0
BC_EPISODES=100
BC_EPOCHS=20
NONWAIT_WEIGHT=20
KICKSTART_COEF=1.0
TIMESTEPS=100000
REWARD_MODES=vent_first
```

Use `sbatch`, not the login node. Record both job IDs.

- [ ] **Step 5: Monitor to terminal state and retrieve artifacts**

For each job, verify `COMPLETED`, exit code `0:0`, absence of traceback, presence of its `_weatherrate...` model/report files, and the expected evaluation rows. Retrieve the two CSVs, Markdown reports, models, and logs with `scp`.

- [ ] **Step 6: Produce the side-by-side result summary**

Create `output/rl_ppo/weather_rate_0.3_vs_1.0.md` containing, for each rate, idle, greedy, BC-only, and PPO storage rate, loss rate, stored/vented tonnes, actual total cost, and cost per stored tonne. State deltas both versus greedy within each rate and between rates for the same controller.

---

### Task 6: Final verification and handoff

**Files:**
- Verify: all modified source/test/HPC files
- Verify: paired output reports and models

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: evidence-backed final report.

- [ ] **Step 1: Re-run task-owned tests**

Run:

```bash
python -m pytest tests/test_project_structure.py tests/test_env.py tests/test_env_scenarios.py tests/test_gym_env.py tests/test_compare_reward_modes_bc.py tests/test_train.py -q
```

Expected: all pass.

- [ ] **Step 2: Verify observation shapes from executable code**

Instantiate the three-vessel environment with no weather observation, window weather, and a route-specific weather mode. Verify exact shapes 38, 55, and 110 and absence of annual clock feature names.

- [ ] **Step 3: Verify scoped diff and artifacts**

Confirm no unrelated dirty files were staged or modified by this work. List both Borg job IDs, logs, report paths, model paths, and the paired comparison path.

- [ ] **Step 4: Report the outcome**

Summarize whether increased weather frequency changes greedy, BC, and PPO performance; whether PPO beats greedy at each rate; and whether the change is meaningful across the five paired evaluation seeds.
