# MPC Forecast-Encoding RL Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a controlled comparison of current-state, flat 168 h forecast, and TCN-encoded 168 h forecast MaskablePPO agents, all warm-started and kickstarted from the same replay-validated `RollingNativeMpcController` demonstrations.

**Architecture:** A shared forecast module exposes current state plus the exact future `t+1..t+168` disturbance tensor. A dedicated Gym adapter presents state-only, flat, or structured observations; the structured variant uses a custom TCN feature extractor. MPC demonstrations are generated once into a schema-checked cache, then the same cache drives BC and kickstarting for every model.

**Tech Stack:** Python 3.10+, NumPy, Gymnasium 0.29-1.2, Stable-Baselines3 2.7, sb3-contrib 2.7, PyTorch, pytest/unittest, SLURM on Imperial Borg.

## Global Constraints

- Scenario is `northern_lights_phase1_3vessels` with its registered 15,000 t Yara buffer.
- RL episodes are 720 h; sampled disturbance trajectories are 889 h so hour 719 and the hour-720 timeout observation both expose `t+1..t+168`.
- Forecast shape is exactly `[168, 9]`: 3 capture, 3 emitter availability, 1 well availability, 1 injectivity, 1 global weather channel.
- Global weather uses 24 h block updates and occupies forecast channel index 8.
- All variants use `vent_first`, partial-load dispatch, the same demonstrations, BC settings, kickstarting schedule, PPO hyperparameters, and paired seeds.
- The first comparison does not add downstream-inventory reward shaping.
- MPC replay mismatch is fatal; no greedy fallback is permitted.
- Preserve unrelated worktree files and changes.

## File Structure

- Create `src/sim/environment/forecast.py`: forecast schema, current-state extraction, and `t+1..t+168` tensor construction.
- Create `src/sim/environment/forecast_gym.py`: three observation modes and learned-policy wrapper.
- Create `src/sim/environment/forecast_encoder.py`: TCN feature extractor for structured observations.
- Create `src/sim/control/demonstrations.py`: MPC dataset generation, strict replay audit, cache save/load, and metadata validation.
- Modify `src/sim/environment/env.py`: expose current global weather without future summaries.
- Modify `src/sim/environment/__init__.py`: export forecast interfaces.
- Modify `src/sim/train.py`: separate RL episode length from scenario forecast context.
- Modify `src/sim/control/imitation.py`: support arrays and dictionary observations through BC and kickstarting.
- Create `scripts/compare_forecast_encoders_rl.py`: cache generation, per-variant training, evaluation, and reporting CLI.
- Modify `src/sim/metrics.py`: report end inventory separately for emitters, vessels, and terminals.
- Create `hpc/submit_forecast_mpc_demos.sh`: CPU demonstration job.
- Create `hpc/submit_forecast_encoder_rl.sh`: three-variant GPU array job.
- Create `tests/test_forecast_observation.py`, `tests/test_forecast_gym.py`, `tests/test_forecast_encoder.py`, `tests/test_demonstrations.py`, and `tests/test_compare_forecast_encoders_rl.py`.
- Modify `tests/test_metrics.py` for end-inventory stage totals.
- Modify `tests/test_imitation.py`, `tests/test_train.py`, and `tests/test_project_structure.py`.

---

### Task 1: Build the shared 168 h forecast observation

**Files:**
- Create: `src/sim/environment/forecast.py`
- Modify: `src/sim/environment/env.py`
- Modify: `src/sim/environment/__init__.py`
- Modify: `src/sim/train.py`
- Create: `tests/test_forecast_observation.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Produces: `FORECAST_HORIZON_H`, `forecast_channel_names(env)`, `current_state_feature_names(env)`, `current_state_observation(env)`, and `future_forecast_observation(env, horizon_h=168)`.
- Produces: `make_native_env(..., scenario_context_hours=169)` so a 720 h RL environment samples an 889 h scenario by default.
- Consumes: `CCSEnv.scenario`, current simulator time, entity definitions, and existing global weather helpers.

- [ ] **Step 1: Write failing forecast shape and timing tests**

```python
# tests/test_forecast_observation.py
import numpy as np

from sim.environment.forecast import (
    current_state_observation,
    forecast_channel_names,
    future_forecast_observation,
)
from sim.train import make_native_env


def _env(hours=2):
    return make_native_env(
        episode_hours=hours,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
        include_weather_obs=False,
    )


def test_three_vessel_forecast_is_168_by_9_and_starts_next_hour():
    env = _env()
    env.reset(seed=7)
    forecast = np.asarray(future_forecast_observation(env), dtype=np.float32)
    assert forecast.shape == (168, 9)
    assert forecast_channel_names(env)[8] == "weather.global_speed_factor"
    vessel_id = env.vessel_ids[0]
    assert forecast[0, 8] == env.scenario.vessel_speed_factor[vessel_id][1]


def test_current_state_has_current_weather_but_no_future_summaries():
    env = _env()
    env.reset(seed=7)
    state = current_state_observation(env)
    assert len(state) == 51
    assert all(np.isfinite(state))


def test_last_rl_step_still_has_full_forecast_context():
    env = _env(hours=2)
    env.reset(seed=7)
    env.step({"vessels": [0] * len(env.vessel_ids), "wells": [0] * len(env.well_ids)})
    assert np.asarray(future_forecast_observation(env)).shape == (168, 9)
```

- [ ] **Step 2: Run the tests and confirm the missing module/API failures**

Run: `pytest -q tests/test_forecast_observation.py`

Expected: FAIL during import because `sim.environment.forecast` does not exist.

- [ ] **Step 3: Add current-only global weather helpers**

Add to `CCSEnv` in `src/sim/environment/env.py` and reuse them from the existing full global-weather method:

```python
def _global_current_weather_feature_names(self) -> list[str]:
    names = ["weather.speed_now"]
    for vessel_id in self.vessel_ids:
        names += [
            f"{vessel_id}.{label}.travel_hours_now"
            for label, _destination_id in self._weather_destination_slots()
        ]
    return names

def _global_current_weather_observation(self) -> list[float]:
    vessel_id = self.vessel_ids[0]
    now = self._weather_speed_at("", vessel_id, 0)
    values = [now]
    for current_vessel_id in self.vessel_ids:
        route = self._routes[current_vessel_id]
        origin_id = self._weather_reference_origin(current_vessel_id)
        for _label, destination_id in self._weather_destination_slots():
            values.append(
                self._normalized_travel_hours(origin_id, destination_id, route, now)
            )
    return values
```

- [ ] **Step 4: Implement the forecast module**

```python
# src/sim/environment/forecast.py
from __future__ import annotations

from .env import CCSEnv

FORECAST_HORIZON_H = 168


def forecast_channel_names(env: CCSEnv) -> tuple[str, ...]:
    names = [f"capture.{emitter_id}" for emitter_id in env.emitter_ids]
    names += [f"emitter_available.{emitter_id}" for emitter_id in env.emitter_ids]
    names += [f"well_available.{well_id}" for well_id in env.well_ids]
    names += [f"injectivity.{well_id}" for well_id in env.well_ids]
    names += ["weather.global_speed_factor"]
    return tuple(names)


def current_state_feature_names(env: CCSEnv) -> tuple[str, ...]:
    if env.config.include_weather_obs:
        raise ValueError("forecast experiment requires include_weather_obs=False")
    return tuple([*env.feature_names, *env._global_current_weather_feature_names()])


def current_state_observation(env: CCSEnv) -> list[float]:
    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before requesting forecast observations.")
    if env.config.include_weather_obs:
        raise ValueError("forecast experiment requires include_weather_obs=False")
    return [*env._observation(), *env._global_current_weather_observation()]


def future_forecast_observation(
    env: CCSEnv,
    horizon_h: int = FORECAST_HORIZON_H,
) -> list[list[float]]:
    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before requesting forecast observations.")
    if len(env.emitter_ids) != 3 or len(env.well_ids) != 1:
        raise ValueError("the comparison forecast schema requires 3 emitters and 1 well")
    now_index = env.scenario.step_index(env.simulator.state.time_h)
    final_index = now_index + int(horizon_h)
    if final_index >= env.scenario.n_steps:
        raise RuntimeError(
            f"forecast requires scenario index {final_index}, "
            f"but trajectory ends at {env.scenario.n_steps - 1}"
        )
    vessel_id = env.vessel_ids[0]
    rows: list[list[float]] = []
    for index in range(now_index + 1, final_index + 1):
        capture = []
        emitter_online = []
        for emitter_id in env.emitter_ids:
            emitter = env.network.entities[emitter_id]
            multiplier = float(env.scenario.emitter_availability[emitter_id][index])
            capture.append(
                min(emitter.max_production_tph, emitter.nominal_capture_tph * multiplier)
                / max(1e-9, emitter.max_production_tph)
            )
            emitter_online.append(1.0 if multiplier > 0.0 else 0.0)
        well_available = [
            1.0 if env.scenario.well_available[well_id][index] else 0.0
            for well_id in env.well_ids
        ]
        injectivity = [
            float(env.scenario.injectivity_factor[well_id][index])
            for well_id in env.well_ids
        ]
        weather = [float(env.scenario.vessel_speed_factor[vessel_id][index])]
        rows.append([*capture, *emitter_online, *well_available, *injectivity, *weather])
    return rows
```

- [ ] **Step 5: Separate episode length from scenario context**

Add `scenario_context_hours: int = 169` to `make_native_env` in `src/sim/train.py` and change only the scenario generator horizon:

```python
scenario_config=ScenarioConfig(
    episode_hours=episode_hours + scenario_context_hours,
    warm_start=warm_start,
    capture_noise_std=capture_noise_std,
    emitter_initial_fill_range=(0.0, initial_inventory_fill_max),
    terminal_initial_fill_range=(0.0, initial_inventory_fill_max),
    reservoir_initial_pressure_fill_range=(0.0, initial_inventory_fill_max),
    weather_window_rate_per_week=weather_window_rate_per_week,
    leg_wave_slowdown_multiplier=leg_wave_slowdown_multiplier,
    leg_wave_speed_factor_floor=leg_wave_speed_factor_floor,
)
```

Add a `tests/test_train.py` assertion that `CCSEnvConfig.episode_hours == 720` while `ScenarioConfig.episode_hours == 889` when context is 169.

- [ ] **Step 6: Export the forecast functions and run targeted tests**

Update `src/sim/environment/__init__.py`, then run:

`pytest -q tests/test_forecast_observation.py tests/test_train.py`

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/sim/environment/forecast.py src/sim/environment/env.py src/sim/environment/__init__.py src/sim/train.py tests/test_forecast_observation.py tests/test_train.py
git commit -m "Add shared 168-hour forecast observations"
```

---

### Task 2: Add the three Gym observation modes

**Files:**
- Create: `src/sim/environment/forecast_gym.py`
- Create: `tests/test_forecast_gym.py`
- Modify: `src/sim/environment/__init__.py`

**Interfaces:**
- Consumes: Task 1 forecast/current-state functions and existing flat native-action helpers.
- Produces: `ForecastGymEnv(env, variant)` where `variant` is `state`, `flat`, or `tcn`.
- Produces: `forecast_policy_observation(env, variant)` and `make_forecast_ppo_policy(model, variant, deterministic=False)`.

- [ ] **Step 1: Write failing observation-space tests**

```python
# tests/test_forecast_gym.py
import numpy as np

from sim.environment.forecast_gym import ForecastGymEnv
from sim.train import make_native_env


def _native():
    return make_native_env(
        episode_hours=2,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
    )


def test_state_flat_and_tcn_observation_shapes():
    state_env = ForecastGymEnv(_native(), "state")
    flat_env = ForecastGymEnv(_native(), "flat")
    tcn_env = ForecastGymEnv(_native(), "tcn")
    state, _ = state_env.reset(seed=4)
    flat, _ = flat_env.reset(seed=4)
    structured, _ = tcn_env.reset(seed=4)
    assert state.shape == (51,)
    assert flat.shape == (51 + 168 * 9,)
    assert structured["state"].shape == (51,)
    assert structured["forecast"].shape == (168, 9)
    assert np.allclose(flat[:51], state)
    assert np.allclose(flat[51:].reshape(168, 9), structured["forecast"])
```

- [ ] **Step 2: Run and confirm import failure**

Run: `pytest -q tests/test_forecast_gym.py`

Expected: FAIL because `forecast_gym.py` is missing.

- [ ] **Step 3: Implement the Gym adapter**

```python
# src/sim/environment/forecast_gym.py
from __future__ import annotations

from typing import Literal
import numpy as np
from gymnasium import Env, spaces

from .env import CCSEnv
from .forecast import (
    current_state_feature_names,
    current_state_observation,
    future_forecast_observation,
)
from .gym_adapter import flat_action_mask, native_action_from_flat

ObservationVariant = Literal["state", "flat", "tcn"]


def forecast_policy_observation(
    env: CCSEnv,
    variant: ObservationVariant,
    *,
    timeout: bool = False,
):
    state = np.asarray(current_state_observation(env), dtype=np.float32)
    if timeout:
        index = env.scenario.step_index(env.simulator.state.time_h)
        feature_index = {
            name: position
            for position, name in enumerate(current_state_feature_names(env))
        }
        for emitter_id in env.emitter_ids:
            state[feature_index[f"{emitter_id}.availability"]] = (
                env.scenario.emitter_availability[emitter_id][index]
            )
        for well_id in env.well_ids:
            state[feature_index[f"{well_id}.available"]] = float(
                env.scenario.well_available[well_id][index]
            )
            state[feature_index[f"{well_id}.injectivity"]] = (
                env.scenario.injectivity_factor[well_id][index]
            )
    if variant == "state":
        return state
    forecast = np.asarray(future_forecast_observation(env), dtype=np.float32)
    if variant == "flat":
        return np.concatenate((state, forecast.reshape(-1))).astype(np.float32)
    if variant == "tcn":
        return {"state": state, "forecast": forecast}
    raise ValueError(f"unknown forecast observation variant: {variant}")


class ForecastGymEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, env: CCSEnv, variant: ObservationVariant):
        super().__init__()
        self.env = env
        self.variant = variant
        self.action_space = spaces.MultiDiscrete(
            env.vessel_action_dims + env.well_rate_action_dims
        )
        state_size = len(current_state_feature_names(env))
        if variant == "state":
            self.observation_space = spaces.Box(-10.0, 10.0, (state_size,), np.float32)
        elif variant == "flat":
            self.observation_space = spaces.Box(
                -10.0, 10.0, (state_size + 168 * 9,), np.float32
            )
        elif variant == "tcn":
            self.observation_space = spaces.Dict({
                "state": spaces.Box(-10.0, 10.0, (state_size,), np.float32),
                "forecast": spaces.Box(-10.0, 10.0, (168, 9), np.float32),
            })
        else:
            raise ValueError(f"unknown forecast observation variant: {variant}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.env.reset(seed=episode_seed)
        return forecast_policy_observation(self.env, self.variant), {}

    def step(self, action):
        _obs, reward, terminated, truncated, info = self.env.step(
            native_action_from_flat(self.env, action)
        )
        return (
            forecast_policy_observation(
                self.env,
                self.variant,
                timeout=truncated,
            ),
            float(reward), terminated, truncated, info,
        )

    def action_masks(self):
        return flat_action_mask(
            self.env.vessel_action_mask(), self.env.well_rate_action_mask()
        )
```

- [ ] **Step 4: Add policy wrapper, exports, and terminal-step tests**

```python
def make_forecast_ppo_policy(model, variant: ObservationVariant, deterministic=False):
    def policy(env: CCSEnv):
        observation = forecast_policy_observation(env, variant)
        masks = flat_action_mask(env.vessel_action_mask(), env.well_rate_action_mask())
        action, _state = model.predict(
            observation,
            deterministic=deterministic,
            action_masks=masks,
        )
        return native_action_from_flat(env, action)
    return policy
```

Add a test that executes the two-hour environment through truncation and verifies the returned terminal observation retains the mode's declared shape and real `t+1..t+168` forecast. SB3 uses this observation for timeout bootstrapping, so keep `truncated=True` and test it through `DummyVecEnv`. Also force an availability/injectivity transition at the timeout and verify the terminal current-state fields use that hour's values without mutating the ended native environment. The pre-action observation at hour 719 also contains the complete real forecast. Export the optional-RL interfaces lazily so importing the core package does not require NumPy/Gymnasium, then run:

`pytest -q tests/test_forecast_gym.py tests/test_gym_env.py`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/sim/environment/forecast_gym.py src/sim/environment/__init__.py tests/test_forecast_gym.py
git commit -m "Add forecast-aware Gym observation modes"
```

---

### Task 3: Add the TCN-64 forecast feature extractor

**Files:**
- Create: `src/sim/environment/forecast_encoder.py`
- Create: `tests/test_forecast_encoder.py`

**Interfaces:**
- Consumes: Dict observation `{"state": [B, 51], "forecast": [B, 168, 9]}`.
- Produces: `TCNForecastExtractor` with a 128-dimensional combined output.

- [ ] **Step 1: Write the failing extractor test**

```python
# tests/test_forecast_encoder.py
import torch

from sim.environment.forecast_encoder import TCNForecastExtractor
from sim.environment.forecast_gym import ForecastGymEnv
from sim.train import make_native_env


def test_tcn_encoder_maps_structured_forecast_to_128_features():
    native = make_native_env(
        episode_hours=2,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
    )
    gym_env = ForecastGymEnv(native, "tcn")
    extractor = TCNForecastExtractor(gym_env.observation_space)
    output = extractor({
        "state": torch.zeros(2, 51),
        "forecast": torch.zeros(2, 168, 9),
    })
    assert output.shape == (2, 128)
```

- [ ] **Step 2: Run and confirm import failure**

Run: `pytest -q tests/test_forecast_encoder.py`

Expected: FAIL because the extractor is missing.

- [ ] **Step 3: Implement the minimal TCN extractor**

```python
# src/sim/environment/forecast_encoder.py
from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TCNForecastExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, state_features=64, forecast_features=64):
        super().__init__(observation_space, state_features + forecast_features)
        state_dim = observation_space["state"].shape[0]
        channels = observation_space["forecast"].shape[1]
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, state_features), nn.ReLU()
        )
        self.forecast_conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        )
        with torch.no_grad():
            conv_size = self.forecast_conv(torch.zeros(1, channels, 168)).numel()
        self.forecast_projection = nn.Sequential(
            nn.Flatten(), nn.Linear(conv_size, forecast_features), nn.ReLU()
        )

    def forward(self, observations):
        state_features = self.state_net(observations["state"])
        forecast = observations["forecast"].transpose(1, 2)
        forecast_features = self.forecast_projection(self.forecast_conv(forecast))
        return torch.cat((state_features, forecast_features), dim=1)
```

- [ ] **Step 4: Verify gradients reach the forecast encoder**

Extend the test with `output.sum().backward()` and assert every trainable convolution parameter has a gradient. Then run:

`pytest -q tests/test_forecast_encoder.py`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/sim/environment/forecast_encoder.py tests/test_forecast_encoder.py
git commit -m "Add TCN forecast feature extractor"
```

---

### Task 4: Generate and validate reusable MPC demonstrations

**Files:**
- Create: `src/sim/control/demonstrations.py`
- Create: `tests/test_demonstrations.py`

**Interfaces:**
- Produces: `MpcDemonstrationBatch`, `collect_mpc_demonstrations(env_factory, seeds, episode_hours=720)`, `save_demonstrations(batch, path)`, and `load_demonstrations(path, expected_metadata)`.
- Consumes: Task 1 observations, `RollingNativeMpcController`, existing native-action flattening, action masks, and replay helpers.

- [ ] **Step 1: Write cache round-trip and mismatch tests**

```python
def sample_batch():
    return MpcDemonstrationBatch(
        state=np.zeros((2, 51), dtype=np.float32),
        forecast=np.zeros((2, 168, 9), dtype=np.float32),
        actions=np.zeros((2, 4), dtype=np.int64),
        masks=np.ones((2, 21), dtype=bool),
        seeds=np.asarray([0, 0], dtype=np.int64),
        hours=np.asarray([0, 1], dtype=np.int64),
        metadata={"scenario": "northern_lights_phase1_3vessels", "horizon_h": 168},
    )


def test_demo_cache_round_trip_preserves_structured_arrays(tmp_path):
    batch = sample_batch()
    path = tmp_path / "demos.npz"
    save_demonstrations(batch, path)
    loaded = load_demonstrations(path, expected_metadata=batch.metadata)
    np.testing.assert_array_equal(loaded.state, batch.state)
    np.testing.assert_array_equal(loaded.forecast, batch.forecast)
    np.testing.assert_array_equal(loaded.actions, batch.actions)


def test_demo_cache_rejects_schema_mismatch(tmp_path):
    batch = sample_batch()
    path = tmp_path / "demos.npz"
    save_demonstrations(batch, path)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_demonstrations(path, expected_metadata={**batch.metadata, "horizon_h": 24})
```

- [ ] **Step 2: Run and confirm import failure**

Run: `pytest -q tests/test_demonstrations.py`

Expected: FAIL because `sim.control.demonstrations` is missing.

- [ ] **Step 3: Implement the batch and cache contract**

```python
@dataclass(frozen=True)
class MpcDemonstrationBatch:
    state: np.ndarray
    forecast: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    seeds: np.ndarray
    hours: np.ndarray
    metadata: dict[str, object]

    def observations(self, variant: str):
        if variant == "state":
            return self.state
        if variant == "flat":
            return np.concatenate(
                (self.state, self.forecast.reshape(len(self.state), -1)), axis=1
            ).astype(np.float32)
        if variant == "tcn":
            return {"state": self.state, "forecast": self.forecast}
        raise ValueError(f"unknown observation variant: {variant}")


def save_demonstrations(batch: MpcDemonstrationBatch, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state=batch.state,
        forecast=batch.forecast,
        actions=batch.actions,
        masks=batch.masks,
        seeds=batch.seeds,
        hours=batch.hours,
        metadata=np.asarray(json.dumps(batch.metadata, sort_keys=True)),
    )


def load_demonstrations(path: Path, expected_metadata: dict[str, object]):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"demonstration metadata mismatch: {mismatches}")
        batch = MpcDemonstrationBatch(
            state=archive["state"].astype(np.float32),
            forecast=archive["forecast"].astype(np.float32),
            actions=archive["actions"].astype(np.int64),
            masks=archive["masks"].astype(bool),
            seeds=archive["seeds"].astype(np.int64),
            hours=archive["hours"].astype(np.int64),
            metadata=metadata,
        )
    if batch.forecast.ndim != 3 or batch.forecast.shape[1:] != (168, 9):
        raise ValueError(f"invalid forecast shape: {batch.forecast.shape}")
    if not np.isfinite(batch.state).all() or not np.isfinite(batch.forecast).all():
        raise ValueError("demonstration observations must be finite")
    return batch
```

- [ ] **Step 4: Implement MPC collection with strict replay**

```python
def _expectation_from_snapshot(snapshot):
    names = tuple(field.name for field in dataclasses.fields(ReplaySnapshot))
    return ReplayExpectation(
        required_fields=frozenset(names),
        **{name: getattr(snapshot, name) for name in names},
    )


def collect_mpc_demonstrations(env_factory, seeds, episode_hours=720):
    state_rows, forecast_rows, action_rows, mask_rows = [], [], [], []
    seed_rows, hour_rows = [], []
    for seed in seeds:
        env = env_factory(demonstration=True)
        env.reset(seed=int(seed))
        initial_env = copy.deepcopy(env)
        controller = RollingNativeMpcController(
            env, replan_every=24, planning_horizon_h=168
        )
        native_actions = []
        for hour in range(episode_hours):
            state_rows.append(current_state_observation(env))
            forecast_rows.append(future_forecast_observation(env))
            mask_rows.append(flat_action_mask(
                env.vessel_action_mask(), env.well_rate_action_mask()
            ))
            action = controller(env)
            if not controller.last_trace_replay_is_exact:
                raise RuntimeError(f"MPC candidate replay mismatch at seed={seed}, hour={hour}")
            native_actions.append(action)
            action_rows.append(flat_action_from_native(env, action))
            seed_rows.append(seed)
            hour_rows.append(hour)
            env.step(action)

        first = replay_native_actions(
            initial_env, native_actions, horizon_h=episode_hours
        )
        second = replay_native_actions(
            initial_env,
            native_actions,
            horizon_h=episode_hours,
            expected=_expectation_from_snapshot(first.actual),
        )
        if not first.is_executable or not second.is_exact:
            raise RuntimeError(
                f"full MPC trace replay failed for seed={seed}: {second.mismatches}"
            )

    return MpcDemonstrationBatch(
        state=np.asarray(state_rows, dtype=np.float32),
        forecast=np.asarray(forecast_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.int64),
        masks=np.asarray(mask_rows, dtype=bool),
        seeds=np.asarray(seed_rows, dtype=np.int64),
        hours=np.asarray(hour_rows, dtype=np.int64),
        metadata=env_factory.metadata(),
    )
```

The environment factory must return a fresh 889 h demonstration environment and expose deterministic schema metadata through `metadata()`.

- [ ] **Step 5: Run cache and short-episode collection tests**

Run: `pytest -q tests/test_demonstrations.py tests/test_solver_replay_adapters.py`

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/sim/control/demonstrations.py tests/test_demonstrations.py
git commit -m "Add replay-validated MPC demonstration cache"
```

---

### Task 5: Generalize BC and kickstarting to structured observations

**Files:**
- Modify: `src/sim/control/imitation.py`
- Modify: `tests/test_imitation.py`

**Interfaces:**
- Consumes: either `np.ndarray` observations or `dict[str, np.ndarray]` observations.
- Preserves: existing array-only BC behaviour and dimension-specific non-WAIT weights.
- Produces: structured-observation support in `behavior_clone` and `make_kickstart_callback`.

- [ ] **Step 1: Write failing dictionary-observation helper tests**

```python
def test_index_observations_preserves_dictionary_keys():
    observations = {
        "state": np.arange(12, dtype=np.float32).reshape(3, 4),
        "forecast": np.zeros((3, 168, 9), dtype=np.float32),
    }
    indexed = imitation._index_observations(observations, np.array([2, 0]))
    assert list(indexed) == ["state", "forecast"]
    assert indexed["state"].tolist() == [observations["state"][2].tolist(), observations["state"][0].tolist()]


def test_observation_count_rejects_misaligned_dictionary():
    with pytest.raises(ValueError, match="same leading dimension"):
        imitation._observation_count({
            "state": np.zeros((2, 4)),
            "forecast": np.zeros((3, 168, 9)),
        })
```

- [ ] **Step 2: Run and confirm helper failures**

Run: `pytest -q tests/test_imitation.py`

Expected: FAIL because the structured helpers are missing.

- [ ] **Step 3: Implement shared observation helpers**

```python
def _observation_count(observations) -> int:
    if isinstance(observations, dict):
        lengths = {len(value) for value in observations.values()}
        if len(lengths) != 1:
            raise ValueError("observation arrays must share the same leading dimension")
        return lengths.pop()
    return len(observations)


def _tensor_observations(observations, device):
    import torch
    if isinstance(observations, dict):
        return {
            key: torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)
            for key, value in observations.items()
        }
    return torch.as_tensor(np.asarray(observations, dtype=np.float32), device=device)


def _index_observations(observations, index):
    if isinstance(observations, dict):
        return {key: value[index] for key, value in observations.items()}
    return observations[index]
```

- [ ] **Step 4: Use the helpers in BC and kickstarting**

In `behavior_clone`, replace the observation conversion and indexing with:

```python
obs_t = _tensor_observations(observations, device)
n = _observation_count(observations)
for start in range(0, n, batch_size):
    idx = perm[start : start + batch_size]
    batch_obs = _index_observations(obs_t, idx)
    batch_masks = mask_t[idx] if mask_t is not None else None
    if w_t is not None and w_t.ndim == 2:
        log_prob = _masked_action_log_probs(
            policy, batch_obs, act_t[idx], batch_masks
        )
    else:
        _values, log_prob, _entropy = policy.evaluate_actions(
            batch_obs, act_t[idx], action_masks=batch_masks
        )
```

In `_KickstartBC._on_training_start`, set `self._obs = _tensor_observations(observations, device)`. In `_on_rollout_end`, set `n = _observation_count(observations)` and use `batch_obs = _index_observations(self._obs, idx)` in both log-probability branches. Keep actions, masks, and weights unchanged.

- [ ] **Step 5: Add a fake structured policy test and run imitation tests**

Verify that one `behavior_clone` update calls `evaluate_actions` with both `state` and `forecast`, and that the TCN convolution parameters change after the update.

Run: `pytest -q tests/test_imitation.py tests/test_forecast_encoder.py`

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add src/sim/control/imitation.py tests/test_imitation.py
git commit -m "Support structured observations in BC kickstarting"
```

---

### Task 6: Build the three-model experiment runner

**Files:**
- Create: `scripts/compare_forecast_encoders_rl.py`
- Create: `tests/test_compare_forecast_encoders_rl.py`
- Modify: `src/sim/metrics.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_project_structure.py`

**Interfaces:**
- Produces CLI subcommands: `generate-demos`, `train`, and `report`.
- Consumes the same demonstration cache and run manifest for all variants.
- Produces BC-only and PPO checkpoints, per-seed CSV, aggregate CSV/Markdown, and immutable JSON manifests.

- [ ] **Step 1: Write failing CLI/default tests**

```python
def test_train_defaults_lock_the_comparison_protocol():
    args = parse_args([
        "train", "--variant", "tcn", "--demo-cache", "demos.npz"
    ])
    assert args.scenario == "northern_lights_phase1_3vessels"
    assert args.episode_hours == 720
    assert args.forecast_horizon_h == 168
    assert args.weather_mode == "block"
    assert args.reward_mode == "vent_first"
    assert args.kickstart_coef == 1.0
    assert args.timesteps == 100_000


def test_tcn_model_uses_multi_input_policy_and_custom_extractor():
    policy, policy_kwargs = model_policy_config("tcn")
    assert policy == "MultiInputPolicy"
    assert policy_kwargs["features_extractor_class"] is TCNForecastExtractor
```

- [ ] **Step 2: Run and confirm script import failure**

Run: `pytest -q tests/test_compare_forecast_encoders_rl.py`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement locked environment and policy factories**

```python
def make_experiment_env(args, *, demonstration=False):
    episode_hours = args.episode_hours + args.forecast_horizon_h + 1 if demonstration else args.episode_hours
    context_hours = 0 if demonstration else args.forecast_horizon_h + 1
    return make_native_env(
        episode_hours=episode_hours,
        scenario_context_hours=context_hours,
        warm_start=True,
        scenario=args.scenario,
        weather_mode="block",
        include_weather_obs=False,
        reward_mode="vent_first",
        vent_first_vent_eur_per_t=10_000.0,
        overflow_risk_eur_per_t=100.0,
        operating_cost_weight=1.0,
        enforce_full_load_dispatch=False,
    )


class ExperimentEnvFactory:
    def __init__(self, args):
        self.args = args

    def __call__(self, demonstration=False):
        return make_experiment_env(self.args, demonstration=demonstration)

    def metadata(self):
        return {
            "scenario": self.args.scenario,
            "episode_hours": self.args.episode_hours,
            "horizon_h": self.args.forecast_horizon_h,
            "forecast_shape": [168, 9],
            "forecast_channels": [
                "capture.brevik", "capture.celsio", "capture.yara_sluiskil",
                "emitter_available.brevik", "emitter_available.celsio",
                "emitter_available.yara_sluiskil",
                "well_available.aurora_well_a7_ah",
                "injectivity.aurora_well_a7_ah",
                "weather.global_speed_factor",
            ],
            "weather_mode": "block",
            "reward_mode": "vent_first",
        }


def model_policy_config(variant):
    if variant == "tcn":
        return "MultiInputPolicy", {
            "features_extractor_class": TCNForecastExtractor,
            "features_extractor_kwargs": {
                "state_features": 64,
                "forecast_features": 64,
            },
        }
    return "MlpPolicy", {}


def checkpoint_path(args, stage):
    path = Path(args.out_dir) / (
        f"{stage}_{args.variant}_seed{args.model_seed}.zip"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    demos = subparsers.add_parser("generate-demos")
    demos.add_argument("--demo-cache", required=True)
    demos.add_argument("--demo-seeds", type=int, nargs="+", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--variant", choices=["state", "flat", "tcn"], required=True)
    train.add_argument("--demo-cache", required=True)
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--bc-epochs", type=int, default=20)
    train.add_argument("--kickstart-coef", type=float, default=1.0)
    train.add_argument("--nonwait-weight", type=float, default=10.0)
    train.add_argument("--model-seed", type=int, default=0)
    train.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    train.add_argument("--device", default="auto")
    train.add_argument("--out-dir", default="output/rl_forecast")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--out-dir", default="output/rl_forecast")
    for subparser in (demos, train):
        subparser.set_defaults(
            scenario="northern_lights_phase1_3vessels",
            episode_hours=720,
            forecast_horizon_h=168,
            weather_mode="block",
            reward_mode="vent_first",
        )
    return parser.parse_args(argv)
```

- [ ] **Step 4: Implement the three subcommands**

```python
def train_variant(args):
    native_env = make_experiment_env(args, demonstration=False)
    gym_env = ForecastGymEnv(native_env, args.variant)
    policy, policy_kwargs = model_policy_config(args.variant)
    model = MaskablePPO(
        policy,
        gym_env,
        policy_kwargs=policy_kwargs,
        seed=args.model_seed,
        gamma=0.999,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        device=args.device,
        verbose=1,
    )
    expected = ExperimentEnvFactory(args).metadata()
    batch = load_demonstrations(Path(args.demo_cache), expected)
    observations = batch.observations(args.variant)
    weights = action_dimension_weights(
        batch.actions,
        vessel_count=len(native_env.vessel_ids),
        nonwait_weight=args.nonwait_weight,
    )
    behavior_clone(
        model,
        observations,
        batch.actions,
        masks=batch.masks,
        weights=weights,
        epochs=args.bc_epochs,
        batch_size=256,
        lr=1e-3,
    )
    bc_path = checkpoint_path(args, "bc")
    model.save(str(bc_path))
    callback = make_kickstart_callback(
        observations,
        batch.actions,
        batch.masks,
        weights,
        total_timesteps=args.timesteps,
        coef0=args.kickstart_coef,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    final_path = checkpoint_path(args, "ppo")
    model.save(str(final_path))
    return bc_path, final_path


def generate_demos(args):
    batch = collect_mpc_demonstrations(
        ExperimentEnvFactory(args),
        seeds=args.demo_seeds,
        episode_hours=args.episode_hours,
    )
    save_demonstrations(batch, Path(args.demo_cache))


def report(args):
    rows = []
    for path in sorted(Path(args.out_dir).glob("results_*.csv")):
        rows.extend(csv.DictReader(path.open(encoding="utf-8")))
    write_paired_report(rows, Path(args.out_dir))


def write_paired_report(rows, out_dir):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)
    summary_path = out_dir / "forecast_encoder_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["policy", "episodes", "vented_t_mean", "vented_t_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for policy, policy_rows in sorted(grouped.items()):
            values = np.asarray([float(row["vented_t"]) for row in policy_rows])
            writer.writerow({
                "policy": policy,
                "episodes": len(values),
                "vented_t_mean": float(values.mean()),
                "vented_t_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            })
```

`generate-demos` creates the cache and its manifest. `train` uses the exact flow above for every variant. `report` concatenates variant result files and computes paired deltas.

The manifest must include git commit, model variant, model seed, demonstration seeds, evaluation seeds, reward parameters, observation schema/channel order, cache SHA-256, and all PPO/BC hyperparameters.

- [ ] **Step 5: Add reference-controller and KPI rows**

Add fields to `EpisodeMetrics` and populate them in `_MetricsRecorder.result`:

```python
emitter_inventory_t: float = 0.0
vessel_inventory_t: float = 0.0
terminal_inventory_t: float = 0.0

# inside _MetricsRecorder.result
emitter_inventory_t = sum(
    env.simulator.state.entity_inventory_t.get(entity_id, 0.0)
    for entity_id in env.emitter_ids
)
vessel_inventory_t = sum(
    env.simulator.state.entity_inventory_t.get(entity_id, 0.0)
    for entity_id in env.vessel_ids
)
terminal_inventory_t = sum(
    env.simulator.state.entity_inventory_t.get(entity_id, 0.0)
    for entity_id in env.terminal_ids
)
```

Add `tests/test_metrics.py::test_episode_metrics_split_end_inventory_by_stage`, then evaluate idle, greedy, and `RollingNativeMpcController`, plus BC stochastic/deterministic and PPO stochastic/deterministic rows. Write vented tonnes as the primary column and include the three inventory totals, loss/storage rates, costs, streaks, waiting, throttling, runtime, parameter count, and inference latency.

- [ ] **Step 6: Run script and existing comparison tests**

Run: `pytest -q tests/test_compare_forecast_encoders_rl.py tests/test_compare_reward_modes_bc.py tests/test_metrics.py tests/test_project_structure.py`

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add scripts/compare_forecast_encoders_rl.py src/sim/metrics.py tests/test_compare_forecast_encoders_rl.py tests/test_metrics.py tests/test_project_structure.py
git commit -m "Add forecast encoder RL comparison runner"
```

---

### Task 7: Add Borg smoke and pilot jobs

**Files:**
- Create: `hpc/submit_forecast_mpc_demos.sh`
- Create: `hpc/submit_forecast_encoder_rl.sh`
- Modify: `tests/test_project_structure.py`

**Interfaces:**
- Demonstration job writes one audited cache under `output/rl_forecast/demos/`.
- GPU array maps task indices `0,1,2` to `state,flat,tcn` and reads the same cache.

- [ ] **Step 1: Write failing HPC-script contract tests**

Assert the demo script uses `--qos=long`, no GPU request, 720 h episodes, 168 h horizon, and block weather. Assert the training script uses `--array=0-2` by default, one GPU, `long` QoS, and modulo/division mapping that also supports a formal `sbatch --array=0-8` override for three variants by three model seeds.

- [ ] **Step 2: Create the CPU demonstration script**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=ccs_mpc_demos
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/mpc_demos-%j.out
#SBATCH -e logs/mpc_demos-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus
cd "${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
mkdir -p logs output/rl_forecast/demos
python -u scripts/compare_forecast_encoders_rl.py generate-demos \
  --demo-cache output/rl_forecast/demos/mpc_720h_30eps.npz \
  --demo-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
```

- [ ] **Step 3: Create the GPU array training script**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=ccs_forecast_rl
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o logs/forecast_rl-%A_%a.out
#SBATCH -e logs/forecast_rl-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus
cd "${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
mkdir -p logs output/rl_forecast
VARIANTS=(state flat tcn)
MODEL_SEEDS=(0 1 2)
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
MODEL_SEED="${MODEL_SEEDS[$SEED_INDEX]}"
# shellcheck disable=SC2206
EVAL_SEEDS_ARGS=(${EVAL_SEEDS:-101 102 103 104 105})
python -u scripts/compare_forecast_encoders_rl.py train \
  --variant "$VARIANT" \
  --demo-cache output/rl_forecast/demos/mpc_720h_30eps.npz \
  --timesteps "${TIMESTEPS:-100000}" \
  --bc-epochs "${BC_EPOCHS:-20}" \
  --kickstart-coef "${KICKSTART_COEF:-1.0}" \
  --model-seed "$MODEL_SEED" \
  --eval-seeds "${EVAL_SEEDS_ARGS[@]}" \
  --device cuda \
  --out-dir output/rl_forecast
```

- [ ] **Step 4: Run shell-contract tests**

Run: `pytest -q tests/test_project_structure.py`

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

```bash
git add hpc/submit_forecast_mpc_demos.sh hpc/submit_forecast_encoder_rl.sh tests/test_project_structure.py
git commit -m "Add Borg forecast RL experiment jobs"
```

---

### Task 8: Verify locally and launch the pilot safely

**Files:**
- Modify only if verification exposes a defect in files owned by Tasks 1–7.
- Produce at runtime: `output/rl_forecast/` smoke artifacts and manifests.

**Interfaces:**
- Consumes every preceding task.
- Produces test evidence and, after local success, Borg job IDs for demo generation and three pilot variants.

- [ ] **Step 1: Run the focused suite**

Run:

```bash
pytest -q tests/test_forecast_observation.py tests/test_forecast_gym.py tests/test_forecast_encoder.py tests/test_demonstrations.py tests/test_imitation.py tests/test_compare_forecast_encoders_rl.py tests/test_train.py tests/test_metrics.py tests/test_project_structure.py
```

Expected: all tests PASS.

- [ ] **Step 2: Run the full local suite**

Run: `pytest -q`

Expected: all non-dependency-skipped tests PASS.

- [ ] **Step 3: Run a local CPU smoke comparison**

Generate one short cache and run all three variants with one BC epoch and a minimal legal PPO rollout. Use a dedicated `output/rl_forecast/smoke/` directory and verify that state, flat, and TCN checkpoints save and reload.

- [ ] **Step 4: Sync only the committed project state to Borg**

From the local project root, sync to `/scratch_root/hx721/CCS_RLLLM/`, excluding `.git/`, `__pycache__/`, `.venv/`, `wandb/`, and large unrelated output directories. Confirm the remote commit and Python environment before submission.

- [ ] **Step 5: Submit a short-queue Borg smoke job**

Override the formal scripts to use one demonstration seed, one BC epoch, and 2,048 PPO timesteps. Verify `which python`, PyTorch/CUDA, observation shapes, finite BC/PPO losses, checkpoints, and logs before formal work.

- [ ] **Step 6: Submit demonstration and pilot jobs**

Submit `hpc/submit_forecast_mpc_demos.sh`, record its job ID, and wait for successful cache audit. Only then submit `hpc/submit_forecast_encoder_rl.sh`; record the array job ID, git commit, cache hash, and output directory.

- [ ] **Step 7: Monitor and retrieve results**

Use `squeue`, `sacct`, and per-task logs. After all three variants complete, run the report subcommand, retrieve CSV/Markdown/manifests/checkpoints, and summarize paired venting deltas, end inventory, costs, imitation accuracy, and runtime.

- [ ] **Step 8: Launch the formal three-seed array after pilot acceptance**

Submit the same training script with `sbatch --array=0-8 --export=ALL,EVAL_SEEDS='101 102 103 104 105 106 107 108 109 110' hpc/submit_forecast_encoder_rl.sh`. The modulo/division mapping runs `(state, flat, tcn) x (seed 0, seed 1, seed 2)`. Regenerate the aggregate report and record the formal array job ID separately from the pilot.

- [ ] **Step 9: Commit any verification-only fixes separately**

If verification required code changes, stage only the files directly involved and commit with a message that names the verified defect. Do not commit generated checkpoints, large demonstration caches, or unrelated user files.

## Plan Self-Review

- Spec coverage: forecast schema, weather channel, 889/720 boundary handling, three variants, shared MPC cache, structured imitation, fixed reward, paired evaluation, HPC separation, and failure handling all map to explicit tasks.
- Completeness scan: every implementation step names concrete APIs, files, commands, expected results, and failure behaviour.
- Type consistency: forecast is time-major NumPy `[N, 168, 9]`, structured Gym observation is `{"state", "forecast"}`, PyTorch TCN input is transposed to `[N, 9, 168]`, and all three variants consume the same cached actions/masks.
