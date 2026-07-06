"""Generate one episode's time-varying disturbance trajectories.

A :class:`Scenario` is the exogenous part of an episode: randomized initial
conditions plus per-hour trajectories for capture availability, weather speed
factors, well maintenance and injectivity decline. It writes the
current step's values into :class:`PhysicalState`; it never chooses actions.

Runtime operations read those values through
``sim.scenario_generation.disturbance_resolver``, which applies the "state
override first, nominal entity fallback second" rule.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..entities.emitter import Emitter
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell, Reservoir
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel

# Regional weather as a sticky 3-state Markov chain; each state maps to a vessel
# speed multiplier. Shared across vessels because weather is a regional driver.
_WEATHER_SPEED = {"calm": 1.0, "breeze": 0.85, "storm": 0.6}
_WEATHER_TRANSITIONS = {
    "calm": (("calm", 0.90), ("breeze", 0.10)),
    "breeze": (("calm", 0.30), ("breeze", 0.50), ("storm", 0.20)),
    "storm": (("breeze", 0.50), ("storm", 0.50)),
}


@dataclass
class ScenarioConfig:
    """Knobs for the disturbance generators. Defaults are mild but non-trivial."""

    episode_hours: int = 168
    time_step_hours: float = 1.0

    # Capture-plant multiplier: bounded profile jitter + occasional trips.
    # Historical field name retained; interpreted as +/- fractional half-width.
    capture_noise_std: float = 0.10
    capture_outage_rate_per_week: float = 0.5
    capture_outage_mean_hours: float = 12.0

    # Weather -> vessel speed (Markov chain above). Disable by setting False.
    enable_weather: bool = True

    # Injection-well maintenance windows.
    well_maintenance_rate_per_week: float = 0.3
    well_maintenance_mean_hours: float = 12.0

    # Injectivity decline over the episode (time proxy for cumulative injection).
    injectivity_max_decline: float = 0.15
    injectivity_floor: float = 0.3
    injectivity_noise_std: float = 0.01

    # Initial-condition randomization (fraction-of-capacity ranges).
    randomize_initial_inventory: bool = True
    emitter_initial_fill_range: tuple[float, float] = (0.0, 0.5)
    terminal_initial_fill_range: tuple[float, float] = (0.0, 0.5)

    # Warm-start randomization of the *slow* variables (reservoir pressure and
    # injectivity). Off by default. Turning it on lets short episodes start from
    # mid-life reservoir/injectivity states, so a policy trained on 168 h windows
    # has seen the high-pressure / declined-injectivity regimes it will encounter
    # during a long (e.g. one-year) evaluation rollout.
    warm_start: bool = False
    injectivity_warmstart_min: float = 0.5
    reservoir_initial_pressure_fill_range: tuple[float, float] = (0.0, 0.5)


@dataclass
class Scenario:
    """A sampled episode: pre-computed per-step disturbance trajectories."""

    time_step_hours: float
    n_steps: int
    initial_inventory_t: dict[str, float] = field(default_factory=dict)
    emitter_availability: dict[str, list[float]] = field(default_factory=dict)
    vessel_speed_factor: dict[str, list[float]] = field(default_factory=dict)
    leg_speed_factor: dict[str, list[float]] = field(default_factory=dict)
    well_available: dict[str, list[bool]] = field(default_factory=dict)
    injectivity_factor: dict[str, list[float]] = field(default_factory=dict)
    seed: int | None = None

    def step_index(self, time_h: float) -> int:
        """Step index for a wall-clock time, clamped to the episode horizon."""
        index = int(round(time_h / self.time_step_hours))
        return max(0, min(self.n_steps - 1, index))

    def apply_initial(self, state: PhysicalState) -> None:
        """Seed an initial state with this scenario's starting inventories."""
        for entity_id, inventory_t in self.initial_inventory_t.items():
            state.entity_inventory_t[entity_id] = inventory_t

    def apply_to_state(self, state: PhysicalState, time_h: float) -> None:
        """Write the disturbances for the step beginning at ``time_h``."""
        i = self.step_index(time_h)
        state.emitter_availability = {k: v[i] for k, v in self.emitter_availability.items()}
        state.vessel_speed_factor = {k: v[i] for k, v in self.vessel_speed_factor.items()}
        state.leg_speed_factor = {k: v[i] for k, v in self.leg_speed_factor.items()}
        state.well_available = {k: v[i] for k, v in self.well_available.items()}
        state.injectivity_factor = {k: v[i] for k, v in self.injectivity_factor.items()}


class ScenarioGenerator:
    """Samples reproducible :class:`Scenario` objects for a network."""

    def __init__(self, config: ScenarioConfig | None = None, seed: int | None = None) -> None:
        self.config = config or ScenarioConfig()
        self.seed = seed

    def sample(self, network, seed: int | None = None) -> Scenario:
        config = self.config
        episode_seed = seed if seed is not None else self.seed
        master = random.Random(episode_seed)
        # Independent child streams so toggling one channel does not shift others.
        capture_rng = random.Random(master.random())
        weather_rng = random.Random(master.random())
        maintenance_rng = random.Random(master.random())
        injectivity_rng = random.Random(master.random())
        init_rng = random.Random(master.random())

        dt = config.time_step_hours
        n_steps = max(1, int(round(config.episode_hours / dt)))

        emitters = _ids_of_type(network, Emitter)
        vessels = _ids_of_type(network, Vessel)
        wells = _ids_of_type(network, InjectionWell)
        terminals = network._entities_of_type(Terminal)
        reservoirs = network._entities_of_type(Reservoir)

        emitter_availability = {
            emitter_id: _capture_availability_series(capture_rng, n_steps, dt, config)
            for emitter_id in emitters
        }
        weather_speed = _weather_speed_series(weather_rng, n_steps, config)
        vessel_speed_factor = {vessel_id: list(weather_speed) for vessel_id in vessels}
        well_available = {
            well_id: _availability_from_outage(
                _outage_series(
                    maintenance_rng, n_steps, dt,
                    config.well_maintenance_rate_per_week, config.well_maintenance_mean_hours,
                )
            )
            for well_id in wells
        }
        injectivity_factor = {}
        for well_id in wells:
            start_level = (
                injectivity_rng.uniform(config.injectivity_warmstart_min, 1.0)
                if config.warm_start
                else 1.0
            )
            injectivity_factor[well_id] = _injectivity_series(
                injectivity_rng, n_steps, config, start_level=start_level
            )
        initial_inventory_t = self._initial_inventory(
            network, init_rng, emitters, terminals, reservoirs
        )

        return Scenario(
            time_step_hours=dt,
            n_steps=n_steps,
            initial_inventory_t=initial_inventory_t,
            emitter_availability=emitter_availability,
            vessel_speed_factor=vessel_speed_factor,
            well_available=well_available,
            injectivity_factor=injectivity_factor,
            seed=episode_seed,
        )

    def _initial_inventory(self, network, rng, emitters, terminals, reservoirs) -> dict[str, float]:
        inventory: dict[str, float] = {}
        config = self.config
        if config.randomize_initial_inventory:
            lo_e, hi_e = config.emitter_initial_fill_range
            for emitter_id in emitters:
                emitter = network.entities[emitter_id]
                inventory[emitter_id] = rng.uniform(lo_e, hi_e) * emitter.buffer_capacity_t
            lo_t, hi_t = config.terminal_initial_fill_range
            for terminal_id, terminal in terminals.items():
                inventory[terminal_id] = rng.uniform(lo_t, hi_t) * terminal.storage_capacity_t
        if config.warm_start:
            # Pre-fill the reservoir so pressure starts mid-life: this is the slow
            # variable a short cold-start episode would otherwise never expose.
            lo_r, hi_r = config.reservoir_initial_pressure_fill_range
            for reservoir_id, reservoir in reservoirs.items():
                inventory[reservoir_id] = rng.uniform(lo_r, hi_r) * reservoir.pressure_limited_capacity_t()
        return inventory


def _ids_of_type(network, entity_type: type) -> list[str]:
    return list(network._entities_of_type(entity_type))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _outage_series(rng, n_steps: int, dt: float, rate_per_week: float, mean_hours: float) -> list[bool]:
    """A boolean outage trajectory from a simple start/stop random process."""
    if rate_per_week <= 0.0 or mean_hours <= 0.0:
        return [False] * n_steps
    start_p = _clamp(rate_per_week * dt / 168.0, 0.0, 1.0)
    end_p = _clamp(dt / mean_hours, 0.0, 1.0)
    series: list[bool] = []
    in_outage = False
    for _ in range(n_steps):
        if in_outage:
            series.append(True)
            if rng.random() < end_p:
                in_outage = False
        elif rng.random() < start_p:
            in_outage = True
            series.append(True)
        else:
            series.append(False)
    return series


def _availability_from_outage(outage: list[bool]) -> list[bool]:
    return [not is_out for is_out in outage]


def _capture_availability_series(rng, n_steps: int, dt: float, config: ScenarioConfig) -> list[float]:
    outage = _outage_series(
        rng, n_steps, dt, config.capture_outage_rate_per_week, config.capture_outage_mean_hours
    )
    series: list[float] = []
    for is_out in outage:
        if is_out:
            series.append(0.0)
            continue
        if config.capture_noise_std > 0.0:
            noisy = rng.uniform(1.0 - config.capture_noise_std, 1.0 + config.capture_noise_std)
        else:
            noisy = 1.0
        series.append(max(0.0, noisy))
    return series


def _weather_speed_series(rng, n_steps: int, config: ScenarioConfig) -> list[float]:
    if not config.enable_weather:
        return [1.0] * n_steps
    state = "calm"
    series: list[float] = []
    for _ in range(n_steps):
        series.append(_WEATHER_SPEED[state])
        state = _next_weather_state(rng, state)
    return series


def _next_weather_state(rng, state: str) -> str:
    roll = rng.random()
    cumulative = 0.0
    for next_state, probability in _WEATHER_TRANSITIONS[state]:
        cumulative += probability
        if roll < cumulative:
            return next_state
    return state


def _injectivity_series(
    rng, n_steps: int, config: ScenarioConfig, start_level: float = 1.0
) -> list[float]:
    slope = rng.uniform(0.0, config.injectivity_max_decline)
    series: list[float] = []
    for step in range(n_steps):
        progress = step / (n_steps - 1) if n_steps > 1 else 0.0
        noise = rng.gauss(0.0, config.injectivity_noise_std) if config.injectivity_noise_std > 0.0 else 0.0
        series.append(_clamp(start_level - slope * progress + noise, config.injectivity_floor, start_level))
    return series


