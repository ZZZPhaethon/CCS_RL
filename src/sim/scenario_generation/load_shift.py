"""Load-shifting scenario generator: a rotating capture "hot spot".

This is a standalone add-on that does not modify any existing scenario code. It
subclasses :class:`ScenarioGenerator`, keeps everything the base generator
produces (noise, outages, weather, maintenance, injectivity), and only rewrites
the per-emitter capture-availability series so that the *location* of the load
rotates over time: in each phase one emitter runs near full capture while the
others are throttled, cycling to the next emitter each phase.

Why: with a rotating hot spot the total captured CO2 stays roughly the same, but
*which* emitter dominates keeps changing. A fixed vessel->emitter assignment
(static planner) cannot keep up - its vessel idles when its emitter is cold and
is overwhelmed when the hot emitter sits in another vessel's region - while a
dynamic, goal-conditioned policy can reallocate to follow the hot spot. This is
the regime where learning should beat a static heuristic.

Usage:
    from sim.scenario_generation.load_shift import (
        LoadShiftScenarioGenerator, LoadShiftConfig,
    )
    gen = LoadShiftScenarioGenerator(
        config=ScenarioConfig(episode_hours=720),
        load_shift=LoadShiftConfig(phase_hours=120, hot_level=1.0, cold_level=0.2),
    )
    scenario = gen.sample(network, seed=0)
"""

from __future__ import annotations

from dataclasses import dataclass

from .generator import Scenario, ScenarioConfig, ScenarioGenerator


@dataclass
class LoadShiftConfig:
    """Parameters of the rotating capture hot spot.

    ``hot_count`` is how many emitters are hot simultaneously (the hot *set*
    rotates each phase). With a single hot emitter a static per-emitter
    assignment still covers it; setting ``hot_count`` so that two same-region
    emitters can be hot together creates a coverage overload that only a dynamic
    policy can fill.
    """

    phase_hours: float = 120.0  # how long each hot set persists
    hot_level: float = 1.0      # capture-availability multiplier for hot emitters
    cold_level: float = 0.2     # multiplier for the throttled (cold) emitters
    hot_count: int = 1          # number of simultaneously hot emitters


class LoadShiftScenarioGenerator(ScenarioGenerator):
    """``ScenarioGenerator`` that rotates a capture hot spot among emitters."""

    def __init__(
        self,
        config: ScenarioConfig | None = None,
        seed: int | None = None,
        load_shift: LoadShiftConfig | None = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        self.load_shift = load_shift or LoadShiftConfig()

    def sample(self, network, seed: int | None = None) -> Scenario:
        scenario = super().sample(network, seed=seed)
        ls = self.load_shift
        emitters = sorted(scenario.emitter_availability)
        k = len(emitters)
        if k == 0:
            return scenario

        dt = scenario.time_step_hours
        steps_per_phase = max(1, int(round(ls.phase_hours / dt)))

        hot_count = max(1, min(ls.hot_count, k))
        rotated: dict[str, list[float]] = {emitter_id: [] for emitter_id in emitters}
        for t in range(scenario.n_steps):
            phase = (t // steps_per_phase) % k
            hot_set = {(phase + offset) % k for offset in range(hot_count)}
            for i, emitter_id in enumerate(emitters):
                base = scenario.emitter_availability[emitter_id][t]  # keep base noise/outages
                level = ls.hot_level if i in hot_set else ls.cold_level
                rotated[emitter_id].append(base * level)

        for emitter_id in emitters:
            scenario.emitter_availability[emitter_id] = rotated[emitter_id]
        return scenario
