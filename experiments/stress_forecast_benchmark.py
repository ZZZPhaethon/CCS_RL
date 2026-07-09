"""Stress-test 3-vessel Phase 1 dispatch under forecast-limited disturbances.

The scenario is a Northern-Lights-calibrated counterfactual: same emitters,
terminal, routes, vessels, and well model, but with tighter Yara/terminal
buffers plus stochastic weather and emitter production shocks.  Controllers may
only look ahead over the configured forecast horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from statistics import pstdev
from typing import Callable

from sim.control.baselines import greedy_shuttle_policy
from sim.control.rule_based import RuleBasedActionGenerator
from sim.control.rolling_milp import RollingMilpController
from sim.economics import CostModel, EconomicParameters
from sim.environment import (
    CCSEnv,
    CCSEnvConfig,
    OFF_WELL_RATE_INDEX,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    WELL_RATE_LEVELS_MTPA,
)
from sim.entities.emitter import Emitter
from sim.entities.vessel import Vessel
from sim.metrics import EpisodeMetrics, run_episode
from sim.network_scenarios import build_fixed_scenario_demo, fixed_scenario_locations
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator

Policy = Callable[[object], dict[str, list]]
PolicyFactory = Callable[[object], Policy]
RULE_BASED_WELL_RATE_INDEX = list(WELL_RATE_LEVELS_MTPA).index(1.5)


class ForecastStressScenarioGenerator(ScenarioGenerator):
    """Add strong but reproducible weather and emitter shocks to a base scenario."""

    def __init__(
        self,
        config: ScenarioConfig,
        *,
        weather_window_count: int = 4,
        emitter_event_count: int = 6,
    ) -> None:
        super().__init__(config=config)
        self.weather_window_count = weather_window_count
        self.emitter_event_count = emitter_event_count
        self.last_events: list[dict[str, object]] = []

    def sample(self, network, seed: int | None = None) -> Scenario:
        scenario = super().sample(network, seed=seed)
        rng = random.Random(f"forecast-stress:{seed}")
        events: list[dict[str, object]] = []
        self._apply_weather_windows(network, scenario, rng, events)
        self._apply_emitter_events(network, scenario, rng, events)
        self.last_events = events
        return scenario

    def _apply_weather_windows(self, network, scenario: Scenario, rng, events) -> None:
        vessel_ids = list(network._entities_of_type(Vessel))
        for vessel_id in vessel_ids:
            scenario.vessel_speed_factor.setdefault(vessel_id, [1.0] * scenario.n_steps)

        for index in range(self.weather_window_count):
            week_start = min(scenario.n_steps - 1, index * 168)
            start = min(scenario.n_steps - 1, week_start + rng.randint(0, 120))
            duration = rng.randint(18, 60)
            factor = rng.uniform(0.42, 0.68)
            end = min(scenario.n_steps, start + duration)
            for vessel_id in vessel_ids:
                series = scenario.vessel_speed_factor[vessel_id]
                vessel_factor = max(0.35, min(1.0, factor + rng.uniform(-0.05, 0.05)))
                for t in range(start, end):
                    series[t] = min(series[t], vessel_factor)
            events.append(
                {
                    "type": "weather_slowdown",
                    "start_h": start,
                    "duration_h": end - start,
                    "speed_factor": round(factor, 3),
                }
            )

    def _apply_emitter_events(self, network, scenario: Scenario, rng, events) -> None:
        emitter_ids = list(network._entities_of_type(Emitter))
        for emitter_id in emitter_ids:
            scenario.emitter_availability.setdefault(emitter_id, [1.0] * scenario.n_steps)

        for _ in range(self.emitter_event_count):
            emitter_id = rng.choice(emitter_ids)
            event_type = "outage" if rng.random() < 0.55 else "high_output"
            factor = 0.0 if event_type == "outage" else rng.uniform(1.35, 1.75)
            start = rng.randint(0, max(0, scenario.n_steps - 24))
            duration = rng.randint(24, 72)
            end = min(scenario.n_steps, start + duration)
            series = scenario.emitter_availability[emitter_id]
            for t in range(start, end):
                if event_type == "outage":
                    series[t] = 0.0
                else:
                    series[t] = max(series[t], factor)
            events.append(
                {
                    "type": f"emitter_{event_type}",
                    "emitter_id": emitter_id,
                    "start_h": start,
                    "duration_h": end - start,
                    "availability": round(factor, 3),
                }
            )


def make_stress_env(
    *,
    cap_hours: int,
    yara_buffer_t: float,
    terminal_buffer_t: float,
    economics: EconomicParameters,
    config: ScenarioConfig,
) -> tuple[object, ForecastStressScenarioGenerator]:
    network, _state = build_fixed_scenario_demo("northern_lights_phase1_3vessels")
    env = CCSEnv(
        network,
        fixed_scenario_locations("northern_lights_phase1_3vessels"),
        scenario_generator=ScenarioGenerator(config=config),
        cost_model=CostModel(economics),
        config=CCSEnvConfig(episode_hours=cap_hours),
    )
    yara = env.network.entities["yara_sluiskil"]
    env.network.entities["yara_sluiskil"] = replace(yara, buffer_capacity_t=yara_buffer_t)
    terminal = env.network.entities["oygarden_terminal"]
    env.network.entities["oygarden_terminal"] = replace(terminal, storage_capacity_t=terminal_buffer_t)
    generator = ForecastStressScenarioGenerator(config)
    env.scenario_generator = generator
    return env, generator


def forecast_balanced_policy(env) -> dict[str, list]:
    """Greedy variant with one-week overflow scoring and limited milk-run behavior."""
    state = env.simulator.state
    vessel_actions: list[int] = []
    vessel_masks = env.vessel_action_mask()
    for index, vessel_id in enumerate(env.vessel_ids):
        mask = vessel_masks[index]
        cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
        vessel = env.network.entities[vessel_id]
        berth = state.vessel_berths.get(vessel_id)

        if berth in env.terminal_ids and cargo_t > 1e-9:
            vessel_actions.append(VESSEL_WAIT)
            continue

        if mask[VESSEL_GO_TERMINAL] and cargo_t >= vessel.capacity_t - 1e-9:
            if _should_wait_for_weather(env, vessel_id) and not _source_overflow_imminent(env):
                vessel_actions.append(VESSEL_WAIT)
            else:
                vessel_actions.append(VESSEL_GO_TERMINAL)
            continue

        if berth in env.emitter_ids and cargo_t < vessel.capacity_t - 1e-9:
            local_inventory = state.entity_inventory_t.get(str(berth), 0.0)
            if local_inventory > 500.0:
                vessel_actions.append(VESSEL_WAIT)
                continue
            milk_run_action = _best_forecast_emitter_action(env, mask, exclude=str(berth))
            if milk_run_action is not None and cargo_t > 0.1 * vessel.capacity_t:
                vessel_actions.append(milk_run_action)
                continue

        action = _best_forecast_emitter_action(env, mask)
        if action is not None:
            vessel_actions.append(action)
        elif mask[VESSEL_GO_TERMINAL] and cargo_t > 1e-9:
            vessel_actions.append(VESSEL_GO_TERMINAL)
        else:
            vessel_actions.append(VESSEL_WAIT)

    return {
        "vessels": vessel_actions,
        "wells": [env.highest_feasible_well_rate_index(well_id) for well_id in env.well_ids],
    }


def forecast_dispatch_policy(env) -> dict[str, list]:
    """Forecast emitter overflow without weather waiting.

    This keeps the high-throughput behavior that makes greedy strong, while
    changing the emitter choice from current-buffer pressure to one-week
    forecasted overflow pressure.  Partially loaded vessels may top up at
    another emitter when the current berth has little immediate supply.
    """
    state = env.simulator.state
    vessel_actions: list[int] = []
    vessel_masks = env.vessel_action_mask()
    for index, vessel_id in enumerate(env.vessel_ids):
        mask = vessel_masks[index]
        cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
        vessel = env.network.entities[vessel_id]
        berth = state.vessel_berths.get(vessel_id)

        if berth in env.terminal_ids and cargo_t > 1e-9:
            vessel_actions.append(VESSEL_WAIT)
            continue
        if mask[VESSEL_GO_TERMINAL] and cargo_t >= vessel.capacity_t - 1e-9:
            vessel_actions.append(VESSEL_GO_TERMINAL)
            continue
        if berth in env.emitter_ids and cargo_t < vessel.capacity_t - 1e-9:
            berth_id = str(berth)
            local_inventory = state.entity_inventory_t.get(berth_id, 0.0)
            local_score = _forecast_emitter_score(env, berth_id)
            best_other = _best_forecast_emitter_action(env, mask, exclude=berth_id)
            best_other_score = _action_score(env, best_other)
            if (
                cargo_t > 0.15 * vessel.capacity_t
                and best_other is not None
                and local_inventory < 500.0
                and best_other_score > local_score * 1.25
            ):
                vessel_actions.append(best_other)
            else:
                vessel_actions.append(VESSEL_WAIT)
            continue

        action = _best_forecast_emitter_action(env, mask)
        if action is not None:
            vessel_actions.append(action)
        elif mask[VESSEL_GO_TERMINAL] and cargo_t > 1e-9:
            vessel_actions.append(VESSEL_GO_TERMINAL)
        else:
            vessel_actions.append(VESSEL_WAIT)

    return {
        "vessels": vessel_actions,
        "wells": [env.highest_feasible_well_rate_index(well_id) for well_id in env.well_ids],
    }


def milkrun_greedy_policy(env) -> dict[str, list]:
    """Greedy dispatch with a narrow milk-run override for slow local fill-ups."""
    state = env.simulator.state
    vessel_actions: list[int] = []
    vessel_masks = env.vessel_action_mask()
    for index, vessel_id in enumerate(env.vessel_ids):
        mask = vessel_masks[index]
        cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
        vessel = env.network.entities[vessel_id]
        berth = state.vessel_berths.get(vessel_id)

        if berth in env.terminal_ids and cargo_t > 1e-9:
            vessel_actions.append(VESSEL_WAIT)
            continue
        if mask[VESSEL_GO_TERMINAL] and cargo_t >= vessel.capacity_t - 1e-9:
            vessel_actions.append(VESSEL_GO_TERMINAL)
            continue
        if berth in env.emitter_ids and cargo_t < vessel.capacity_t - 1e-9:
            berth_id = str(berth)
            local_fill_h = _hours_until_full_if_wait(env, vessel_id, berth_id)
            local_score = _forecast_emitter_score(env, berth_id)
            other_action = _best_forecast_emitter_action(env, mask, exclude=berth_id)
            other_score = _action_score(env, other_action)
            if (
                cargo_t >= 0.2 * vessel.capacity_t
                and local_fill_h > 36
                and other_action is not None
                and other_score > local_score + 2_000.0
            ):
                vessel_actions.append(other_action)
            elif _current_emitter_supply_score(env, berth_id) > 1e-9:
                vessel_actions.append(VESSEL_WAIT)
            elif other_action is not None:
                vessel_actions.append(other_action)
            else:
                vessel_actions.append(VESSEL_WAIT)
            continue

        greedy_action = _best_current_emitter_action(env, mask)
        if greedy_action is not None:
            vessel_actions.append(greedy_action)
        elif mask[VESSEL_GO_TERMINAL] and cargo_t > 1e-9:
            vessel_actions.append(VESSEL_GO_TERMINAL)
        else:
            vessel_actions.append(VESSEL_WAIT)

    return {
        "vessels": vessel_actions,
        "wells": [env.highest_feasible_well_rate_index(well_id) for well_id in env.well_ids],
    }


def rule_based_env_policy(env) -> Policy:
    generator = RuleBasedActionGenerator(env.network, env._routes)

    def policy(current_env) -> dict[str, list]:
        frame = generator.next_action_frame(current_env.simulator.state)
        vessel_actions = [VESSEL_WAIT] * len(current_env.vessel_ids)
        well_actions = [RULE_BASED_WELL_RATE_INDEX] * len(current_env.well_ids)
        mask = current_env.vessel_action_mask()
        vessel_index = {vessel_id: i for i, vessel_id in enumerate(current_env.vessel_ids)}
        well_index = {well_id: i for i, well_id in enumerate(current_env.well_ids)}

        for proposal in frame.proposals:
            if proposal.verb == "sail_to" and proposal.entity_id in vessel_index:
                i = vessel_index[proposal.entity_id]
                route = current_env._routes[proposal.entity_id]
                destination = str(proposal.params["destination_id"])
                if destination in current_env.emitter_ids:
                    emitter_action = current_env.vessel_go_emitter_action(destination)
                    if mask[i][emitter_action]:
                        vessel_actions[i] = emitter_action
                elif destination == route["destination"] and mask[i][VESSEL_GO_TERMINAL]:
                    vessel_actions[i] = VESSEL_GO_TERMINAL
            elif proposal.verb == "set_well_split":
                for well_id, split in proposal.params["well_splits"].items():
                    if well_id not in well_index:
                        continue
                    well_actions[well_index[well_id]] = (
                        RULE_BASED_WELL_RATE_INDEX if float(split) > 0.0 else OFF_WELL_RATE_INDEX
                    )

        return {"vessels": vessel_actions, "wells": well_actions}

    return policy


def _best_current_emitter_action(env, mask: list[bool]) -> int | None:
    best: tuple[float, int] | None = None
    for emitter_id in env.emitter_ids:
        action = env.vessel_go_emitter_action(emitter_id)
        if not mask[action]:
            continue
        score = _current_emitter_supply_score(env, emitter_id)
        if best is None or score > best[0]:
            best = (score, action)
    return None if best is None or best[0] <= 1e-9 else best[1]


def _current_emitter_supply_score(env, emitter_id: str) -> float:
    emitter = env.network.entities[emitter_id]
    state = env.simulator.state
    availability = state.emitter_availability.get(emitter_id, emitter.availability)
    return state.entity_inventory_t.get(emitter_id, 0.0) + emitter.nominal_capture_tph * max(0.0, float(availability))


def _hours_until_full_if_wait(env, vessel_id: str, emitter_id: str, horizon_h: int = 168) -> float:
    if env.scenario is None:
        return float("inf")
    state = env.simulator.state
    vessel = env.network.entities[vessel_id]
    emitter = env.network.entities[emitter_id]
    cargo = state.entity_inventory_t.get(vessel_id, 0.0)
    source = state.entity_inventory_t.get(emitter_id, 0.0)
    load_rate = min(vessel.loading_rate_tph, emitter.loading_rate_tph) * env.network.time_step_hours
    series = env.scenario.emitter_availability.get(emitter_id, [])
    for offset in range(horizon_h):
        time_h = state.time_h + offset * env.network.time_step_hours
        idx = env.scenario.step_index(time_h)
        availability = series[idx] if series else state.emitter_availability.get(emitter_id, emitter.availability)
        source = min(
            emitter.buffer_capacity_t,
            source + emitter.capture_rate_tph_at(time_h) * max(0.0, float(availability)),
        )
        loaded = min(load_rate, source, max(0.0, vessel.capacity_t - cargo))
        source -= loaded
        cargo += loaded
        if cargo >= vessel.capacity_t - 1e-9:
            return float(offset + 1)
    return float("inf")


def _best_forecast_emitter_action(env, mask: list[bool], *, exclude: str | None = None) -> int | None:
    best: tuple[float, int] | None = None
    for emitter_id in env.emitter_ids:
        if emitter_id == exclude:
            continue
        action = env.vessel_go_emitter_action(emitter_id)
        if not mask[action]:
            continue
        score = _forecast_emitter_score(env, emitter_id)
        if best is None or score > best[0]:
            best = (score, action)
    return None if best is None or best[0] <= 1e-9 else best[1]


def _action_score(env, action: int | None) -> float:
    if action is None:
        return 0.0
    for emitter_id in env.emitter_ids:
        if action == env.vessel_go_emitter_action(emitter_id):
            return _forecast_emitter_score(env, emitter_id)
    return 0.0


def _forecast_emitter_score(env, emitter_id: str, horizon_h: int = 168) -> float:
    overflow = _forecast_overflow_score(env, emitter_id, horizon_h)
    emitter = env.network.entities[emitter_id]
    inventory = env.simulator.state.entity_inventory_t.get(emitter_id, 0.0)
    future_capture = _forecast_capture_t(env, emitter_id, horizon_h)
    return max(0.0, overflow) * 4.0 + inventory + 0.25 * future_capture


def _forecast_overflow_score(env, emitter_id: str, horizon_h: int = 168) -> float:
    emitter = env.network.entities[emitter_id]
    state = env.simulator.state
    inventory = state.entity_inventory_t.get(emitter_id, 0.0)
    future_capture = _forecast_capture_t(env, emitter_id, horizon_h)
    overflow = inventory + future_capture - emitter.buffer_capacity_t
    return max(0.0, overflow)


def _forecast_capture_t(env, emitter_id: str, horizon_h: int) -> float:
    emitter = env.network.entities[emitter_id]
    state = env.simulator.state
    future_capture = 0.0
    if env.scenario is not None:
        series = env.scenario.emitter_availability.get(emitter_id, [])
    else:
        series = []
    for offset in range(min(horizon_h, env.n_steps)):
        time_h = state.time_h + offset * env.network.time_step_hours
        idx = env.scenario.step_index(time_h) if env.scenario is not None else 0
        availability = series[idx] if series else state.emitter_availability.get(emitter_id, emitter.availability)
        future_capture += emitter.capture_rate_tph_at(time_h) * max(0.0, float(availability))
    return future_capture


def _should_wait_for_weather(env, vessel_id: str, wait_h: int = 18) -> bool:
    if env.scenario is None:
        return False
    state = env.simulator.state
    series = env.scenario.vessel_speed_factor.get(vessel_id, [])
    if not series:
        return False
    now_idx = env.scenario.step_index(state.time_h)
    current = float(series[now_idx])
    if current >= 0.65:
        return False
    future = [
        float(series[env.scenario.step_index(state.time_h + offset)])
        for offset in range(1, wait_h + 1)
    ]
    return max(future) >= current + 0.2


def _source_overflow_imminent(env) -> bool:
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        inventory = env.simulator.state.entity_inventory_t.get(emitter_id, 0.0)
        if inventory >= 0.85 * emitter.buffer_capacity_t:
            return True
    return False


def controller_factories(
    *,
    replan_every_h: int,
    planning_horizon_h: int,
    rolling_time_limit_s: float,
    economics: EconomicParameters,
) -> dict[str, PolicyFactory]:
    return {
        "greedy": lambda _env: greedy_shuttle_policy,
        "rule_based": rule_based_env_policy,
        "milkrun_greedy": lambda _env: milkrun_greedy_policy,
        "forecast_dispatch": lambda _env: forecast_dispatch_policy,
        "forecast_balanced": lambda _env: forecast_balanced_policy,
        "rolling_mpc": lambda env: RollingMilpController(
            env,
            replan_every=replan_every_h,
            planning_horizon_h=planning_horizon_h,
            time_limit_s=rolling_time_limit_s,
            economics=economics,
        ),
    }


def metric_row(seed: int, controller: str, metrics: EpisodeMetrics, solve_time_s: float) -> dict[str, object]:
    return {
        "seed": seed,
        "controller": controller,
        "solve_time_s": solve_time_s,
        **metrics.as_dict(),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    controllers = sorted({str(row["controller"]) for row in rows})
    metrics = [
        "captured_t",
        "stored_t",
        "vented_t",
        "in_transit_t",
        "in_transit_growth_t",
        "loss_rate",
        "storage_rate",
        "operating_cost",
        "total_cost",
        "throttle_hours",
        "berth_wait_vessel_hours",
        "pressure_risk_hours",
        "solve_time_s",
    ]
    for controller in controllers:
        subset = [row for row in rows if row["controller"] == controller]
        out: dict[str, object] = {"controller": controller, "episodes": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset if row.get(metric) not in (None, "")]
            out[f"{metric}_mean"] = sum(values) / len(values) if values else ""
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out_rows.append(out)
    greedy = next((row for row in out_rows if row["controller"] == "greedy"), None)
    if greedy is not None:
        for row in out_rows:
            row["stored_delta_vs_greedy_t"] = float(row["stored_t_mean"]) - float(greedy["stored_t_mean"])
            row["vented_delta_vs_greedy_t"] = float(row["vented_t_mean"]) - float(greedy["vented_t_mean"])
            row["cost_delta_vs_greedy_eur"] = float(row["total_cost_mean"]) - float(greedy["total_cost_mean"])
    return out_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-hours", type=int, default=720)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--controllers", nargs="+", default=["greedy", "forecast_dispatch", "rolling_mpc"])
    parser.add_argument("--yara-buffer-t", type=float, default=7500.0)
    parser.add_argument("--terminal-buffer-t", type=float, default=7500.0)
    parser.add_argument("--planning-horizon-h", type=int, default=168)
    parser.add_argument("--replan-every-h", type=int, default=24)
    parser.add_argument("--rolling-time-limit-s", type=float, default=10.0)
    parser.add_argument("--shortfall-penalty-eur-per-t", type=float, default=1000.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/phase1_3vessels_720h_forecast_stress"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ScenarioConfig(
        episode_hours=args.cap_hours,
        randomize_initial_inventory=True,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
    )
    economics = EconomicParameters(storage_shortfall_eur_per_t=args.shortfall_penalty_eur_per_t)
    factories = controller_factories(
        replan_every_h=args.replan_every_h,
        planning_horizon_h=args.planning_horizon_h,
        rolling_time_limit_s=args.rolling_time_limit_s,
        economics=economics,
    )

    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for seed in args.seeds:
        seed_events: list[dict[str, object]] | None = None
        for controller in args.controllers:
            env, generator = make_stress_env(
                cap_hours=args.cap_hours,
                yara_buffer_t=args.yara_buffer_t,
                terminal_buffer_t=args.terminal_buffer_t,
                economics=economics,
                config=config,
            )
            run_start = time.perf_counter()
            metrics = run_episode(env, factories[controller](env), seed=seed)
            run_s = time.perf_counter() - run_start
            rows.append(metric_row(seed, controller, metrics, run_s))
            if seed_events is None:
                seed_events = list(generator.last_events)
                for event in seed_events:
                    event_rows.append({"seed": seed, **event})
            write_csv(args.output_dir / "by_seed.partial.csv", rows)
            print(
                f"seed={seed} {controller}: stored={metrics.stored_t:.1f} "
                f"vented={metrics.vented_t:.1f} cost={metrics.total_cost:.0f} "
                f"time={run_s:.1f}s",
                flush=True,
            )

    summary_rows = summarize(rows)
    write_csv(args.output_dir / "by_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "stress_events.csv", event_rows)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scenario": "northern_lights_phase1_3vessels_forecast_stress",
                "cap_hours": args.cap_hours,
                "seeds": args.seeds,
                "controllers": args.controllers,
                "yara_buffer_t": args.yara_buffer_t,
                "terminal_buffer_t": args.terminal_buffer_t,
                "planning_horizon_h": args.planning_horizon_h,
                "replan_every_h": args.replan_every_h,
                "rolling_time_limit_s": args.rolling_time_limit_s,
                "shortfall_penalty_eur_per_t": args.shortfall_penalty_eur_per_t,
                "scenario_config": config.__dict__,
                "elapsed_s": time.perf_counter() - start,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote {args.output_dir}")
    for row in summary_rows:
        print(
            f"{row['controller']}: stored={float(row['stored_t_mean']):.1f}, "
            f"vented={float(row['vented_t_mean']):.1f}, "
            f"storage_rate={float(row['storage_rate_mean']):.4f}, "
            f"loss_rate={float(row['loss_rate_mean']):.4f}, "
            f"delta_stored={float(row.get('stored_delta_vs_greedy_t', 0.0)):.1f}, "
            f"delta_vented={float(row.get('vented_delta_vs_greedy_t', 0.0)):.1f}, "
            f"delta_cost={float(row.get('cost_delta_vs_greedy_eur', 0.0)):.0f}"
        )


if __name__ == "__main__":
    main()
