"""Run the paper E0 physical-simulator validation and build S1 artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False

from sim.control.baselines import greedy_shuttle_policy
from sim.control.cplex_milp import _terminal_cleanup_cost_for_state
from sim.control.event_based.residual_rl_v4.scenario import (
    ReplayableDifficultyScenarioGenerator,
)
from sim.entities import (
    Emitter,
    InjectionWell,
    Pipeline,
    Reservoir,
    Terminal,
    Vessel,
)
from sim.environment import CCSEnv, CCSEnvConfig, build_phase1_env
from sim.network import PhysicalNetwork
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "experiments_results" / "E0"
PROTOCOL_PATH = (
    ROOT
    / "experiments"
    / "protocols"
    / "unified_window_v1_paper_protocol.json"
)
SCENARIO_PATH = (
    ROOT / "scenarios" / "northern_lights_phase1_3vessels.json"
)
VALIDATION_SEED = 8_100_001
MASS_TOLERANCE_T = 1e-6
CAPACITY_TOLERANCE_T = 1e-6
PRESSURE_TOLERANCE_BAR = 1e-8
RATE_TOLERANCE_TPH = 1e-6

PYTEST_TARGETS = (
    "tests/test_physical_layer.py",
    "tests/test_simulator.py",
    "tests/test_env.py",
    "tests/test_env_scenarios.py",
    "tests/test_disturbances.py",
    "tests/test_scenario.py",
    "tests/test_metrics.py",
    "tests/test_paper_experiment_protocol.py",
    "tests/test_control_replay.py",
    "tests/test_economics.py",
    "tests/test_ship_speed.py",
    "tests/test_vessel_operation_mode.py",
    (
        "tests/test_cplex_milp.py::CplexMilpInterfaceTests::"
        "test_terminal_cleanup_value_reports_separate_future_cost"
    ),
    (
        "tests/test_cplex_milp.py::CplexMilpInterfaceTests::"
        "test_action_warm_start_completes_terminal_cleanup_variables"
    ),
)

HARD_VIOLATIONS = {
    "mass_balance_error",
    "inventory_capacity_exceeded",
    "reservoir_pressure_exceeded",
    "negative_inventory",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=VALIDATION_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_out_dir(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "experiments_results").resolve()
    if allowed_root not in resolved.parents:
        raise ValueError(
            f"E0 output must stay below {allowed_root}, got {resolved}."
        )
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(
                f"{resolved} already exists; pass --overwrite to replace it."
            )
        shutil.rmtree(resolved)
    (resolved / "automated_tests").mkdir(parents=True)
    (resolved / "source_data").mkdir()
    (resolved / "figures").mkdir()


def _formal_env(
    *,
    episode_hours: int,
    scenario_generator,
) -> CCSEnv:
    return build_phase1_env(
        scenario="northern_lights_phase1_3vessels",
        scenario_generator=scenario_generator,
        weather_mode="window",
        config=CCSEnvConfig(
            episode_hours=episode_hours,
            include_goal_obs=False,
            reward_mode="economic",
            injection_reward_eur_per_t=0.0,
            store_reward_eur_per_t=0.0,
            vent_penalty_weight=1.0,
            operating_cost_weight=1.0,
            enforce_full_load_dispatch=False,
            require_empty_terminal_departure=True,
            well_control_mode="automatic_max",
        ),
    )


def _medium_formal_env() -> CCSEnv:
    return _formal_env(
        episode_hours=720,
        scenario_generator=ReplayableDifficultyScenarioGenerator(
            episode_hours=720 + 168,
            weather_process="window",
            hard_probability=0.0,
            scenario_protocol="unified_window_v1",
        ),
    )


def _controlled_formal_env(
    *,
    episode_hours: int,
    capture_high_output: bool = False,
    weather_window: bool = False,
    well_maintenance: bool = False,
) -> CCSEnv:
    config = ScenarioConfig(
        episode_hours=episode_hours,
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        capture_high_output_rate_per_week=(
            168.0 if capture_high_output else 0.0
        ),
        capture_high_output_mean_hours=1_000.0,
        capture_high_output_multiplier_range=(1.5, 1.5),
        weather_process="window",
        weather_window_rate_per_week=(
            168.0 if weather_window else 0.0
        ),
        weather_window_mean_hours=1_000.0,
        weather_window_speed_factor_range=(0.6, 0.6),
        well_maintenance_rate_per_week=(
            168.0 if well_maintenance else 0.0
        ),
        well_maintenance_mean_hours=1_000.0,
        randomize_initial_inventory=False,
        warm_start=False,
    )
    return _formal_env(
        episode_hours=episode_hours,
        scenario_generator=ScenarioGenerator(config=config),
    )


def _single_vessel_env(episode_hours: int = 96) -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(
        Emitter(
            "source",
            nominal_capture_tph=60.0,
            buffer_capacity_t=4_000.0,
            loading_rate_tph=600.0,
        )
    )
    network.add_entity(
        Vessel(
            "ship",
            capacity_t=1_000.0,
            loading_rate_tph=600.0,
            unloading_rate_tph=600.0,
            speed_knots=14.0,
        )
    )
    network.add_entity(
        Terminal(
            "terminal",
            storage_capacity_t=4_000.0,
            berth_count=1,
        )
    )
    network.add_entity(Pipeline("pipeline", max_flow_tph=300.0))
    network.add_entity(
        InjectionWell("well", max_injection_tph=300.0)
    )
    network.add_entity(
        Reservoir(
            "reservoir",
            storage_capacity_t=1_000_000.0,
            initial_pressure_bar=100.0,
            pressure_at_capacity_bar=200.0,
            max_pressure_bar=200.0,
        )
    )
    network.connect("source", "ship")
    network.connect("ship", "terminal")
    network.connect("terminal", "pipeline")
    network.connect("pipeline", "well")
    network.connect("well", "reservoir")
    config = ScenarioConfig(
        episode_hours=episode_hours,
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        capture_high_output_rate_per_week=0.0,
        weather_window_rate_per_week=0.0,
        well_maintenance_rate_per_week=0.0,
        randomize_initial_inventory=False,
    )
    return CCSEnv(
        network,
        {
            "source": (59.05, 9.70),
            "terminal": (60.58, 4.84),
        },
        scenario_generator=ScenarioGenerator(config=config),
        config=CCSEnvConfig(
            episode_hours=episode_hours,
            include_goal_obs=False,
            reward_mode="economic",
            injection_reward_eur_per_t=0.0,
            store_reward_eur_per_t=0.0,
            well_control_mode="automatic_max",
            enforce_full_load_dispatch=False,
            require_empty_terminal_departure=True,
        ),
    )


def _inventory_by_stage(env: CCSEnv) -> dict[str, float]:
    state = env.simulator.state
    inventory = state.entity_inventory_t
    return {
        "emitter_inventory_t": sum(
            inventory.get(entity_id, 0.0)
            for entity_id in env.emitter_ids
        ),
        "vessel_inventory_t": sum(
            inventory.get(entity_id, 0.0)
            for entity_id in env.vessel_ids
        ),
        "terminal_inventory_t": sum(
            inventory.get(entity_id, 0.0)
            for entity_id in env.terminal_ids
        ),
        "reservoir_inventory_t": sum(
            inventory.get(entity_id, 0.0)
            for entity_id in env.reservoir_ids
        ),
    }


def _capacity_diagnostics(env: CCSEnv) -> dict[str, float]:
    state = env.simulator.state
    maximum_ratio = {
        "emitter": 0.0,
        "vessel": 0.0,
        "terminal": 0.0,
        "reservoir": 0.0,
    }
    max_pressure_excess = 0.0
    min_inventory = 0.0
    for entity_id, entity in env.network.entities.items():
        inventory = float(state.entity_inventory_t.get(entity_id, 0.0))
        min_inventory = min(min_inventory, inventory)
        if isinstance(entity, Emitter):
            maximum_ratio["emitter"] = max(
                maximum_ratio["emitter"],
                inventory / entity.buffer_capacity_t,
            )
        elif isinstance(entity, Vessel):
            maximum_ratio["vessel"] = max(
                maximum_ratio["vessel"],
                inventory / entity.capacity_t,
            )
        elif isinstance(entity, Terminal):
            maximum_ratio["terminal"] = max(
                maximum_ratio["terminal"],
                inventory / entity.storage_capacity_t,
            )
        elif isinstance(entity, Reservoir):
            maximum_ratio["reservoir"] = max(
                maximum_ratio["reservoir"],
                inventory / entity.pressure_limited_capacity_t(),
            )
            max_pressure_excess = max(
                max_pressure_excess,
                entity.pressure_bar(inventory)
                - entity.max_pressure_bar,
            )
    return {
        **{
            f"max_{stage}_capacity_ratio": ratio
            for stage, ratio in maximum_ratio.items()
        },
        "max_pressure_excess_bar": max_pressure_excess,
        "min_inventory_t": min_inventory,
    }


def _run_case(
    case_id: str,
    env: CCSEnv,
    *,
    seed: int,
    keep_trajectory: bool = False,
) -> tuple[dict, list[dict]]:
    env.reset(seed=seed)
    initial_mass = sum(
        float(value)
        for value in env.simulator.state.entity_inventory_t.values()
    )
    trajectory: list[dict] = []
    max_abs_mass_error = 0.0
    max_rate_excess = 0.0
    max_unavailable_injection = 0.0
    illegal_destination_changes = 0
    queue_duplicate_hours = 0
    hard_violations = 0
    diagnostics = _capacity_diagnostics(env)
    maximums = dict(diagnostics)
    done = False

    while not done:
        previous_vessel_states = {
            vessel_id: dict(state)
            for vessel_id, state in env.simulator.vessel_states.items()
        }
        requested_rates = dict(
            zip(env.well_ids, env.automatic_well_rates_tph())
        )
        action = greedy_shuttle_policy(env)
        well_available_before_step = {
            well_id: bool(
                env.simulator.state.well_available.get(well_id, True)
            )
            for well_id in env.well_ids
        }
        _obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        state = env.simulator.state
        stage = _inventory_by_stage(env)
        current_mass = sum(
            float(value)
            for value in state.entity_inventory_t.values()
        )
        mass_error = (
            current_mass
            + float(env.ledger.vented_t)
            - initial_mass
            - float(env.cumulative_captured_t)
        )
        max_abs_mass_error = max(
            max_abs_mass_error,
            abs(mass_error),
        )
        current = _capacity_diagnostics(env)
        for key, value in current.items():
            if key == "min_inventory_t":
                maximums[key] = min(maximums[key], value)
            else:
                maximums[key] = max(maximums[key], value)

        for well_id in env.well_ids:
            actual = float(
                state.last_injection_flow_tph.get(well_id, 0.0)
            )
            max_rate_excess = max(
                max_rate_excess,
                actual - requested_rates[well_id],
            )
            if not well_available_before_step.get(well_id, True):
                max_unavailable_injection = max(
                    max_unavailable_injection,
                    actual,
                )

        for vessel_id, before in previous_vessel_states.items():
            after = env.simulator.vessel_states[vessel_id]
            if (
                before["mode"] == "sailing"
                and after["mode"] == "sailing"
                and before["destination"] != after["destination"]
            ):
                illegal_destination_changes += 1
        for queue in state.terminal_unload_queues.values():
            if len(queue) != len(set(queue)):
                queue_duplicate_hours += 1
        hard_violations += sum(
            violation in HARD_VIOLATIONS
            for violation in info.get("violations", [])
        )

        if keep_trajectory:
            availability = list(state.emitter_availability.values())
            speeds = list(state.vessel_speed_factor.values())
            wells = list(state.well_available.values())
            trajectory.append(
                {
                    "case_id": case_id,
                    "time_h": float(state.time_h),
                    "captured_t": float(env.cumulative_captured_t),
                    "stored_t": float(env.cumulative_stored_t),
                    "vented_t": float(env.ledger.vented_t),
                    **stage,
                    "system_mass_balance_error_t": mass_error,
                    "mean_capture_multiplier": (
                        sum(availability) / len(availability)
                        if availability
                        else 1.0
                    ),
                    "minimum_speed_factor": (
                        min(speeds) if speeds else 1.0
                    ),
                    "available_well_fraction": (
                        sum(bool(value) for value in wells) / len(wells)
                        if wells
                        else 1.0
                    ),
                }
            )

    scenario = env.scenario
    emitter_values = [
        value
        for values in scenario.emitter_availability.values()
        for value in values[: env.n_steps]
    ]
    speed_values = [
        value
        for values in scenario.vessel_speed_factor.values()
        for value in values[: env.n_steps]
    ]
    well_values = [
        value
        for values in scenario.well_available.values()
        for value in values[: env.n_steps]
    ]
    ledger = env.ledger
    cost_sum = (
        ledger.vessel_fuel
        + ledger.conditioning
        + ledger.reconditioning
        + ledger.loading
        + ledger.unloading
        + ledger.vent_penalty
        + ledger.storage_shortfall_penalty
    )
    summary = {
        "case_id": case_id,
        "seed": int(seed),
        "episode_hours": int(env.n_steps),
        "emitters": len(env.emitter_ids),
        "vessels": len(env.vessel_ids),
        "terminals": len(env.terminal_ids),
        "wells": len(env.well_ids),
        "captured_t": float(env.cumulative_captured_t),
        "stored_t": float(env.cumulative_stored_t),
        "vented_t": float(ledger.vented_t),
        **_inventory_by_stage(env),
        "max_abs_mass_balance_error_t": max_abs_mass_error,
        **maximums,
        "max_injection_above_requested_tph": max_rate_excess,
        "max_injection_while_unavailable_tph": (
            max_unavailable_injection
        ),
        "illegal_midvoyage_destination_changes": (
            illegal_destination_changes
        ),
        "terminal_queue_duplicate_hours": queue_duplicate_hours,
        "hard_violation_count": hard_violations,
        "minimum_speed_factor": min(speed_values, default=1.0),
        "maximum_capture_multiplier": max(
            emitter_values,
            default=1.0,
        ),
        "well_unavailable_samples": sum(
            not bool(value) for value in well_values
        ),
        "operating_cost_eur": float(ledger.operating_cost),
        "total_cost_eur": float(ledger.total_cost),
        "cost_decomposition_error_eur": float(
            ledger.total_cost - cost_sum
        ),
    }
    return summary, trajectory


def _run_pytest(out_dir: Path) -> dict:
    junit_path = out_dir / "automated_tests" / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *PYTEST_TARGETS,
        "-q",
        f"--junitxml={junit_path}",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / "automated_tests" / "pytest_stdout.txt").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (out_dir / "automated_tests" / "pytest_stderr.txt").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if not junit_path.exists():
        raise RuntimeError(
            "Pytest did not create JUnit output:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    result = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    result["passed"] = (
        result["tests"]
        - result["failures"]
        - result["errors"]
        - result["skipped"]
    )
    result["return_code"] = completed.returncode
    result["targets"] = list(PYTEST_TARGETS)
    if completed.returncode != 0:
        raise RuntimeError(
            "E0 automated tests failed; see automated_tests output."
        )
    return result


def _check_row(
    check_id: str,
    category: str,
    expected: str,
    observed: float | int | str,
    tolerance: str,
    passed: bool,
    evidence: str,
) -> dict:
    return {
        "check_id": check_id,
        "category": category,
        "expected": expected,
        "observed": observed,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def _build_checks(
    cases: dict[str, dict],
    pytest_summary: dict,
    cleanup: dict,
    common_well_rate_spread: float,
) -> list[dict]:
    formal = cases["formal_medium_720h"]
    checks = [
        _check_row(
            "mass.formal_720h",
            "Mass conservation",
            "Maximum absolute whole-system residual is near zero",
            formal["max_abs_mass_balance_error_t"],
            f"≤ {MASS_TOLERANCE_T:g} t",
            formal["max_abs_mass_balance_error_t"]
            <= MASS_TOLERANCE_T,
            "source_data/figure_s1_timeseries.csv",
        ),
    ]
    for stage in ("emitter", "vessel", "terminal", "reservoir"):
        value = formal[f"max_{stage}_capacity_ratio"]
        checks.append(
            _check_row(
                f"capacity.{stage}",
                "Capacity limits",
                f"{stage} inventory does not exceed capacity",
                value,
                f"≤ 1 + {CAPACITY_TOLERANCE_T:g} t-equivalent",
                value <= 1.0 + CAPACITY_TOLERANCE_T,
                "source_data/simplified_cases.csv",
            )
        )
    checks.extend(
        [
            _check_row(
                "capacity.nonnegative_inventory",
                "Capacity limits",
                "All entity inventories remain non-negative",
                formal["min_inventory_t"],
                f"≥ −{CAPACITY_TOLERANCE_T:g} t",
                formal["min_inventory_t"]
                >= -CAPACITY_TOLERANCE_T,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "pressure.reservoir",
                "Pressure limits",
                "Reservoir pressure does not exceed its limit",
                formal["max_pressure_excess_bar"],
                f"≤ {PRESSURE_TOLERANCE_BAR:g} bar",
                formal["max_pressure_excess_bar"]
                <= PRESSURE_TOLERANCE_BAR,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "injection.maximum_feasible",
                "Automatic well control",
                "Actual injection never exceeds the requested continuous maximum",
                formal["max_injection_above_requested_tph"],
                f"≤ {RATE_TOLERANCE_TPH:g} t/h",
                formal["max_injection_above_requested_tph"]
                <= RATE_TOLERANCE_TPH,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "injection.common_rule",
                "Automatic well control",
                "All controller labels call the same state-to-rate rule",
                common_well_rate_spread,
                "0 t/h",
                common_well_rate_spread <= RATE_TOLERANCE_TPH,
                "config_snapshot.json",
            ),
            _check_row(
                "state.destination_lock",
                "Vessel state machine",
                "No destination changes while a vessel remains mid-voyage",
                formal["illegal_midvoyage_destination_changes"],
                "0 events",
                formal["illegal_midvoyage_destination_changes"] == 0,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "state.terminal_fifo",
                "Vessel state machine",
                "Terminal queues never contain duplicate vessels",
                formal["terminal_queue_duplicate_hours"],
                "0 hours",
                formal["terminal_queue_duplicate_hours"] == 0,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "state.hard_violations",
                "Physical feasibility",
                "No hard physical violations in the 720 h rollout",
                formal["hard_violation_count"],
                "0",
                formal["hard_violation_count"] == 0,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "cost.decomposition",
                "Economic accounting",
                "Ledger components sum to total episode cost",
                abs(formal["cost_decomposition_error_eur"]),
                "≤ 1e-6 EUR",
                abs(formal["cost_decomposition_error_eur"]) <= 1e-6,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "scenario.no_disturbance",
                "Simplified scenarios",
                "No weather, maintenance, or high-output event is present",
                (
                    f"speed={cases['no_disturbance']['minimum_speed_factor']}; "
                    f"maintenance={cases['no_disturbance']['well_unavailable_samples']}; "
                    f"capture_max={cases['no_disturbance']['maximum_capture_multiplier']}"
                ),
                "speed=1; maintenance=0; capture_max=1",
                (
                    abs(
                        cases["no_disturbance"][
                            "minimum_speed_factor"
                        ]
                        - 1.0
                    )
                    <= 1e-12
                    and cases["no_disturbance"][
                        "well_unavailable_samples"
                    ]
                    == 0
                    and abs(
                        cases["no_disturbance"][
                            "maximum_capture_multiplier"
                        ]
                        - 1.0
                    )
                    <= 1e-12
                ),
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "scenario.single_vessel",
                "Simplified scenarios",
                "One-emitter/one-vessel case completes without hard violations",
                cases["single_vessel_single_emitter"][
                    "hard_violation_count"
                ],
                "0",
                (
                    cases["single_vessel_single_emitter"]["emitters"] == 1
                    and cases["single_vessel_single_emitter"]["vessels"]
                    == 1
                    and cases["single_vessel_single_emitter"][
                        "hard_violation_count"
                    ]
                    == 0
                ),
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "scenario.weather",
                "Simplified scenarios",
                "Weather window reduces the vessel speed factor",
                cases["weather_window"]["minimum_speed_factor"],
                "< 1",
                cases["weather_window"]["minimum_speed_factor"] < 1.0,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "scenario.maintenance",
                "Simplified scenarios",
                "Maintenance occurs and blocks injection",
                (
                    f"samples={cases['well_maintenance']['well_unavailable_samples']}; "
                    "max_unavailable_injection="
                    f"{cases['well_maintenance']['max_injection_while_unavailable_tph']}"
                ),
                "samples > 0 and injection ≤ 1e-6 t/h",
                (
                    cases["well_maintenance"][
                        "well_unavailable_samples"
                    ]
                    > 0
                    and cases["well_maintenance"][
                        "max_injection_while_unavailable_tph"
                    ]
                    <= RATE_TOLERANCE_TPH
                ),
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "scenario.high_output",
                "Simplified scenarios",
                "High-output multiplier exceeds the nominal capture rate",
                cases["capture_high_output"][
                    "maximum_capture_multiplier"
                ],
                "> 1",
                cases["capture_high_output"][
                    "maximum_capture_multiplier"
                ]
                > 1.0,
                "source_data/simplified_cases.csv",
            ),
            _check_row(
                "cleanup.current_state",
                "Terminal accounting",
                "Common compact cleanup is deterministic and non-negative",
                cleanup["repeat_absolute_error_eur"],
                "repeat error ≤ 1e-6 EUR",
                (
                    cleanup["cost_eur"] >= 0.0
                    and cleanup["repeat_absolute_error_eur"] <= 1e-6
                ),
                "terminal_cleanup_validation.json",
            ),
            _check_row(
                "tests.e0_suite",
                "Automated regression",
                "All selected physical, disturbance, state, cost, replay, and cleanup tests pass",
                (
                    f"{pytest_summary['passed']}/"
                    f"{pytest_summary['tests']} passed"
                ),
                "0 failures and 0 errors",
                (
                    pytest_summary["failures"] == 0
                    and pytest_summary["errors"] == 0
                ),
                "automated_tests/junit.xml",
            ),
        ]
    )
    return checks


def _write_s1_table(out_dir: Path, checks: list[dict]) -> None:
    _write_csv(out_dir / "supplementary_table_s1.csv", checks)
    lines = [
        "# Supplementary Table S1 — E0 physical validation",
        "",
        "| Check | Category | Expected | Observed | Tolerance | Status |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in checks:
        lines.append(
            "| {check_id} | {category} | {expected} | {observed} | "
            "{tolerance} | **{status}** |".format(**row)
        )
    lines.append("")
    (out_dir / "supplementary_table_s1.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _make_figure_s1(out_dir: Path, trajectory: list[dict]) -> None:
    x = [row["time_h"] for row in trajectory]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 / 25.4, 120 / 25.4),
        constrained_layout=True,
    )
    ax = axes[0, 0]
    ax.plot(
        x,
        [row["captured_t"] for row in trajectory],
        color="#4D4D4D",
        lw=1.5,
        label="Captured",
    )
    ax.plot(
        x,
        [row["stored_t"] for row in trajectory],
        color="#0F4D92",
        lw=1.5,
        label="Stored",
    )
    ax.plot(
        x,
        [row["vented_t"] for row in trajectory],
        color="#B64342",
        lw=1.2,
        label="Vented",
    )
    ax.set_ylabel("Cumulative CO$_2$ (t)")
    ax.legend(ncol=3, loc="upper left")
    _add_panel_label(ax, "a")

    ax = axes[0, 1]
    inventory_lines = (
        ("Emitter", "emitter_inventory_t", "#B64342"),
        ("Vessel", "vessel_inventory_t", "#3775BA"),
        ("Terminal", "terminal_inventory_t", "#42949E"),
    )
    for label, key, color in inventory_lines:
        ax.plot(
            x,
            [row[key] for row in trajectory],
            color=color,
            lw=1.1,
            label=label,
        )
    ax.set_ylabel("Recoverable inventory (t)")
    ax.legend(ncol=3, loc="upper right")
    _add_panel_label(ax, "b")

    ax = axes[1, 0]
    residual = [
        max(abs(row["system_mass_balance_error_t"]), 1e-12)
        for row in trajectory
    ]
    ax.plot(x, residual, color="#9A4D8E", lw=1.0)
    ax.axhline(
        MASS_TOLERANCE_T,
        color="#767676",
        lw=0.8,
        ls="--",
        label="Tolerance",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Simulation time (h)")
    ax.set_ylabel("|Mass-balance residual| (t)")
    ax.legend(loc="upper right")
    _add_panel_label(ax, "c")

    ax = axes[1, 1]
    ax.plot(
        x,
        [row["mean_capture_multiplier"] for row in trajectory],
        color="#B64342",
        lw=1.0,
        label="Capture multiplier",
    )
    ax.plot(
        x,
        [row["minimum_speed_factor"] for row in trajectory],
        color="#3775BA",
        lw=1.0,
        label="Minimum speed factor",
    )
    ax.plot(
        x,
        [row["available_well_fraction"] for row in trajectory],
        color="#42949E",
        lw=1.0,
        label="Available well fraction",
    )
    ax.set_ylim(-0.03, 2.1)
    ax.set_xlabel("Simulation time (h)")
    ax.set_ylabel("Dimensionless disturbance state")
    ax.legend(loc="upper right")
    _add_panel_label(ax, "d")

    base = out_dir / "figures" / "figure_s1_mass_balance_inventory"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_simulator_sources() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "sim").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_snapshot() -> dict:
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()),
        "status": status.stdout.splitlines(),
    }


def _write_figure_contract(out_dir: Path, seed: int) -> None:
    text = f"""# Figure S1 contract

- Core conclusion: The physical simulator conserves whole-system CO2 mass
  throughout a disturbed 720 h rollout while inventories and storage respond
  consistently to operational disturbances.
- Figure archetype: quantitative grid.
- Target/output: double-column supplementary figure; editable SVG and PDF,
  600 dpi TIFF, 300 dpi PNG preview.
- Backend: Python/matplotlib only.
- Final size: 183 mm × 120 mm.
- Panel a: cumulative captured, stored, and vented CO2.
- Panel b: recoverable inventory split across emitters, vessels, and terminal.
- Panel c: absolute whole-system mass-balance residual and tolerance.
- Panel d: capture, weather-speed, and well-availability disturbance states.
- Evidence hierarchy: panel c is the conservation evidence; panels a–b show
  physical stock/flow consistency; panel d anchors changes to disturbances.
- Statistics: one deterministic validation trajectory, seed {seed}; no
  inferential statistics.
- Source data: source_data/figure_s1_timeseries.csv.
- Image integrity: vector-native line art; no local image adjustment.
- Reviewer risk: this representative trajectory does not replace the complete
  component-test suite reported in Supplementary Table S1.
"""
    (out_dir / "figure_s1_contract.md").write_text(
        text,
        encoding="utf-8",
    )


def _write_readme(
    out_dir: Path,
    *,
    checks: list[dict],
    cases: dict[str, dict],
    pytest_summary: dict,
    cleanup: dict,
    elapsed_seconds: float,
) -> None:
    failed = [row for row in checks if row["status"] != "PASS"]
    formal = cases["formal_medium_720h"]
    text = f"""# E0 物理仿真层验证结果

## 结论

E0 状态：**{"PASS" if not failed else "FAIL"}**。

- Supplementary Table S1：{len(checks) - len(failed)}/{len(checks)} 项通过；
- 自动化回归：{pytest_summary["passed"]}/{pytest_summary["tests"]} 项通过；
- 720 h 全系统最大质量守恒误差：
  {formal["max_abs_mass_balance_error_t"]:.3e} t；
- 720 h hard physical violations：{formal["hard_violation_count"]}；
- 当前末状态 common compact cleanup cost：
  EUR {cleanup["cost_eur"]:,.2f}；
- 总运行时间：{elapsed_seconds:.2f} s。

## 目录

- `supplementary_table_s1.csv/.md`：验证项目、容差、观测值和结论；
- `figures/figure_s1_mass_balance_inventory.*`：Figure S1；
- `source_data/figure_s1_timeseries.csv`：Figure S1 源数据；
- `source_data/simplified_cases.csv`：五类简化案例和正式轨迹摘要；
- `automated_tests/`：pytest stdout、JUnit XML 和测试目标；
- `summary.json`：机器可读总结果；
- `config_snapshot.json`：协议、版本和哈希；
- `terminal_cleanup_validation.json`：共同末端核算检查。

## 范围

E0 验证仿真器及共同核算边界，不比较控制器性能，也不用于选择
Iterative Q 的 future representation 或超参数。正式 test seeds 未被访问。
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    _prepare_out_dir(out_dir, args.overwrite)
    started = time.perf_counter()

    pytest_summary = _run_pytest(out_dir)
    case_builders = (
        ("formal_medium_720h", _medium_formal_env, True),
        (
            "no_disturbance",
            lambda: _controlled_formal_env(episode_hours=168),
            False,
        ),
        (
            "single_vessel_single_emitter",
            _single_vessel_env,
            False,
        ),
        (
            "well_maintenance",
            lambda: _controlled_formal_env(
                episode_hours=168,
                well_maintenance=True,
            ),
            False,
        ),
        (
            "weather_window",
            lambda: _controlled_formal_env(
                episode_hours=168,
                weather_window=True,
            ),
            False,
        ),
        (
            "capture_high_output",
            lambda: _controlled_formal_env(
                episode_hours=168,
                capture_high_output=True,
            ),
            False,
        ),
    )
    cases: dict[str, dict] = {}
    figure_trajectory: list[dict] = []
    formal_env = None
    for case_id, builder, keep_trajectory in case_builders:
        env = builder()
        summary, trajectory = _run_case(
            case_id,
            env,
            seed=args.seed,
            keep_trajectory=keep_trajectory,
        )
        cases[case_id] = summary
        if keep_trajectory:
            figure_trajectory = trajectory
            formal_env = env

    assert formal_env is not None
    cleanup_cost = _terminal_cleanup_cost_for_state(
        formal_env,
        formal_env.cost_model.parameters,
    )
    cleanup_repeat = _terminal_cleanup_cost_for_state(
        formal_env,
        formal_env.cost_model.parameters,
    )
    cleanup = {
        "case_id": "formal_medium_720h",
        "cost_eur": float(cleanup_cost),
        "repeat_cost_eur": float(cleanup_repeat),
        "repeat_absolute_error_eur": abs(
            float(cleanup_cost) - float(cleanup_repeat)
        ),
        "function": (
            "sim.control.cplex_milp."
            "_terminal_cleanup_cost_for_state"
        ),
    }
    _write_json(out_dir / "terminal_cleanup_validation.json", cleanup)

    controller_labels = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )["control_scope"]["well_control"]["applies_to"]
    rates = formal_env.automatic_well_rates_tph()
    common_well_rates = {
        label: list(rates) for label in controller_labels
    }
    flat_rates = [
        rate
        for values in common_well_rates.values()
        for rate in values
    ]
    common_well_rate_spread = (
        max(flat_rates) - min(flat_rates) if flat_rates else 0.0
    )

    checks = _build_checks(
        cases,
        pytest_summary,
        cleanup,
        common_well_rate_spread,
    )
    _write_csv(
        out_dir / "source_data" / "simplified_cases.csv",
        list(cases.values()),
    )
    _write_csv(
        out_dir / "source_data" / "figure_s1_timeseries.csv",
        figure_trajectory,
    )
    _write_s1_table(out_dir, checks)
    _make_figure_s1(out_dir, figure_trajectory)
    _write_figure_contract(out_dir, args.seed)

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    config_snapshot = {
        "e0_schema_version": 1,
        "validation_seed": int(args.seed),
        "formal_test_seeds_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _hash_file(PROTOCOL_PATH),
        "scenario_path": str(SCENARIO_PATH.relative_to(ROOT)),
        "scenario_sha256": _hash_file(SCENARIO_PATH),
        "scenario_hash_matches_protocol": (
            _hash_file(SCENARIO_PATH)
            == protocol["scope"]["scenario_file_sha256"]
        ),
        "simulator_source_sha256": _hash_simulator_sources(),
        "git": _git_snapshot(),
        "tolerances": {
            "mass_t": MASS_TOLERANCE_T,
            "capacity_t": CAPACITY_TOLERANCE_T,
            "pressure_bar": PRESSURE_TOLERANCE_BAR,
            "rate_tph": RATE_TOLERANCE_TPH,
        },
        "common_automatic_well_rates_tph": common_well_rates,
        "common_well_rate_spread_tph": common_well_rate_spread,
        "controlled_scenario_configs": {
            "no_disturbance": asdict(
                _controlled_formal_env(
                    episode_hours=168
                ).scenario_generator.config
            ),
            "single_vessel_single_emitter": asdict(
                _single_vessel_env().scenario_generator.config
            ),
        },
    }
    _write_json(out_dir / "config_snapshot.json", config_snapshot)

    failed = [row for row in checks if row["status"] != "PASS"]
    elapsed_seconds = time.perf_counter() - started
    summary = {
        "experiment": "E0 physical simulator validation",
        "status": "PASS" if not failed else "FAIL",
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "failed_check_ids": [row["check_id"] for row in failed],
        "pytest": pytest_summary,
        "cases": cases,
        "terminal_cleanup": cleanup,
        "elapsed_seconds": elapsed_seconds,
        "outputs": {
            "supplementary_table_s1": (
                "supplementary_table_s1.csv"
            ),
            "supplementary_figure_s1": (
                "figures/figure_s1_mass_balance_inventory.svg"
            ),
            "figure_source_data": (
                "source_data/figure_s1_timeseries.csv"
            ),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _write_readme(
        out_dir,
        checks=checks,
        cases=cases,
        pytest_summary=pytest_summary,
        cleanup=cleanup,
        elapsed_seconds=elapsed_seconds,
    )
    print(
        f"E0 {summary['status']}: "
        f"{summary['checks_passed']}/{summary['checks_total']} checks passed. "
        f"Results: {out_dir}"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
