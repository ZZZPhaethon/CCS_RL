from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from sim.control.event_based.rl.observation_encoder import FORECAST_WINDOWS_H
from sim.control.event_based.residual_rl_v4.scenario import (
    ReplayableDifficultyScenarioGenerator,
)
from sim.economics import EconomicParameters


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "experiments" / "protocols" / "unified_window_v1_paper_protocol.json"
)
SEED_MANIFEST_PATH = (
    ROOT / "experiments" / "protocols" / "unified_window_v1_seed_manifest.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inclusive_range(spec: list[int]) -> set[int]:
    start, end = spec
    assert start <= end
    return set(range(start, end + 1))


def test_locked_protocol_matches_shared_source_constants() -> None:
    protocol = _load(PROTOCOL_PATH)
    economics = protocol["economic_objective"]
    parameters = economics["parameters"]
    defaults = EconomicParameters()

    assert protocol["status"] == "design_locked_implementation_pending"
    assert protocol["scope"]["episode_hours"] == 720
    assert protocol["scope"]["scenario_hours"] == 888
    assert protocol["scope"]["time_step_hours"] == 1.0
    assert protocol["scope"]["forecast_context_hours"] == 168
    assert (
        protocol["environment"]["terminal_boundary"]["mode"]
        == "common_compact_trip_cleanup_value"
    )
    assert protocol["environment"]["terminal_boundary"][
        "applies_to_training_terminal_return"
    ]
    assert tuple(
        protocol["forecast_protocol"]["learning_methods"]["windows_hours"]
    ) == FORECAST_WINDOWS_H
    assert not protocol["forecast_protocol"]["learning_methods"][
        "valid_fraction_feature"
    ]
    assert (
        protocol["forecast_protocol"][
            "execution_and_scoring_boundary_hours"
        ]
        == 720
    )
    well_control = protocol["control_scope"]["well_control"]
    assert (
        well_control["mode"]
        == "automatic_continuous_maximum_feasible_rate"
    )
    assert well_control["rate_unit"] == "tonnes_per_hour"
    assert parameters["carbon_price_eur_per_t"] == defaults.carbon_price_eur_per_t
    assert parameters["conditioning_eur_per_t"] == defaults.conditioning_eur_per_t
    assert (
        parameters["reconditioning_eur_per_t"]
        == defaults.reconditioning_eur_per_t
    )
    assert (
        economics["derived_rates"]["vessel_fuel_eur_per_h_sailing"]
        == pytest.approx(defaults.vessel_fuel_eur_per_h_sailing)
    )
    assert economics["derived_rates"]["hoteling_fuel_eur_per_h"] == pytest.approx(
        defaults.hoteling_fuel_eur_per_h
    )


def test_locked_scenario_and_disturbance_definition_have_not_drifted() -> None:
    protocol = _load(PROTOCOL_PATH)
    scope = protocol["scope"]
    scenario_path = ROOT / scope["scenario_file"]
    assert hashlib.sha256(scenario_path.read_bytes()).hexdigest() == scope[
        "scenario_file_sha256"
    ]

    generator = ReplayableDifficultyScenarioGenerator(
        episode_hours=scope["episode_hours"] + scope["forecast_context_hours"],
        weather_process="window",
        hard_probability=0.5,
        scenario_protocol=scope["scenario_protocol"],
    )
    actual = generator.normal.config
    locked = protocol["medium_stress_disturbance"]
    for field in (
        "capture_noise_std",
        "capture_outage_rate_per_week",
        "capture_outage_mean_hours",
        "capture_high_output_rate_per_week",
        "capture_high_output_mean_hours",
        "weather_window_rate_per_week",
        "weather_window_mean_hours",
        "well_maintenance_rate_per_week",
        "well_maintenance_mean_hours",
    ):
        assert getattr(actual, field) == locked[field]
    for field in (
        "capture_high_output_multiplier_range",
        "weather_window_speed_factor_range",
        "emitter_initial_fill_range",
        "terminal_initial_fill_range",
        "reservoir_initial_pressure_fill_range",
    ):
        assert tuple(getattr(actual, field)) == tuple(locked[field])


def test_protocol_removes_well_control_from_every_compared_method() -> None:
    protocol = _load(PROTOCOL_PATH)
    scope = protocol["control_scope"]
    well = scope["well_control"]

    assert scope["upper_level_actions"] == ["vessel_dispatch"]
    assert "well_injection_rate" in scope["excluded_upper_level_actions"]
    expected = set(protocol["formal_online_controllers"])
    expected.add(protocol["offline_reference"])
    assert set(well["applies_to"]) == expected


def test_seed_manifest_keeps_formal_test_unseen_and_disjoint() -> None:
    manifest = _load(SEED_MANIFEST_PATH)
    assert manifest["protocol_id"] == _load(PROTOCOL_PATH)["protocol_id"]

    validation = _inclusive_range(
        manifest["controller_validation"]["range_inclusive"]
    )
    formal_test = _inclusive_range(manifest["formal_test"]["range_inclusive"])
    legacy = _inclusive_range(
        manifest["legacy_development_only"]["range_inclusive"]
    )

    assert len(validation) == manifest["controller_validation"]["count"]
    assert len(formal_test) == manifest["formal_test"]["count"]
    assert validation.isdisjoint(formal_test)
    assert legacy.isdisjoint(formal_test)
    assert legacy.isdisjoint(validation)


def test_iterative_q_schedule_reaches_locked_cumulative_roots() -> None:
    protocol = _load(PROTOCOL_PATH)
    manifest = _load(SEED_MANIFEST_PATH)
    cumulative = 0
    observed: dict[str, int] = {}

    for index, group in enumerate(("G0", "G1", "G2", "G3"), start=1):
        stage = manifest["iterative_q_data"][group]
        train_seed_count = len(_inclusive_range(stage["train_range_inclusive"]))
        cumulative += train_seed_count * stage["roots_per_train_seed"]
        observed[f"P{index}"] = cumulative

    assert observed == protocol["training_budget"]["iterative_q_cumulative_roots"]
