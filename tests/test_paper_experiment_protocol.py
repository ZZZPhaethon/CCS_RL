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

    assert protocol["status"] == "e1_completed_analysis_pending"
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
    full_milp = protocol["forecast_protocol"]["full_horizon_milp"]
    assert full_milp["planning_horizon_hours"] == 720
    assert full_milp["terminal_cleanup_boundary_hours"] == 720
    assert not full_milp["post_720h_forecast_used"]
    milp_compute = protocol["milp_compute_protocol"]
    assert milp_compute["solver_threads_per_process"] == 4
    assert milp_compute["rolling_milp"][
        "validation_time_limits_seconds_per_replan"
    ] == [30, 300]
    assert milp_compute["rolling_milp"][
        "formal_time_limit_seconds_per_replan"
    ] == 600
    assert milp_compute["rolling_milp"][
        "superseded_time_limit_seconds_per_replan"
    ] == 300
    assert milp_compute["full_horizon_milp"][
        "time_limit_seconds_per_seed"
    ] == 18000
    assert milp_compute["full_horizon_milp"][
        "superseded_time_limit_seconds_per_seed"
    ] == 7200
    promotion = milp_compute["promoted_result_provenance"]
    assert promotion["rolling_array_job_id"] == 34375
    assert promotion["full_array_job_id"] == 34376
    assert promotion["superseded_results_retained_for_provenance"]
    assert not promotion["single_factor_time_limit_comparison"]
    assert promotion["source_hashes_differ_from_superseded_run"]
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


def test_hourly_ppo_protocol_is_direct_one_hour_baseline() -> None:
    protocol = _load(PROTOCOL_PATH)
    hourly = protocol["learning_controller_interfaces"][
        "Hourly Centralized Maskable PPO"
    ]

    assert hourly["policy_transition_hours"] == 1
    assert hourly["physical_steps_per_policy_transition"] == 1
    assert hourly["action_space"]["type"] == "MultiDiscrete"
    assert hourly["action_space"]["three_vessel_dimensions"] == [5, 5, 5]
    assert hourly["action_space"]["directly_applied_to_native_environment"]
    assert hourly["action_space"]["legal_action_masks"]
    assert hourly["gamma"] == 1.0
    assert "event trigger" in hourly["excluded_structure"]
    assert "rule or MPC executor" in hourly["excluded_structure"]
    assert "Greedy default" in hourly["excluded_structure"]
    assert "residual action" in hourly["excluded_structure"]


def test_seed_manifest_tracks_active_and_deprecated_test_sets() -> None:
    manifest = _load(SEED_MANIFEST_PATH)
    protocol = _load(PROTOCOL_PATH)
    assert manifest["protocol_id"] == protocol["protocol_id"]
    assert (
        manifest["formal_test"]["range_inclusive"]
        == protocol["test_set_revision"]["active_range_inclusive"]
    )
    assert (
        manifest["deprecated_formal_test"]["range_inclusive"]
        == protocol["test_set_revision"]["deprecated_range_inclusive"]
    )

    validation = _inclusive_range(
        manifest["controller_validation"]["range_inclusive"]
    )
    formal_test = _inclusive_range(manifest["formal_test"]["range_inclusive"])
    deprecated_test = _inclusive_range(
        manifest["deprecated_formal_test"]["range_inclusive"]
    )
    legacy = _inclusive_range(
        manifest["legacy_development_only"]["range_inclusive"]
    )

    assert manifest["manifest_version"] == 5
    assert formal_test == set(range(9000031, 9000061))
    assert deprecated_test == set(range(9000001, 9000031))
    assert len(validation) == manifest["controller_validation"]["count"]
    assert len(formal_test) == manifest["formal_test"]["count"]
    assert len(deprecated_test) == manifest["deprecated_formal_test"]["count"]
    assert validation.isdisjoint(formal_test)
    assert validation.isdisjoint(deprecated_test)
    assert formal_test.isdisjoint(deprecated_test)
    assert legacy.isdisjoint(formal_test)
    assert legacy.isdisjoint(deprecated_test)
    assert legacy.isdisjoint(validation)
    assert manifest["formal_test"]["access_status"] == "formal_test_completed"
    assert (
        manifest["deprecated_formal_test"]["result_status"]
        == "deprecated_for_future_primary_comparison_but_retained_for_provenance"
    )
    for spec in manifest["ppo_training_episode_seeds"].values():
        if not isinstance(spec, dict):
            continue
        training = _inclusive_range(spec["range_inclusive"])
        assert training.isdisjoint(validation)
        assert training.isdisjoint(formal_test)
        assert training.isdisjoint(deprecated_test)
        assert training.isdisjoint(legacy)


def test_iterative_q_schedule_reaches_locked_cumulative_roots() -> None:
    protocol = _load(PROTOCOL_PATH)
    manifest = _load(SEED_MANIFEST_PATH)
    cumulative = 0
    observed: dict[str, int] = {}

    for index, stage in enumerate(manifest["iterative_q_data"].values(), start=1):
        train_seed_count = len(_inclusive_range(stage["train_range_inclusive"]))
        cumulative += train_seed_count * stage["roots_per_train_seed"]
        observed[f"P{index}"] = cumulative

    assert observed == protocol["training_budget"]["iterative_q_cumulative_roots"]
