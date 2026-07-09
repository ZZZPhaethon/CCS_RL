"""Load fixed network scenarios from project data files.

This module turns JSON scenario data and capture-rate profiles into a
:class:`PhysicalNetwork` plus its initial :class:`PhysicalState`. It does not
sample random disturbances; that belongs to ``sim.scenario_generation``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .entities.emitter import Emitter
from .entities.manifold import SubseaManifold
from .entities.pipeline import Pipeline
from .entities.state import PhysicalState
from .entities.storage import InjectionWell, Reservoir
from .entities.terminal import Terminal
from .entities.vessel import Vessel
from .line_source import LineSourceParameters
from .network import PhysicalNetwork

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = ROOT / "scenarios"
CAPTURE_RATE_ROOT = ROOT / "data" / "capture_rates"
NORTHERN_LIGHTS_PHASE1_DATA_PATH = SCENARIO_ROOT / "northern_lights_phase1.json"
NORTHERN_LIGHTS_PHASE1_3VESSELS_DATA_PATH = SCENARIO_ROOT / "northern_lights_phase1_3vessels.json"
NORTHERN_LIGHTS_PHASE1_2WELL_DATA_PATH = SCENARIO_ROOT / "northern_lights_phase1_2well.json"
NORTHERN_LIGHTS_PHASE2_DATA_PATH = SCENARIO_ROOT / "northern_lights_phase2.json"
TOY_DATA_PATH = SCENARIO_ROOT / "toy.json"
MILK_RUN_STRESS_DATA_PATH = SCENARIO_ROOT / "milk_run_stress.json"
NORTHERN_LIGHTS_PHASE1_MILKRUN_DATA_PATH = SCENARIO_ROOT / "northern_lights_phase1_milkrun.json"
NORTHERN_LIGHTS_PHASE1_MILKRUN_IMBALANCED_DATA_PATH = SCENARIO_ROOT / "northern_lights_phase1_milkrun_imbalanced.json"
NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH = CAPTURE_RATE_ROOT / "phase1plus_emitters_capture_rate_profile_hourly.csv"

_FIXED_SCENARIO_PATHS = {
    "toy": TOY_DATA_PATH,
    "milk_run_stress": MILK_RUN_STRESS_DATA_PATH,
    "northern_lights_phase1": NORTHERN_LIGHTS_PHASE1_DATA_PATH,
    "northern_lights_phase1_3vessels": NORTHERN_LIGHTS_PHASE1_3VESSELS_DATA_PATH,
    "northern_lights_phase1_milkrun": NORTHERN_LIGHTS_PHASE1_MILKRUN_DATA_PATH,
    "northern_lights_phase1_milkrun_imbalanced": NORTHERN_LIGHTS_PHASE1_MILKRUN_IMBALANCED_DATA_PATH,
    "northern_lights_phase1_2well": NORTHERN_LIGHTS_PHASE1_2WELL_DATA_PATH,
    "northern_lights_phase2": NORTHERN_LIGHTS_PHASE2_DATA_PATH,
}


def _load_phase1_data() -> dict:
    return _load_fixed_scenario_data("northern_lights_phase1")


def _load_phase2_data() -> dict:
    return _load_fixed_scenario_data("northern_lights_phase2")


def _load_phase1_2well_data() -> dict:
    return _load_fixed_scenario_data("northern_lights_phase1_2well")


def _load_toy_data() -> dict:
    return _load_fixed_scenario_data("toy")


def available_fixed_scenario_ids() -> tuple[str, ...]:
    """Canonical fixed-network scenario ids backed by JSON files."""
    return tuple(sorted(_FIXED_SCENARIO_PATHS))


def available_fixed_scenario_choices() -> tuple[str, ...]:
    """CLI-friendly scenario names matching JSON file stems."""
    return available_fixed_scenario_ids()


def resolve_fixed_scenario_id(identifier: str) -> str:
    if identifier not in _FIXED_SCENARIO_PATHS:
        choices = ", ".join(available_fixed_scenario_choices())
        raise ValueError(f"Unknown fixed scenario '{identifier}'. Choose one of: {choices}")
    return identifier


def _load_fixed_scenario_data(identifier: str) -> dict:
    scenario_id = resolve_fixed_scenario_id(identifier)
    with _FIXED_SCENARIO_PATHS[scenario_id].open(encoding="utf-8") as handle:
        return json.load(handle)


def fixed_scenario_locations(identifier: str) -> dict[str, tuple[float, float]]:
    return {
        location_id: _coordinate(values)
        for location_id, values in _load_fixed_scenario_data(identifier)["locations"].items()
    }


def recommended_fixed_scenario_episode_hours(identifier: str, default: int = 600) -> int:
    """Scenario-specific recommended rollout horizon, if declared in JSON."""

    payload = _load_fixed_scenario_data(identifier)
    return int(payload.get("recommended_episode_hours", default))


def _load_hourly_capture_profiles(path: Path) -> dict[str, tuple[float, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    profile_columns = {
        column.removesuffix("_capture_tph"): column
        for column in rows[0]
        if column.endswith("_capture_tph") and column != "total_capture_tph"
    }
    return {
        entity_id: tuple(float(row[column]) for row in rows)
        for entity_id, column in profile_columns.items()
    }


def _coordinate(values: list[float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


PHASE1_DATA = _load_phase1_data()
PHASE1_2WELL_DATA = _load_phase1_2well_data()
NATURGASSPARKEN = _coordinate(PHASE1_DATA["locations"]["oygarden_terminal"])
EOS_SUBSEA_TEMPLATE_LOCATION = _coordinate(PHASE1_DATA["locations"]["eos_subsea_template"])
C1_INJECTION_WELL_LOCATION = _coordinate(PHASE1_2WELL_DATA["locations"]["c1_h_contingent_well"])
OFFSHORE_PIPELINE_ROUTE = [
    _coordinate(point)
    for point in PHASE1_DATA["offshore_pipeline_route"]
]


def build_northern_lights_phase1_demo() -> tuple[PhysicalNetwork, PhysicalState]:
    """Northern Lights Phase 1 commercial scenario used by demos and tests."""

    return build_fixed_scenario_demo("northern_lights_phase1")


def build_northern_lights_phase2_demo() -> tuple[PhysicalNetwork, PhysicalState]:
    """Phase 2-style topology from the public-data scenario JSON.

    The scenario keeps the current two public well locations for visualization and
    high-level operations checks; it does not instantiate placeholder Phase 2 wells
    whose coordinates are not public in the reviewed sources.
    """

    return build_fixed_scenario_demo("northern_lights_phase2")


def build_toy_demo() -> tuple[PhysicalNetwork, PhysicalState]:
    """Two-emitter toy comparison scenario used by controller experiments."""

    return build_fixed_scenario_demo("toy")


def build_milk_run_stress_demo() -> tuple[PhysicalNetwork, PhysicalState]:
    """One-vessel case where milk-run planning should beat greedy waiting."""

    return build_fixed_scenario_demo("milk_run_stress")


def build_fixed_scenario_demo(identifier: str) -> tuple[PhysicalNetwork, PhysicalState]:
    """Load any fixed-network scenario from the project ``scenarios`` folder."""
    scenario_id = resolve_fixed_scenario_id(identifier)
    hourly_profiles = {}
    if scenario_id in {
        "northern_lights_phase1",
        "northern_lights_phase1_3vessels",
        "northern_lights_phase1_2well",
    }:
        hourly_profiles = _load_hourly_capture_profiles(NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH)
    return _build_network_from_scenario_data(
        _load_fixed_scenario_data(scenario_id),
        hourly_capture_profiles=hourly_profiles,
    )


def toy_locations() -> dict[str, tuple[float, float]]:
    return fixed_scenario_locations("toy")


def milk_run_stress_locations() -> dict[str, tuple[float, float]]:
    return fixed_scenario_locations("milk_run_stress")


def _build_network_from_scenario_data(
    data: dict,
    hourly_capture_profiles: dict[str, tuple[float, ...]] | None = None,
) -> tuple[PhysicalNetwork, PhysicalState]:
    network = PhysicalNetwork(time_step_hours=float(data["time_step_hours"]))
    hourly_capture_profiles = hourly_capture_profiles or {}

    for emitter in data["emitters"]:
        annual_target_export_tpy = float(emitter["annual_target_export_tpy"])
        network.add_entity(
            Emitter(
                emitter["entity_id"],
                nominal_capture_tph=float(emitter.get("nominal_capture_tph", annual_target_export_tpy / 8760.0)),
                buffer_capacity_t=float(emitter["buffer_capacity_t"]),
                min_utilization=float(emitter["min_utilization"]),
                loading_rate_tph=float(emitter["loading_rate_tph"]),
                annual_target_export_tpy=annual_target_export_tpy,
                max_production_tph=float(emitter["max_production_tph"]),
                reference_name=emitter["reference_name"],
                hourly_capture_profile_tph=hourly_capture_profiles.get(emitter["entity_id"]),
            )
        )

    for vessel in data["vessels"]:
        network.add_entity(
            Vessel(
                vessel["entity_id"],
                capacity_t=float(vessel["capacity_t"]),
                loading_rate_tph=float(vessel["loading_rate_tph"]),
                unloading_rate_tph=float(vessel["unloading_rate_tph"]),
                volume_capacity_m3=float(vessel["volume_capacity_m3"]),
                speed_knots=float(vessel["speed_knots"]),
            )
        )

    terminal = data["terminal"]
    network.add_entity(
        Terminal(
            terminal["entity_id"],
            storage_capacity_t=float(terminal["storage_capacity_t"]),
            berth_count=int(terminal["berth_count"]),
            site_name=terminal["site_name"],
        )
    )

    pipeline = data["pipeline"]
    network.add_entity(
        Pipeline(
            pipeline["entity_id"],
            max_flow_tph=float(pipeline["max_flow_tph"]),
            annual_capacity_tpy=float(pipeline["annual_capacity_tpy"]),
            length_km=float(pipeline["length_km"]),
            route_color=pipeline["route_color"],
            route_coordinates=[_coordinate(point) for point in data["offshore_pipeline_route"]],
        )
    )

    manifold = data["manifold"]
    network.add_entity(
        SubseaManifold(
            manifold["entity_id"],
            max_flow_tph=float(manifold["max_flow_tph"]),
        )
    )

    for well in data["injection_wells"]:
        network.add_entity(
            InjectionWell(
                well["entity_id"],
                max_injection_tph=float(well["max_injection_tph"]),
            )
        )

    reservoir = data["reservoir"]
    line_source_parameters = data["line_source_parameters"]
    network.add_entity(
        Reservoir(
            reservoir["entity_id"],
            storage_capacity_t=float(reservoir["storage_capacity_t"]),
            initial_pressure_bar=float(reservoir["reservoir_initial_pressure_bar"]),
            pressure_at_capacity_bar=float(reservoir["reservoir_average_pressure_at_capacity_bar"]),
            max_pressure_bar=float(reservoir["reservoir_average_pressure_limit_bar"]),
            depth_m=float(reservoir["reservoir_depth_below_seabed_m"]),
            reservoir_pressure_model=reservoir["reservoir_pressure_model"],
            seawater_depth_m=float(reservoir["seawater_depth_m"]),
            well_fracture_gradient_psi_per_ft=float(reservoir["well_fracture_gradient_psi_per_ft"]),
            well_fracture_gradient_reference_depth_m=float(reservoir["well_fracture_gradient_reference_depth_m"]),
            well_fracture_pressure_bar=float(reservoir["well_fracture_pressure_bar"]),
            well_bottomhole_pressure_safety_factor=float(reservoir["well_bottomhole_pressure_safety_factor"]),
            well_bottomhole_pressure_limit_bar=float(reservoir["well_bottomhole_pressure_limit_bar"]),
            line_source_parameters=LineSourceParameters(
                initial_pressure_bar=float(line_source_parameters["initial_reservoir_pressure_bar"]),
                permeability_md=float(line_source_parameters["permeability_md"]),
                thickness_m=float(line_source_parameters["thickness_m"]),
                porosity_fraction=float(line_source_parameters["porosity_fraction"]),
                total_compressibility_1_pa=float(line_source_parameters["total_compressibility_1_pa"]),
                viscosity_pa_s=float(line_source_parameters["viscosity_pa_s"]),
                co2_density_kg_m3=float(line_source_parameters["co2_density_kg_m3"]),
                well_radius_m=float(line_source_parameters["well_radius_m"]),
                skin=float(line_source_parameters["skin"]),
            ),
            line_source_observation_radii_m=tuple(
                float(radius_m)
                for radius_m in reservoir["line_source_observation_radii_m"]
            ),
            line_source_well_distances_m=reservoir["line_source_well_distances_m"],
            line_source_parameter_status=dict(data["line_source_parameter_status"]),
        )
    )

    for source, target in data["connections"]:
        network.connect(source, target)

    state = PhysicalState(
        entity_inventory_t={
            entity_id: 0.0
            for entity_id in network.entities
        }
    )
    return network, state
