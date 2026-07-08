"""Toy network fixtures for fast unit tests.

These fixtures are intentionally not physically calibrated. They keep small,
round numbers so environment, metrics, and MILP tests can exercise mechanics
quickly without implying a real Northern Lights / Yara benchmark.
"""

from __future__ import annotations

from sim.entities import (
    Emitter,
    InjectionWell,
    Pipeline,
    Reservoir,
    SubseaManifold,
    Terminal,
    Vessel,
)
from sim.network import PhysicalNetwork

TOY_TWO_SOURCE_LOCATIONS = {
    "source_a": (59.05, 9.70),
    "source_b": (59.86, 10.84),
    "terminal": (60.58, 4.84),
}


def make_toy_two_source_network() -> PhysicalNetwork:
    """A compact two-source shipping network for unit tests only."""
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source_a", nominal_capture_tph=80.0, buffer_capacity_t=4_000.0))
    network.add_entity(Emitter("source_b", nominal_capture_tph=60.0, buffer_capacity_t=4_000.0))
    network.add_entity(
        Vessel(
            "vessel_a",
            capacity_t=800.0,
            loading_rate_tph=800.0,
            unloading_rate_tph=800.0,
            speed_knots=12.0,
        )
    )
    network.add_entity(
        Vessel(
            "vessel_b",
            capacity_t=800.0,
            loading_rate_tph=800.0,
            unloading_rate_tph=800.0,
            speed_knots=12.0,
        )
    )
    network.add_entity(Terminal("terminal", storage_capacity_t=6_000.0, berth_count=2))
    network.add_entity(Pipeline("pipeline", max_flow_tph=400.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=400.0))
    network.add_entity(InjectionWell("well_a", max_injection_tph=200.0))
    network.add_entity(InjectionWell("well_b", max_injection_tph=200.0))
    network.add_entity(
        Reservoir(
            "reservoir",
            storage_capacity_t=1e7,
            initial_pressure_bar=100.0,
            pressure_at_capacity_bar=200.0,
            max_pressure_bar=200.0,
        )
    )
    network.connect("source_a", "vessel_a")
    network.connect("source_b", "vessel_b")
    network.connect("vessel_a", "terminal")
    network.connect("vessel_b", "terminal")
    network.connect("terminal", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well_a")
    network.connect("manifold", "well_b")
    network.connect("well_a", "reservoir")
    network.connect("well_b", "reservoir")
    return network
