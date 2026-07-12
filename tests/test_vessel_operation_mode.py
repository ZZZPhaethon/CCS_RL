from __future__ import annotations

import copy

import numpy as np

from sim.environment import CCSEnv, CCSEnvConfig
from sim.environment import vessel_mode
from sim.environment.vessel_mode import (
    VESSEL_OPERATION_MODES,
    vessel_operation_mode_feature_names,
    vessel_operation_mode_observation,
    vessel_operation_modes,
)
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _env() -> CCSEnv:
    env = CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=48)),
        config=CCSEnvConfig(episode_hours=48),
    )
    env.reset(seed=0)
    return env


def _berth(env: CCSEnv, vessel_id: str, location_id: str) -> None:
    env.simulator.state.vessel_berths[vessel_id] = location_id
    env.simulator.vessel_states[vessel_id] = {
        "mode": "berthed",
        "berth": location_id,
        "origin": location_id,
        "destination": location_id,
        "progress": 0.0,
    }


def test_mode_observation_is_vessel_major_one_hot_and_read_only():
    env = _env()
    before_state = copy.deepcopy(env.simulator.state.as_dict())
    before_vessels = copy.deepcopy(env.simulator.vessel_states)

    values = np.asarray(vessel_operation_mode_observation(env), dtype=np.float32).reshape(
        len(env.vessel_ids), len(VESSEL_OPERATION_MODES)
    )

    assert VESSEL_OPERATION_MODES == ("sailing", "loading", "unloading", "queued", "idle")
    np.testing.assert_array_equal(values.sum(axis=1), np.ones(len(env.vessel_ids)))
    assert np.logical_or(values == 0.0, values == 1.0).all()
    assert env.simulator.state.as_dict() == before_state
    assert env.simulator.vessel_states == before_vessels
    assert vessel_operation_mode_feature_names(env) == tuple(
        f"{vessel_id}.mode_{mode}"
        for vessel_id in env.vessel_ids
        for mode in VESSEL_OPERATION_MODES
    )


def test_sailing_vessel_is_sailing():
    env = _env()
    vessel_id = env.vessel_ids[0]
    env.simulator.vessel_states[vessel_id] = {
        "mode": "sailing",
        "berth": None,
        "origin": "source_a",
        "destination": "terminal",
        "progress": 0.5,
    }
    env.simulator.state.vessel_berths.pop(vessel_id, None)

    assert vessel_operation_modes(env)[0] == "sailing"


def test_sailing_destination_observation_is_vessel_major_one_hot():
    feature_names = getattr(vessel_mode, "vessel_sailing_destination_feature_names", None)
    observation = getattr(vessel_mode, "vessel_sailing_destination_observation", None)
    assert callable(feature_names)
    assert callable(observation)
    env = _env()
    vessel_id = env.vessel_ids[0]
    env.simulator.vessel_states[vessel_id] = {
        "mode": "sailing",
        "berth": None,
        "origin": "source_a",
        "destination": "source_b",
        "progress": 0.5,
    }
    env.simulator.state.vessel_berths.pop(vessel_id, None)
    destinations = [*env.terminal_ids, *env.emitter_ids]

    values = np.asarray(observation(env), dtype=np.float32).reshape(
        len(env.vessel_ids), len(destinations)
    )

    assert feature_names(env) == tuple(
        f"{current_vessel_id}.sailing_to_{destination_id}"
        for current_vessel_id in env.vessel_ids
        for destination_id in destinations
    )
    np.testing.assert_array_equal(values[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(values[1], np.zeros(len(destinations)))


def test_berthed_vessels_have_no_sailing_destination():
    observation = getattr(vessel_mode, "vessel_sailing_destination_observation", None)
    assert callable(observation)
    env = _env()

    values = np.asarray(observation(env), dtype=np.float32)

    np.testing.assert_array_equal(
        values,
        np.zeros(len(env.vessel_ids) * (len(env.terminal_ids) + len(env.emitter_ids))),
    )


def test_distinct_emitter_vessels_are_loading_when_transfer_is_possible():
    env = _env()
    state = env.simulator.state
    _berth(env, "vessel_a", "source_a")
    _berth(env, "vessel_b", "source_b")
    state.entity_inventory_t.update(
        {"source_a": 100.0, "source_b": 100.0, "vessel_a": 0.0, "vessel_b": 0.0}
    )

    assert vessel_operation_modes(env) == ("loading", "loading")


def test_second_eligible_vessel_at_same_emitter_is_queued():
    env = _env()
    state = env.simulator.state
    _berth(env, "vessel_a", "source_a")
    _berth(env, "vessel_b", "source_a")
    state.entity_inventory_t.update({"source_a": 100.0, "vessel_a": 0.0, "vessel_b": 0.0})

    assert vessel_operation_modes(env) == ("loading", "queued")


def test_terminal_fifo_head_is_unloading_and_follower_is_queued_without_mutation():
    env = _env()
    state = env.simulator.state
    terminal_id = env.terminal_ids[0]
    _berth(env, "vessel_a", terminal_id)
    _berth(env, "vessel_b", terminal_id)
    state.entity_inventory_t.update({"vessel_a": 100.0, "vessel_b": 100.0})
    state.terminal_unload_queues[terminal_id] = ["vessel_b", "vessel_a"]
    before = copy.deepcopy(state.terminal_unload_queues)

    assert vessel_operation_modes(env) == ("queued", "unloading")
    assert state.terminal_unload_queues == before


def test_zero_terminal_berths_make_loaded_vessels_queued():
    env = _env()
    state = env.simulator.state
    terminal_id = env.terminal_ids[0]
    _berth(env, "vessel_a", terminal_id)
    _berth(env, "vessel_b", terminal_id)
    state.entity_inventory_t.update({"vessel_a": 100.0, "vessel_b": 100.0})
    state.terminal_unload_queues[terminal_id] = ["vessel_a", "vessel_b"]
    state.berth_count_override[terminal_id] = 0

    assert vessel_operation_modes(env) == ("queued", "queued")


def test_nonservice_berthed_states_are_idle():
    env = _env()
    state = env.simulator.state
    terminal_id = env.terminal_ids[0]
    _berth(env, "vessel_a", "source_a")
    _berth(env, "vessel_b", terminal_id)
    state.entity_inventory_t["source_a"] = 100.0
    state.entity_inventory_t["vessel_a"] = env.network.entities["vessel_a"].capacity_t
    state.entity_inventory_t["vessel_b"] = 0.0

    assert vessel_operation_modes(env) == ("idle", "idle")


def test_empty_emitter_inventory_makes_waiting_vessel_idle():
    env = _env()
    state = env.simulator.state
    _berth(env, "vessel_a", "source_a")
    state.entity_inventory_t["source_a"] = 0.0
    state.entity_inventory_t["vessel_a"] = 0.0

    assert vessel_operation_modes(env)[0] == "idle"
