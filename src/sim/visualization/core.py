from __future__ import annotations

import json
import math
from typing import Any

from ..actions import ActionFrame, ActionProposal
from ..entities.emitter import Emitter
from ..entities.pipeline import Pipeline
from ..entities.storage import InjectionWell, Reservoir
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..routes import route_distance_km, sea_route
from ..network_scenarios import (
    C1_INJECTION_WELL_LOCATION,
    EOS_SUBSEA_TEMPLATE_LOCATION,
    NATURGASSPARKEN,
    NORTHERN_LIGHTS_PHASE1_DATA_PATH,
    NORTHERN_LIGHTS_PHASE2_DATA_PATH,
    OFFSHORE_PIPELINE_ROUTE,
    build_northern_lights_phase1_demo,
    build_northern_lights_phase2_demo,
)
from ..simulator import PhysicalSimulator

Coordinate = tuple[float, float]
MAX_DISPLAY_ROUTE_SEGMENT_KM = 25.0
CORNER_SMOOTHING_FRACTION = 0.12
CORNER_SMOOTHING_STEPS = 5

LOCATIONS: dict[str, dict[str, float | str]] = {
    "brevik": {"lat": 59.05, "lon": 9.70, "label": "Brevik"},
    "celsio": {"lat": 59.86, "lon": 10.84, "label": "Celsio Oslo"},
    "oygarden_terminal": {
        "lat": NATURGASSPARKEN[0],
        "lon": NATURGASSPARKEN[1],
        "label": "Naturgassparken (Northern Lights Carbon Capture Plant Site)",
    },
    "oygarden_pipeline": {"lat": 60.58, "lon": 3.65, "label": "Offshore CO2 Pipeline"},
    "aurora_subsea_manifold": {
        "lat": EOS_SUBSEA_TEMPLATE_LOCATION[0],
        "lon": EOS_SUBSEA_TEMPLATE_LOCATION[1],
        "label": "31/5-7 EOS Subsea Distribution Manifold",
    },
    "aurora_well_a7_ah": {
        "lat": EOS_SUBSEA_TEMPLATE_LOCATION[0],
        "lon": EOS_SUBSEA_TEMPLATE_LOCATION[1],
        "label": "31/5-A-7 AH Injection Well",
    },
    "aurora_well_c1_h": {
        "lat": C1_INJECTION_WELL_LOCATION[0],
        "lon": C1_INJECTION_WELL_LOCATION[1],
        "label": "31/5-C-1 H Injection Well",
    },
    "aurora_reservoir": {"lat": 60.55, "lon": 3.46, "label": "Aurora Reservoir"},
}

VESSEL_ROUTES = {
    "northern_pioneer": ("brevik", "oygarden_terminal", 0.05),
    "northern_pathfinder": ("celsio", "oygarden_terminal", 0.45),
}

def build_demo_trajectory(
    hours: int = 48,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> dict[str, Any]:
    network, state = build_northern_lights_phase1_demo()
    with NORTHERN_LIGHTS_PHASE1_DATA_PATH.open(encoding="utf-8") as handle:
        scenario_data = json.load(handle)
    return build_trajectory(
        network,
        state,
        locations=_locations_from_scenario_data(scenario_data),
        hours=hours,
        action_frames=action_frames,
        action_generator_factory=action_generator_factory,
        title="CCS Physical Layer Dashboard",
    )


def build_northern_lights_phase2_trajectory(
    hours: int = 240,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> dict[str, Any]:
    network, state = build_northern_lights_phase2_demo()
    with NORTHERN_LIGHTS_PHASE2_DATA_PATH.open(encoding="utf-8") as handle:
        scenario_data = json.load(handle)
    return build_trajectory(
        network,
        state,
        locations=_locations_from_phase2_data(scenario_data),
        hours=hours,
        action_frames=action_frames,
        action_generator_factory=action_generator_factory,
        title="Northern Lights Phase 2 Dashboard",
    )


def build_northern_lights_phase1_trajectory(
    hours: int = 72,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> dict[str, Any]:
    network, state = build_northern_lights_phase1_demo()
    with NORTHERN_LIGHTS_PHASE1_DATA_PATH.open(encoding="utf-8") as handle:
        scenario_data = json.load(handle)
    return build_trajectory(
        network,
        state,
        locations=_locations_from_scenario_data(scenario_data),
        hours=hours,
        action_frames=action_frames,
        action_generator_factory=action_generator_factory,
        title="Northern Lights Phase 1 Dashboard",
    )
def _locations_from_phase2_data(data: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    return _locations_from_scenario_data(data)


def _locations_from_scenario_data(data: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    locations: dict[str, dict[str, float | str]] = {}

    for emitter in data["emitters"]:
        lat, lon = emitter["coordinates"]
        locations[emitter["entity_id"]] = {
            "lat": float(lat),
            "lon": float(lon),
            "label": emitter["reference_name"],
        }

    terminal = data["terminal"]
    terminal_lat, terminal_lon = data["locations"][terminal["entity_id"]]
    locations[terminal["entity_id"]] = {
        "lat": float(terminal_lat),
        "lon": float(terminal_lon),
        "label": terminal["site_name"],
    }

    route = data["offshore_pipeline_route"]
    midpoint = route[len(route) // 2]
    locations[data["pipeline"]["entity_id"]] = {
        "lat": float(midpoint[0]),
        "lon": float(midpoint[1]),
        "label": "Offshore CO2 Pipeline",
    }

    manifold = data["manifold"]
    manifold_lat, manifold_lon = data["locations"]["eos_subsea_template"]
    locations[manifold["entity_id"]] = {
        "lat": float(manifold_lat),
        "lon": float(manifold_lon),
        "label": "31/5-7 EOS Subsea Distribution Manifold",
    }

    for well in data["injection_wells"]:
        coordinates = well.get("coordinates")
        if coordinates is None:
            continue
        lat, lon = coordinates
        locations[well["entity_id"]] = {
            "lat": float(lat),
            "lon": float(lon),
            "label": well.get("public_name") or well["entity_id"],
        }

    locations[data["reservoir"]["entity_id"]] = {
        "lat": 60.55,
        "lon": 3.46,
        "label": "Aurora Reservoir",
    }
    return locations


def build_trajectory(
    network: Any,
    state: Any,
    locations: dict[str, dict[str, float | str]],
    hours: int = 48,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
    title: str = "CCS Physical Layer Dashboard",
) -> dict[str, Any]:
    if hours < 0:
        raise ValueError("hours must be non-negative")

    routes = _build_routes(network, locations)
    simulator = PhysicalSimulator(network, state, routes, _dashboard_locations(locations))
    action_generator = action_generator_factory(network, routes) if action_generator_factory else None
    frames = [
        _frame_from_state(
            network,
            simulator.state,
            flows={},
            violations=[],
            actions={},
            vessel_positions=simulator.vessel_positions(),
        )
    ]

    for hour in range(hours):
        if action_generator is not None:
            action_frame = action_generator.next_action_frame(simulator.state)
        else:
            actions = _actions_at(action_frames, hour)
            action_frame = ActionFrame(time_h=simulator.state.time_h, proposals=_legacy_action_proposals(actions))
        record = simulator.step(action_frame)
        frames.append(
            _frame_from_state(
                network,
                record.step_result.state,
                flows=record.step_result.as_dict()["flows_t"],
                violations=record.step_result.as_dict()["violations"],
                actions=record.committed_action_frame.actions,
                vessel_positions=record.vessel_positions,
            )
        )

    first_route = next(iter(routes.values()), {})
    return {
        "title": title,
        "time_step_hours": network.time_step_hours,
        "route": {
            "provider": first_route.get("provider"),
            "distance_km": first_route.get("distance_km"),
            "coordinates": first_route.get("coordinates", []),
        },
        "map": {
            "bbox": _map_bbox(routes, network, locations),
            "locations": locations,
            "routes": routes,
            "pipeline_segments": _build_pipeline_segments(network, locations),
            "injection_links": _build_injection_links(network, locations),
        },
        "connections": [
            {"source": connection.source, "target": connection.target}
            for connection in network.connections
        ],
        "storage_targets": _storage_targets(network),
        "frames": frames,
    }


def _build_routes(network: Any, locations: dict[str, dict[str, float | str]]) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    origin_usage: dict[str, int] = {}
    for vessel_id, entity in network.entities.items():
        if not isinstance(entity, Vessel):
            continue
        origin_id = _route_origin_for_vessel(network, vessel_id, origin_usage)
        destination_id = _route_destination_for_vessel(network, vessel_id)
        if origin_id is None or destination_id is None:
            continue
        origin_usage[origin_id] = origin_usage.get(origin_id, 0) + 1
        origin = _location_tuple(origin_id, locations)
        destination = _location_tuple(destination_id, locations)
        route = sea_route(origin, destination)
        if route.provider != "searoute":
            raise RuntimeError("Dashboard route generation requires searoute. Install it with `python -m pip install searoute`.")
        display_coordinates = _connect_route_to_facilities(route.coordinates, origin, destination)
        routes[vessel_id] = {
            "vessel_id": vessel_id,
            "origin": origin_id,
            "destination": destination_id,
            "provider": route.provider,
            "distance_km": round(route.distance_km, 2),
            "speed_knots": _vessel_speed_knots(network, vessel_id),
            "sea_coordinates": route.coordinates,
            "coordinates": display_coordinates,
            "return_coordinates": list(reversed(display_coordinates)),
            "return_policy": "same_corridor_reverse",
        }
    return routes


def _route_origin_for_vessel(network: Any, vessel_id: str, origin_usage: dict[str, int]) -> str | None:
    emitter_ids = [
        upstream_id
        for upstream_id in network.upstream_of(vessel_id)
        if isinstance(network.entities.get(upstream_id), Emitter)
    ]
    if not emitter_ids:
        return None
    return min(emitter_ids, key=lambda emitter_id: (origin_usage.get(emitter_id, 0), emitter_ids.index(emitter_id)))


def _route_destination_for_vessel(network: Any, vessel_id: str) -> str | None:
    for downstream_id in network.downstream_of(vessel_id):
        if isinstance(network.entities.get(downstream_id), Terminal):
            return downstream_id
    return None


def _vessel_speed_knots(network: Any, vessel_id: str) -> float | None:
    vessel = network.entities.get(vessel_id)
    if isinstance(vessel, Vessel) and vessel.speed_knots:
        return float(vessel.speed_knots)
    return None


def _connect_route_to_facilities(
    sea_coordinates: list[Coordinate],
    origin: Coordinate,
    destination: Coordinate,
) -> list[Coordinate]:
    coordinates = list(sea_coordinates)
    if not coordinates:
        return [origin, destination]
    if coordinates[0] != origin:
        coordinates.insert(0, origin)
    if coordinates[-1] != destination:
        coordinates.append(destination)
    coordinates = _smooth_route_corners(coordinates)
    return _densify_route(coordinates, MAX_DISPLAY_ROUTE_SEGMENT_KM)


def _smooth_route_corners(coordinates: list[Coordinate]) -> list[Coordinate]:
    if len(coordinates) < 3:
        return list(coordinates)

    smoothed = [coordinates[0]]
    for previous, current, following in zip(coordinates, coordinates[1:], coordinates[2:]):
        entry = _interpolate_coordinate(current, previous, CORNER_SMOOTHING_FRACTION)
        exit_point = _interpolate_coordinate(current, following, CORNER_SMOOTHING_FRACTION)
        smoothed.append(entry)
        for step in range(1, CORNER_SMOOTHING_STEPS + 1):
            t = step / CORNER_SMOOTHING_STEPS
            smoothed.append(_quadratic_bezier(entry, current, exit_point, t))
    smoothed.append(coordinates[-1])
    return smoothed


def _quadratic_bezier(start: Coordinate, control: Coordinate, end: Coordinate, t: float) -> Coordinate:
    lat_start, lon_start = start
    lat_control, lon_control = control
    lat_end, lon_end = end
    inverse = 1.0 - t
    return (
        inverse * inverse * lat_start + 2 * inverse * t * lat_control + t * t * lat_end,
        inverse * inverse * lon_start + 2 * inverse * t * lon_control + t * t * lon_end,
    )


def _interpolate_coordinate(start: Coordinate, end: Coordinate, fraction: float) -> Coordinate:
    lat_start, lon_start = start
    lat_end, lon_end = end
    return (lat_start + (lat_end - lat_start) * fraction, lon_start + (lon_end - lon_start) * fraction)


def _densify_route(coordinates: list[Coordinate], max_segment_km: float) -> list[Coordinate]:
    if len(coordinates) < 2:
        return list(coordinates)

    densified = [coordinates[0]]
    for start, end in zip(coordinates, coordinates[1:]):
        segment_km = route_distance_km([start, end])
        steps = max(1, math.ceil(segment_km / max_segment_km))
        lat_a, lon_a = start
        lat_b, lon_b = end
        for step in range(1, steps + 1):
            fraction = step / steps
            densified.append((lat_a + (lat_b - lat_a) * fraction, lon_a + (lon_b - lon_a) * fraction))
    return densified


def _build_pipeline_segments(network: Any, locations: dict[str, dict[str, float | str]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for pipeline_id, pipeline in network.entities.items():
        if not isinstance(pipeline, Pipeline):
            continue
        upstream_terminal = _first_upstream_of_type(network, pipeline_id, Terminal)
        downstream_target = network.downstream_of(pipeline_id)[0] if network.downstream_of(pipeline_id) else None
        if upstream_terminal is None or downstream_target is None:
            continue
        coordinates = list(pipeline.route_coordinates) or _fallback_pipeline_coordinates(
            upstream_terminal,
            pipeline_id,
            downstream_target,
            locations,
        )
        segment_id = (
            "naturgassparken_to_eos_subsea_manifold"
            if pipeline.route_coordinates
            else f"{upstream_terminal}_to_{downstream_target}"
        )
        label = (
            "Offshore pipeline from Naturgassparken to 31/5-7 EOS subsea manifold"
            if pipeline.route_coordinates
            else f"Pipeline from {upstream_terminal} to {downstream_target}"
        )
        segments.append(
            {
                "id": segment_id,
                "pipeline_id": pipeline.entity_id,
                "component_id": pipeline.entity_id,
                "source": upstream_terminal,
                "target": downstream_target,
                "label": label,
                "color": pipeline.route_color or "#ff0000",
                "coordinates": coordinates,
                "length_km": pipeline.length_km,
            }
        )
    for manifold_id, entity in network.entities.items():
        if entity.__class__.__name__ != "SubseaManifold":
            continue
        for well_id in network.downstream_of(manifold_id):
            if not isinstance(network.entities.get(well_id), InjectionWell):
                continue
            source = _location_tuple(manifold_id, locations)
            target = _location_tuple(well_id, locations)
            segments.append(
                {
                    "id": f"{manifold_id}_to_{well_id}",
                    "pipeline_id": "oygarden_pipeline",
                    "component_id": manifold_id,
                    "source": manifold_id,
                    "target": well_id,
                    "label": f"{manifold_id} to {well_id}",
                    "color": "#7c3aed",
                    "style": "subsea_connection",
                    "coordinates": [source, target],
                }
            )
    return segments


def _fallback_pipeline_coordinates(
    upstream_id: str,
    pipeline_id: str,
    downstream_id: str,
    locations: dict[str, dict[str, float | str]],
) -> list[Coordinate]:
    coordinates = list(OFFSHORE_PIPELINE_ROUTE)
    coordinates[0] = _location_tuple(upstream_id, locations)
    coordinates[-1] = _location_tuple(downstream_id, locations)
    return coordinates


def _first_upstream_of_type(network: Any, entity_id: str, entity_type: type) -> str | None:
    for upstream_id in network.upstream_of(entity_id):
        if isinstance(network.entities.get(upstream_id), entity_type):
            return upstream_id
    return None


def _build_injection_links(network: Any, locations: dict[str, dict[str, float | str]]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for connection in network.connections:
        source_entity = network.entities.get(connection.source)
        target_entity = network.entities.get(connection.target)
        if not isinstance(source_entity, InjectionWell) or not isinstance(target_entity, Reservoir):
            continue
        source = _location_tuple(connection.source, locations)
        target = _location_tuple(connection.target, locations)
        links.append(
            {
                "id": f"{connection.source}_to_{connection.target}",
                "component_id": connection.target,
                "source": connection.source,
                "target": connection.target,
                "label": f"{connection.source} injection target: {connection.target}",
                "color": "#64748b",
                "relation": "injection_target",
                "style": "geologic",
                "coordinates": [source, target],
            }
        )
    return links


def _storage_targets(network: Any) -> dict[str, str]:
    targets: dict[str, str] = {}
    for connection in network.connections:
        source = network.entities.get(connection.source)
        target = network.entities.get(connection.target)
        if isinstance(source, InjectionWell) and isinstance(target, Reservoir):
            targets[connection.source] = connection.target
    return targets


def _frame_from_state(
    network: Any,
    state: Any,
    flows: dict[str, float],
    violations: list[dict[str, Any]],
    actions: dict[str, Any],
    vessel_positions: dict[str, dict[str, object]],
) -> dict[str, Any]:
    snapshot = network.snapshot(state)
    return {
        "time_h": state.time_h,
        "entities": snapshot["entities"],
        "flows_t": flows,
        "violations": violations,
        "actions": actions,
        "vessel_positions": vessel_positions,
    }


def _actions_at(
    action_frames: list[dict[str, dict[str, Any]]] | None,
    hour: int,
) -> dict[str, dict[str, Any]]:
    if action_frames is None or hour >= len(action_frames):
        return {}
    return {
        entity_id: dict(action)
        for entity_id, action in action_frames[hour].items()
    }


def _legacy_action_proposals(actions: dict[str, dict[str, Any]]) -> list[ActionProposal]:
    proposals: list[ActionProposal] = []
    for entity_id, action in actions.items():
        if "utilization" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="set_capture_utilization",
                    params={"utilization": action["utilization"]},
                )
            )
        if "load_vessel" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="load_vessel",
                    params={"vessel_id": action["load_vessel"]},
                )
            )
        if "sail_to" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="sail_to",
                    params={"destination_id": action["sail_to"]},
                )
            )
        if "unload_vessel" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="unload_vessel",
                    params={"vessel_id": action["unload_vessel"]},
                )
            )
        if "flow_tph" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="set_flow",
                    params={"flow_tph": action["flow_tph"]},
                )
            )
    return proposals


def _dashboard_locations(locations: dict[str, dict[str, float | str]]) -> dict[str, Coordinate]:
    return {
        location_id: (float(location["lat"]), float(location["lon"]))
        for location_id, location in locations.items()
    }


def _physical_actions(
    actions: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        entity_id: {
            key: value
            for key, value in action.items()
            if key != "sail_to"
        }
        for entity_id, action in actions.items()
        if entity_id not in routes
    }


def _initial_vessel_states(routes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        vessel_id: {
            "mode": "berthed",
            "berth": route["origin"],
            "origin": route["origin"],
            "destination": route["origin"],
            "progress": 0.0,
        }
        for vessel_id, route in routes.items()
    }


def _start_vessel_voyages(
    vessel_states: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
    actions: dict[str, dict[str, Any]],
) -> None:
    for vessel_id, route in routes.items():
        action = actions.get(vessel_id, {})
        destination = action.get("sail_to")
        state = vessel_states[vessel_id]
        if not destination or state["mode"] != "berthed" or destination == state["berth"]:
            continue
        if destination not in {route["origin"], route["destination"]}:
            continue
        state.update(
            {
                "mode": "sailing",
                "origin": state["berth"],
                "destination": destination,
                "berth": None,
                "progress": 0.0,
            }
        )


def _advance_vessel_voyages(
    vessel_states: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
    hours: float,
) -> None:
    for vessel_id, state in vessel_states.items():
        if state["mode"] != "sailing":
            continue
        route = routes[vessel_id]
        speed_knots = route.get("speed_knots")
        distance_km = float(route.get("distance_km") or 0.0)
        if not speed_knots or distance_km <= 0.0:
            state["progress"] = 1.0
        else:
            state["progress"] = min(1.0, state["progress"] + (float(speed_knots) * 1.852 * hours) / distance_km)
        if state["progress"] >= 1.0:
            state.update(
                {
                    "mode": "berthed",
                    "berth": state["destination"],
                    "origin": state["destination"],
                    "progress": 0.0,
                }
            )


def _vessel_berths_from_states(vessel_states: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        vessel_id: str(state["berth"])
        for vessel_id, state in vessel_states.items()
        if state["mode"] == "berthed" and state.get("berth")
    }


def _vessel_positions_from_states(
    vessel_states: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, object]]:
    positions: dict[str, dict[str, object]] = {}
    for vessel_id, state in vessel_states.items():
        route = routes[vessel_id]
        if state["mode"] == "berthed":
            berth = str(state["berth"])
            lat, lon = _location_tuple(berth)
            if berth == route["destination"]:
                leg = "unloading_at_terminal"
                berth_id = f"{berth}_unloading_berth_1"
                berth_label = f"Unloading berth 1 at {berth}"
            else:
                leg = "loading_at_origin"
                berth_id = f"{berth}_loading_berth"
                berth_label = f"Loading berth at {berth}"
            leg_label = f"Berthed at {berth}"
            at_berth = True
        else:
            origin = str(state["origin"])
            destination = str(state["destination"])
            coordinates = _route_coordinates_for_leg(route, origin, destination)
            leg = "outbound_to_terminal" if destination == route["destination"] else "return_to_origin"
            leg_label = "Outbound voyage to terminal" if leg == "outbound_to_terminal" else "Return voyage to emitter"
            lat, lon = _interpolate_route(coordinates, float(state["progress"]))
            at_berth = False
            berth_id = None
            berth_label = None
        positions[vessel_id] = {
            "lat": lat,
            "lon": lon,
            "route_id": vessel_id,
            "leg": leg,
            "leg_label": leg_label,
            "at_berth": at_berth,
            "berth_id": berth_id,
            "berth_label": berth_label,
        }
    return positions


def _route_coordinates_for_leg(route: dict[str, Any], origin: str, destination: str) -> list[Coordinate]:
    if origin == route["origin"] and destination == route["destination"]:
        return route["coordinates"]
    if origin == route["destination"] and destination == route["origin"]:
        return route["return_coordinates"]
    return [_location_tuple(origin), _location_tuple(destination)]


def _interpolate_route(coordinates: list[Coordinate], progress: float) -> Coordinate:
    if len(coordinates) == 1:
        return coordinates[0]
    distances = [
        route_distance_km([a, b])
        for a, b in zip(coordinates, coordinates[1:])
    ]
    total = sum(distances) or 1.0
    target = max(0.0, min(1.0, progress)) * total
    covered = 0.0
    for index, segment in enumerate(distances):
        if covered + segment >= target:
            local = (target - covered) / (segment or 1.0)
            lat_a, lon_a = coordinates[index]
            lat_b, lon_b = coordinates[index + 1]
            return (lat_a + (lat_b - lat_a) * local, lon_a + (lon_b - lon_a) * local)
        covered += segment
    return coordinates[-1]


def _map_bbox(
    routes: dict[str, dict[str, Any]],
    network: Any,
    locations: dict[str, dict[str, float | str]],
) -> dict[str, float]:
    coordinates: list[Coordinate] = []
    for location in locations.values():
        coordinates.append((float(location["lat"]), float(location["lon"])))
    for route in routes.values():
        coordinates.extend(route["coordinates"])
        coordinates.extend(route["return_coordinates"])
        for dynamic_leg in route.get("dynamic_leg_routes", {}).values():
            coordinates.extend(dynamic_leg["coordinates"])
    for segment in _build_pipeline_segments(network, locations):
        coordinates.extend(segment["coordinates"])
    for link in _build_injection_links(network, locations):
        coordinates.extend(link["coordinates"])
    lats = [lat for lat, _ in coordinates]
    lons = [lon for _, lon in coordinates]
    return {
        "min_lat": min(lats) - 0.7,
        "max_lat": max(lats) + 0.7,
        "min_lon": min(lons) - 0.9,
        "max_lon": max(lons) + 0.9,
    }


def _location_tuple(location_id: str, locations: dict[str, dict[str, float | str]] = LOCATIONS) -> Coordinate:
    location = locations[location_id]
    return (float(location["lat"]), float(location["lon"]))
