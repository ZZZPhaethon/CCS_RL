from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Any

from ..entities.emitter import Emitter
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..network_scenarios import (
    NORTHERN_LIGHTS_PHASE2_DATA_PATH,
    build_northern_lights_phase2_demo,
)
from ..routes import route_distance_km, sea_route
from .core import (
    _build_injection_links,
    _build_pipeline_segments,
    _connect_route_to_facilities,
    _locations_from_phase2_data,
    _map_bbox,
    _interpolate_route,
)


E1_VESSEL_COLORS = ("#f4b942", "#64c7c4", "#7f9cf5")
E1_EMITTER_COLORS = ("#e9a23b", "#72c7a0", "#7ca7df")


def build_e1_cinematic_payload(
    trace_csv: str | Path,
    *,
    title: str = "Iterative Action-Q · Northern Lights operations",
) -> dict[str, Any]:
    """Adapt an E1 hourly trace CSV to the cinematic dashboard contract."""
    source_path = Path(trace_csv)
    rows = _read_trace_rows(source_path)
    controller = str(rows[0]["controller"])
    test_seed = int(rows[0]["test_seed"])
    hours = [float(row["hour"]) for row in rows]
    _validate_hours(hours)

    network, _state = build_northern_lights_phase2_demo()
    with NORTHERN_LIGHTS_PHASE2_DATA_PATH.open(encoding="utf-8") as handle:
        import json

        locations = _locations_from_phase2_data(json.load(handle))

    emitter_ids = _entity_ids_from_suffix(
        rows[0],
        "_capture_availability",
    )
    vessel_ids = _entity_ids_from_suffix(
        rows[0],
        "_operational_state",
    )
    well_ids = _entity_ids_from_suffix(
        rows[0],
        "_available",
    )
    terminal_ids = [
        entity_id
        for entity_id, entity in network.entities.items()
        if isinstance(entity, Terminal)
        and f"{entity_id}_inventory_t" in rows[0]
    ]
    if len(terminal_ids) != 1:
        raise ValueError(
            "E1 cinematic traces require exactly one terminal inventory column; "
            f"found {terminal_ids}."
        )
    terminal_id = terminal_ids[0]

    _validate_trace_entities(network, emitter_ids, vessel_ids, terminal_id)
    service_routes = _build_service_routes(
        [*emitter_ids, terminal_id],
        locations,
    )
    vessel_positions = _build_vessel_positions(
        rows,
        vessel_ids,
        terminal_id,
        locations,
        service_routes,
    )

    emitter_metadata = [
        {
            "id": emitter_id,
            "label": str(locations[emitter_id]["label"]),
            "capacity_t": float(network.entities[emitter_id].buffer_capacity_t),
            "color": E1_EMITTER_COLORS[index % len(E1_EMITTER_COLORS)],
            "lat": float(locations[emitter_id]["lat"]),
            "lon": float(locations[emitter_id]["lon"]),
        }
        for index, emitter_id in enumerate(emitter_ids)
    ]
    vessel_metadata = [
        {
            "id": vessel_id,
            "label": _friendly_vessel_name(vessel_id),
            "capacity_t": float(network.entities[vessel_id].capacity_t),
            "color": E1_VESSEL_COLORS[index % len(E1_VESSEL_COLORS)],
        }
        for index, vessel_id in enumerate(vessel_ids)
    ]
    terminal_entity = network.entities[terminal_id]
    terminal_metadata = {
        "id": terminal_id,
        "label": str(locations[terminal_id]["label"]),
        "capacity_t": float(terminal_entity.storage_capacity_t),
        "lat": float(locations[terminal_id]["lat"]),
        "lon": float(locations[terminal_id]["lon"]),
    }
    transport_storage_components = [
        {
            **terminal_metadata,
            "type": "terminal",
            "short_label": "Terminal",
        },
        *[
            {
                "id": component_id,
                "label": str(locations[component_id]["label"]),
                "short_label": short_label,
                "type": component_type,
                "lat": float(locations[component_id]["lat"]),
                "lon": float(locations[component_id]["lon"]),
            }
            for component_id, component_type, short_label in [
                ("oygarden_pipeline", "pipeline", "Pipeline"),
                ("aurora_subsea_manifold", "manifold", "Manifold"),
            ]
        ],
        *[
            {
                "id": well_id,
                "label": str(locations[well_id]["label"]),
                "short_label": "Well A7",
                "type": "well",
                "lat": float(locations[well_id]["lat"]),
                "lon": float(locations[well_id]["lon"]),
            }
            for well_id in well_ids
        ],
        {
            "id": "aurora_reservoir",
            "label": str(locations["aurora_reservoir"]["label"]),
            "short_label": "Reservoir",
            "type": "reservoir",
            "lat": float(locations["aurora_reservoir"]["lat"]),
            "lon": float(locations["aurora_reservoir"]["lon"]),
        },
    ]

    frames = _build_frames(
        rows,
        emitter_metadata,
        vessel_metadata,
        terminal_metadata,
        vessel_positions,
    )
    events = _build_events(frames, emitter_ids, vessel_ids)

    relevant_locations = {
        entity_id: location
        for entity_id, location in locations.items()
        if entity_id in {
            *emitter_ids,
            terminal_id,
            "oygarden_pipeline",
            "aurora_subsea_manifold",
            "aurora_well_a7_ah",
            "aurora_well_c1_h",
            "aurora_reservoir",
        }
    }
    route_bbox_payload = {
        route_id: {
            "coordinates": route["coordinates"],
            "return_coordinates": list(reversed(route["coordinates"])),
        }
        for route_id, route in service_routes.items()
    }

    return {
        "title": title,
        "subtitle": (
            f"E1 representative trajectory · {controller} · "
            f"test seed {test_seed}"
        ),
        "controller": controller,
        "test_seed": test_seed,
        "source_trace": source_path.as_posix(),
        "duration_hours": hours[-1],
        "time_step_hours": hours[1] - hours[0] if len(hours) > 1 else 1.0,
        "emitters": emitter_metadata,
        "vessels": vessel_metadata,
        "terminal": terminal_metadata,
        "components": {
            "capture_sites": emitter_metadata,
            "fleet": vessel_metadata,
            "transport_storage": transport_storage_components,
        },
        "map": {
            "locations": relevant_locations,
            "service_routes": service_routes,
            "pipeline_segments": _build_pipeline_segments(network, locations),
            "injection_links": _build_injection_links(network, locations),
            "bbox": _map_bbox(route_bbox_payload, network, relevant_locations),
        },
        "frames": frames,
        "events": events,
    }


def _read_trace_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"E1 hourly trace does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("E1 hourly trace must contain at least two frames.")

    controllers = {row.get("controller", "") for row in rows}
    seeds = {row.get("test_seed", "") for row in rows}
    if len(controllers) != 1 or "" in controllers:
        raise ValueError(
            "E1 cinematic input must contain exactly one non-empty controller."
        )
    if len(seeds) != 1 or "" in seeds:
        raise ValueError(
            "E1 cinematic input must contain exactly one non-empty test seed."
        )
    return rows


def _validate_hours(hours: list[float]) -> None:
    if hours[0] != 0.0:
        raise ValueError(f"E1 cinematic trace must begin at hour 0, got {hours[0]}.")
    for previous, current in zip(hours, hours[1:]):
        if current <= previous:
            raise ValueError("E1 cinematic trace hours must be strictly increasing.")
        if not math.isclose(current - previous, 1.0, abs_tol=1e-9):
            raise ValueError(
                "E1 cinematic trace currently requires contiguous one-hour frames."
            )


def _entity_ids_from_suffix(
    row: dict[str, str],
    suffix: str,
) -> list[str]:
    ids = [
        column[: -len(suffix)]
        for column in row
        if column.endswith(suffix)
    ]
    if not ids:
        raise ValueError(f"E1 trace has no columns ending in {suffix!r}.")
    return ids


def _validate_trace_entities(
    network: Any,
    emitter_ids: list[str],
    vessel_ids: list[str],
    terminal_id: str,
) -> None:
    for emitter_id in emitter_ids:
        if not isinstance(network.entities.get(emitter_id), Emitter):
            raise ValueError(f"Unknown Phase 2 emitter in trace: {emitter_id}")
    for vessel_id in vessel_ids:
        if not isinstance(network.entities.get(vessel_id), Vessel):
            raise ValueError(f"Unknown Phase 2 vessel in trace: {vessel_id}")
    if not isinstance(network.entities.get(terminal_id), Terminal):
        raise ValueError(f"Unknown Phase 2 terminal in trace: {terminal_id}")


def _build_service_routes(
    facility_ids: list[str],
    locations: dict[str, dict[str, float | str]],
) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for origin_id, destination_id in itertools.combinations(facility_ids, 2):
        origin = _coordinate(locations[origin_id])
        destination = _coordinate(locations[destination_id])
        route = sea_route(origin, destination)
        if route.provider != "searoute":
            raise RuntimeError(
                "Cinematic route generation requires searoute. "
                "Install it with `python -m pip install searoute`."
            )
        coordinates = _connect_route_to_facilities(
            route.coordinates,
            origin,
            destination,
        )
        route_id = _leg_id(origin_id, destination_id)
        routes[route_id] = {
            "id": route_id,
            "label": (
                f"{locations[origin_id]['label']} → "
                f"{locations[destination_id]['label']}"
            ),
            "origin": origin_id,
            "destination": destination_id,
            "provider": route.provider,
            "distance_km": round(route_distance_km(coordinates), 2),
            "coordinates": coordinates,
        }
    return routes


def _build_vessel_positions(
    rows: list[dict[str, str]],
    vessel_ids: list[str],
    terminal_id: str,
    locations: dict[str, dict[str, float | str]],
    service_routes: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    positions_by_vessel: dict[str, list[dict[str, Any]]] = {}
    for vessel_id in vessel_ids:
        positions: list[dict[str, Any] | None] = [None] * len(rows)
        last_berth = str(rows[0][f"{vessel_id}_destination"])
        index = 0
        while index < len(rows):
            state = str(rows[index][f"{vessel_id}_operational_state"])
            destination = str(rows[index][f"{vessel_id}_destination"])
            if not state.startswith("sailing"):
                if destination not in locations:
                    raise ValueError(
                        f"Trace destination {destination!r} has no map location."
                    )
                lat, lon = _coordinate(locations[destination])
                positions[index] = {
                    "lat": lat,
                    "lon": lon,
                    "bearing_deg": 0.0,
                    "origin": destination,
                    "destination": destination,
                    "progress": 1.0,
                }
                last_berth = destination
                index += 1
                continue

            run_start = index
            run_destination = destination
            while (
                index < len(rows)
                and str(
                    rows[index][f"{vessel_id}_operational_state"]
                ).startswith("sailing")
                and str(
                    rows[index][f"{vessel_id}_destination"]
                ) == run_destination
            ):
                index += 1
            run_end = index
            coordinates = _coordinates_for_leg(
                last_berth,
                run_destination,
                terminal_id,
                service_routes,
            )
            run_length = run_end - run_start
            for offset, frame_index in enumerate(range(run_start, run_end)):
                progress = (offset + 1) / (run_length + 1)
                lat, lon = _interpolate_route(coordinates, progress)
                next_lat, next_lon = _interpolate_route(
                    coordinates,
                    min(1.0, progress + 0.01),
                )
                positions[frame_index] = {
                    "lat": lat,
                    "lon": lon,
                    "bearing_deg": _bearing_degrees(
                        (lat, lon),
                        (next_lat, next_lon),
                    ),
                    "origin": last_berth,
                    "destination": run_destination,
                    "progress": progress,
                }
            last_berth = run_destination

        if any(position is None for position in positions):
            raise RuntimeError(f"Failed to position every frame for {vessel_id}.")
        positions_by_vessel[vessel_id] = [
            dict(position) for position in positions if position is not None
        ]
    return positions_by_vessel


def _coordinates_for_leg(
    origin: str,
    destination: str,
    terminal_id: str,
    service_routes: dict[str, dict[str, Any]],
) -> list[tuple[float, float]]:
    if origin == destination:
        raise ValueError(
            f"Unexpected sailing leg with identical endpoints: {origin!r}."
        )
    direct_id = _leg_id(origin, destination)
    reverse_id = _leg_id(destination, origin)
    if direct_id in service_routes:
        return service_routes[direct_id]["coordinates"]
    if reverse_id in service_routes:
        return list(reversed(service_routes[reverse_id]["coordinates"]))
    raise ValueError(
        f"Unsupported E1 vessel leg {origin!r} → {destination!r}; "
        f"known service routes are {sorted(service_routes)}."
    )


def _build_frames(
    rows: list[dict[str, str]],
    emitters: list[dict[str, Any]],
    vessels: list[dict[str, Any]],
    terminal: dict[str, Any],
    positions: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    previous_vent = 0.0
    for index, row in enumerate(rows):
        cumulative_vent = float(row["cumulative_vent_t"])
        vent_rate = max(0.0, cumulative_vent - previous_vent)
        previous_vent = cumulative_vent

        emitter_states: dict[str, dict[str, Any]] = {}
        for emitter in emitters:
            emitter_id = str(emitter["id"])
            inventory = float(row[f"{emitter_id}_inventory_t"])
            emitter_states[emitter_id] = {
                "inventory_t": inventory,
                "fill_fraction": _safe_fraction(
                    inventory,
                    float(emitter["capacity_t"]),
                ),
                "capture_availability": float(
                    row[f"{emitter_id}_capture_availability"]
                ),
                "capture_outage": bool(
                    int(row[f"{emitter_id}_capture_outage"])
                ),
                "capture_high_output": bool(
                    int(row[f"{emitter_id}_capture_high_output"])
                ),
            }

        vessel_states: dict[str, dict[str, Any]] = {}
        for vessel in vessels:
            vessel_id = str(vessel["id"])
            inventory = float(row[f"{vessel_id}_inventory_t"])
            vessel_states[vessel_id] = {
                **positions[vessel_id][index],
                "inventory_t": inventory,
                "fill_fraction": _safe_fraction(
                    inventory,
                    float(vessel["capacity_t"]),
                ),
                "mode": str(row[f"{vessel_id}_mode"]),
                "operational_state": str(
                    row[f"{vessel_id}_operational_state"]
                ),
            }

        terminal_inventory = float(row[f"{terminal['id']}_inventory_t"])
        frames.append(
            {
                "hour": float(row["hour"]),
                "cumulative_vent_t": cumulative_vent,
                "vent_rate_tph": vent_rate,
                "weather_speed_factor": float(
                    row["weather_speed_factor"]
                ),
                "terminal_inventory_t": terminal_inventory,
                "terminal_fill_fraction": _safe_fraction(
                    terminal_inventory,
                    float(terminal["capacity_t"]),
                ),
                "total_emitter_inventory_t": sum(
                    state["inventory_t"]
                    for state in emitter_states.values()
                ),
                "total_vessel_inventory_t": sum(
                    state["inventory_t"]
                    for state in vessel_states.values()
                ),
                "emitters": emitter_states,
                "vessels": vessel_states,
                "well_available": {
                    column[: -len("_available")]: bool(int(value))
                    for column, value in row.items()
                    if column.endswith("_available")
                },
            }
        )
    return frames


def _build_events(
    frames: list[dict[str, Any]],
    emitter_ids: list[str],
    vessel_ids: list[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous = frames[0]
    for frame in frames[1:]:
        hour = float(frame["hour"])
        if frame["vent_rate_tph"] > 1e-9:
            events.append(
                {
                    "hour": hour,
                    "type": "vent",
                    "entity_id": "system",
                    "label": f"Venting +{frame['vent_rate_tph']:.0f} t",
                }
            )
        for vessel_id in vessel_ids:
            current_state = frame["vessels"][vessel_id]["operational_state"]
            previous_state = previous["vessels"][vessel_id][
                "operational_state"
            ]
            destination = frame["vessels"][vessel_id]["destination"]
            if current_state != previous_state:
                events.append(
                    {
                        "hour": hour,
                        "type": "vessel",
                        "entity_id": vessel_id,
                        "label": (
                            f"{_friendly_vessel_name(vessel_id)} · "
                            f"{_friendly_state(current_state)} · "
                            f"{_friendly_location(destination)}"
                        ),
                    }
                )
        for emitter_id in emitter_ids:
            current = frame["emitters"][emitter_id]
            old = previous["emitters"][emitter_id]
            if current["capture_outage"] and not old["capture_outage"]:
                events.append(
                    {
                        "hour": hour,
                        "type": "outage",
                        "entity_id": emitter_id,
                        "label": (
                            f"{_friendly_location(emitter_id)} capture outage"
                        ),
                    }
                )
            if current["capture_high_output"] and not old["capture_high_output"]:
                events.append(
                    {
                        "hour": hour,
                        "type": "capture",
                        "entity_id": emitter_id,
                        "label": (
                            f"{_friendly_location(emitter_id)} high output"
                        ),
                    }
                )
        previous = frame
    return events


def _coordinate(
    location: dict[str, float | str],
) -> tuple[float, float]:
    return float(location["lat"]), float(location["lon"])


def _bearing_degrees(
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    lat_a, lon_a = map(math.radians, start)
    lat_b, lon_b = map(math.radians, end)
    delta_lon = lon_b - lon_a
    x_value = math.sin(delta_lon) * math.cos(lat_b)
    y_value = (
        math.cos(lat_a) * math.sin(lat_b)
        - math.sin(lat_a) * math.cos(lat_b) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(x_value, y_value)) + 360.0) % 360.0


def _safe_fraction(value: float, capacity: float) -> float:
    if capacity <= 0.0:
        return 0.0
    return max(0.0, min(1.0, value / capacity))


def _friendly_vessel_name(entity_id: str) -> str:
    return entity_id.replace("_", " ").title()


def _friendly_location(entity_id: str) -> str:
    names = {
        "brevik": "Brevik",
        "celsio": "Celsio",
        "yara_sluiskil": "Yara Sluiskil",
        "oygarden_terminal": "Øygarden",
    }
    return names.get(entity_id, entity_id.replace("_", " ").title())


def _friendly_state(state: str) -> str:
    return state.replace("_", " ").replace("to", "→").title()


def _leg_id(origin: str, destination: str) -> str:
    return f"{origin}__{destination}"
