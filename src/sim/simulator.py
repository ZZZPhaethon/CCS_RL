from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .actions import ActionFrame, ActionResolver, CommittedActionFrame
from .scenario_generation.disturbance_resolver import leg_speed_factor, vessel_speed_factor
from .entities.state import PhysicalState, StepResult
from .network import PhysicalNetwork
from .routes import route_distance_km, sea_route

Coordinate = tuple[float, float]


@dataclass
class SimulationStepRecord:
    action_frame: ActionFrame
    committed_action_frame: CommittedActionFrame
    step_result: StepResult
    observation: dict[str, object]
    vessel_positions: dict[str, dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return {
            "time_h": self.step_result.state.time_h,
            "proposed": asdict(self.action_frame),
            "committed": asdict(self.committed_action_frame),
            "executed": self.step_result.as_dict(),
            "observation": self.observation,
            "vessel_positions": self.vessel_positions,
        }


class PhysicalSimulator:
    """Step-level wrapper for action frames, vessel movement, and execution logs."""

    def __init__(
        self,
        network: PhysicalNetwork,
        state: PhysicalState,
        routes: dict[str, dict[str, Any]] | None = None,
        locations: dict[str, Coordinate] | None = None,
    ) -> None:
        self.network = network
        self.state = state.copy()
        self.routes = routes or {}
        self.locations = locations or {}
        self.resolver = ActionResolver(network)
        self.vessel_states = self._initial_vessel_states()
        self.state.vessel_berths = self._vessel_berths_from_states()

    def step(self, action_frame: ActionFrame, compute_observation: bool = True) -> SimulationStepRecord:
        committed = self.resolver.resolve(action_frame)
        self._start_vessel_voyages(committed.actions)
        self.state.vessel_berths = self._vessel_berths_from_states()
        result = self.network.step(self.state, self._physical_actions(committed.actions))
        self.state = result.state
        self._advance_vessel_voyages(self.network.time_step_hours)
        self.state.vessel_berths = self._vessel_berths_from_states()
        # The full network snapshot recomputes the line-source bottomhole-pressure
        # model (CoolProp + O(t^2) rate history), which is needed for dashboards but
        # not for RL, whose observation is built separately. Skipping it in the
        # training loop speeds each step up by ~10-50x.
        observation = self.network.snapshot(self.state) if compute_observation else {}
        return SimulationStepRecord(
            action_frame=action_frame,
            committed_action_frame=committed,
            step_result=result,
            observation=observation,
            vessel_positions=self.vessel_positions(),
        )

    def vessel_positions(self) -> dict[str, dict[str, object]]:
        positions: dict[str, dict[str, object]] = {}
        for vessel_id, state in self.vessel_states.items():
            route = self.routes[vessel_id]
            if state["mode"] == "berthed":
                berth = str(state["berth"])
                lat, lon = self._location_tuple(berth, route)
                if berth == route["destination"]:
                    leg = "unloading_at_terminal"
                    berth_id = f"{berth}_unloading_berth_1"
                    berth_label = f"Unloading berth 1 at {berth}"
                else:
                    leg = "loading_at_origin"
                    berth_id = f"{berth}_loading_berth"
                    berth_label = f"Loading berth at {berth}"
                positions[vessel_id] = {
                    "lat": lat,
                    "lon": lon,
                    "route_id": vessel_id,
                    "leg": leg,
                    "leg_label": f"Berthed at {berth}",
                    "at_berth": True,
                    "berth_id": berth_id,
                    "berth_label": berth_label,
                }
                continue
            origin = str(state["origin"])
            destination = str(state["destination"])
            coordinates = self._route_coordinates_for_leg(route, origin, destination)
            leg = "outbound_to_terminal" if destination == route["destination"] else "return_to_origin"
            leg_label = "Outbound voyage to terminal" if leg == "outbound_to_terminal" else "Return voyage to emitter"
            lat, lon = self._interpolate_route(coordinates, float(state["progress"]))
            positions[vessel_id] = {
                "lat": lat,
                "lon": lon,
                "route_id": vessel_id,
                "leg": leg,
                "leg_label": leg_label,
                "at_berth": False,
                "berth_id": None,
                "berth_label": None,
            }
        return positions

    def _initial_vessel_states(self) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for vessel_id, route in self.routes.items():
            berth = self.state.vessel_berths.get(vessel_id, route["origin"])
            states[vessel_id] = {
                "mode": "berthed",
                "berth": berth,
                "origin": berth,
                "destination": berth,
                "progress": 0.0,
            }
        return states

    def _start_vessel_voyages(self, actions: dict[str, dict[str, Any]]) -> None:
        for vessel_id, route in self.routes.items():
            action = actions.get(vessel_id, {})
            destination = action.get("sail_to")
            state = self.vessel_states[vessel_id]
            if not destination or state["mode"] != "berthed" or destination == state["berth"]:
                continue
            if not self._known_destination(str(destination), route):
                continue
            origin = str(state["berth"])
            destination = str(destination)
            coordinates = self._route_coordinates_for_leg(route, origin, destination)
            distance_km = (
                float(route.get("distance_km") or 0.0)
                if {origin, destination} == {route["origin"], route["destination"]}
                else route_distance_km(coordinates)
            )
            state.update(
                {
                    "mode": "sailing",
                    "origin": origin,
                    "destination": destination,
                    "berth": None,
                    "progress": 0.0,
                    "distance_km": distance_km,
                }
            )

    def _advance_vessel_voyages(self, hours: float) -> None:
        for vessel_id, state in self.vessel_states.items():
            if state["mode"] != "sailing":
                continue
            route = self.routes[vessel_id]
            speed_knots = route.get("speed_knots")
            distance_km = float(state.get("distance_km") or route.get("distance_km") or 0.0)
            vessel_factor = vessel_speed_factor(self.state, vessel_id)
            speed_factor = leg_speed_factor(
                self.state,
                str(state["origin"]),
                str(state["destination"]),
                fallback=vessel_factor,
            )
            effective_speed_knots = float(speed_knots) * speed_factor if speed_knots else 0.0
            if effective_speed_knots <= 0.0 or distance_km <= 0.0:
                # No nominal speed/distance falls back to instant arrival; a
                # weather factor of 0 instead stalls the vessel in place.
                state["progress"] = state["progress"] if speed_knots and speed_factor <= 0.0 else 1.0
            else:
                distance_covered_km = effective_speed_knots * 1.852 * hours
                state["progress"] = min(1.0, state["progress"] + distance_covered_km / distance_km)
            if state["progress"] >= 1.0:
                state.update(
                    {
                        "mode": "berthed",
                        "berth": state["destination"],
                        "origin": state["destination"],
                        "progress": 0.0,
                        "distance_km": 0.0,
                    }
                )

    def _known_destination(self, destination: str, route: dict[str, Any]) -> bool:
        return destination in self.locations or destination in {route["origin"], route["destination"]}

    def _vessel_berths_from_states(self) -> dict[str, str]:
        return {
            vessel_id: str(state["berth"])
            for vessel_id, state in self.vessel_states.items()
            if state["mode"] == "berthed" and state.get("berth")
        }

    def _physical_actions(self, actions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        physical_actions: dict[str, dict[str, Any]] = {}
        for entity_id, action in actions.items():
            translated = {key: value for key, value in action.items() if key != "sail_to"}
            if translated:
                physical_actions[entity_id] = translated
        return physical_actions

    def _route_coordinates_for_leg(self, route: dict[str, Any], origin: str, destination: str) -> list[Coordinate]:
        if origin == route["origin"] and destination == route["destination"]:
            return route["coordinates"]
        if origin == route["destination"] and destination == route["origin"]:
            return route["return_coordinates"]
        return self._dynamic_leg_route(route, origin, destination)["coordinates"]

    def _dynamic_leg_route(self, route: dict[str, Any], origin: str, destination: str) -> dict[str, Any]:
        leg_routes = route.setdefault("dynamic_leg_routes", {})
        leg_id = f"{origin}->{destination}"
        if leg_id not in leg_routes:
            origin_coordinate = self._location_tuple(origin, route)
            destination_coordinate = self._location_tuple(destination, route)
            maritime_route = sea_route(origin_coordinate, destination_coordinate)
            coordinates = self._connect_route_to_endpoints(
                maritime_route.coordinates,
                origin_coordinate,
                destination_coordinate,
            )
            leg_routes[leg_id] = {
                "id": leg_id,
                "origin": origin,
                "destination": destination,
                "provider": maritime_route.provider,
                "distance_km": round(route_distance_km(coordinates), 2),
                "coordinates": coordinates,
            }
        return leg_routes[leg_id]

    @staticmethod
    def _connect_route_to_endpoints(
        coordinates: list[Coordinate],
        origin: Coordinate,
        destination: Coordinate,
    ) -> list[Coordinate]:
        connected = list(coordinates)
        if not connected:
            return [origin, destination]
        if connected[0] != origin:
            connected.insert(0, origin)
        if connected[-1] != destination:
            connected.append(destination)
        return connected

    def _location_tuple(self, location_id: str, route: dict[str, Any]) -> Coordinate:
        if location_id in self.locations:
            return self.locations[location_id]
        if location_id == route["origin"]:
            return tuple(route["coordinates"][0])  # type: ignore[return-value]
        if location_id == route["destination"]:
            return tuple(route["coordinates"][-1])  # type: ignore[return-value]
        raise KeyError(f"Unknown location: {location_id}")

    def _interpolate_route(self, coordinates: list[Coordinate], progress: float) -> Coordinate:
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
