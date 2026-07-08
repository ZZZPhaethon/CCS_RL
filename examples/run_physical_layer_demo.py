from __future__ import annotations

import json

from sim.routes import sea_route
from sim.network_scenarios import build_northern_lights_phase1_demo


def main() -> None:
    network, state = build_northern_lights_phase1_demo()
    state.vessel_berths["northern_pioneer"] = "brevik"

    result = network.step(
        state,
        actions={
            "brevik": {"load_vessel": "northern_pioneer"},
        },
    )
    route = sea_route((59.05, 9.70), (60.62, 4.84))

    print(json.dumps(network.snapshot(result.state), indent=2))
    print(
        json.dumps(
            {
                "route_provider": route.provider,
                "route_distance_km": round(route.distance_km, 2),
                "route_points": route.coordinates,
                "step": result.as_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
