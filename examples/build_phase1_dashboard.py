from __future__ import annotations

from pathlib import Path

from sim.visualization import write_northern_lights_phase1_dashboard

DASHBOARD_HOURS = 24 * 30


def main() -> None:
    output = write_northern_lights_phase1_dashboard(
        Path("docs") / "physical_layer_phase1_dashboard.html",
        hours=DASHBOARD_HOURS,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
