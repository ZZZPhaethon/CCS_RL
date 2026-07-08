from __future__ import annotations

from pathlib import Path

from sim.control.rule_based import RuleBasedActionGenerator
from sim.visualization import (
    write_northern_lights_phase1_dashboard,
    write_northern_lights_phase2_dashboard,
)


def rule_based_factory(network, routes):
    return RuleBasedActionGenerator(network, routes)


def main() -> None:
    outputs = [
        write_northern_lights_phase1_dashboard(
            Path("docs") / "physical_layer_phase1_rule_based_dashboard.html",
            hours=24 * 30,
            action_generator_factory=rule_based_factory,
        ),
        write_northern_lights_phase2_dashboard(
            Path("docs") / "physical_layer_phase2_rule_based_dashboard.html",
            hours=240,
            action_generator_factory=rule_based_factory,
        ),
    ]
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
