from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import (
    build_demo_trajectory,
    build_northern_lights_phase1_trajectory,
    build_northern_lights_phase2_trajectory,
)
from .html import render_dashboard_html
def write_dashboard(
    path: str | Path,
    hours: int = 48,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_dashboard_html(build_demo_trajectory(hours, action_frames, action_generator_factory)),
        encoding="utf-8",
    )
    return output_path


def write_northern_lights_phase2_dashboard(
    path: str | Path,
    hours: int = 240,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_dashboard_html(build_northern_lights_phase2_trajectory(hours, action_frames, action_generator_factory)),
        encoding="utf-8",
    )
    return output_path


def write_northern_lights_phase1_dashboard(
    path: str | Path,
    hours: int = 72,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_dashboard_html(build_northern_lights_phase1_trajectory(hours, action_frames, action_generator_factory)),
        encoding="utf-8",
    )
    return output_path


