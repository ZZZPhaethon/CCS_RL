from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from experiments import evaluate_iterative_action_q as iterative_q_eval
from sim.environment.vessel_mode import vessel_operation_modes
from sim.scenario_generation import generator as scenario_generator


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments_results" / "E1" / "figures"
TEST_SEED = 9000056
CHECKPOINT = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "models"
    / "iterative_q"
    / "g60_p4_model_seed_0"
    / "iterative_action_q.pt"
)
ITERATIVE_Q_RESULTS = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "algorithms"
    / "formal_iterative_action_q_g60_p4_seeds_9000031-9000060_run01"
    / "model_seed_0"
    / "evaluation.csv"
)
POLICY_WINDOWS = (
    "108-155,156-203,204-251,252-299,300-347,348-395,"
    "396-443,444-491,492-539,540-587,588-635,636-680"
)

EMITTER_COLORS = ("#0F4D92", "#D5892F", "#2E8B8B")
VESSEL_COLORS = ("#596A9E", "#D58CA3", "#42949E")
TERMINAL_COLOR = "#7C6CCF"
VENT_COLOR = "#B64342"
DISTURBANCE_COLORS = {
    "weather": "#4C78A8",
    "capture_outage": "#F28E2B",
    "capture_high_output": "#59A14F",
    "well": "#8F63B8",
}
OPERATIONAL_STATES = (
    "idle",
    "queued",
    "loading",
    "sailing_to_terminal",
    "unloading",
    "sailing_to_emitter",
)
OPERATIONAL_LABELS = {
    "idle": "Idle",
    "queued": "Queued",
    "loading": "Loading",
    "sailing_to_terminal": "Sailing → terminal",
    "unloading": "Unloading",
    "sailing_to_emitter": "Sailing → emitter",
}
OPERATIONAL_COLORS = (
    "#E8E8E8",
    "#BDBDBD",
    "#7884B4",
    "#42949E",
    "#D58CA3",
    "#E6A15C",
)
EMITTER_DISPLAY_NAMES = {
    "brevik": "Brevik",
    "celsio": "Celsio",
    "yara_sluiskil": "Yara Sluiskil",
}
TERMINAL_DISPLAY_NAMES = {
    "oygarden_terminal": "Øygarden terminal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the restructured E1 Figure 4 from a trace-only replay of "
            "Iterative Action-Q model seed 0 on test seed 9000056."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--iterative-q-results",
        type=Path,
        default=ITERATIVE_Q_RESULTS,
    )
    return parser.parse_args()


def evaluation_args(checkpoint: Path) -> argparse.Namespace:
    return iterative_q_eval.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            "unused-figure-4-trace",
            "--eval-seeds",
            str(TEST_SEED),
            "--episode-hours",
            "720",
            "--reward-scale",
            "0.00001",
            "--gates",
            f"figure4_trace:4:0.40:12:{POLICY_WINDOWS}",
            "--scenario-protocol",
            "unified_window_v1",
            "--hard-scenario-probability",
            "0.5",
            "--forecast-context-hours",
            "168",
            "--device",
            "cpu",
        ]
    )


def load_formal_result(iterative_q_results: Path) -> dict[str, str]:
    with iterative_q_results.open("r", encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if int(row["seed"]) == TEST_SEED
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one formal Iterative Action-Q row for seed {TEST_SEED}, "
            f"found {len(matches)}."
        )
    return matches[0]


def friendly_emitter_name(entity_id: str) -> str:
    return EMITTER_DISPLAY_NAMES.get(
        entity_id,
        entity_id.replace("_", " ").title(),
    )


def friendly_terminal_name(entity_id: str) -> str:
    return TERMINAL_DISPLAY_NAMES.get(
        entity_id,
        entity_id.replace("_", " ").title(),
    )


class TraceRecorder:
    def __init__(
        self,
        capture_high_output_factor: dict[str, list[float]],
    ) -> None:
        self.frames: list[dict[str, object]] = []
        self.emitter_ids: tuple[str, ...] = ()
        self.vessel_ids: tuple[str, ...] = ()
        self.terminal_ids: tuple[str, ...] = ()
        self.well_ids: tuple[str, ...] = ()
        self.capacities_t: dict[str, float] = {}
        self.capture_high_output_factor = capture_high_output_factor

    def record(self, env) -> None:
        if env.simulator is None:
            raise RuntimeError("Cannot record a trace before environment reset.")
        if not self.frames:
            self.emitter_ids = tuple(env.emitter_ids)
            self.vessel_ids = tuple(env.vessel_ids)
            self.terminal_ids = tuple(env.terminal_ids)
            self.well_ids = tuple(env.well_ids)
            for entity_id in (
                *self.emitter_ids,
                *self.vessel_ids,
                *self.terminal_ids,
            ):
                entity = env.network.entities[entity_id]
                capacity = getattr(
                    entity,
                    "buffer_capacity_t",
                    getattr(
                        entity,
                        "capacity_t",
                        getattr(entity, "storage_capacity_t", None),
                    ),
                )
                if capacity is None:
                    raise ValueError(f"No inventory capacity for {entity_id}.")
                self.capacities_t[entity_id] = float(capacity)

        state = env.simulator.state
        hour_index = int(round(float(state.time_h)))
        frame: dict[str, object] = {
            "controller": "iterative_action_q_g60_p4_model_seed_0",
            "test_seed": TEST_SEED,
            "hour": float(state.time_h),
            "cumulative_vent_t": float(env.ledger.vented_t),
            "weather_speed_factor": min(
                (
                    float(state.vessel_speed_factor.get(vessel_id, 1.0))
                    for vessel_id in self.vessel_ids
                ),
                default=1.0,
            ),
        }
        for emitter_id in self.emitter_ids:
            high_output_series = self.capture_high_output_factor[emitter_id]
            if hour_index >= len(high_output_series):
                raise ValueError(
                    f"Capture high-output trace for {emitter_id} ends before "
                    f"hour {hour_index}."
                )
            high_output_factor = float(high_output_series[hour_index])
            availability = float(
                state.emitter_availability.get(emitter_id, 1.0)
            )
            frame[f"{emitter_id}_inventory_t"] = float(
                state.entity_inventory_t.get(emitter_id, 0.0)
            )
            frame[f"{emitter_id}_capture_availability"] = availability
            frame[f"{emitter_id}_capture_outage"] = int(
                abs(availability) <= 1e-12
            )
            frame[f"{emitter_id}_capture_high_output_factor"] = (
                high_output_factor
            )
            frame[f"{emitter_id}_capture_high_output"] = int(
                high_output_factor > 1.0 + 1e-12
            )

        modes = vessel_operation_modes(env)
        for vessel_id, mode in zip(self.vessel_ids, modes):
            frame[f"{vessel_id}_inventory_t"] = float(
                state.entity_inventory_t.get(vessel_id, 0.0)
            )
            vessel_state = env.simulator.vessel_states[vessel_id]
            if mode == "sailing":
                destination = str(vessel_state["destination"])
                operational_state = (
                    "sailing_to_terminal"
                    if destination in self.terminal_ids
                    else "sailing_to_emitter"
                )
            else:
                destination = str(vessel_state.get("berth", ""))
                operational_state = mode
            if operational_state not in OPERATIONAL_STATES:
                raise ValueError(
                    f"Unknown operational state for {vessel_id}: "
                    f"{operational_state}"
                )
            frame[f"{vessel_id}_mode"] = mode
            frame[f"{vessel_id}_destination"] = destination
            frame[f"{vessel_id}_operational_state"] = operational_state

        for terminal_id in self.terminal_ids:
            frame[f"{terminal_id}_inventory_t"] = float(
                state.entity_inventory_t.get(terminal_id, 0.0)
            )
        for well_id in self.well_ids:
            frame[f"{well_id}_available"] = int(
                bool(state.well_available.get(well_id, True))
            )
        self.frames.append(frame)


def validate_trace(recorder: TraceRecorder) -> None:
    if len(recorder.frames) != 721:
        raise ValueError(
            f"Figure 4 trace has {len(recorder.frames)} frames; expected 721."
        )
    hours = [float(frame["hour"]) for frame in recorder.frames]
    if hours != [float(hour) for hour in range(721)]:
        raise ValueError("Figure 4 trace is not contiguous from hour 0 to 720.")


def validate_metric(field: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-6):
        raise ValueError(
            f"Trace-only replay changed {field}: actual={actual}, "
            f"archived={expected}."
        )


def reset_with_capture_event_trace(wrapper):
    high_output_series: list[list[float]] = []
    original_factor_window_series = scenario_generator._factor_window_series

    def traced_factor_window_series(
        rng,
        n_steps,
        dt,
        rate_per_week,
        mean_hours,
        value_range,
        *,
        inactive_value,
    ):
        values = original_factor_window_series(
            rng,
            n_steps,
            dt,
            rate_per_week,
            mean_hours,
            value_range,
            inactive_value=inactive_value,
        )
        if float(value_range[0]) > 1.0:
            high_output_series.append(list(values))
        return values

    scenario_generator._factor_window_series = traced_factor_window_series
    try:
        observation, info = wrapper.residual_env.reset_native_seed(TEST_SEED)
    finally:
        scenario_generator._factor_window_series = original_factor_window_series

    emitter_ids = tuple(wrapper.env.emitter_ids)
    if len(high_output_series) != len(emitter_ids):
        raise ValueError(
            "Expected one capture high-output factor series per emitter; "
            f"found {len(high_output_series)} for {len(emitter_ids)} emitters."
        )
    factor_by_emitter = {
        emitter_id: values
        for emitter_id, values in zip(emitter_ids, high_output_series)
    }
    return observation, info, factor_by_emitter


def replay_iterative_q(
    args,
    iterative_q_results: Path,
) -> tuple[TraceRecorder, dict[str, float], int, int]:
    device = torch.device("cpu")
    model, metadata = iterative_q_eval._load_model(args, device)
    variant = str(metadata["observation_variant"])
    follow_index = int(metadata["follow_action_index"])
    gate = args.gates[0]
    wrapper = iterative_q_eval.make_event_env(args, variant)

    observation, info, high_output_factor = reset_with_capture_event_trace(
        wrapper
    )
    recorder = TraceRecorder(high_output_factor)
    recorder.record(wrapper.env)
    original_step = wrapper.env.step

    def traced_step(action):
        result = original_step(action)
        recorder.record(wrapper.env)
        return result

    wrapper.env.step = traced_step
    observation, info = wrapper._after_reset(observation, info)

    done = False
    event_count = 0
    override_events = 0
    used_windows: set[int] = set()
    while not done:
        expected_q = iterative_q_eval.expected_q_for_observation(
            model,
            observation,
            wrapper.env,
            device,
            args.future_ablation,
        )
        action, _decision = iterative_q_eval.select_safe_action(
            expected_q,
            wrapper.action_masks(),
            follow_index,
            required_heads=int(gate["required_heads"]),
            margin=float(gate["margin"]),
            uncertainty_beta=float(gate.get("uncertainty_beta", 0.0)),
        )
        active_window = None
        if gate.get("windows") is not None:
            active_window = next(
                (
                    index
                    for index, (start, end) in enumerate(gate["windows"])
                    if float(start) <= float(wrapper.env.t) <= float(end)
                ),
                None,
            )
            if (
                action != follow_index
                and (active_window is None or active_window in used_windows)
            ):
                action = follow_index
        if (
            gate.get("max_overrides") is not None
            and override_events >= int(gate["max_overrides"])
            and action != follow_index
        ):
            action = follow_index
        if action != follow_index and active_window is not None:
            used_windows.add(active_window)

        observation, _reward, terminated, truncated, _info = wrapper.step(action)
        event_count += 1
        override_events += int(action != follow_index)
        done = bool(terminated or truncated)

    validate_trace(recorder)
    metrics = iterative_q_eval._metrics(wrapper.env)
    formal = load_formal_result(iterative_q_results)
    for field in (
        "total_cost_eur",
        "episode_total_cost_eur",
        "terminal_cleanup_operating_cost_eur",
        "vented_t",
        "stored_t",
    ):
        validate_metric(field, float(metrics[field]), float(formal[field]))
    if event_count != int(formal["event_count"]):
        raise ValueError(
            f"Event count changed: {event_count} vs {formal['event_count']}."
        )
    if override_events != int(formal["override_events"]):
        raise ValueError(
            f"Intervention count changed: {override_events} vs "
            f"{formal['override_events']}."
        )
    return recorder, metrics, event_count, override_events


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contiguous_ranges(
    frames: list[dict[str, object]],
    predicate,
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start: float | None = None
    for frame in frames[:-1]:
        hour = float(frame["hour"])
        active = bool(predicate(frame))
        if active and start is None:
            start = hour
        elif not active and start is not None:
            ranges.append((start, hour))
            start = None
    if start is not None:
        ranges.append((start, 720.0))
    return ranges


def capacity_guides(
    ax,
    capacity_values_t: list[float],
) -> None:
    for capacity_t in sorted(set(capacity_values_t)):
        capacity_kt = capacity_t / 1_000.0
        ax.axhline(
            capacity_kt,
            color="#767676",
            linewidth=0.65,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        ax.text(
            714,
            capacity_kt,
            f"capacity {capacity_kt:g} kt",
            fontsize=4.8,
            color="#606060",
            ha="right",
            va="bottom",
        )


def style_line_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=6.8, fontweight="bold", pad=3)
    ax.set_ylabel(ylabel, fontsize=6.0)
    ax.tick_params(axis="both", labelsize=5.4, length=2.3, width=0.55)
    ax.yaxis.grid(True, color="#D8D8D8", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.set_xlim(0.0, 720.0)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.11,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_figure(recorder: TraceRecorder, output_dir: Path) -> list[Path]:
    width_in = 183.0 / 25.4
    height_in = 155.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=(1.05, 1.45, 1.45, 1.30),
        hspace=0.47,
        wspace=0.55,
    )
    ax_disturbance = fig.add_subplot(grid[0, :])
    ax_emitter = fig.add_subplot(grid[1, :2])
    ax_terminal = fig.add_subplot(grid[1, 2])
    ax_vessel = fig.add_subplot(grid[2, :2], sharex=ax_emitter)
    ax_vent = fig.add_subplot(grid[2, 2], sharex=ax_terminal)
    ax_state = fig.add_subplot(grid[3, :])

    hours = np.asarray([float(row["hour"]) for row in recorder.frames])

    event_rows: list[
        tuple[str, list[tuple[str, list[tuple[float, float]]]]]
    ] = [
        (
            "Weather slowdown",
            [
                (
                    "weather",
                    contiguous_ranges(
                        recorder.frames,
                        lambda row: float(row["weather_speed_factor"])
                        < 1.0 - 1e-9,
                    ),
                )
            ],
        )
    ]
    for emitter_id in recorder.emitter_ids:
        event_rows.append(
            (
                f"{friendly_emitter_name(emitter_id)} capture",
                [
                    (
                        "capture_high_output",
                        contiguous_ranges(
                            recorder.frames,
                            lambda row, entity_id=emitter_id: int(
                                row[f"{entity_id}_capture_high_output"]
                            )
                            == 1,
                        ),
                    ),
                    (
                        "capture_outage",
                        contiguous_ranges(
                            recorder.frames,
                            lambda row, entity_id=emitter_id: int(
                                row[f"{entity_id}_capture_outage"]
                            )
                            == 1,
                        ),
                    ),
                ],
            )
        )
    for well_id in recorder.well_ids:
        event_rows.append(
            (
                "Injection-well maintenance",
                [
                    (
                        "well",
                        contiguous_ranges(
                            recorder.frames,
                            lambda row, entity_id=well_id: int(
                                row[f"{entity_id}_available"]
                            )
                            == 0,
                        ),
                    )
                ],
            )
        )

    for row_index, (_label, event_channels) in enumerate(event_rows):
        for event_type, ranges in event_channels:
            ax_disturbance.broken_barh(
                [(start, end - start) for start, end in ranges],
                (row_index - 0.34, 0.68),
                facecolors=DISTURBANCE_COLORS[event_type],
                edgecolors="none",
                alpha=0.92,
            )
    ax_disturbance.set_yticks(np.arange(len(event_rows)))
    ax_disturbance.set_yticklabels(
        [label for label, _channels in event_rows],
        fontsize=5.5,
    )
    ax_disturbance.set_ylim(len(event_rows) - 0.5, -0.5)
    ax_disturbance.set_xlim(0.0, 720.0)
    ax_disturbance.set_xticks(np.arange(0, 721, 120))
    ax_disturbance.tick_params(axis="x", labelbottom=False, length=2.2, width=0.55)
    ax_disturbance.tick_params(axis="y", length=0, pad=3)
    ax_disturbance.xaxis.grid(True, color="#E0E0E0", linewidth=0.4)
    for side in ("top", "right", "left"):
        ax_disturbance.spines[side].set_visible(False)
    ax_disturbance.spines["bottom"].set_linewidth(0.65)
    ax_disturbance.set_title(
        "Exogenous event schedule",
        loc="left",
        fontsize=6.8,
        fontweight="bold",
        pad=3,
    )
    ax_disturbance.legend(
        handles=[
            Patch(color=DISTURBANCE_COLORS["weather"], label="Weather slowdown"),
            Patch(
                color=DISTURBANCE_COLORS["capture_outage"],
                label="Capture outage",
            ),
            Patch(
                color=DISTURBANCE_COLORS["capture_high_output"],
                label="Capture high-output",
            ),
            Patch(color=DISTURBANCE_COLORS["well"], label="Well maintenance"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.22),
        ncol=4,
        frameon=False,
        fontsize=4.9,
        handlelength=1.2,
        columnspacing=0.7,
        handletextpad=0.35,
    )

    emitter_capacities = [
        recorder.capacities_t[entity_id] for entity_id in recorder.emitter_ids
    ]
    for emitter_id, color in zip(recorder.emitter_ids, EMITTER_COLORS):
        values = np.asarray(
            [
                float(row[f"{emitter_id}_inventory_t"])
                for row in recorder.frames
            ]
        )
        capacity = recorder.capacities_t[emitter_id]
        ax_emitter.plot(
            hours,
            values / 1_000.0,
            color=color,
            linewidth=1.05,
            label=(
                f"{friendly_emitter_name(emitter_id)} "
                f"(cap. {capacity / 1_000.0:g} kt)"
            ),
        )
    capacity_guides(ax_emitter, emitter_capacities)
    style_line_axis(ax_emitter, "Emitter buffer inventories", "Inventory (kt)")
    ax_emitter.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=5.1,
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    ax_emitter.tick_params(axis="x", labelbottom=False)

    terminal_id = recorder.terminal_ids[0]
    terminal_capacity = recorder.capacities_t[terminal_id]
    terminal_values = np.asarray(
        [
            float(row[f"{terminal_id}_inventory_t"])
            for row in recorder.frames
        ]
    )
    ax_terminal.plot(
        hours,
        terminal_values / 1_000.0,
        color=TERMINAL_COLOR,
        linewidth=1.1,
    )
    capacity_guides(ax_terminal, [terminal_capacity])
    style_line_axis(
        ax_terminal,
        friendly_terminal_name(terminal_id),
        "Inventory (kt)",
    )
    ax_terminal.tick_params(axis="x", labelbottom=False)

    vessel_display_names = {
        vessel_id: f"Vessel {index + 1}"
        for index, vessel_id in enumerate(recorder.vessel_ids)
    }
    vessel_capacities = [
        recorder.capacities_t[vessel_id] for vessel_id in recorder.vessel_ids
    ]
    for vessel_id, color in zip(recorder.vessel_ids, VESSEL_COLORS):
        values = np.asarray(
            [
                float(row[f"{vessel_id}_inventory_t"])
                for row in recorder.frames
            ]
        )
        ax_vessel.plot(
            hours,
            values / 1_000.0,
            color=color,
            linewidth=1.05,
            label=vessel_display_names[vessel_id],
        )
    capacity_guides(ax_vessel, vessel_capacities)
    style_line_axis(ax_vessel, "Vessel cargo inventories", "Cargo (kt)")
    ax_vessel.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=5.2,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    ax_vessel.set_xticks(np.arange(0, 721, 120))
    ax_vessel.set_xlabel("Time (h)", fontsize=6.0)

    vent_values = np.asarray(
        [float(row["cumulative_vent_t"]) for row in recorder.frames]
    )
    ax_vent.plot(hours, vent_values, color=VENT_COLOR, linewidth=1.1)
    vent_upper = max(50.0, float(vent_values.max()) * 1.08)
    ax_vent.set_ylim(-0.02 * vent_upper, vent_upper)
    style_line_axis(ax_vent, "Cumulative vent", r"Vented CO$_2$ (t)")
    ax_vent.set_xticks((0, 360, 720))
    ax_vent.set_xlabel("Time (h)", fontsize=6.0)
    ax_vent.text(
        0.97,
        0.84,
        f"Final: {vent_values[-1]:,.0f} t",
        transform=ax_vent.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=VENT_COLOR,
        fontweight="bold",
    )

    state_to_index = {
        state: index for index, state in enumerate(OPERATIONAL_STATES)
    }
    mode_matrix = np.asarray(
        [
            [
                state_to_index[
                    str(frame[f"{vessel_id}_operational_state"])
                ]
                for frame in recorder.frames[:-1]
            ]
            for vessel_id in recorder.vessel_ids
        ],
        dtype=float,
    )
    ax_state.imshow(
        mode_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(OPERATIONAL_COLORS),
        vmin=-0.5,
        vmax=len(OPERATIONAL_STATES) - 0.5,
        extent=(0.0, 720.0, len(recorder.vessel_ids) - 0.5, -0.5),
    )
    ax_state.set_yticks(np.arange(len(recorder.vessel_ids)))
    ax_state.set_yticklabels(
        [vessel_display_names[vessel_id] for vessel_id in recorder.vessel_ids],
        fontsize=5.8,
    )
    ax_state.set_xticks(np.arange(0, 721, 120))
    ax_state.set_xlim(0.0, 720.0)
    ax_state.set_xlabel("Time (h)", fontsize=6.0)
    ax_state.set_title(
        "Vessel operational states",
        loc="left",
        fontsize=6.8,
        fontweight="bold",
        pad=3,
    )
    ax_state.tick_params(axis="x", labelsize=5.5, length=2.3, width=0.55)
    ax_state.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax_state.spines[side].set_visible(False)
    ax_state.spines["left"].set_linewidth(0.65)
    ax_state.spines["bottom"].set_linewidth(0.65)
    ax_state.legend(
        handles=[
            Patch(color=color, label=OPERATIONAL_LABELS[state])
            for state, color in zip(OPERATIONAL_STATES, OPERATIONAL_COLORS)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=6,
        frameon=False,
        fontsize=5.2,
        handlelength=1.15,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    for label, ax in zip(
        ("a", "b", "c", "d", "e", "f"),
        (
            ax_disturbance,
            ax_emitter,
            ax_terminal,
            ax_vessel,
            ax_vent,
            ax_state,
        ),
    ):
        panel_label(ax, label)

    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.95,
        bottom=0.14,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "figure_4_representative_trajectory"
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def outage_hours(
    recorder: TraceRecorder,
    field: str,
    outage_value: int,
) -> int:
    return sum(
        int(frame[field]) == outage_value for frame in recorder.frames[:-1]
    )


def high_output_windows(
    recorder: TraceRecorder,
    emitter_id: str,
) -> list[dict[str, float]]:
    factor_field = f"{emitter_id}_capture_high_output_factor"
    windows: list[dict[str, float]] = []
    start: float | None = None
    multiplier: float | None = None
    for frame in recorder.frames[:-1]:
        hour = float(frame["hour"])
        factor = float(frame[factor_field])
        active = factor > 1.0 + 1e-12
        factor_changed = (
            multiplier is not None
            and not np.isclose(factor, multiplier, rtol=0.0, atol=1e-12)
        )
        if active and start is None:
            start = hour
            multiplier = factor
        elif active and factor_changed:
            windows.append(
                {
                    "start_hour": start,
                    "end_hour": hour,
                    "multiplier": multiplier,
                }
            )
            start = hour
            multiplier = factor
        elif not active and start is not None:
            windows.append(
                {
                    "start_hour": start,
                    "end_hour": hour,
                    "multiplier": multiplier,
                }
            )
            start = None
            multiplier = None
    if start is not None:
        windows.append(
            {
                "start_hour": start,
                "end_hour": 720.0,
                "multiplier": multiplier,
            }
        )
    return windows


def main() -> None:
    args = parse_args()
    recorder, metrics, event_count, override_events = replay_iterative_q(
        evaluation_args(args.checkpoint),
        args.iterative_q_results,
    )

    source_data_dir = args.output_dir / "source_data"
    trace_csv = source_data_dir / "figure_4_hourly_trace.csv"
    metadata_json = source_data_dir / "figure_4_metadata.json"
    write_csv(trace_csv, recorder.frames)

    vessel_display_names = {
        vessel_id: f"Vessel {index + 1}"
        for index, vessel_id in enumerate(recorder.vessel_ids)
    }
    capture_outage_hours = {
        friendly_emitter_name(emitter_id): outage_hours(
            recorder,
            f"{emitter_id}_capture_outage",
            1,
        )
        for emitter_id in recorder.emitter_ids
    }
    capture_high_output_hours = {
        friendly_emitter_name(emitter_id): outage_hours(
            recorder,
            f"{emitter_id}_capture_high_output",
            1,
        )
        for emitter_id in recorder.emitter_ids
    }
    capture_high_output_windows = {
        friendly_emitter_name(emitter_id): high_output_windows(
            recorder,
            emitter_id,
        )
        for emitter_id in recorder.emitter_ids
    }
    well_outage_hours = {
        well_id: outage_hours(
            recorder,
            f"{well_id}_available",
            0,
        )
        for well_id in recorder.well_ids
    }
    weather_slowdown_hours = sum(
        float(frame["weather_speed_factor"]) < 1.0 - 1e-9
        for frame in recorder.frames[:-1]
    )
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "figure": "Figure 4",
                "test_seed": TEST_SEED,
                "iterative_q_model_seed": 0,
                "trace_role": "trace-only replay; excluded from E1 statistics",
                "checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)),
                "formal_results": str(
                    args.iterative_q_results.relative_to(REPO_ROOT)
                ),
                "hourly_frames": len(recorder.frames),
                "emitter_display_names": {
                    emitter_id: friendly_emitter_name(emitter_id)
                    for emitter_id in recorder.emitter_ids
                },
                "vessel_display_names": vessel_display_names,
                "terminal_display_names": {
                    terminal_id: friendly_terminal_name(terminal_id)
                    for terminal_id in recorder.terminal_ids
                },
                "inventory_capacities_t": recorder.capacities_t,
                "event_definition": {
                    "weather_slowdown": "vessel speed factor < 1",
                    "capture_outage": "capture availability == 0",
                    "capture_high_output": (
                        "sampled high-output multiplier > 1; ordinary hourly "
                        "capture noise is excluded"
                    ),
                    "well_maintenance": "well available == false",
                    "capture_noise_not_labelled_as_outage": True,
                },
                "event_hours": {
                    "weather_slowdown": weather_slowdown_hours,
                    "capture_outage_by_emitter": capture_outage_hours,
                    "capture_high_output_by_emitter": (
                        capture_high_output_hours
                    ),
                    "well_maintenance_by_well": well_outage_hours,
                },
                "capture_high_output_windows_by_emitter": (
                    capture_high_output_windows
                ),
                "replay_validation": {
                    "total_cost_eur": metrics["total_cost_eur"],
                    "vented_t": metrics["vented_t"],
                    "stored_t": metrics["stored_t"],
                    "event_count": event_count,
                    "status": "exact_match_to_archived_metrics",
                },
                "output_formats": ["pdf", "png"],
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    outputs = draw_figure(recorder, args.output_dir)
    print(f"Wrote {trace_csv}")
    print(f"Wrote {metadata_json}")
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
