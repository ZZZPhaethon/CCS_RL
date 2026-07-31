"""Build the E6 Greedy-versus-Iterative-Q mechanism case study."""

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
from experiments import plot_e1_figure4 as legacy_figure


TEST_SEED = 9_000_056
MODEL_SEED = 0
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments_results" / "E6"
FORMAL_COMPARISON = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "algorithms"
    / "formal_comparison"
    / "e1_formal_per_episode.csv"
)
Q_COLOR = "#2166AC"
GREEDY_COLOR = "#666666"
VENT_COLOR = "#B2182B"
TERMINAL_COLOR = "#7B61A8"
EMITTER_COLORS = ("#2B6CB0", "#D97706", "#0F8B8D")
LOCATION_IDLE = "Idle / queue"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


class MechanismTraceRecorder(legacy_figure.TraceRecorder):
    def __init__(self, high_output_factor, controller: str) -> None:
        super().__init__(high_output_factor)
        self.controller = controller

    def record(self, env) -> None:
        super().record(env)
        frame = self.frames[-1]
        frame["controller"] = self.controller
        frame["cumulative_operating_cost_eur"] = float(
            env.ledger.operating_cost
        )
        frame["cumulative_vent_penalty_eur"] = float(
            env.ledger.vent_penalty
        )
        frame["cumulative_episode_cost_eur"] = float(env.ledger.total_cost)


def _formal_row(algorithm: str, model_seed: str = "") -> dict[str, str]:
    with FORMAL_COMPARISON.open("r", encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row["algorithm"] == algorithm
            and row["model_seed"] == model_seed
            and int(row["test_seed"]) == TEST_SEED
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one formal row for {algorithm}/{model_seed}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _action_label(action: int, emitter_ids: tuple[str, ...]) -> str:
    if action == 0:
        return "Wait"
    if action == 1:
        return "Terminal"
    emitter_index = action - 2
    if 0 <= emitter_index < len(emitter_ids):
        return legacy_figure.friendly_emitter_name(
            emitter_ids[emitter_index]
        )
    return f"Action {action}"


def _gated_q_action(
    wrapper,
    observation,
    model,
    device,
    args,
    gate,
    follow_index: int,
    used_windows: set[int],
    override_events: int,
):
    expected_q = iterative_q_eval.expected_q_for_observation(
        model,
        observation,
        wrapper.env,
        device,
        args.future_ablation,
    )
    action, decision = iterative_q_eval.select_safe_action(
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
    return action, decision, active_window


def replay_controller(
    controller: str,
) -> tuple[
    MechanismTraceRecorder,
    dict[str, float],
    list[dict[str, object]],
    int,
    int,
]:
    args = legacy_figure.evaluation_args(legacy_figure.CHECKPOINT)
    device = torch.device("cpu")
    model, metadata = iterative_q_eval._load_model(args, device)
    variant = str(metadata["observation_variant"])
    follow_index = int(metadata["follow_action_index"])
    gate = args.gates[0]
    wrapper = iterative_q_eval.make_event_env(
        args,
        variant,
        greedy_control_variate=False,
    )
    observation, info, high_output_factor = (
        legacy_figure.reset_with_capture_event_trace(wrapper)
    )
    recorder = MechanismTraceRecorder(high_output_factor, controller)
    recorder.record(wrapper.env)
    original_step = wrapper.env.step

    def traced_step(action):
        result = original_step(action)
        recorder.record(wrapper.env)
        return result

    wrapper.env.step = traced_step
    observation, info = wrapper._after_reset(observation, info)

    decisions: list[dict[str, object]] = []
    event_count = 0
    override_events = 0
    used_windows: set[int] = set()
    done = False
    while not done:
        decision_hour = float(wrapper.env.t)
        if controller == "greedy":
            action = follow_index
            decision = {
                "candidate": follow_index,
                "agreement": 0,
                "positive_heads": 0,
                "ensemble_advantage": 0.0,
                "advantage_std": 0.0,
                "lower_confidence_advantage": 0.0,
            }
            active_window = None
        else:
            action, decision, active_window = _gated_q_action(
                wrapper,
                observation,
                model,
                device,
                args,
                gate,
                follow_index,
                used_windows,
                override_events,
            )

        greedy_native = [
            int(value)
            for value in wrapper.residual_env._base_action["vessels"]
        ]
        selected_native = [
            int(value)
            for value in wrapper.residual_env.native_action(
                wrapper.decode_action(action)
            )["vessels"]
        ]
        for vessel_index, vessel_id in enumerate(wrapper.env.vessel_ids):
            greedy_action = greedy_native[vessel_index]
            selected_action = selected_native[vessel_index]
            decisions.append(
                {
                    "controller": controller,
                    "test_seed": TEST_SEED,
                    "event_index": event_count,
                    "hour": decision_hour,
                    "vessel_id": vessel_id,
                    "greedy_native_action": greedy_action,
                    "greedy_action_label": _action_label(
                        greedy_action,
                        tuple(wrapper.env.emitter_ids),
                    ),
                    "selected_native_action": selected_action,
                    "selected_action_label": _action_label(
                        selected_action,
                        tuple(wrapper.env.emitter_ids),
                    ),
                    "physical_action_changed": int(
                        selected_action != greedy_action
                    ),
                    "joint_action_index": int(action),
                    "candidate_joint_action_index": int(
                        decision["candidate"]
                    ),
                    "accepted_override": int(action != follow_index),
                    "active_policy_window": (
                        "" if active_window is None else int(active_window)
                    ),
                    "head_agreement": int(decision["agreement"]),
                    "positive_heads": int(decision["positive_heads"]),
                    "ensemble_advantage": float(
                        decision["ensemble_advantage"]
                    ),
                    "advantage_std": float(decision["advantage_std"]),
                    "lower_confidence_advantage": float(
                        decision["lower_confidence_advantage"]
                    ),
                }
            )
        observation, _reward, terminated, truncated, _info = wrapper.step(
            action
        )
        event_count += 1
        override_events += int(action != follow_index)
        done = bool(terminated or truncated)

    legacy_figure.validate_trace(recorder)
    metrics = iterative_q_eval._metrics(wrapper.env)
    expected = (
        _formal_row("greedy")
        if controller == "greedy"
        else _formal_row("iterative_action_q_g60_p4", str(MODEL_SEED))
    )
    for field in (
        "total_cost_eur",
        "episode_total_cost_eur",
        "terminal_cleanup_operating_cost_eur",
        "vented_t",
        "stored_t",
    ):
        legacy_figure.validate_metric(
            field,
            float(metrics[field]),
            float(expected[field]),
        )
    if controller == "iterative_q":
        if event_count != int(expected["decision_count"]):
            raise ValueError("Iterative-Q event count changed during E6 replay")
        if override_events != int(expected["override_events"]):
            raise ValueError("Iterative-Q override count changed during E6 replay")
    return recorder, metrics, decisions, event_count, override_events


def _style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=7.0, fontweight="bold", pad=3)
    ax.set_ylabel(ylabel, fontsize=6.2)
    ax.tick_params(labelsize=5.7, length=2.5, width=0.55)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def _intervention_hours(decisions: list[dict[str, object]]) -> list[float]:
    return sorted(
        {
            float(row["hour"])
            for row in decisions
            if int(row["accepted_override"]) == 1
        }
    )


def _save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, dpi in (("svg", 300), ("pdf", 300), ("tiff", 600), ("png", 300)):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        paths.append(path)
    return paths


def draw_system_states(
    greedy: MechanismTraceRecorder,
    iterative_q: MechanismTraceRecorder,
    greedy_metrics: dict[str, float],
    q_metrics: dict[str, float],
    q_decisions: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(183 / 25.4, 126 / 25.4),
        sharex=True,
        constrained_layout=True,
    )
    hours_d = np.asarray([float(row["hour"]) / 24 for row in greedy.frames])
    q_hours_d = np.asarray(
        [float(row["hour"]) / 24 for row in iterative_q.frames]
    )
    intervention_days = [hour / 24 for hour in _intervention_hours(q_decisions)]

    for panel_index, emitter_id in enumerate(greedy.emitter_ids):
        ax = axes.flat[panel_index]
        capacity = greedy.capacities_t[emitter_id]
        greedy_fill = np.asarray(
            [
                float(row[f"{emitter_id}_inventory_t"]) / capacity * 100
                for row in greedy.frames
            ]
        )
        q_fill = np.asarray(
            [
                float(row[f"{emitter_id}_inventory_t"]) / capacity * 100
                for row in iterative_q.frames
            ]
        )
        ax.plot(
            hours_d,
            greedy_fill,
            color=GREEDY_COLOR,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            label="Greedy",
        )
        ax.plot(q_hours_d, q_fill, color=Q_COLOR, linewidth=1.15, label="Iterative-Q")
        _style_axis(
            ax,
            f"{chr(97 + panel_index)}  "
            f"{legacy_figure.friendly_emitter_name(emitter_id)} buffer",
            "Fill level (%)",
        )
        ax.axhline(100, color=VENT_COLOR, linewidth=0.65, linestyle=":")

    terminal_id = greedy.terminal_ids[0]
    ax_terminal = axes.flat[3]
    terminal_capacity = greedy.capacities_t[terminal_id]
    for recorder, color, line_style, label, hours in (
        (greedy, GREEDY_COLOR, (0, (3, 2)), "Greedy", hours_d),
        (iterative_q, Q_COLOR, "-", "Iterative-Q", q_hours_d),
    ):
        terminal_fill = np.asarray(
            [
                float(row[f"{terminal_id}_inventory_t"])
                / terminal_capacity
                * 100
                for row in recorder.frames
            ]
        )
        ax_terminal.plot(
            hours,
            terminal_fill,
            color=color,
            linewidth=1.1,
            linestyle=line_style,
            label=label,
        )
    _style_axis(ax_terminal, "d  Terminal buffer", "Fill level (%)")

    ax_vent = axes.flat[4]
    ax_vent.plot(
        hours_d,
        [float(row["cumulative_vent_t"]) for row in greedy.frames],
        color=GREEDY_COLOR,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
    )
    ax_vent.plot(
        q_hours_d,
        [float(row["cumulative_vent_t"]) for row in iterative_q.frames],
        color=Q_COLOR,
        linewidth=1.15,
    )
    _style_axis(ax_vent, "e  Cumulative venting", "Vented CO$_2$ (t)")

    ax_cost = axes.flat[5]
    for recorder, metrics, color, line_style in (
        (greedy, greedy_metrics, GREEDY_COLOR, (0, (3, 2))),
        (iterative_q, q_metrics, Q_COLOR, "-"),
    ):
        values = (
            np.asarray(
                [
                    float(row["cumulative_episode_cost_eur"])
                    for row in recorder.frames
                ]
            )
            / 1e6
        )
        ax_cost.plot(
            hours_d,
            values,
            color=color,
            linewidth=1.1,
            linestyle=line_style,
        )
        ax_cost.scatter(
            [30],
            [float(metrics["total_cost_eur"]) / 1e6],
            s=14,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
    _style_axis(ax_cost, "f  Cumulative cost", "Cost (€ million)")
    ax_cost.text(
        0.99,
        0.04,
        "End markers include terminal cleanup",
        transform=ax_cost.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#555555",
    )

    for ax in axes.flat:
        ymin, ymax = ax.get_ylim()
        marker_y = ymin + 0.025 * (ymax - ymin)
        ax.plot(
            intervention_days,
            [marker_y] * len(intervention_days),
            linestyle="none",
            marker="|",
            markersize=4.0,
            markeredgewidth=0.65,
            color=Q_COLOR,
            alpha=0.75,
            clip_on=True,
        )
    for ax in axes[-1]:
        ax.set_xlabel("Episode day", fontsize=6.2)

    saving = greedy_metrics["total_cost_eur"] - q_metrics["total_cost_eur"]
    saving_percent = 100 * saving / greedy_metrics["total_cost_eur"]
    fig.suptitle(
        "Iterative-Q prevents buffer overflow and venting through sparse interventions\n"
        f"Seed {TEST_SEED}: {len(intervention_days)} interventions, "
        f"€{saving / 1e3:,.0f}k "
        f"lower final cost ({saving_percent:.2f}%)",
        fontsize=8.1,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            plt.Line2D(
                [0], [0], color=GREEDY_COLOR, linestyle=(0, (3, 2)),
                linewidth=1.1, label="Greedy",
            ),
            plt.Line2D(
                [0], [0], color=Q_COLOR, linewidth=1.2, label="Iterative-Q",
            ),
            plt.Line2D(
                [0], [0], color=Q_COLOR, marker="|", linestyle="none",
                label="Iterative-Q intervention",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=5.8,
    )
    return _save_figure(fig, output_dir, "e6_system_states")


def _service_location(
    frame: dict[str, object],
    vessel_id: str,
    emitter_ids: tuple[str, ...],
    terminal_ids: tuple[str, ...],
) -> str:
    destination = str(frame[f"{vessel_id}_destination"])
    if destination in terminal_ids:
        return "Terminal"
    if destination in emitter_ids:
        return legacy_figure.friendly_emitter_name(destination)
    return LOCATION_IDLE


def draw_vessel_actions(
    greedy: MechanismTraceRecorder,
    iterative_q: MechanismTraceRecorder,
    q_decisions: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    emitter_names = [
        legacy_figure.friendly_emitter_name(value)
        for value in greedy.emitter_ids
    ]
    categories = [LOCATION_IDLE, *emitter_names, "Terminal"]
    colors = ["#E5E5E5", *EMITTER_COLORS[: len(emitter_names)], TERMINAL_COLOR]
    category_code = {name: index for index, name in enumerate(categories)}

    def matrix(recorder):
        values = np.zeros(
            (len(recorder.vessel_ids), len(recorder.frames) - 1),
            dtype=int,
        )
        for vessel_index, vessel_id in enumerate(recorder.vessel_ids):
            for hour_index, frame in enumerate(recorder.frames[:-1]):
                location = _service_location(
                    frame,
                    vessel_id,
                    recorder.emitter_ids,
                    recorder.terminal_ids,
                )
                values[vessel_index, hour_index] = category_code[location]
        return values

    greedy_values = matrix(greedy)
    q_values = matrix(iterative_q)
    difference = (greedy_values != q_values).astype(int)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(183 / 25.4, 105 / 25.4),
        gridspec_kw={"height_ratios": (1, 1, 0.55)},
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.12,
        top=0.76,
        hspace=0.33,
    )
    extent = (0, 30, len(greedy.vessel_ids) - 0.5, -0.5)
    cmap = ListedColormap(colors)
    vessel_labels = [
        f"Vessel {index + 1}" for index in range(len(greedy.vessel_ids))
    ]
    for ax, values, title in (
        (axes[0], greedy_values, "a  Greedy vessel execution"),
        (axes[1], q_values, "b  Iterative-Q vessel execution"),
    ):
        ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(categories) - 0.5,
        )
        ax.set_yticks(range(len(vessel_labels)), vessel_labels)
        ax.set_title(title, loc="left", fontsize=7.0, fontweight="bold", pad=3)
        ax.tick_params(labelsize=5.8, length=2.3)

    axes[2].imshow(
        difference,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=ListedColormap(["#F1F1F1", Q_COLOR]),
        vmin=-0.5,
        vmax=1.5,
    )
    axes[2].set_yticks(range(len(vessel_labels)), vessel_labels)
    axes[2].set_title(
        "c  Hours when executed service locations differ",
        loc="left",
        fontsize=7.0,
        fontweight="bold",
        pad=3,
    )
    axes[2].set_xlabel("Episode day", fontsize=6.2)
    axes[2].tick_params(labelsize=5.8, length=2.3)
    for ax in axes:
        ax.set_xlim(0, 30)
        for hour in _intervention_hours(q_decisions):
            ax.axvline(
                hour / 24,
                color="#111111",
                linewidth=0.38,
                alpha=0.42,
            )
    fig.suptitle(
        "Sparse Iterative-Q commands alter the subsequent vessel service sequence",
        y=0.985,
        fontsize=8.1,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            *[
                Patch(facecolor=color, edgecolor="none", label=label)
                for label, color in zip(categories, colors)
            ],
            Patch(facecolor=Q_COLOR, edgecolor="none", label="Different hour"),
            plt.Line2D(
                [0], [0], color="#111111", linewidth=0.7,
                label="Accepted Q intervention",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        fontsize=5.6,
    )
    return _save_figure(fig, output_dir, "e6_vessel_actions")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_interventions(
    decisions: list[dict[str, object]],
    trace: MechanismTraceRecorder,
) -> list[dict[str, object]]:
    accepted_events = sorted(
        {
            int(row["event_index"])
            for row in decisions
            if int(row["accepted_override"]) == 1
        }
    )
    vessel_names = {
        vessel_id: f"Vessel {index + 1}"
        for index, vessel_id in enumerate(trace.vessel_ids)
    }
    rows = []
    for intervention_index, event_index in enumerate(accepted_events, start=1):
        event_rows = [
            row
            for row in decisions
            if int(row["event_index"]) == event_index
        ]
        changed = [
            row for row in event_rows if int(row["physical_action_changed"]) == 1
        ]
        hour = float(event_rows[0]["hour"])
        frame = trace.frames[int(round(hour))]
        emitter_fill = {
            emitter_id: (
                100
                * float(frame[f"{emitter_id}_inventory_t"])
                / trace.capacities_t[emitter_id]
            )
            for emitter_id in trace.emitter_ids
        }
        rows.append(
            {
                "intervention_index": intervention_index,
                "event_index": event_index,
                "hour": hour,
                "day": hour / 24,
                "changed_vessel_count": len(changed),
                "action_changes": "; ".join(
                    f"{vessel_names[str(row['vessel_id'])]}: "
                    f"{row['greedy_action_label']} -> "
                    f"{row['selected_action_label']}"
                    for row in changed
                ),
                "max_emitter_fill_percent": max(emitter_fill.values()),
                **{
                    f"{emitter_id}_fill_percent": value
                    for emitter_id, value in emitter_fill.items()
                },
                "terminal_fill_percent": (
                    100
                    * float(
                        frame[f"{trace.terminal_ids[0]}_inventory_t"]
                    )
                    / trace.capacities_t[trace.terminal_ids[0]]
                ),
                "cumulative_vent_t_before": float(
                    frame["cumulative_vent_t"]
                ),
                "head_agreement": int(event_rows[0]["head_agreement"]),
                "lower_confidence_advantage": float(
                    event_rows[0]["lower_confidence_advantage"]
                ),
            }
        )
    if not rows:
        raise ValueError("E6 replay contains no accepted Iterative-Q interventions")
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output directory: {args.output_dir}"
        )
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        legacy_figure.CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_metadata = checkpoint["metadata"]
    greedy, greedy_metrics, greedy_decisions, greedy_events, _ = (
        replay_controller("greedy")
    )
    q_trace, q_metrics, q_decisions, q_events, q_overrides = (
        replay_controller("iterative_q")
    )
    _write_csv(source_dir / "e6_greedy_hourly_trace.csv", greedy.frames)
    _write_csv(
        source_dir / "e6_iterative_q_hourly_trace.csv",
        q_trace.frames,
    )
    _write_csv(
        source_dir / "e6_event_decisions.csv",
        [*greedy_decisions, *q_decisions],
    )
    interventions = _summarize_interventions(q_decisions, q_trace)
    _write_csv(
        source_dir / "e6_interventions.csv",
        interventions,
    )
    _write_csv(
        source_dir / "e6_outcome_comparison.csv",
        [
            {
                "controller": controller,
                "total_cost_eur": metrics["total_cost_eur"],
                "episode_total_cost_eur": metrics["episode_total_cost_eur"],
                "terminal_cleanup_operating_cost_eur": metrics[
                    "terminal_cleanup_operating_cost_eur"
                ],
                "vented_t": metrics["vented_t"],
                "stored_t": metrics["stored_t"],
                "unit_cost_eur_per_stored_t": metrics[
                    "unit_cost_eur_per_t"
                ],
            }
            for controller, metrics in (
                ("greedy", greedy_metrics),
                ("iterative_q", q_metrics),
            )
        ],
    )
    system_paths = draw_system_states(
        greedy,
        q_trace,
        greedy_metrics,
        q_metrics,
        q_decisions,
        figure_dir,
    )
    action_paths = draw_vessel_actions(
        greedy,
        q_trace,
        q_decisions,
        figure_dir,
    )
    cost_saving = greedy_metrics["total_cost_eur"] - q_metrics["total_cost_eur"]
    metadata = {
        "experiment": "E6 mechanism case study",
        "test_seed": TEST_SEED,
        "iterative_q_model_seed": MODEL_SEED,
        "iterative_q_checkpoint": str(
            legacy_figure.CHECKPOINT.relative_to(REPO_ROOT)
        ),
        "iterative_q_observation_schema": {
            "source_state_features": len(
                checkpoint_metadata.get(
                    "source_state_feature_names",
                    checkpoint_metadata["state_feature_names"],
                )
            ),
            "state_features": len(checkpoint_metadata["state_feature_names"]),
            "excluded_state_features": checkpoint_metadata.get(
                "excluded_state_feature_names",
                [],
            ),
        },
        "comparison": "same exogenous scenario; Greedy versus Iterative-Q",
        "trace_role": "mechanistic case study; excluded from E1 statistics",
        "greedy": {
            **greedy_metrics,
            "event_count": greedy_events,
        },
        "iterative_q": {
            **q_metrics,
            "event_count": q_events,
            "override_events": q_overrides,
        },
        "cost_saving_eur": cost_saving,
        "cost_reduction_vs_greedy_percent": (
            100 * cost_saving / greedy_metrics["total_cost_eur"]
        ),
        "source_files": [
            "source_data/e6_greedy_hourly_trace.csv",
            "source_data/e6_iterative_q_hourly_trace.csv",
            "source_data/e6_event_decisions.csv",
            "source_data/e6_interventions.csv",
            "source_data/e6_outcome_comparison.csv",
        ],
        "figures": [
            str(path.relative_to(args.output_dir))
            for path in [*system_paths, *action_paths]
        ],
    }
    (source_dir / "e6_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    metadata = run(parse_args())
    print(
        "E6_COMPLETE "
        f"cost_saving_eur={metadata['cost_saving_eur']:.2f} "
        "figures=2",
        flush=True,
    )


if __name__ == "__main__":
    main()
