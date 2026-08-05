"""Aggregate E7 horizon-generalization shards and draw the formal figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_e7_temporal_generalization import (
    CONTROLLERS,
    FORMAL_SEEDS,
    HORIZONS,
    expanded_policy_windows,
)


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "experiments_results" / "E7" / "formal_run01"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments_results"
    / "E7"
    / "formal_summary_direct_global"
)
DEFAULT_DIRECT_INPUT_ROOT = (
    REPO_ROOT
    / "experiments_results"
    / "E7"
    / "formal_run05_direct_global"
)
DEFAULT_ADDITIONAL_INPUT_ROOT = (
    REPO_ROOT
    / "experiments_results"
    / "E7"
    / "formal_run06_h4320"
)
DISPLAY_NAMES = {
    "fixed_assignment": "Fixed-Assignment",
    "greedy": "Greedy",
    "iterative_q_direct": "Iterative-Q (direct-global)",
    "iterative_q_receding": "Iterative-Q (receding-cyclic)",
}
COLORS = {
    "fixed_assignment": "#777777",
    "greedy": "#555555",
    "iterative_q_direct": "#7AA6D8",
    "iterative_q_receding": "#2166AC",
}
MARKERS = {
    "fixed_assignment": "o",
    "greedy": "s",
    "iterative_q_direct": "^",
    "iterative_q_receding": "o",
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--direct-input-root",
        type=Path,
        default=DEFAULT_DIRECT_INPUT_ROOT,
    )
    parser.add_argument(
        "--additional-input-root",
        type=Path,
        default=DEFAULT_ADDITIONAL_INPUT_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_and_validate(
    input_root: Path,
    direct_input_root: Path | None = None,
    additional_input_root: Path | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, object]], list[dict[str, str]]]:
    paths = sorted(input_root.rglob("per_episode.csv"))
    if direct_input_root is not None:
        paths = [
            path for path in paths if "iterative_q_direct" not in path.parts
        ]
        direct_paths = sorted(direct_input_root.rglob("per_episode.csv"))
        if any("iterative_q_direct" not in path.parts for path in direct_paths):
            raise ValueError(
                "direct-input-root must contain only iterative_q_direct shards"
            )
        paths.extend(direct_paths)
    if additional_input_root is not None:
        additional_paths = sorted(
            additional_input_root.rglob("per_episode.csv")
        )
        if any("h4320" not in path.parts for path in additional_paths):
            raise ValueError(
                "additional-input-root must contain only h4320 shards"
            )
        paths.extend(additional_paths)
    expected_shards = 48 * len(HORIZONS)
    if len(paths) != expected_shards:
        raise ValueError(
            f"expected {expected_shards} E7 shards, found {len(paths)}"
        )
    rows: list[dict[str, str]] = []
    metadata: list[dict[str, object]] = []
    manifest: list[dict[str, str]] = []
    for path in paths:
        shard_rows = _read_csv(path)
        if not shard_rows:
            raise ValueError(f"empty E7 shard: {path}")
        rows.extend(shard_rows)
        metadata_path = path.with_name("metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        manifest.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "records": str(len(shard_rows)),
            }
        )

    expected_seeds = set(FORMAL_SEEDS)
    unique_keys: set[tuple[str, str, int, int]] = set()
    for row in rows:
        key = (
            row["controller"],
            row["model_seed"],
            int(row["horizon_hours"]),
            int(row["test_seed"]),
        )
        if key in unique_keys:
            raise ValueError(f"duplicate E7 record: {key}")
        unique_keys.add(key)
    for controller in CONTROLLERS:
        for horizon in HORIZONS:
            selected = [
                row
                for row in rows
                if row["controller"] == controller
                and int(row["horizon_hours"]) == horizon
            ]
            expected_records = 90 if controller.startswith("iterative_q") else 30
            if len(selected) != expected_records:
                raise ValueError(
                    f"incomplete E7 grid for {controller}/{horizon}: "
                    f"{len(selected)} vs {expected_records}"
                )
            if {int(row["test_seed"]) for row in selected} != expected_seeds:
                raise ValueError(f"test-seed mismatch for {controller}/{horizon}")
            model_seeds = {row["model_seed"] for row in selected}
            expected_models = (
                {"0", "1", "2"}
                if controller.startswith("iterative_q")
                else {""}
            )
            if model_seeds != expected_models:
                raise ValueError(
                    f"model-seed mismatch for {controller}/{horizon}: "
                    f"{model_seeds}"
                )
    if any(int(item["scenario_hours"]) != 8928 for item in metadata):
        raise ValueError("all E7 shards must use the same 8928 h scenario")
    if any(not bool(item["nested_scenario_prefix"]) for item in metadata):
        raise ValueError("E7 shards must declare nested scenario prefixes")
    if any(item.get("host") != "rootrunner" for item in metadata):
        raise ValueError("all E7 shards must run on rootrunner")
    if any(
        str(item.get("slurm_cpus_per_task")) != "4" for item in metadata
    ):
        raise ValueError("all E7 shards must use four allocated CPUs")
    for item in metadata:
        controller = str(item["controller"])
        if not controller.startswith("iterative_q"):
            continue
        horizon = int(item["execution_horizon_hours"])
        expected_windows = [
            list(window) for window in expanded_policy_windows(horizon)
        ]
        if controller == "iterative_q_direct":
            adapter = item.get("direct_global_adapter")
            if not isinstance(adapter, dict):
                raise ValueError("Direct-global shard lacks adapter metadata")
            if adapter.get("episode_progress") != "t / H":
                raise ValueError("Direct-global must use t / H episode progress")
            if adapter.get("policy_windows") != expected_windows:
                raise ValueError(
                    "Direct-global policy windows must repeat every 720 h"
                )
            if item.get("direct_policy_windows") != expected_windows:
                raise ValueError("Direct-global window audit fields disagree")
        elif controller == "iterative_q_receding":
            adapter = item.get("receding_adapter")
            if not isinstance(adapter, dict):
                raise ValueError("Receding-cyclic shard lacks adapter metadata")
            if adapter.get("episode_progress") != "(t mod 720) / 720":
                raise ValueError(
                    "Receding-cyclic must use modulo episode progress"
                )
            if adapter.get("policy_windows") != expected_windows:
                raise ValueError(
                    "Receding-cyclic policy windows must repeat every 720 h"
                )
    for model_seed in (0, 1, 2):
        checkpoint_hashes = {
            str(item.get("checkpoint_sha256"))
            for item in metadata
            if str(item["controller"]).startswith("iterative_q")
            and int(item["model_seed"]) == model_seed
        }
        if len(checkpoint_hashes) != 1:
            raise ValueError(
                "Direct-global and receding-cyclic must use the same checkpoint "
                f"for model seed {model_seed}: {sorted(checkpoint_hashes)}"
            )

    captured: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        captured.setdefault(
            (int(row["horizon_hours"]), int(row["test_seed"])),
            [],
        ).append(float(row["captured_t"]))
    for key, values in captured.items():
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
            raise ValueError(
                "controllers did not share the same exogenous captured-CO2 "
                f"realization for horizon/test seed {key}"
            )
    short_direct = {
        (row["model_seed"], int(row["test_seed"])): row
        for row in rows
        if row["controller"] == "iterative_q_direct"
        and int(row["horizon_hours"]) == 720
    }
    short_receding = {
        (row["model_seed"], int(row["test_seed"])): row
        for row in rows
        if row["controller"] == "iterative_q_receding"
        and int(row["horizon_hours"]) == 720
    }
    if set(short_direct) != set(short_receding):
        raise ValueError("720 h direct/receding Q grids do not match")
    for key in short_direct:
        for field in ("total_cost_eur", "stored_t", "vented_t"):
            if not np.isclose(
                float(short_direct[key][field]),
                float(short_receding[key][field]),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "receding adapter changed the 720 h frozen-policy "
                    f"reference for {key}/{field}"
                )
    return rows, metadata, manifest


def _model_grid(
    rows: list[dict[str, str]],
    controller: str,
    horizon: int,
    value_field: str,
) -> dict[str, dict[int, float]]:
    selected = [
        row
        for row in rows
        if row["controller"] == controller
        and int(row["horizon_hours"]) == horizon
    ]
    grid: dict[str, dict[int, float]] = {}
    for row in selected:
        model_seed = row["model_seed"] or "not_applicable"
        grid.setdefault(model_seed, {})[int(row["test_seed"])] = float(
            row[value_field]
        )
    return grid


def _paired_reduction_grid(
    rows: list[dict[str, str]],
    controller: str,
    horizon: int,
) -> dict[str, dict[int, float]]:
    fixed = {
        int(row["test_seed"]): float(row["total_cost_eur"])
        for row in rows
        if row["controller"] == "fixed_assignment"
        and int(row["horizon_hours"]) == horizon
    }
    selected = [
        row
        for row in rows
        if row["controller"] == controller
        and int(row["horizon_hours"]) == horizon
    ]
    grid: dict[str, dict[int, float]] = {}
    for row in selected:
        model_seed = row["model_seed"] or "not_applicable"
        test_seed = int(row["test_seed"])
        grid.setdefault(model_seed, {})[test_seed] = (
            100.0
            * (fixed[test_seed] - float(row["total_cost_eur"]))
            / fixed[test_seed]
        )
    return grid


def _hierarchical_mean_ci(
    grid: dict[str, dict[int, float]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    model_keys = tuple(sorted(grid))
    test_seeds = tuple(sorted(next(iter(grid.values()))))
    if any(set(values) != set(test_seeds) for values in grid.values()):
        raise ValueError("incomplete model/test-seed grid")
    point = float(
        np.mean(
            [
                grid[model][test_seed]
                for model in model_keys
                for test_seed in test_seeds
            ]
        )
    )
    estimates = np.empty(samples, dtype=float)
    learned = model_keys != ("not_applicable",)
    for index in range(samples):
        sampled_models = (
            rng.choice(model_keys, size=len(model_keys), replace=True)
            if learned
            else model_keys
        )
        sampled_seeds = rng.choice(
            test_seeds,
            size=len(test_seeds),
            replace=True,
        )
        estimates[index] = np.mean(
            [
                grid[str(model)][int(test_seed)]
                for model in sampled_models
                for test_seed in sampled_seeds
            ]
        )
    low, high = np.quantile(estimates, (0.025, 0.975))
    return point, float(low), float(high)


def _hierarchical_ratio_ci(
    numerator: dict[str, dict[int, float]],
    denominator: dict[str, dict[int, float]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if set(numerator) != set(denominator):
        raise ValueError("ratio grids have different model seeds")
    model_keys = tuple(sorted(numerator))
    test_seeds = tuple(sorted(next(iter(numerator.values()))))
    if any(
        set(numerator[model]) != set(test_seeds)
        or set(denominator[model]) != set(test_seeds)
        for model in model_keys
    ):
        raise ValueError("incomplete paired ratio grid")

    def estimate(models, seeds) -> float:
        numerator_values = [
            numerator[str(model)][int(test_seed)]
            for model in models
            for test_seed in seeds
        ]
        denominator_values = [
            denominator[str(model)][int(test_seed)]
            for model in models
            for test_seed in seeds
        ]
        return 100.0 * float(np.mean(numerator_values)) / float(
            np.mean(denominator_values)
        )

    point = estimate(model_keys, test_seeds)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_models = rng.choice(
            model_keys,
            size=len(model_keys),
            replace=True,
        )
        sampled_seeds = rng.choice(
            test_seeds,
            size=len(test_seeds),
            replace=True,
        )
        estimates[index] = estimate(sampled_models, sampled_seeds)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return point, float(low), float(high)


def _paired_q_difference_grids(
    rows: list[dict[str, str]],
    horizon: int,
) -> dict[str, dict[str, dict[int, float]]]:
    direct = {
        (row["model_seed"], int(row["test_seed"])): row
        for row in rows
        if row["controller"] == "iterative_q_direct"
        and int(row["horizon_hours"]) == horizon
    }
    receding = {
        (row["model_seed"], int(row["test_seed"])): row
        for row in rows
        if row["controller"] == "iterative_q_receding"
        and int(row["horizon_hours"]) == horizon
    }
    if set(direct) != set(receding):
        raise ValueError(f"Direct/receding paired grid mismatch at {horizon} h")
    grids: dict[str, dict[str, dict[int, float]]] = {
        "cost_saving_eur": {},
        "direct_cost_eur": {},
        "vent_decrease_t_per_720h": {},
        "receding_cost_win": {},
    }
    for (model_seed, test_seed), direct_row in direct.items():
        receding_row = receding[(model_seed, test_seed)]
        direct_cost = float(direct_row["total_cost_eur"])
        receding_cost = float(receding_row["total_cost_eur"])
        saving = direct_cost - receding_cost
        values = {
            "cost_saving_eur": saving,
            "direct_cost_eur": direct_cost,
            "vent_decrease_t_per_720h": (
                float(direct_row["normalized_vented_t_per_720h"])
                - float(receding_row["normalized_vented_t_per_720h"])
            ),
            "receding_cost_win": float(receding_cost < direct_cost),
        }
        for metric, value in values.items():
            grids[metric].setdefault(model_seed, {})[test_seed] = value
    return grids


def _direct_vs_receding_rows(
    rows: list[dict[str, str]],
    *,
    samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    comparison = []
    metric_names = (
        "cost_saving_eur",
        "vent_decrease_t_per_720h",
        "receding_cost_win",
    )
    for horizon_index, horizon in enumerate(HORIZONS):
        grids = _paired_q_difference_grids(rows, horizon)
        estimates = {}
        for metric_index, metric in enumerate(metric_names):
            estimates[metric] = _hierarchical_mean_ci(
                grids[metric],
                samples=samples,
                rng=np.random.default_rng(
                    bootstrap_seed + 100 + horizon_index * 10 + metric_index
                ),
            )
        saving, saving_low, saving_high = estimates["cost_saving_eur"]
        percent, percent_low, percent_high = _hierarchical_ratio_ci(
            grids["cost_saving_eur"],
            grids["direct_cost_eur"],
            samples=samples,
            rng=np.random.default_rng(
                bootstrap_seed + 100 + horizon_index * 10 + 3
            ),
        )
        vent, vent_low, vent_high = estimates[
            "vent_decrease_t_per_720h"
        ]
        win, win_low, win_high = estimates["receding_cost_win"]
        comparison.append(
            {
                "horizon_hours": horizon,
                "horizon_days": horizon / 24,
                "paired_records": 90,
                "mean_receding_cost_saving_eur": saving,
                "cost_saving_95pct_ci_low_eur": saving_low,
                "cost_saving_95pct_ci_high_eur": saving_high,
                "mean_receding_cost_saving_percent": percent,
                "cost_saving_95pct_ci_low_percent": percent_low,
                "cost_saving_95pct_ci_high_percent": percent_high,
                "mean_receding_vent_decrease_t_per_720h": vent,
                "vent_decrease_95pct_ci_low_t_per_720h": vent_low,
                "vent_decrease_95pct_ci_high_t_per_720h": vent_high,
                "receding_lower_cost_rate_percent": 100.0 * win,
                "lower_cost_rate_95pct_ci_low_percent": 100.0 * win_low,
                "lower_cost_rate_95pct_ci_high_percent": 100.0 * win_high,
            }
        )
    return comparison


def _summary_rows(
    rows: list[dict[str, str]],
    *,
    samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(HORIZONS):
        for controller in CONTROLLERS:
            selected = [
                row
                for row in rows
                if row["controller"] == controller
                and int(row["horizon_hours"]) == horizon
            ]
            if controller == "fixed_assignment":
                reduction = reduction_low = reduction_high = 0.0
            else:
                reduction, reduction_low, reduction_high = (
                    _hierarchical_mean_ci(
                        _paired_reduction_grid(rows, controller, horizon),
                        samples=samples,
                        rng=np.random.default_rng(
                            bootstrap_seed + horizon_index * 10
                        ),
                    )
                )
            vent, vent_low, vent_high = _hierarchical_mean_ci(
                _model_grid(
                    rows,
                    controller,
                    horizon,
                    "normalized_vented_t_per_720h",
                ),
                samples=samples,
                rng=np.random.default_rng(
                    bootstrap_seed + horizon_index * 10 + 1
                ),
            )
            normalized_cost, cost_low, cost_high = _hierarchical_mean_ci(
                _model_grid(
                    rows,
                    controller,
                    horizon,
                    "normalized_cost_eur_per_720h",
                ),
                samples=samples,
                rng=np.random.default_rng(
                    bootstrap_seed + horizon_index * 10 + 2
                ),
            )
            summary.append(
                {
                    "controller": controller,
                    "controller_display_name": DISPLAY_NAMES[controller],
                    "horizon_hours": horizon,
                    "horizon_days": horizon / 24,
                    "records": len(selected),
                    "mean_total_cost_eur": float(
                        np.mean(
                            [float(row["total_cost_eur"]) for row in selected]
                        )
                    ),
                    "mean_normalized_cost_eur_per_720h": normalized_cost,
                    "normalized_cost_95pct_ci_low_eur_per_720h": cost_low,
                    "normalized_cost_95pct_ci_high_eur_per_720h": cost_high,
                    "mean_cost_reduction_vs_fixed_percent": reduction,
                    "cost_reduction_95pct_ci_low_percent": reduction_low,
                    "cost_reduction_95pct_ci_high_percent": reduction_high,
                    "mean_normalized_vented_t_per_720h": vent,
                    "normalized_vented_95pct_ci_low_t_per_720h": vent_low,
                    "normalized_vented_95pct_ci_high_t_per_720h": vent_high,
                    "mean_storage_rate": float(
                        np.mean([float(row["storage_rate"]) for row in selected])
                    ),
                    "mean_captured_t": float(
                        np.mean([float(row["captured_t"]) for row in selected])
                    ),
                    "mean_stored_t": float(
                        np.mean([float(row["stored_t"]) for row in selected])
                    ),
                    "mean_vented_t": float(
                        np.mean([float(row["vented_t"]) for row in selected])
                    ),
                    "mean_unit_cost_eur_per_captured_t": float(
                        np.mean(
                            [
                                float(row["unit_cost_eur_per_captured_t"])
                                for row in selected
                            ]
                        )
                    ),
                    "mean_episode_wall_time_s": float(
                        np.mean(
                            [float(row["episode_wall_time_s"]) for row in selected]
                        )
                    ),
                    "mean_override_events": (
                        float(
                            np.mean(
                                [
                                    float(row["override_events"])
                                    for row in selected
                                ]
                            )
                        )
                        if controller.startswith("iterative_q")
                        else ""
                    ),
                }
            )
    return summary


def _style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold", pad=5)
    ax.set_ylabel(ylabel, fontsize=7.8)
    ax.tick_params(labelsize=7.2, length=3.0, width=0.6)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def _draw_figure(
    summary: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 82 / 25.4),
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.17,
        top=0.72,
        wspace=0.20,
    )
    positions = np.arange(len(HORIZONS), dtype=float)
    horizon_labels = tuple(f"{horizon // 24} days" for horizon in HORIZONS)
    plotted = (
        "greedy",
        "iterative_q_receding",
    )
    for controller in plotted:
        selected = [
            row for row in summary if row["controller"] == controller
        ]
        reduction = np.asarray(
            [float(row["mean_cost_reduction_vs_fixed_percent"]) for row in selected]
        )
        reduction_low = np.asarray(
            [float(row["cost_reduction_95pct_ci_low_percent"]) for row in selected]
        )
        reduction_high = np.asarray(
            [float(row["cost_reduction_95pct_ci_high_percent"]) for row in selected]
        )
        axes[0].errorbar(
            positions,
            reduction,
            yerr=np.vstack((reduction - reduction_low, reduction_high - reduction)),
            color=COLORS[controller],
            marker=MARKERS[controller],
            markersize=4.8,
            linewidth=1.4,
            capsize=2.2,
            capthick=0.65,
            label=(
                "Iterative Action-Q"
                if controller == "iterative_q_receding"
                else DISPLAY_NAMES[controller]
            ),
        )
        vent = np.asarray(
            [float(row["mean_normalized_vented_t_per_720h"]) for row in selected]
        )
        vent_low = np.asarray(
            [
                float(row["normalized_vented_95pct_ci_low_t_per_720h"])
                for row in selected
            ]
        )
        vent_high = np.asarray(
            [
                float(row["normalized_vented_95pct_ci_high_t_per_720h"])
                for row in selected
            ]
        )
        axes[1].errorbar(
            positions,
            vent,
            yerr=np.vstack((vent - vent_low, vent_high - vent)),
            color=COLORS[controller],
            marker=MARKERS[controller],
            markersize=4.8,
            linewidth=1.4,
            capsize=2.2,
            capthick=0.65,
            label=(
                "Iterative Action-Q"
                if controller == "iterative_q_receding"
                else DISPLAY_NAMES[controller]
            ),
        )
    axes[0].axhline(0, color="#888888", linewidth=0.7, linestyle=":")
    _style_axis(
        axes[0],
        "a  Cost advantage over Fixed-Assignment",
        "Paired cost reduction (%)",
    )
    _style_axis(
        axes[1],
        "b  CO$_2$ venting at longer horizons",
        "Vented CO$_2$ (t per 720 h)",
    )
    for ax in axes:
        ax.set_xticks(positions, horizon_labels)
        ax.set_xlabel("Evaluation horizon", fontsize=7.8)
    fig.suptitle(
        "Frozen Iterative-Q temporal deployment from 30 days to one year",
        y=0.985,
        fontsize=10.0,
        fontweight="bold",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
        fontsize=7.2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, dpi in (("svg", 300), ("pdf", 300), ("tiff", 600), ("png", 300)):
        path = output_dir / (
            f"e7_temporal_generalization_iterative_action_q.{suffix}"
        )
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        paths.append(path)
    plt.close(fig)
    return paths


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata, manifest = _load_and_validate(
        args.input_root,
        args.direct_input_root,
        args.additional_input_root,
    )
    summary = _summary_rows(
        rows,
        samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    direct_comparison = _direct_vs_receding_rows(
        rows,
        samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_csv(args.output_dir / "e7_per_episode.csv", rows)
    _write_csv(args.output_dir / "e7_summary.csv", summary)
    _write_csv(
        args.output_dir / "e7_direct_vs_receding.csv",
        direct_comparison,
    )
    markdown = [
        "| Method | Horizon | Mean total cost (EUR) | Cost / 720 h (EUR) | Cost reduction vs Fixed (95% CI) | Vent / 720 h (t) | Storage rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        markdown.append(
            "| {controller_display_name} | {horizon_days:.0f} d | "
            "{mean_total_cost_eur:,.0f} | "
            "{mean_normalized_cost_eur_per_720h:,.0f} | "
            "{mean_cost_reduction_vs_fixed_percent:.2f}% "
            "[{cost_reduction_95pct_ci_low_percent:.2f}, "
            "{cost_reduction_95pct_ci_high_percent:.2f}] | "
            "{mean_normalized_vented_t_per_720h:,.1f} | "
            "{mean_storage_rate:.3f} |".format(**row)
        )
    (args.output_dir / "e7_summary.md").write_text(
        "\n".join(markdown)
        + "\n\nAll costs and venting are normalized to 720 h. "
        "Direct-global and receding-cyclic use identical repeated policy windows; "
        "their only controller difference is episode progress t / H versus "
        "(t mod 720) / 720. "
        "Rolling MILP is intentionally excluded from this initial E7 run.\n",
        encoding="utf-8",
    )
    comparison_markdown = [
        "| Horizon | Receding cost saving vs Direct-global (95% CI) | Mean saving (EUR) | Vent decrease / 720 h (t) | Receding lower-cost rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in direct_comparison:
        comparison_markdown.append(
            "| {horizon_days:.0f} d | "
            "{mean_receding_cost_saving_percent:.3f}% "
            "[{cost_saving_95pct_ci_low_percent:.3f}, "
            "{cost_saving_95pct_ci_high_percent:.3f}] | "
            "{mean_receding_cost_saving_eur:,.0f} | "
            "{mean_receding_vent_decrease_t_per_720h:,.1f} | "
            "{receding_lower_cost_rate_percent:.1f}% |".format(**row)
        )
    (args.output_dir / "e7_direct_vs_receding.md").write_text(
        "\n".join(comparison_markdown)
        + "\n\nPositive values favour receding-cyclic. Confidence intervals use "
        "hierarchical bootstrap resampling over model and test seeds.\n",
        encoding="utf-8",
    )
    figure_paths = _draw_figure(summary, args.output_dir / "figures")
    audit = {
        "experiment": "E7 temporal-horizon generalization",
        "input_root": str(args.input_root),
        "direct_input_root": str(args.direct_input_root),
        "additional_input_root": str(args.additional_input_root),
        "controllers": list(CONTROLLERS),
        "horizons_hours": list(HORIZONS),
        "rolling_milp_included": False,
        "same_8928h_scenario_then_prefix_truncation": True,
        "frozen_q_weights": True,
        "direct_and_receding_share_policy_windows": True,
        "only_q_difference": "episode progress: t / H vs (t mod 720) / 720",
        "iterative_q_state_feature_exclusions": ["hour_of_week"],
        "iterative_q_state_feature_count": 93,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "records": len(rows),
        "direct_vs_receding_paired_records_per_horizon": 90,
        "shards": manifest,
        "metadata_records": len(metadata),
        "figures": [
            str(path.relative_to(args.output_dir)) for path in figure_paths
        ],
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    rows = run(parse_args())
    print(f"E7_AGGREGATION_COMPLETE summary_rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
