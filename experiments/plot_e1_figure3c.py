from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "formal_comparison"
    / "e1_formal_per_episode.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments_results" / "E1" / "figures"

METHODS = (
    "fixed_assignment",
    "greedy",
    "ppo_hourly",
    "ppo_high_level",
    "iterative_action_q_g60_p4",
    "rolling_milp",
)
DISPLAY_NAMES = {
    "fixed_assignment": "Fixed-Assignment",
    "greedy": "Greedy",
    "ppo_hourly": "Hourly PPO",
    "ppo_high_level": "High-level PPO",
    "iterative_action_q_g60_p4": "Iterative Action-Q",
    "rolling_milp": "Rolling MILP (600 s/replan)",
}
COMPONENTS = (
    (
        "episode_vessel_fuel_eur",
        "mean_vessel_fuel_eur_per_captured_t",
        "Vessel fuel",
        "#596A9E",
    ),
    (
        "episode_conditioning_eur",
        "mean_conditioning_eur_per_captured_t",
        "Conditioning",
        "#8CA6D8",
    ),
    (
        "episode_reconditioning_eur",
        "mean_reconditioning_eur_per_captured_t",
        "Reconditioning",
        "#A894C7",
    ),
    (
        "episode_loading_eur",
        "mean_loading_eur_per_captured_t",
        "Loading",
        "#63A8A6",
    ),
    (
        "episode_unloading_eur",
        "mean_unloading_eur_per_captured_t",
        "Unloading",
        "#9CC8B6",
    ),
    (
        "episode_vent_penalty_eur",
        "mean_vent_penalty_eur_per_captured_t",
        "Vent penalty",
        "#D58CA3",
    ),
    (
        "terminal_cleanup_operating_cost_eur",
        "mean_terminal_cleanup_eur_per_captured_t",
        "Terminal cleanup",
        "#D3D7E8",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create E1 Figure 3c from episode-level formal results."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_rows(input_csv: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {
        algorithm: [] for algorithm in METHODS
    }
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    for row in all_rows:
        algorithm = row["algorithm"]
        if algorithm in grouped:
            grouped[algorithm].append(row)

    missing = [algorithm for algorithm, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"Missing formal E1 algorithms: {missing}")

    captured_by_test_seed: dict[str, float] = {}
    for row in grouped["greedy"]:
        test_seed = row["test_seed"]
        if test_seed in captured_by_test_seed:
            raise ValueError(f"Duplicate Greedy test seed: {test_seed}")
        captured_t = float(row["captured_t"])
        if captured_t <= 0.0:
            raise ValueError(
                f"Greedy has non-positive captured CO2 for test seed {test_seed}."
            )
        captured_by_test_seed[test_seed] = captured_t

    output: list[dict[str, object]] = []
    for algorithm in METHODS:
        episode_rows = grouped[algorithm]
        component_unit_costs = {
            output_field: [] for _input_field, output_field, _label, _color in COMPONENTS
        }
        total_unit_costs: list[float] = []
        model_seeds: set[str] = set()

        for row in episode_rows:
            test_seed = row["test_seed"]
            captured_t = captured_by_test_seed[test_seed]
            exported_captured_t = row["captured_t"].strip()
            if exported_captured_t and not np.isclose(
                float(exported_captured_t),
                captured_t,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{algorithm} captured CO2 differs from the paired exogenous "
                    f"scenario for test seed {test_seed}."
                )
            storage_shortfall = float(
                row["episode_storage_shortfall_penalty_eur"]
            )
            if abs(storage_shortfall) > 1e-6:
                raise ValueError(
                    f"{algorithm} has a non-zero storage-shortfall component "
                    "that Figure 3c does not encode."
                )

            component_values = {
                output_field: float(row[input_field])
                for input_field, output_field, _label, _color in COMPONENTS
            }
            total_cost = float(row["total_cost_eur"])
            if not np.isclose(
                total_cost,
                sum(component_values.values()),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{algorithm} episode {row['test_seed']} cost components "
                    "do not sum to total cost."
                )

            for output_field, value in component_values.items():
                component_unit_costs[output_field].append(value / captured_t)
            total_unit_costs.append(total_cost / captured_t)
            if row["model_seed"]:
                model_seeds.add(row["model_seed"])

        mean_components = {
            field: float(np.mean(values))
            for field, values in component_unit_costs.items()
        }
        mean_total = float(np.mean(total_unit_costs))
        if not np.isclose(
            mean_total,
            sum(mean_components.values()),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                f"{algorithm} unit-cost components do not sum to the mean "
                f"unit total cost: {sum(mean_components.values())} vs "
                f"{mean_total}."
            )

        output.append(
            {
                "algorithm": algorithm,
                "algorithm_display_name": DISPLAY_NAMES[algorithm],
                **mean_components,
                "mean_unit_total_cost_eur_per_captured_t": mean_total,
                "episode_records": len(episode_rows),
                "model_instance_count": max(1, len(model_seeds)),
            }
        )

    output.sort(
        key=lambda item: float(
            item["mean_unit_total_cost_eur_per_captured_t"]
        )
    )
    return output


def write_source_data(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    width_in = 89.0 / 25.4
    height_in = 82.0 / 25.4
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(rows))
    left = np.zeros(len(rows), dtype=float)
    for _input_field, output_field, label, color in COMPONENTS:
        values = np.asarray([float(row[output_field]) for row in rows])
        ax.barh(
            y,
            values,
            left=left,
            height=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        left += values

    max_total = float(left.max())
    for yi, total in zip(y, left):
        ax.text(
            total + 0.025 * max_total,
            yi,
            f"€{total:.1f}/t",
            ha="left",
            va="center",
            fontsize=5.5,
            color="#272727",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [str(row["algorithm_display_name"]) for row in rows],
        fontsize=6.2,
    )
    ax.invert_yaxis()
    ax.set_xlim(0.0, max_total * 1.17)
    ax.set_xlabel(
        r"Mean unit cost (€ per captured t CO$_2$)",
        fontsize=6.2,
        labelpad=4,
    )
    ax.tick_params(axis="x", labelsize=5.8, length=2.5, width=0.6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        frameon=False,
        fontsize=4.8,
        handlelength=1.1,
        columnspacing=0.65,
        handletextpad=0.32,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.text(0.015, 0.975, "c", ha="left", va="top", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.36, right=0.985, top=0.94, bottom=0.30)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "figure_3c_unit_cost_decomposition"
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    source_data_dir = args.output_dir / "source_data"
    source_csv = source_data_dir / "figure_3c_unit_cost_decomposition.csv"
    metadata_json = source_data_dir / "figure_3c_metadata.json"
    write_source_data(source_csv, rows)
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "figure": "Figure 3c",
                "method_count": len(rows),
                "sort_order": "ascending mean episode-level unit total cost",
                "normalization": (
                    "Each component is divided by all captured_t entering the "
                    "system within each episode before averaging across episode "
                    "records. Missing Iterative Action-Q captured_t values use "
                    "the paired exogenous-scenario value from the same test seed."
                ),
                "component_definition": {
                    output_field: label
                    for _input_field, output_field, label, _color in COMPONENTS
                },
                "rolling_milp_time_limit_seconds_per_replan": 600,
                "input_csv": str(args.input_csv.relative_to(REPO_ROOT)),
                "output_formats": ["pdf", "png"],
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    outputs = draw_figure(rows, args.output_dir)
    print(f"Wrote {source_csv}")
    print(f"Wrote {metadata_json}")
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
