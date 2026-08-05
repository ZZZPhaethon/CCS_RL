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
    / "algorithms"
    / "formal_comparison"
    / "e1_formal_per_algorithm.csv"
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
    ("mean_episode_vessel_fuel_eur", "Vessel fuel", "#596A9E"),
    ("mean_episode_conditioning_eur", "Conditioning", "#8CA6D8"),
    ("mean_episode_reconditioning_eur", "Reconditioning", "#A894C7"),
    ("mean_episode_loading_eur", "Loading", "#63A8A6"),
    ("mean_episode_unloading_eur", "Unloading", "#9CC8B6"),
    ("mean_episode_vent_penalty_eur", "Vent penalty", "#D58CA3"),
    (
        "mean_terminal_cleanup_operating_cost_eur",
        "Terminal cleanup",
        "#D3D7E8",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create E1 Figure 3b from the formal comparison dataset."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_rows(input_csv: Path) -> list[dict[str, object]]:
    by_algorithm: dict[str, dict[str, str]] = {}
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            algorithm = row["algorithm"]
            if algorithm in METHODS:
                if algorithm in by_algorithm:
                    raise ValueError(f"Duplicate algorithm row: {algorithm}")
                by_algorithm[algorithm] = row

    if set(by_algorithm) != set(METHODS):
        missing = sorted(set(METHODS) - set(by_algorithm))
        raise ValueError(f"Missing formal E1 algorithms: {missing}")

    output: list[dict[str, object]] = []
    for algorithm in METHODS:
        row = by_algorithm[algorithm]
        storage_shortfall = float(
            row["mean_episode_storage_shortfall_penalty_eur"]
        )
        if abs(storage_shortfall) > 1e-6:
            raise ValueError(
                f"{algorithm} has a non-zero storage-shortfall component "
                "that Figure 3b does not encode."
            )

        values = {
            field: float(row[field])
            for field, _label, _color in COMPONENTS
        }
        total = float(row["mean_total_cost_eur"])
        component_sum = sum(values.values())
        if not np.isclose(total, component_sum, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{algorithm} cost decomposition does not sum to total cost: "
                f"{component_sum} vs {total}."
            )

        output.append(
            {
                "algorithm": algorithm,
                "algorithm_display_name": DISPLAY_NAMES[algorithm],
                **values,
                "mean_total_cost_eur": total,
                "episode_records": int(row["episode_records"]),
                "model_instance_count": int(row["model_instances"]),
            }
        )

    output.sort(key=lambda item: float(item["mean_total_cost_eur"]))
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
    for field, label, color in COMPONENTS:
        values = np.asarray([float(row[field]) for row in rows]) / 1_000_000.0
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
            total + 0.055 * max_total,
            yi,
            f"€{total:.2f}M",
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
    ax.set_xlim(0.0, max_total * 1.24)
    ax.set_xlabel("Mean total cost (€ million)", fontsize=6.2, labelpad=4)
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
    fig.text(0.015, 0.975, "b", ha="left", va="top", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.36, right=0.985, top=0.94, bottom=0.30)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "figure_3b_cost_decomposition"
    outputs = [
        base.with_suffix(".pdf"),
        base.with_suffix(".svg"),
        base.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    source_data_dir = args.output_dir / "source_data"
    source_csv = source_data_dir / "figure_3b_cost_decomposition.csv"
    metadata_json = source_data_dir / "figure_3b_metadata.json"
    write_source_data(source_csv, rows)
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "figure": "Figure 3b",
                "method_count": len(rows),
                "sort_order": "ascending mean total cost",
                "component_definition": {
                    field: label for field, label, _color in COMPONENTS
                },
                "rolling_milp_time_limit_seconds_per_replan": 600,
                "input_csv": str(args.input_csv.relative_to(REPO_ROOT)),
                "output_formats": ["pdf", "svg", "png"],
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
