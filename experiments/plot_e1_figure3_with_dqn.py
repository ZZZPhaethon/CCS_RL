from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import plot_e1_figure3a as figure3a
import plot_e1_figure3b as figure3b
import plot_e1_figure3c as figure3c


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE_INPUT = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "algorithms"
    / "formal_comparison"
    / "e1_formal_per_episode.csv"
)
DEFAULT_ALGORITHM_INPUT = DEFAULT_EPISODE_INPUT.with_name(
    "e1_formal_per_algorithm.csv"
)
DEFAULT_DQN_RESULTS = (
    REPO_ROOT
    / "experiments_results"
    / "E1_addendum_masked_double_dqn_20260804"
    / "formal_test"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments_results" / "E1" / "figures_with_dqn"
)

DQN_ALGORITHM = "masked_double_dqn"
DQN_DISPLAY_NAME = "Masked Double DQN"
EXPECTED_MODEL_SEEDS = (0, 1, 2)
EXPECTED_TEST_SEEDS = tuple(range(9000031, 9000061))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create E1 Figure 3a-c with the post-hoc Masked Double-DQN baseline."
    )
    parser.add_argument(
        "--episode-input-csv", type=Path, default=DEFAULT_EPISODE_INPUT
    )
    parser.add_argument(
        "--algorithm-input-csv", type=Path, default=DEFAULT_ALGORITHM_INPUT
    )
    parser.add_argument(
        "--dqn-results-dir", type=Path, default=DEFAULT_DQN_RESULTS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty source-data table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_dqn_episode_rows(
    reference_rows: list[dict[str, str]],
    dqn_results_dir: Path,
) -> list[dict[str, object]]:
    headers = list(reference_rows[0])
    greedy_by_test_seed = {
        int(row["test_seed"]): float(row["total_cost_eur"])
        for row in reference_rows
        if row["algorithm"] == "greedy"
    }
    if set(greedy_by_test_seed) != set(EXPECTED_TEST_SEEDS):
        raise ValueError("Greedy reference does not cover the locked E1 test seeds.")

    output: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for model_seed in EXPECTED_MODEL_SEEDS:
        evaluation_csv = dqn_results_dir / f"model_seed_{model_seed}" / "evaluation.csv"
        rows = read_csv(evaluation_csv)
        actual_test_seeds = tuple(int(row["seed"]) for row in rows)
        if actual_test_seeds != EXPECTED_TEST_SEEDS:
            raise ValueError(
                f"DQN model seed {model_seed} has incorrect formal-test coverage."
            )

        for raw in rows:
            test_seed = int(raw["seed"])
            key = (model_seed, test_seed)
            if key in seen:
                raise ValueError(f"Duplicate DQN evaluation row: {key}")
            seen.add(key)

            row: dict[str, object] = {field: "" for field in headers}
            for field in headers:
                if field in raw:
                    row[field] = raw[field]
            total_cost = float(raw["total_cost_eur"])
            greedy_cost = greedy_by_test_seed[test_seed]
            delta = total_cost - greedy_cost
            row.update(
                {
                    "algorithm": DQN_ALGORITHM,
                    "algorithm_display_name": DQN_DISPLAY_NAME,
                    "method_class": "reinforcement_learning",
                    "model_seed": model_seed,
                    "test_seed": test_seed,
                    "episode_hours": int(round(float(raw["simulated_hours"]))),
                    "decision_count": int(raw["decisions"]),
                    "mean_decision_interval_h": (
                        float(raw["simulated_hours"]) / int(raw["decisions"])
                    ),
                    "greedy_total_cost_eur": greedy_cost,
                    "delta_total_cost_vs_greedy_eur": delta,
                    "paired_outcome_vs_greedy": "win" if delta < 0 else "loss",
                    "source_file": str(
                        evaluation_csv.relative_to(REPO_ROOT)
                    ).replace("\\", "/"),
                }
            )
            output.append(row)

    expected_records = len(EXPECTED_MODEL_SEEDS) * len(EXPECTED_TEST_SEEDS)
    if len(output) != expected_records:
        raise ValueError(f"Expected {expected_records} DQN records; found {len(output)}.")
    return output


AGGREGATE_FIELDS = {
    "mean_episode_vessel_fuel_eur": "episode_vessel_fuel_eur",
    "mean_episode_conditioning_eur": "episode_conditioning_eur",
    "mean_episode_reconditioning_eur": "episode_reconditioning_eur",
    "mean_episode_loading_eur": "episode_loading_eur",
    "mean_episode_unloading_eur": "episode_unloading_eur",
    "mean_episode_operating_cost_eur": "episode_operating_cost_eur",
    "mean_episode_vent_penalty_eur": "episode_vent_penalty_eur",
    "mean_episode_storage_shortfall_penalty_eur": (
        "episode_storage_shortfall_penalty_eur"
    ),
    "mean_episode_total_cost_eur": "episode_total_cost_eur",
    "mean_terminal_cleanup_operating_cost_eur": (
        "terminal_cleanup_operating_cost_eur"
    ),
    "mean_operating_cost_eur": "operating_cost_eur",
    "mean_total_cost_eur": "total_cost_eur",
    "mean_captured_t": "captured_t",
    "mean_stored_t": "stored_t",
    "mean_vented_t": "vented_t",
    "mean_storage_rate": "storage_rate",
    "mean_loss_rate": "loss_rate",
    "mean_unit_total_cost_eur_per_t": "unit_total_cost_eur_per_t",
    "mean_delta_total_cost_vs_greedy_eur": "delta_total_cost_vs_greedy_eur",
}


def build_dqn_algorithm_row(
    algorithm_headers: list[str],
    dqn_rows: list[dict[str, object]],
) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in algorithm_headers}
    row.update(
        {
            "algorithm": DQN_ALGORITHM,
            "algorithm_display_name": DQN_DISPLAY_NAME,
            "method_class": "reinforcement_learning",
            "model_instances": len(EXPECTED_MODEL_SEEDS),
            "trained_model_seed_count": len(EXPECTED_MODEL_SEEDS),
            "episode_records": len(dqn_rows),
        }
    )
    for aggregate_field, episode_field in AGGREGATE_FIELDS.items():
        row[aggregate_field] = float(
            np.mean([float(item[episode_field]) for item in dqn_rows])
        )

    total_costs = np.asarray(
        [float(item["total_cost_eur"]) for item in dqn_rows], dtype=float
    )
    model_means = np.asarray(
        [
            np.mean(
                [
                    float(item["total_cost_eur"])
                    for item in dqn_rows
                    if int(item["model_seed"]) == model_seed
                ]
            )
            for model_seed in EXPECTED_MODEL_SEEDS
        ],
        dtype=float,
    )
    deltas = np.asarray(
        [float(item["delta_total_cost_vs_greedy_eur"]) for item in dqn_rows],
        dtype=float,
    )
    tolerance = 1e-9
    row.update(
        {
            "pooled_episode_sd_total_cost_eur": float(total_costs.std(ddof=1)),
            "between_model_seed_sd_mean_total_cost_eur": float(
                model_means.std(ddof=1)
            ),
            "wins_vs_greedy": int(np.count_nonzero(deltas < -tolerance)),
            "ties_vs_greedy": int(np.count_nonzero(np.abs(deltas) <= tolerance)),
            "losses_vs_greedy": int(np.count_nonzero(deltas > tolerance)),
        }
    )
    return row


def configure_figure_modules() -> None:
    figure3a.METHODS = (
        "greedy",
        "ppo_hourly",
        "ppo_high_level",
        DQN_ALGORITHM,
        "iterative_action_q_g60_p4",
        "rolling_milp",
    )
    figure3a.LEARNED_METHODS = {
        "ppo_hourly",
        "ppo_high_level",
        DQN_ALGORITHM,
        "iterative_action_q_g60_p4",
    }
    figure3a.DISPLAY_NAMES[DQN_ALGORITHM] = DQN_DISPLAY_NAME
    figure3a.METHOD_CLASSES[DQN_ALGORITHM] = "reinforcement_learning"
    figure3a.COLORS[DQN_ALGORITHM] = "#C4A46B"

    for module in (figure3b, figure3c):
        module.METHODS = (*module.METHODS, DQN_ALGORITHM)
        module.DISPLAY_NAMES[DQN_ALGORITHM] = DQN_DISPLAY_NAME


def build_composite_preview(output_dir: Path) -> Path:
    panel_a = Image.open(
        output_dir / "figure_3a_fixed_assignment_baseline.png"
    ).convert("RGB")
    panel_b = Image.open(output_dir / "figure_3b_cost_decomposition.png").convert(
        "RGB"
    )
    panel_c = Image.open(
        output_dir / "figure_3c_unit_cost_decomposition.png"
    ).convert("RGB")
    gap = 24
    target_width = panel_b.width + gap + panel_c.width
    target_height = round(panel_a.height * target_width / panel_a.width)
    panel_a = panel_a.resize(
        (target_width, target_height), resample=Image.Resampling.LANCZOS
    )
    canvas = Image.new(
        "RGB",
        (target_width, target_height + gap + max(panel_b.height, panel_c.height)),
        "white",
    )
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (0, target_height + gap))
    canvas.paste(panel_c, (panel_b.width + gap, target_height + gap))
    output = output_dir / "figure_3_with_dqn_preview.png"
    canvas.save(output, dpi=(300, 300))
    return output


def main() -> None:
    args = parse_args()
    reference_episode_rows = read_csv(args.episode_input_csv)
    reference_algorithm_rows = read_csv(args.algorithm_input_csv)
    dqn_rows = build_dqn_episode_rows(reference_episode_rows, args.dqn_results_dir)

    source_dir = args.output_dir / "source_data"
    combined_episode_csv = source_dir / "e1_formal_per_episode_with_dqn.csv"
    combined_algorithm_csv = source_dir / "e1_formal_per_algorithm_with_dqn.csv"
    combined_episode_rows: list[dict[str, object]] = [
        *reference_episode_rows,
        *dqn_rows,
    ]
    dqn_algorithm_row = build_dqn_algorithm_row(
        list(reference_algorithm_rows[0]), dqn_rows
    )
    combined_algorithm_rows: list[dict[str, object]] = [
        *reference_algorithm_rows,
        dqn_algorithm_row,
    ]
    write_csv(combined_episode_csv, combined_episode_rows)
    write_csv(combined_algorithm_csv, combined_algorithm_rows)

    configure_figure_modules()

    costs = figure3a.load_costs(combined_episode_csv)
    scenario_rows, summary_rows = figure3a.build_source_data(costs)
    figure3a.write_csv(
        source_dir / "figure_3a_paired_scenario_differences.csv", scenario_rows
    )
    figure3a.write_csv(
        source_dir / "figure_3a_summary_statistics.csv", summary_rows
    )
    outputs = figure3a.draw_figure(scenario_rows, summary_rows, args.output_dir)

    decomposition_rows = figure3b.load_rows(combined_algorithm_csv)
    figure3b.write_source_data(
        source_dir / "figure_3b_cost_decomposition.csv", decomposition_rows
    )
    outputs.extend(figure3b.draw_figure(decomposition_rows, args.output_dir))

    unit_cost_rows = figure3c.load_rows(combined_episode_csv)
    figure3c.write_source_data(
        source_dir / "figure_3c_unit_cost_decomposition.csv", unit_cost_rows
    )
    outputs.extend(figure3c.draw_figure(unit_cost_rows, args.output_dir))
    composite_preview = build_composite_preview(args.output_dir)

    metadata = {
        "figure": "E1 Figure 3a-c with post-hoc Masked Double DQN",
        "backend": "Python/matplotlib",
        "formal_test_seeds": [EXPECTED_TEST_SEEDS[0], EXPECTED_TEST_SEEDS[-1]],
        "model_seeds": list(EXPECTED_MODEL_SEEDS),
        "dqn_episode_records": len(dqn_rows),
        "point_definition_figure_3a": (
            "Per-test-scenario paired total-cost difference; learned-method points "
            "average model seeds 0, 1, and 2."
        ),
        "interval_definition_figure_3a": (
            "95% percentile hierarchical bootstrap confidence interval for the mean; "
            "test scenarios and learned model instances are resampled independently."
        ),
        "posthoc_disclosure": (
            "Masked Double DQN was locked and evaluated as a post-hoc addendum after "
            "the original E1 formal test set had been accessed for other controllers."
        ),
        "reference_episode_csv": str(args.episode_input_csv.relative_to(REPO_ROOT)),
        "reference_algorithm_csv": str(
            args.algorithm_input_csv.relative_to(REPO_ROOT)
        ),
        "dqn_results_dir": str(args.dqn_results_dir.relative_to(REPO_ROOT)),
        "output_formats": ["pdf", "svg", "png"],
    }
    metadata_json = source_dir / "figure_3_with_dqn_metadata.json"
    metadata_json.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {combined_episode_csv}")
    print(f"Wrote {combined_algorithm_csv}")
    print(f"Wrote {metadata_json}")
    for output in outputs:
        print(f"Wrote {output}")
    print(f"Wrote {composite_preview}")


if __name__ == "__main__":
    main()
