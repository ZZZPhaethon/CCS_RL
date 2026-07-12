"""Generate paired A/B/C summaries for the BC objective ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


OBJECTIVES = ("current", "decision_only", "decision_balanced")
VARIANTS = ("state_mode", "tcn_mode")
COMPARISONS = (
    ("decision_only", "current"),
    ("decision_balanced", "current"),
    ("decision_balanced", "decision_only"),
)
METRICS = ("vented_t", "stored_t", "total_cost")
DEMO_METRICS = (
    "voluntary_wait_accuracy",
    "dispatch_recall",
    "conditional_destination_accuracy",
    "mean_wait_probability",
)
ROLLOUT_METRICS = (
    "dispatch_count",
    "partial_load_departure_count",
    "milk_run_departure_count",
    "longest_berthed_no_dispatch_streak",
    "mean_wait_probability",
)
T_CRITICAL_DF4 = 2.7764451051977987


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


def _key(row) -> tuple[str, bool, int, int]:
    return (
        str(row["variant"]),
        _bool_value(row["deterministic"]),
        int(row["model_seed"]),
        int(row["eval_seed"]),
    )


def _metric_maps(rows_by_objective, metric: str):
    maps = {}
    for objective in OBJECTIVES:
        mapping = {}
        for row in rows_by_objective[objective]:
            if row.get("stage") != "bc" or row.get("variant") not in VARIANTS:
                continue
            key = _key(row)
            if key in mapping:
                raise ValueError(f"duplicate pairing key for {objective}: {key}")
            mapping[key] = float(row[metric])
        maps[objective] = mapping
    reference = set(maps["current"])
    if not reference:
        raise ValueError("missing paired keys: current has no BC mode rows")
    for objective in OBJECTIVES[1:]:
        keys = set(maps[objective])
        if keys != reference:
            raise ValueError(
                f"missing paired keys for {objective}: "
                f"missing={sorted(reference - keys)}, unmatched={sorted(keys - reference)}"
            )
    return maps


def paired_metric_rows(rows_by_objective, metric: str) -> list[dict[str, object]]:
    maps = _metric_maps(rows_by_objective, metric)
    output = []
    for treatment, baseline in COMPARISONS:
        for variant in VARIANTS:
            for deterministic in (False, True):
                model_deltas = []
                for model_seed in range(5):
                    keys = [
                        key
                        for key in maps["current"]
                        if key[0] == variant
                        and key[1] is deterministic
                        and key[2] == model_seed
                    ]
                    if not keys:
                        continue
                    model_deltas.append(
                        statistics.fmean(
                            maps[treatment][key] - maps[baseline][key]
                            for key in keys
                        )
                    )
                if not model_deltas:
                    continue
                model_sd = statistics.stdev(model_deltas) if len(model_deltas) > 1 else 0.0
                interval = (
                    T_CRITICAL_DF4 * model_sd / len(model_deltas) ** 0.5
                    if len(model_deltas) == 5
                    else 0.0
                )
                output.append(
                    {
                        "metric": metric,
                        "comparison": f"{treatment}-{baseline}",
                        "variant": variant,
                        "deterministic": deterministic,
                        "model_seeds": len(model_deltas),
                        "mean_delta": statistics.fmean(model_deltas),
                        "model_sd": model_sd,
                        "ci95_half_width": interval,
                        "per_model_deltas": json.dumps(model_deltas, separators=(",", ":")),
                    }
                )
    return output


def load_result_rows(directory: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(Path(directory).glob("results_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"no results_*.csv files found in {directory}")
    return rows


def objective_summary_rows(rows_by_objective) -> list[dict[str, object]]:
    output = []
    for objective, rows in rows_by_objective.items():
        for variant in VARIANTS:
            for deterministic in (False, True):
                selected = [
                    row for row in rows
                    if row.get("stage") == "bc"
                    and row.get("variant") == variant
                    and _bool_value(row.get("deterministic")) is deterministic
                ]
                if not selected:
                    continue
                model_values = {metric: [] for metric in METRICS}
                for model_seed in range(5):
                    model_rows = [
                        row for row in selected if int(row["model_seed"]) == model_seed
                    ]
                    for metric in METRICS:
                        model_values[metric].append(
                            statistics.fmean(float(row[metric]) for row in model_rows)
                        )
                summary = {
                    "objective": objective,
                    "variant": variant,
                    "deterministic": deterministic,
                    "model_seeds": 5,
                    "eval_episodes": len(selected),
                }
                for metric, values in model_values.items():
                    sd = statistics.stdev(values)
                    summary[f"{metric}_mean"] = statistics.fmean(values)
                    summary[f"{metric}_model_sd"] = sd
                    summary[f"{metric}_ci95_half_width"] = (
                        T_CRITICAL_DF4 * sd / 5**0.5
                    )
                output.append(summary)
    return output


def _uncertainty(values) -> tuple[float, float, float]:
    values = list(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    interval = T_CRITICAL_DF4 * sd / 5**0.5 if len(values) == 5 else 0.0
    return mean, sd, interval


def demo_summary_rows(rows_by_objective) -> list[dict[str, object]]:
    output = []
    for objective, rows in rows_by_objective.items():
        for variant in VARIANTS:
            modes = sorted(
                {
                    row["mode"] for row in rows
                    if row.get("stage") == "bc"
                    and row.get("variant") == variant
                    and row.get("vessel") == "all"
                }
            )
            for mode in modes:
                selected = [
                    row for row in rows
                    if row.get("stage") == "bc"
                    and row.get("variant") == variant
                    and row.get("vessel") == "all"
                    and row.get("mode") == mode
                ]
                summary = {
                    "objective": objective,
                    "variant": variant,
                    "mode": mode,
                    "model_seeds": len(selected),
                }
                for metric in DEMO_METRICS:
                    values = [float(row[metric]) for row in selected if row.get(metric)]
                    if values:
                        mean, sd, interval = _uncertainty(values)
                        summary[f"{metric}_mean"] = mean
                        summary[f"{metric}_model_sd"] = sd
                        summary[f"{metric}_ci95_half_width"] = interval
                    else:
                        summary[f"{metric}_mean"] = ""
                        summary[f"{metric}_model_sd"] = ""
                        summary[f"{metric}_ci95_half_width"] = ""
                output.append(summary)
    return output


def rollout_summary_rows(rows_by_objective) -> list[dict[str, object]]:
    output = []
    for objective, rows in rows_by_objective.items():
        for variant in VARIANTS:
            for deterministic in (False, True):
                selected = [
                    row for row in rows
                    if row.get("stage") == "bc"
                    and row.get("variant") == variant
                    and row.get("vessel") == "all"
                    and row.get("mode") == "all"
                    and _bool_value(row.get("deterministic")) is deterministic
                ]
                if not selected:
                    continue
                summary = {
                    "objective": objective,
                    "variant": variant,
                    "deterministic": deterministic,
                    "model_seeds": len({int(row["model_seed"]) for row in selected}),
                    "episodes": len(selected),
                }
                for metric in ROLLOUT_METRICS:
                    model_means = []
                    for model_seed in sorted({int(row["model_seed"]) for row in selected}):
                        values = [
                            float(row[metric]) for row in selected
                            if int(row["model_seed"]) == model_seed and row.get(metric)
                        ]
                        if values:
                            model_means.append(statistics.fmean(values))
                    mean, sd, interval = _uncertainty(model_means)
                    summary[f"{metric}_mean"] = mean
                    summary[f"{metric}_model_sd"] = sd
                    summary[f"{metric}_ci95_half_width"] = interval
                output.append(summary)
    return output


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def load_diagnostic_rows(directory: Path, prefix: str) -> list[dict[str, str]]:
    rows = []
    for variant in VARIANTS:
        for path in sorted(Path(directory).glob(f"{prefix}_{variant}*_seed*.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append({"variant": variant, **row})
    if not rows:
        raise FileNotFoundError(f"no {prefix}_*.csv files found in {directory}")
    return rows


def write_report(
    rows_by_objective,
    out_dir: Path,
    demo_rows_by_objective=None,
    rollout_rows_by_objective=None,
):
    out_dir = Path(out_dir)
    summaries = objective_summary_rows(rows_by_objective)
    paired = [
        row for metric in METRICS
        for row in paired_metric_rows(rows_by_objective, metric)
    ]
    summary_path = out_dir / "bc_objective_summary.csv"
    paired_path = out_dir / "bc_objective_paired_deltas.csv"
    _write_csv(summary_path, summaries)
    _write_csv(paired_path, paired)
    if demo_rows_by_objective is not None:
        _write_csv(
            out_dir / "bc_objective_demo_summary.csv",
            demo_summary_rows(demo_rows_by_objective),
        )
    if rollout_rows_by_objective is not None:
        _write_csv(
            out_dir / "bc_objective_rollout_summary.csv",
            rollout_summary_rows(rollout_rows_by_objective),
        )
    markdown = [
        "# BC Objective Ablation",
        "",
        "A=current, B=decision-only loss, C=decision-only loss plus balanced sampling.",
        "",
        "| objective | variant | deterministic | vented t mean | 95% CI half-width |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        markdown.append(
            f"| {row['objective']} | {row['variant']} | {row['deterministic']} | "
            f"{row['vented_t_mean']} | {row['vented_t_ci95_half_width']} |"
        )
    markdown_path = out_dir / "bc_objective_summary.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary_path, paired_path, markdown_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-dir", required=True)
    parser.add_argument("--b-dir", required=True)
    parser.add_argument("--c-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    directories = {
        "current": Path(args.a_dir),
        "decision_only": Path(args.b_dir),
        "decision_balanced": Path(args.c_dir),
    }
    rows = {objective: load_result_rows(path) for objective, path in directories.items()}
    demo_rows = {
        objective: load_diagnostic_rows(path, "demo_mode_diagnostics")
        for objective, path in directories.items()
    }
    rollout_rows = {
        objective: load_diagnostic_rows(path, "rollout_mode_diagnostics")
        for objective, path in directories.items()
    }
    return write_report(
        rows,
        Path(args.out_dir),
        demo_rows_by_objective=demo_rows,
        rollout_rows_by_objective=rollout_rows,
    )


if __name__ == "__main__":
    main()
