"""Summarize paired Event-Residual PPO future-information ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_VARIANTS = (
    "state_only",
    "summary_24_72",
    "summary_168",
    "summary_24_72_168",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_path(run_root: Path, variant: str) -> Path:
    matches = sorted(
        (run_root / variant / "evaluation").glob(
            "best__hardprob0__seeds_*/results.json"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one formal evaluation for {variant}, found {matches}"
        )
    return matches[0]


def _records_by_seed(payload: dict) -> dict[int, dict]:
    return {
        int(record["seed"]): record
        for record in payload["per_seed"]
    }


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(samples, len(values)),
    )
    means = values[indices].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def analyze(args) -> dict:
    greedy_payload = _load_json(args.run_root / "greedy" / "results.json")
    greedy = _records_by_seed(greedy_payload)
    seeds = tuple(sorted(greedy))
    rows = []

    for variant in args.variants:
        config = _load_json(args.run_root / variant / "config.json")
        payload = _load_json(_evaluation_path(args.run_root, variant))
        ppo = _records_by_seed(payload)
        if tuple(sorted(ppo)) != seeds:
            raise ValueError(f"test seeds differ for {variant}")

        greedy_cost = np.asarray(
            [float(greedy[seed]["total_cost_eur"]) for seed in seeds]
        )
        ppo_cost = np.asarray(
            [float(ppo[seed]["total_cost_eur"]) for seed in seeds]
        )
        cost_delta = ppo_cost - greedy_cost
        relative_saving = -100.0 * cost_delta / greedy_cost
        delta_ci = _bootstrap_ci(
            cost_delta,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        relative_ci = _bootstrap_ci(
            relative_saving,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        rows.append(
            {
                "variant": variant,
                "future_summary_windows_h": ",".join(
                    str(value)
                    for value in config["future_summary_windows_h"]
                )
                or "none",
                "test_episodes": len(seeds),
                "ppo_mean_total_cost_eur": float(ppo_cost.mean()),
                "greedy_mean_total_cost_eur": float(greedy_cost.mean()),
                "mean_cost_delta_vs_greedy_eur": float(
                    cost_delta.mean()
                ),
                "mean_cost_delta_ci95_low_eur": delta_ci[0],
                "mean_cost_delta_ci95_high_eur": delta_ci[1],
                "relative_cost_improvement_percent": float(
                    100.0
                    * (greedy_cost.mean() - ppo_cost.mean())
                    / greedy_cost.mean()
                ),
                "paired_relative_improvement_mean_percent": float(
                    relative_saving.mean()
                ),
                "paired_relative_improvement_ci95_low_percent": (
                    relative_ci[0]
                ),
                "paired_relative_improvement_ci95_high_percent": (
                    relative_ci[1]
                ),
                "wins_vs_greedy": int(np.sum(cost_delta < 0.0)),
                "ties_vs_greedy": int(np.sum(cost_delta == 0.0)),
                "losses_vs_greedy": int(np.sum(cost_delta > 0.0)),
                "ppo_mean_vented_t": float(
                    np.mean(
                        [float(ppo[seed]["vented_t"]) for seed in seeds]
                    )
                ),
                "ppo_mean_stored_t": float(
                    np.mean(
                        [float(ppo[seed]["stored_t"]) for seed in seeds]
                    )
                ),
                "ppo_mean_selected_interventions": float(
                    np.mean(
                        [
                            float(ppo[seed]["selected_interventions"])
                            for seed in seeds
                        ]
                    )
                ),
            }
        )

    args.run_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.run_root / "future_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "greedy_results": str(
            (args.run_root / "greedy" / "results.json").resolve()
        ),
        "test_seeds": list(seeds),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "variants": rows,
    }
    (args.run_root / "future_ablation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Event-Residual PPO future-information ablation",
        "",
        "Negative cost delta means Event-Residual PPO is cheaper than Greedy.",
        "",
        "| input | PPO cost (EUR) | delta vs Greedy (EUR) | relative improvement | 95% paired bootstrap CI | wins | vented (t) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['future_summary_windows_h']} | "
            f"{row['ppo_mean_total_cost_eur']:,.0f} | "
            f"{row['mean_cost_delta_vs_greedy_eur']:+,.0f} | "
            f"{row['relative_cost_improvement_percent']:+.2f}% | "
            f"[{row['paired_relative_improvement_ci95_low_percent']:+.2f}%, "
            f"{row['paired_relative_improvement_ci95_high_percent']:+.2f}%] | "
            f"{row['wins_vs_greedy']}/{row['test_episodes']} | "
            f"{row['ppo_mean_vented_t']:,.1f} |"
        )
    (args.run_root / "future_ablation_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def main():
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
