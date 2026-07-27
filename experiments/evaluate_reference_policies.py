"""Evaluate Greedy and native MPC references on fixed environment seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from scripts import compare_forecast_encoders_rl as compare
else:  # pragma: no cover - direct CLI execution
    import compare_forecast_encoders_rl as compare


POLICY_NAMES = {
    "greedy": "greedy",
    "mpc": "RollingNativeMpcController",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--policies", choices=tuple(POLICY_NAMES), nargs="+", default=list(POLICY_NAMES)
    )
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error("--eval-seeds must not contain duplicates")
    return args


def run(args) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    results_path = output_dir / "combined.csv"
    manifest_path = output_dir / "manifest.json"
    collisions = [path for path in (results_path, manifest_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing output collision: " + ", ".join(str(path) for path in collisions)
        )

    compare_args = compare.parse_args(
        [
            "train",
            "--variant",
            "state",
            "--demo-cache",
            "unused-reference-cache.npz",
            "--bc-only",
            "--timesteps",
            "0",
            "--eval-seeds",
            *(str(seed) for seed in args.eval_seeds),
            "--episode-hours",
            str(args.episode_hours),
        ]
    )
    selected = {POLICY_NAMES[name] for name in args.policies}
    rows = [
        row
        for row in compare.evaluate_reference_rows(compare_args)
        if row["policy"] in selected
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    compare.write_results_csv(results_path, rows)
    manifest = {
        "kind": "reference_policy_validation",
        "policies": list(args.policies),
        "eval_seeds": [int(seed) for seed in args.eval_seeds],
        "episode_hours": int(args.episode_hours),
        "forecast_horizon_h": 168,
        "mpc_objective_mode": "economic",
        "results": str(results_path.resolve()),
        "row_count": len(rows),
    }
    compare.write_json_immutable(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, allow_nan=False), flush=True)
    return manifest


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
