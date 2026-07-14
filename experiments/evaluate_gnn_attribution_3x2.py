from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULT_ROOT = ROOT / "output" / "rl_forecast" / "gnn_attribution_3x2"
CHECKPOINT_ROOT = RESULT_ROOT / "results"
OUTPUT_ROOT = RESULT_ROOT / "eval_101_120"
EVAL_SEEDS = list(range(101, 121))
VARIANTS = (
    "larger_mlp_mode_destination",
    "fixed_scale_larger_mlp_mode_destination",
    "edge_gnn_mode_destination",
    "fixed_scale_edge_gnn_mode_destination",
)


def evaluate(task):
    variant, model_seed = task
    from scripts import compare_forecast_encoders_rl as compare
    from sim.environment.forecast_gym import make_forecast_ppo_policy

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_ROOT / f"bc_{variant}_decision_only_seed{model_seed}.zip"
    source_manifest_path = (
        CHECKPOINT_ROOT / f"run_{variant}_decision_only_seed{model_seed}.manifest.json"
    )
    result_path = OUTPUT_ROOT / f"results_{variant}_seed{model_seed}.csv"
    manifest_path = OUTPUT_ROOT / f"run_{variant}_seed{model_seed}.manifest.json"
    if result_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite evaluation for {variant} seed {model_seed}"
        )

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    args = compare.parse_args(
        [
            "train",
            "--variant",
            variant,
            "--demo-cache",
            str(
                ROOT
                / "output/rl_forecast/corrected_forecast_cache/"
                "destination_mask_train_0_99_v4.npz"
            ),
            "--heldout-demo-cache",
            str(
                ROOT
                / "output/rl_forecast/corrected_forecast_cache/"
                "destination_mask_heldout_121_140_v4.npz"
            ),
            "--model-seed",
            str(model_seed),
            "--eval-seeds",
            *[str(seed) for seed in EVAL_SEEDS],
            "--device",
            "cuda",
        ]
    )
    model = compare.MaskablePPO.load(checkpoint, device="cuda")
    exact_match = float(source_manifest["demonstration_accuracy"]["bc_exact_match"])
    action_accuracy = source_manifest["demonstration_accuracy"][
        "bc_action_dimensions"
    ]
    parameter_count = int(source_manifest["trainable_parameters"])

    rows = []
    started = time.perf_counter()
    for deterministic in (False, True):
        for eval_seed in EVAL_SEEDS:
            model.set_random_seed(eval_seed)
            env = compare.ExperimentEnvFactory(args)()
            policy = make_forecast_ppo_policy(
                model,
                variant,
                deterministic=deterministic,
            )
            metrics, runtime, latency = compare._timed_episode(env, policy, eval_seed)
            rows.append(
                compare.metric_result_row(
                    metrics,
                    policy=f"learned_{variant}",
                    family="learned",
                    variant=variant,
                    stage="bc_v4_gnn_attribution_eval_101_120",
                    deterministic=deterministic,
                    model_seed=model_seed,
                    eval_seed=eval_seed,
                    episode_runtime_s=runtime,
                    mean_inference_latency_s=latency,
                    trainable_parameters=parameter_count,
                    demonstration_exact_match=exact_match,
                    demonstration_action_accuracy=action_accuracy,
                )
            )
    compare.write_results_csv(result_path, rows)
    compare.write_json_immutable(
        manifest_path,
        {
            "kind": "gnn_attribution_3x2_v4_evaluation",
            "variant": variant,
            "model_seed": model_seed,
            "eval_seeds": EVAL_SEEDS,
            "deterministic_modes": [False, True],
            "row_count": len(rows),
            "device": "cuda",
            "runtime_s": time.perf_counter() - started,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": compare.file_sha256(checkpoint),
            "source_manifest": str(source_manifest_path),
            "train_cache_sha256": source_manifest["demo_cache_sha256"],
            "heldout_cache_sha256": source_manifest["heldout_demo_cache_sha256"],
            "forecast_schema_version": 4,
            "forecast_start_offset_h": 0,
            "forecast_end_offset_h": 167,
            "forecast_capture_source": "uncapped_hourly_profile_times_availability",
            "results_csv": str(result_path),
        },
    )
    return variant, model_seed, len(rows)


def main():
    tasks = [
        (variant, model_seed)
        for variant in VARIANTS
        for model_seed in range(5)
    ]
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(evaluate, task) for task in tasks]
        for future in as_completed(futures):
            variant, model_seed, row_count = future.result()
            print(
                f"DONE variant={variant} seed={model_seed} rows={row_count}",
                flush=True,
            )


if __name__ == "__main__":
    main()
