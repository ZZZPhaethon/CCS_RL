import csv
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scripts import compare_forecast_encoders_rl as compare
from sim.environment.forecast import current_state_feature_names, forecast_channel_names
from sim.environment.forecast_encoder import (
    EdgeGNNForecastExtractor,
    FixedScaleEdgeGNNForecastExtractor,
    FixedScaleLargerMLPForecastExtractor,
    FixedScaleTCNForecastExtractor,
    GNNForecastExtractor,
    LargerMLPForecastExtractor,
    StableTCNForecastExtractor,
    TCNForecastExtractor,
)
from sim.environment.vessel_mode import vessel_sailing_destination_feature_names
from sim.metrics import EpisodeMetrics
from sim.control.demonstrations import MpcDemonstrationBatch, save_demonstrations


def _train_args(tmp_path: Path, variant: str = "state", *extra: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache = tmp_path / "demos.npz"
    if not cache.exists():
        cache.write_bytes(b"shared-cache")
    return compare.parse_args(
        [
            "train",
            "--variant",
            variant,
            "--demo-cache",
            str(cache),
            "--out-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_cli_has_all_subcommands_and_locks_formal_defaults(tmp_path):
    cache = tmp_path / "demos.npz"
    train = compare.parse_args(["train", "--variant", "tcn", "--demo-cache", str(cache)])
    demos = compare.parse_args(
        ["generate-demos", "--demo-cache", str(cache), "--demo-seeds", "1", "2"]
    )
    report = compare.parse_args(["report"])
    merge = compare.parse_args(
        [
            "merge-demos",
            "--shards",
            str(tmp_path / "a.npz"),
            str(tmp_path / "b.npz"),
            "--demo-cache",
            str(tmp_path / "merged.npz"),
            "--expected-seeds",
            "0",
            "1",
        ]
    )

    assert {train.command, demos.command, merge.command, report.command} == {
        "train",
        "generate-demos",
        "merge-demos",
        "report",
    }
    assert train.scenario == "northern_lights_phase1_3vessels"
    assert train.episode_hours == 720
    assert train.forecast_horizon_h == 168
    assert train.weather_mode == "block"
    assert train.reward_mode == "vent_first"
    assert train.timesteps == 100_000
    assert train.gamma == 0.999
    assert train.n_steps == 512
    assert train.batch_size == 64
    assert train.learning_rate == 3e-4
    assert train.bc_epochs == 20
    assert train.bc_batch_size == 256
    assert train.bc_lr == 1e-3
    assert train.nonwait_weight == 10.0
    assert train.replan_action_weight == 1.0
    assert not train.imitation_only
    assert train.kickstart_coef == 1.0
    assert train.eval_seeds == [101, 102, 103, 104, 105]
    assert train.model_seed == 0
    assert merge.episode_hours == 720


def test_bc_objective_cli_defaults_and_objective_specific_paths(tmp_path):
    current = _train_args(tmp_path / "current", "state_mode")
    assert current.bc_objective == "current"
    assert not current.bc_only
    assert compare.checkpoint_path(current, "bc").name == "bc_state_mode_seed0.zip"

    decision = _train_args(
        tmp_path / "decision",
        "state_mode",
        "--bc-objective",
        "decision_only",
        "--bc-only",
    )
    assert decision.bc_objective == "decision_only"
    assert decision.bc_only
    assert compare.checkpoint_path(decision, "bc").name == (
        "bc_state_mode_decision_only_seed0.zip"
    )
    assert compare.results_path(decision).name == (
        "results_state_mode_decision_only_seed0.csv"
    )


def _tiny_demo_shard(seed: int):
    return MpcDemonstrationBatch(
        state=np.full((2, 2), seed, dtype=np.float32),
        forecast=np.full((2, 168, 9), seed, dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.int64),
        masks=np.ones((2, 4), dtype=bool),
        seeds=np.full(2, seed, dtype=np.int64),
        hours=np.asarray([0, 1], dtype=np.int64),
        metadata={"schema": "merge-cli-v2"},
        operation_modes=np.tile(
            np.asarray([[[1, 0, 0, 0, 0]]], dtype=np.float32),
            (2, 1, 1),
        ),
    )


def test_merge_demos_writes_sorted_cache_and_hash_manifest(tmp_path):
    shard_paths = [tmp_path / "seed1.npz", tmp_path / "seed0.npz"]
    save_demonstrations(_tiny_demo_shard(1), shard_paths[0])
    save_demonstrations(_tiny_demo_shard(0), shard_paths[1])
    output = tmp_path / "merged.npz"
    args = compare.parse_args(
        [
            "merge-demos",
            "--shards",
            *(str(path) for path in shard_paths),
            "--demo-cache",
            str(output),
            "--expected-seeds",
            "0",
            "1",
            "--episode-hours",
            "2",
        ]
    )

    result = compare.merge_demos(args)

    assert output.exists()
    assert result["row_count"] == 4
    assert result["demo_seeds"] == [0, 1]
    manifest = json.loads(compare.demo_manifest_path(output).read_text(encoding="utf-8"))
    assert manifest["cache_sha256"] == compare.file_sha256(output)
    assert len(manifest["shards"]) == 2


def test_cli_rejects_negative_timesteps(tmp_path):
    with pytest.raises(SystemExit):
        compare.parse_args(
            [
                "train",
                "--variant",
                "state",
                "--demo-cache",
                str(tmp_path / "demos.npz"),
                "--timesteps",
                "-1",
            ]
        )


def test_environment_factory_uses_demo_and_training_context_contract(tmp_path):
    args = _train_args(tmp_path)
    sentinel = object()

    with patch.object(compare, "make_native_env", return_value=sentinel) as make_native_env:
        assert compare.make_experiment_env(args, demonstration=True) is sentinel
        demo_call = make_native_env.call_args.kwargs
        assert demo_call["episode_hours"] == 889
        assert demo_call["scenario_context_hours"] == 0

        assert compare.make_experiment_env(args, demonstration=False) is sentinel
        train_call = make_native_env.call_args.kwargs

    assert train_call["episode_hours"] == 720
    assert train_call["scenario_context_hours"] == 169
    for call in (demo_call, train_call):
        assert call["scenario"] == "northern_lights_phase1_3vessels"
        assert call["weather_mode"] == "block"
        assert call["include_weather_obs"] is False
        assert call["reward_mode"] == "vent_first"
        assert call["vent_first_vent_eur_per_t"] == 10_000.0
        assert call["overflow_risk_eur_per_t"] == 100.0
        assert call["overflow_risk_lookahead_h"] == 24.0
        assert call["operating_cost_weight"] == 1.0
        assert call["enforce_full_load_dispatch"] is False
        assert call["require_empty_terminal_departure"] is True


def test_metadata_is_derived_from_environment_helpers_without_schema_drift(tmp_path):
    args = _train_args(tmp_path)
    factory = compare.ExperimentEnvFactory(args)
    metadata = factory.metadata()
    env = factory()

    assert metadata["forecast_channels"] == list(forecast_channel_names(env))
    assert metadata["forecast_shape"] == [168, 9]
    assert metadata["forecast_schema_version"] == 4
    assert (
        metadata["forecast_capture_source"]
        == "uncapped_hourly_profile_times_availability"
    )
    assert metadata["forecast_start_offset_h"] == 0
    assert metadata["forecast_end_offset_h"] == 167
    assert metadata["state_feature_names"] == list(current_state_feature_names(env))
    assert metadata["state_size"] == len(current_state_feature_names(env)) == 51
    assert metadata["action_dimensions"] == [*env.vessel_action_dims, *env.well_rate_action_dims]
    assert metadata["weather_mode"] == "block"
    assert metadata["weather_observation_layout"] == "global"
    assert metadata["reward"]["mode"] == "vent_first"
    assert metadata["partial_load_dispatch"] is True
    assert metadata["require_empty_terminal_departure"] is True
    assert metadata["vessel_destination_feature_names"] == list(
        vessel_sailing_destination_feature_names(env)
    )
    assert metadata["vessel_destination_shape"] == [3, 4]
    assert metadata["warm_start"] is True
    assert metadata["scenario_context_hours"] == 169
    assert metadata["emitter_buffer_capacity_t"]["yara_sluiskil"] == 15_000.0


def test_policy_mapping_uses_custom_extractor_only_for_tcn():
    assert compare.model_policy_config("state") == ("MlpPolicy", {})
    assert compare.model_policy_config("state_mode") == ("MlpPolicy", {})
    assert compare.model_policy_config("flat") == ("MlpPolicy", {})
    for variant in ("tcn", "tcn_mode", "tcn_mode_destination"):
        policy, kwargs = compare.model_policy_config(variant)
        assert policy == "MultiInputPolicy"
        assert kwargs["features_extractor_class"] is TCNForecastExtractor
        assert kwargs["features_extractor_kwargs"] == {
            "state_features": 64,
            "forecast_features": 64,
        }
    policy, kwargs = compare.model_policy_config("gnn_mode_destination")
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is GNNForecastExtractor
    assert kwargs["features_extractor_kwargs"] == {
        "state_features": 64,
        "forecast_features": 64,
    }
    policy, kwargs = compare.model_policy_config("larger_mlp_mode_destination")
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is LargerMLPForecastExtractor
    policy, kwargs = compare.model_policy_config("edge_gnn_mode_destination")
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is EdgeGNNForecastExtractor
    policy, kwargs = compare.model_policy_config(
        "fixed_scale_larger_mlp_mode_destination"
    )
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is FixedScaleLargerMLPForecastExtractor
    policy, kwargs = compare.model_policy_config(
        "fixed_scale_edge_gnn_mode_destination"
    )
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is FixedScaleEdgeGNNForecastExtractor
    policy, kwargs = compare.model_policy_config("stable_tcn_mode_destination")
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is StableTCNForecastExtractor
    policy, kwargs = compare.model_policy_config("fixed_scale_tcn_mode_destination")
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is FixedScaleTCNForecastExtractor
    phase_policy, phase_kwargs = compare.model_policy_config(
        "fixed_scale_tcn_mode_destination_replan_phase"
    )
    assert phase_policy == "MultiInputPolicy"
    assert phase_kwargs["features_extractor_class"] is FixedScaleTCNForecastExtractor


def test_cli_accepts_tcn_mode_destination_variant(tmp_path):
    args = _train_args(tmp_path, "tcn_mode_destination")

    assert args.variant == "tcn_mode_destination"


def test_cli_accepts_replan_phase_variant_and_weight(tmp_path):
    args = _train_args(
        tmp_path,
        "fixed_scale_tcn_mode_destination_replan_phase",
        "--bc-objective",
        "decision_only",
        "--bc-only",
        "--replan-action-weight",
        "3",
    )

    assert args.variant == "fixed_scale_tcn_mode_destination_replan_phase"
    assert args.replan_action_weight == 3.0


def test_cli_accepts_oracle_candidate_imitation_only_variant(tmp_path):
    args = _train_args(
        tmp_path,
        "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
        "--bc-objective",
        "decision_only",
        "--bc-only",
        "--imitation-only",
    )

    assert args.imitation_only
    policy, kwargs = compare.model_policy_config(args.variant)
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is FixedScaleTCNForecastExtractor


def test_cli_accepts_learned_plan_context_imitation_only_variant(tmp_path):
    args = _train_args(
        tmp_path,
        "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
        "--bc-objective",
        "decision_only",
        "--bc-only",
        "--imitation-only",
    )

    assert args.imitation_only
    policy, kwargs = compare.model_policy_config(args.variant)
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is FixedScaleTCNForecastExtractor


@pytest.mark.parametrize(
    ("variant", "extractor_name"),
    [
        ("tcn_mode_destination", "TCNForecastExtractor"),
        ("gnn_mode_destination", "GNNForecastExtractor"),
        ("larger_mlp_mode_destination", "LargerMLPForecastExtractor"),
        ("edge_gnn_mode_destination", "EdgeGNNForecastExtractor"),
        (
            "fixed_scale_larger_mlp_mode_destination",
            "FixedScaleLargerMLPForecastExtractor",
        ),
        (
            "fixed_scale_edge_gnn_mode_destination",
            "FixedScaleEdgeGNNForecastExtractor",
        ),
        ("stable_tcn_mode_destination", "StableTCNForecastExtractor"),
        ("fixed_scale_tcn_mode_destination", "FixedScaleTCNForecastExtractor"),
    ],
)
def test_policy_manifest_records_selected_encoder(variant, extractor_name):
    assert compare.policy_manifest(variant) == {
        "name": "MultiInputPolicy",
        "features_extractor": extractor_name,
        "state_features": 64,
        "forecast_features": 64,
    }


def test_cli_accepts_gnn_mode_destination_variant(tmp_path):
    args = _train_args(tmp_path, "gnn_mode_destination")

    assert args.variant == "gnn_mode_destination"


@pytest.mark.parametrize(
    "variant",
    [
        "larger_mlp_mode_destination",
        "edge_gnn_mode_destination",
        "fixed_scale_larger_mlp_mode_destination",
        "fixed_scale_edge_gnn_mode_destination",
        "stable_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination",
    ],
)
def test_cli_accepts_new_encoder_variants(tmp_path, variant):
    args = _train_args(tmp_path, variant)

    assert args.variant == variant


def test_cli_accepts_heldout_demo_cache_and_uses_distinct_diagnostic_path(tmp_path):
    heldout = tmp_path / "heldout.npz"
    args = _train_args(
        tmp_path,
        "tcn_mode_destination",
        "--heldout-demo-cache",
        str(heldout),
    )

    assert args.heldout_demo_cache == str(heldout)
    assert compare.heldout_demo_diagnostics_path(args).name == (
        "heldout_demo_mode_diagnostics_tcn_mode_destination_seed0.csv"
    )


def test_heldout_diagnostics_add_stage_and_model_seed(monkeypatch):
    expected_rows = [{"vessel": "all", "mode": "all"}]
    monkeypatch.setattr(
        compare,
        "demonstration_mode_diagnostics",
        lambda *args, **kwargs: expected_rows,
    )
    batch = SimpleNamespace(
        observations=lambda variant: {"variant": variant},
        actions=np.zeros((1, 4), dtype=np.int64),
        masks=np.ones((1, 21), dtype=bool),
        operation_modes=np.ones((1, 3, 5), dtype=np.float32),
    )

    rows = compare.heldout_demonstration_diagnostics(
        object(),
        batch,
        variant="tcn_mode_destination",
        vessel_count=3,
        stage="bc",
        model_seed=2,
    )

    assert rows == [
        {
            "stage": "bc",
            "model_seed": 2,
            "vessel": "all",
            "mode": "all",
        }
    ]


def test_train_variant_loads_and_passes_heldout_demonstrations(tmp_path):
    heldout_path = tmp_path / "heldout.npz"
    heldout_path.write_bytes(b"heldout")
    args = _train_args(
        tmp_path,
        "tcn_mode_destination",
        "--heldout-demo-cache",
        str(heldout_path),
    )
    training_batch = SimpleNamespace(
        observations=lambda variant: {"training": variant}
    )
    heldout_batch = object()
    factory = Mock()
    factory.metadata.return_value = {"schema": "v3"}

    with (
        patch.object(compare, "file_sha256", return_value="sha"),
        patch.object(compare, "ExperimentEnvFactory", return_value=factory),
        patch.object(
            compare,
            "load_demonstrations",
            side_effect=[training_batch, heldout_batch],
        ) as load,
        patch.object(compare, "make_experiment_env", return_value=object()),
        patch.object(compare, "_train_loaded_batch", return_value={}) as train,
    ):
        compare.train_variant(args)

    assert load.call_count == 2
    assert train.call_args.kwargs["heldout_batch"] is heldout_batch
    assert train.call_args.kwargs["heldout_cache_sha256"] == "sha"


@pytest.mark.parametrize("variant", ["state", "state_mode", "tcn", "tcn_mode"])
def test_formal_cli_accepts_operation_mode_matrix(tmp_path, variant):
    args = _train_args(tmp_path / variant, variant)
    assert args.variant == variant


def test_formal_hpc_task_mappings_cover_all_seeds_and_variants_once():
    demonstration_seeds = [
        seed for task_id in range(10) for seed in compare.demonstration_task_seeds(task_id)
    ]
    assert demonstration_seeds == list(range(100))

    training_tasks = [compare.formal_training_task(task_id) for task_id in range(20)]
    assert len(set(training_tasks)) == 20
    assert set(training_tasks) == {
        (variant, seed)
        for seed in range(5)
        for variant in ("state", "state_mode", "tcn", "tcn_mode")
    }
    with pytest.raises(ValueError, match="0..9"):
        compare.demonstration_task_seeds(10)
    with pytest.raises(ValueError, match="0..19"):
        compare.formal_training_task(20)


def test_bc_objective_array_mapping_covers_two_variants_and_five_seeds():
    tasks = [compare.bc_objective_training_task(task_id) for task_id in range(10)]
    assert len(set(tasks)) == 10
    assert set(tasks) == {
        (variant, seed)
        for seed in range(5)
        for variant in ("state_mode", "tcn_mode")
    }
    with pytest.raises(ValueError, match="0..9"):
        compare.bc_objective_training_task(10)


def test_bc_objective_hpc_script_locks_bc_only_formal_protocol():
    root = Path(compare.__file__).resolve().parents[1]
    source = (root / "hpc/submit_bc_objective_ablation.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-9%5" in source
    assert "VARIANTS=(state_mode tcn_mode)" in source
    assert "MODEL_SEEDS=(0 1 2 3 4)" in source
    assert 'BC_EPOCHS="${BC_EPOCHS:-50}"' in source
    assert "--bc-objective \"$BC_OBJECTIVE\"" in source
    assert "--bc-only" in source
    assert "--timesteps 0" in source
    assert 'EVAL_SEEDS="${EVAL_SEEDS:-101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120}"' in source


def test_demo_shard_scripts_support_disjoint_training_and_heldout_seed_ranges():
    root = Path(compare.__file__).resolve().parents[1]
    shard = (root / "hpc/submit_forecast_mpc_demo_shards.sh").read_text(
        encoding="utf-8"
    )
    merge = (root / "hpc/submit_forecast_mpc_demo_merge.sh").read_text(
        encoding="utf-8"
    )

    assert 'SEED_START="${SEED_START:-0}"' in shard
    assert 'SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"' in shard
    assert 'SEED_START="${SEED_START:-0}"' in merge
    assert 'TASK_COUNT="${TASK_COUNT:-10}"' in merge
    assert 'SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"' in merge


def test_destination_bc_hpc_script_locks_mask_destination_comparison():
    root = Path(compare.__file__).resolve().parents[1]
    source = (root / "hpc/submit_destination_bc.sh").read_text(encoding="utf-8")

    assert "#SBATCH --array=0-9%5" in source
    assert "VARIANTS=(tcn_mode tcn_mode_destination)" in source
    assert "MODEL_SEEDS=(0 1 2 3 4)" in source
    assert 'BC_EPOCHS="${BC_EPOCHS:-50}"' in source
    assert '--bc-objective decision_only' in source
    assert '--heldout-demo-cache "$HELDOUT_DEMO_CACHE"' in source
    assert "--bc-only" in source
    assert "--timesteps 0" in source


def test_gnn_bc_hpc_script_locks_encoder_only_comparison():
    root = Path(compare.__file__).resolve().parents[1]
    source = (root / "hpc/submit_gnn_bc.sh").read_text(encoding="utf-8")

    assert "#SBATCH --array=0-9%5" in source
    assert "VARIANTS=(tcn_mode_destination gnn_mode_destination)" in source
    assert "MODEL_SEEDS=(0 1 2 3 4)" in source
    assert 'BC_EPOCHS="${BC_EPOCHS:-50}"' in source
    assert '--bc-objective decision_only' in source
    assert '--heldout-demo-cache "$HELDOUT_DEMO_CACHE"' in source
    assert "--bc-only" in source
    assert "--timesteps 0" in source


def _bc_ablation_rows(offset):
    rows = []
    for variant in ("state_mode", "tcn_mode"):
        for model_seed in range(5):
            for eval_seed in (101, 102):
                rows.append(
                    {
                        "variant": variant,
                        "stage": "bc",
                        "deterministic": "true",
                        "model_seed": str(model_seed),
                        "eval_seed": str(eval_seed),
                        "vented_t": str(10.0 + model_seed + offset),
                        "stored_t": str(100.0 - offset),
                        "total_cost": str(1000.0 + offset),
                    }
                )
    return rows


def test_bc_ablation_report_computes_exact_paired_objective_deltas():
    from experiments import report_bc_objective_ablation as report

    rows_by_objective = {
        "current": _bc_ablation_rows(0.0),
        "decision_only": _bc_ablation_rows(-2.0),
        "decision_balanced": _bc_ablation_rows(-3.0),
    }

    paired = report.paired_metric_rows(rows_by_objective, "vented_t")
    lookup = {
        (row["comparison"], row["variant"]): row
        for row in paired
        if row["deterministic"] is True
    }
    assert lookup[("decision_only-current", "state_mode")]["mean_delta"] == -2.0
    assert lookup[("decision_balanced-current", "tcn_mode")]["mean_delta"] == -3.0
    assert lookup[("decision_balanced-decision_only", "state_mode")]["mean_delta"] == -1.0
    assert lookup[("decision_only-current", "state_mode")]["model_sd"] == 0.0
    assert lookup[("decision_only-current", "state_mode")]["ci95_half_width"] == 0.0


def test_bc_ablation_report_rejects_missing_pair_keys():
    from experiments import report_bc_objective_ablation as report

    rows_by_objective = {
        "current": _bc_ablation_rows(0.0),
        "decision_only": _bc_ablation_rows(-2.0)[:-1],
        "decision_balanced": _bc_ablation_rows(-3.0),
    }
    with pytest.raises(ValueError, match="missing paired keys"):
        report.paired_metric_rows(rows_by_objective, "vented_t")


def test_bc_ablation_report_summarizes_decision_and_rollout_diagnostics():
    from experiments import report_bc_objective_ablation as report

    demo = {}
    rollout = {}
    for objective_index, objective in enumerate(report.OBJECTIVES):
        demo[objective] = [
            {
                "variant": "state_mode",
                "stage": "bc",
                "model_seed": str(seed),
                "vessel": "all",
                "mode": "loading",
                "dispatch_recall": str(0.4 + 0.1 * objective_index),
                "conditional_destination_accuracy": "0.5",
                "voluntary_wait_accuracy": "0.9",
                "mean_wait_probability": "0.8",
            }
            for seed in range(5)
        ]
        rollout[objective] = [
            {
                "variant": "state_mode",
                "stage": "bc",
                "deterministic": "true",
                "model_seed": str(seed),
                "eval_seed": str(eval_seed),
                "vessel": "all",
                "mode": "all",
                "dispatch_count": str(30 + objective_index),
                "partial_load_departure_count": "20",
                "milk_run_departure_count": "5",
                "longest_berthed_no_dispatch_streak": "100",
                "mean_wait_probability": "0.95",
            }
            for seed in range(5)
            for eval_seed in (101, 102)
        ]

    demo_rows = report.demo_summary_rows(demo)
    rollout_rows = report.rollout_summary_rows(rollout)

    balanced_demo = next(
        row for row in demo_rows
        if row["objective"] == "decision_balanced" and row["variant"] == "state_mode"
    )
    assert balanced_demo["dispatch_recall_mean"] == pytest.approx(0.6)
    balanced_rollout = next(
        row for row in rollout_rows
        if row["objective"] == "decision_balanced" and row["variant"] == "state_mode"
    )
    assert balanced_rollout["dispatch_count_mean"] == 32.0
    assert balanced_rollout["episodes"] == 10


def test_hpc_scripts_lock_formal_protocol_defaults():
    root = Path(compare.__file__).resolve().parents[1]
    training = (root / "hpc/submit_forecast_encoder_rl.sh").read_text(encoding="utf-8")
    shards = (root / "hpc/submit_forecast_mpc_demo_shards.sh").read_text(encoding="utf-8")
    merge = (root / "hpc/submit_forecast_mpc_demo_merge.sh").read_text(encoding="utf-8")

    assert "#SBATCH --array=0-19%5" in training
    assert "BC_EPOCHS=\"${BC_EPOCHS:-50}\"" in training
    assert "EVAL_SEEDS=\"${EVAL_SEEDS:-101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120}\"" in training
    assert "#SBATCH --array=0-9" in shards
    assert "--cpus-per-task=4" in shards
    assert "--mem=32G" in shards
    assert "merge-demos" in merge
    assert 'SEED_START="${SEED_START:-0}"' in merge
    assert 'TASK_COUNT="${TASK_COUNT:-10}"' in merge
    assert 'SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"' in merge


def test_manifest_write_accepts_identical_bytes_and_rejects_different_content(tmp_path):
    path = tmp_path / "manifest.json"
    compare.write_json_immutable(path, {"b": 2, "a": 1})
    original = path.read_bytes()

    compare.write_json_immutable(path, {"a": 1, "b": 2})
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError, match="different content"):
        compare.write_json_immutable(path, {"a": 9})


def test_all_variants_use_same_cache_hash_and_observation_batch_contract(tmp_path):
    cache = tmp_path / "demos.npz"
    cache.write_bytes(b"one immutable cache")
    expected_hash = compare.file_sha256(cache)
    batch = Mock()
    batch.metadata = {"demo": "metadata"}
    batch.seeds = np.asarray([7, 7])
    batch.actions = np.zeros((2, 4), dtype=np.int64)
    batch.masks = np.ones((2, 21), dtype=bool)
    batch.observations.side_effect = lambda variant: {"variant": variant}

    for variant in ("state", "flat", "tcn"):
        args = _train_args(tmp_path / variant, variant)
        Path(args.demo_cache).write_bytes(cache.read_bytes())
        with (
            patch.object(compare, "make_experiment_env", return_value=SimpleNamespace(vessel_ids=[1, 2, 3])),
            patch.object(compare.ExperimentEnvFactory, "metadata", return_value={"schema": 1}),
            patch.object(compare, "load_demonstrations", return_value=batch) as load,
            patch.object(compare, "_train_loaded_batch", return_value={"cache_sha256": expected_hash}),
        ):
            result = compare.train_variant(args)
        load.assert_called_once_with(Path(args.demo_cache), {"schema": 1})
        assert result["cache_sha256"] == expected_hash

    assert [call.args[0] for call in batch.observations.call_args_list] == ["state", "flat", "tcn"]


def test_training_orchestration_runs_bc_before_ppo_and_preserves_tcn_dict(tmp_path):
    args = _train_args(tmp_path, "tcn", "--timesteps", "4", "--eval-seeds", "9")
    events = []
    observations = {
        "state": np.zeros((2, 3), dtype=np.float32),
        "forecast": np.zeros((2, 168, 9), dtype=np.float32),
    }
    batch = SimpleNamespace(
        observations=lambda variant: events.append(("observations", variant)) or observations,
        actions=np.zeros((2, 4), dtype=np.int64),
        masks=np.ones((2, 21), dtype=bool),
        seeds=np.asarray([4, 4]),
        metadata={"schema": 1},
    )

    class FakeModel:
        policy = SimpleNamespace(parameters=lambda: [])

        def save(self, path):
            events.append(("save", Path(path).name))

        def learn(self, **kwargs):
            events.append(("learn", kwargs))

        def set_random_seed(self, seed):
            events.append(("seed", seed))

    def fake_model(*model_args, **model_kwargs):
        events.append(("model", model_args, model_kwargs))
        return FakeModel()

    def fake_bc(model, obs, actions, **kwargs):
        assert obs is observations
        events.append(("bc", kwargs))

    def fake_evaluate(_args, model, *, stage, **_kwargs):
        model.set_random_seed(9)
        events.append(("eval", stage))
        return []

    callback = object()

    with (
        patch.object(compare, "make_experiment_env", return_value=SimpleNamespace(vessel_ids=[1, 2, 3])) as make_env,
        patch.object(compare, "ForecastGymEnv", return_value=object()),
        patch.object(compare.ExperimentEnvFactory, "metadata", return_value={"schema": 1}),
        patch.object(compare, "load_demonstrations", return_value=batch),
        patch.object(compare, "MaskablePPO", side_effect=fake_model),
        patch.object(compare, "behavior_clone", side_effect=fake_bc),
        patch.object(compare, "action_dimension_weights", return_value=np.ones((2, 4))),
        patch.object(compare, "make_kickstart_callback", return_value=callback) as make_callback,
        patch.object(compare, "demonstration_accuracy", side_effect=[(0.5, [0.5] * 4), (0.75, [0.75] * 4)]),
        patch.object(compare, "evaluate_learned_stage", side_effect=fake_evaluate),
        patch.object(compare, "evaluate_reference_rows", return_value=[]),
        patch.object(compare, "count_trainable_parameters", return_value=12),
        patch.object(compare, "write_results_csv"),
        patch.object(compare, "write_json_immutable") as write_manifest,
        patch.object(compare, "git_commit", return_value="abc123"),
    ):
        compare.train_variant(args)

    names = [event[0] for event in events]
    assert names.index("bc") < names.index("save") < names.index("eval") < names.index("learn")
    assert [event[1] for event in events if event[0] == "save"] == [
        "bc_tcn_seed0.zip",
        "ppo_tcn_seed0.zip",
    ]
    assert [event[1] for event in events if event[0] == "eval"] == ["bc", "ppo"]
    make_callback.assert_called_once()
    assert make_callback.call_args.args[0] is observations
    assert events[[event[0] for event in events].index("learn")][1]["callback"] is callback
    learn_index = [event[0] for event in events].index("learn")
    assert events[learn_index - 1] == ("seed", 0)
    assert make_env.call_count == 1
    manifest = write_manifest.call_args.args[1]
    assert manifest["policy"] == {
        "name": "MultiInputPolicy",
        "features_extractor": "TCNForecastExtractor",
        "state_features": 64,
        "forecast_features": 64,
    }
    assert manifest["kickstart"] == {
        "coefficient": 1.0,
        "decay": "linear",
        "n_batches": 4,
        "batch_size": 256,
        "learning_rate": 1e-3,
    }


def test_zero_timesteps_skips_learn_but_still_saves_and_evaluates_ppo(tmp_path):
    args = _train_args(tmp_path, "state", "--timesteps", "0", "--eval-seeds", "9")
    batch = SimpleNamespace(
        observations=lambda _variant: np.zeros((1, 2), dtype=np.float32),
        actions=np.zeros((1, 4), dtype=np.int64),
        masks=np.ones((1, 21), dtype=bool),
        seeds=np.asarray([1]),
        metadata={},
    )
    model = Mock()
    model.policy.parameters.return_value = []
    with (
        patch.object(compare, "make_experiment_env", return_value=SimpleNamespace(vessel_ids=[1, 2, 3])),
        patch.object(compare, "ForecastGymEnv", return_value=object()),
        patch.object(compare.ExperimentEnvFactory, "metadata", return_value={}),
        patch.object(compare, "load_demonstrations", return_value=batch),
        patch.object(compare, "MaskablePPO", return_value=model),
        patch.object(compare, "behavior_clone"),
        patch.object(compare, "demonstration_accuracy", return_value=(1.0, [1.0] * 4)),
        patch.object(compare, "evaluate_learned_stage", return_value=[]),
        patch.object(compare, "evaluate_reference_rows", return_value=[]),
        patch.object(compare, "write_results_csv"),
        patch.object(compare, "write_json_immutable"),
        patch.object(compare, "git_commit", return_value="abc123"),
    ):
        compare.train_variant(args)

    model.learn.assert_not_called()
    assert model.save.call_count == 2


@pytest.mark.parametrize("objective", ["decision_only", "decision_balanced"])
def test_bc_only_objectives_use_expected_training_path_and_skip_ppo(tmp_path, objective):
    args = _train_args(
        tmp_path / objective,
        "state_mode",
        "--bc-objective",
        objective,
        "--bc-only",
        "--eval-seeds",
        "9",
    )
    args.heldout_demo_cache = "heldout.npz"
    observations = np.zeros((2, 4), dtype=np.float32)
    batch = SimpleNamespace(
        actions=np.zeros((2, 4), dtype=np.int64),
        masks=np.ones((2, 21), dtype=bool),
        seeds=np.asarray([1, 1]),
        operation_modes=None,
    )
    native_env = SimpleNamespace(vessel_ids=["a", "b", "c"])
    model = Mock()
    model.policy.parameters.return_value = []
    decision_weights = np.ones((2, 4), dtype=np.float32)
    sampler_audit = {
        "wait_pairs": 3,
        "dispatch_pairs": 1,
        "sampled_wait_pairs": 3,
        "sampled_dispatch_pairs": 3,
        "well_pairs": 2,
        "total_targets": 8,
    }
    evaluated = []
    heldout_rows = [{"stage": "bc", "model_seed": 0}]
    heldout_batch = object()

    def fake_evaluate(_args, _model, *, stage, **_kwargs):
        evaluated.append(stage)
        return []

    with (
        patch.object(compare, "ForecastGymEnv", return_value=object()),
        patch.object(compare, "MaskablePPO", return_value=model),
        patch.object(
            compare,
            "decision_only_action_weights",
            return_value=decision_weights,
            create=True,
        ) as decision_only,
        patch.object(compare, "behavior_clone") as standard_clone,
        patch.object(
            compare,
            "behavior_clone_balanced_decisions",
            return_value=sampler_audit,
            create=True,
        ) as balanced_clone,
        patch.object(compare, "demonstration_accuracy", return_value=(0.5, [0.5] * 4)),
        patch.object(compare, "evaluate_learned_stage", side_effect=fake_evaluate),
        patch.object(compare, "evaluate_reference_rows", return_value=[]),
        patch.object(compare, "count_trainable_parameters", return_value=12),
        patch.object(compare, "write_results_csv"),
        patch.object(
            compare,
            "heldout_demonstration_diagnostics",
            return_value=heldout_rows,
        ) as heldout_diagnostics,
        patch.object(compare, "write_diagnostics_csv") as write_diagnostics,
        patch.object(compare, "write_json_immutable") as write_manifest,
        patch.object(compare, "git_commit", return_value="abc123"),
    ):
        manifest = compare._train_loaded_batch(
            args,
            batch=batch,
            observations=observations,
            native_env=native_env,
            metadata={"action_dimensions": [5, 5, 5, 6]},
            cache_sha256="cache-sha",
            heldout_batch=heldout_batch,
        )

    assert evaluated == ["bc"]
    model.learn.assert_not_called()
    model.save.assert_called_once_with(str(compare.checkpoint_path(args, "bc")))
    assert manifest is write_manifest.call_args.args[1]
    assert manifest["bc"]["objective"] == objective
    assert manifest["ppo"]["explicitly_skipped"] is True
    assert manifest["checkpoints"]["ppo"] is None
    heldout_diagnostics.assert_called_once()
    write_diagnostics.assert_called_once_with(
        compare.heldout_demo_diagnostics_path(args),
        heldout_rows,
    )
    assert manifest["diagnostics"]["heldout_demonstration_mode_csv"] == str(
        compare.heldout_demo_diagnostics_path(args)
    )
    if objective == "decision_only":
        decision_only.assert_called_once()
        standard_clone.assert_called_once()
        balanced_clone.assert_not_called()
        assert manifest["bc"]["sampler_audit"] is None
    else:
        decision_only.assert_not_called()
        standard_clone.assert_not_called()
        balanced_clone.assert_called_once()
        assert manifest["bc"]["sampler_audit"] == sampler_audit


def test_result_schema_checkpoint_names_and_canonical_baseline_rule(tmp_path):
    args = _train_args(tmp_path, "state")
    assert list(compare.RESULT_COLUMNS)[0] == "vented_t"
    assert compare.checkpoint_path(args, "bc").name == "bc_state_seed0.zip"
    assert compare.checkpoint_path(args, "ppo").name == "ppo_state_seed0.zip"
    assert compare.should_emit_baselines(args)
    args.variant = "flat"
    assert not compare.should_emit_baselines(args)
    args.variant = "state"
    args.model_seed = 3
    assert not compare.should_emit_baselines(args)

    row = compare.metric_result_row(
        EpisodeMetrics(vented_t=2.0, emitter_inventory_t=3.0),
        policy="learned_state",
        family="learned",
        variant="state",
        stage="bc",
        deterministic=True,
        model_seed=0,
        eval_seed=101,
        episode_runtime_s=1.0,
        mean_inference_latency_s=0.01,
        trainable_parameters=20,
        demonstration_exact_match=0.5,
        demonstration_action_accuracy=[0.5, 1.0],
    )
    assert list(row) == list(compare.RESULT_COLUMNS)
    assert row["eval_seed"] == 101
    assert json.loads(row["demonstration_action_accuracy"]) == [0.5, 1.0]


def test_canonical_reference_rows_are_per_seed_and_recreate_mpc(tmp_path):
    args = _train_args(tmp_path, "state", "--eval-seeds", "3", "4")

    class FakeFactory:
        def __call__(self):
            return object()

    with (
        patch.object(compare, "ExperimentEnvFactory", return_value=FakeFactory()),
        patch.object(
            compare,
            "_timed_episode",
            return_value=(EpisodeMetrics(vented_t=1.0), 2.0, 0.1),
        ),
        patch.object(compare, "RollingNativeMpcController", side_effect=lambda *a, **k: object()) as mpc,
    ):
        rows = compare.evaluate_reference_rows(args)

    assert len(rows) == 6
    assert {(row["policy"], row["eval_seed"]) for row in rows} == {
        (policy, seed)
        for policy in ("idle", "greedy", "RollingNativeMpcController")
        for seed in (3, 4)
    }
    assert all(list(row) == list(compare.RESULT_COLUMNS) for row in rows)
    assert mpc.call_count == 2


def _paired_row(
    variant: str,
    vented: float,
    *,
    seed: int = 101,
    model_seed: int = 0,
    stage: str = "ppo",
    deterministic: bool = True,
):
    return {
        "policy": f"learned_{variant}",
        "family": "learned",
        "variant": variant,
        "stage": stage,
        "deterministic": str(deterministic).lower(),
        "model_seed": str(model_seed),
        "eval_seed": str(seed),
        "vented_t": str(vented),
    }


def test_paired_report_uses_model_seed_means_and_mode_matched_deltas(tmp_path):
    rows = []
    for model_seed in range(5):
        for eval_seed in (101, 102):
            rows.extend(
                [
                    _paired_row("state", 10 + model_seed, seed=eval_seed, model_seed=model_seed),
                    _paired_row("state_mode", 8 + model_seed, seed=eval_seed, model_seed=model_seed),
                    _paired_row("tcn", 20 + model_seed, seed=eval_seed, model_seed=model_seed),
                    _paired_row("tcn_mode", 21 + model_seed, seed=eval_seed, model_seed=model_seed),
                ]
            )
    summary_path, markdown_path = compare.write_paired_report(rows, tmp_path)

    with summary_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        summary = {row["variant"]: row for row in reader}
    assert summary["state_mode"]["paired_baseline_variant"] == "state"
    assert float(summary["state_mode"]["paired_vented_delta_mean"]) == -2.0
    assert summary["tcn_mode"]["paired_baseline_variant"] == "tcn"
    assert float(summary["tcn_mode"]["paired_vented_delta_mean"]) == 1.0
    assert int(summary["state"]["model_seeds"]) == 5
    assert float(summary["state"]["vented_t_model_sd"]) == pytest.approx(
        np.std([10, 11, 12, 13, 14], ddof=1)
    )
    assert float(summary["state"]["vented_t_ci95_half_width"]) == pytest.approx(
        2.7764451051977987 * np.std([10, 11, 12, 13, 14], ddof=1) / np.sqrt(5)
    )
    episode_path = tmp_path / "forecast_encoder_episode_summary.csv"
    with episode_path.open(encoding="utf-8", newline="") as handle:
        episode_rows = list(csv.DictReader(handle))
    assert len(episode_rows) == 20
    assert all(int(row["eval_episodes"]) == 2 for row in episode_rows)
    assert "Lower venting is the primary outcome" in markdown_path.read_text(encoding="utf-8")
    assert "model-seed" in markdown_path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    "rows, message",
    [
        ([_paired_row("state", 10.0), _paired_row("state_mode", 7.0), _paired_row("tcn", 12.0)], "missing paired keys"),
        (
            [
                _paired_row("state", 10.0),
                _paired_row("state", 11.0),
                _paired_row("state_mode", 7.0),
                _paired_row("tcn", 12.0),
                _paired_row("tcn_mode", 13.0),
            ],
            "duplicate pairing key",
        ),
    ],
)
def test_paired_report_rejects_missing_or_duplicate_pair_keys(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        compare.write_paired_report(rows, tmp_path)


def test_results_csv_is_immutable_and_per_seed_rows_are_not_aggregated(tmp_path):
    path = tmp_path / "results_state_seed0.csv"
    base = {column: "" for column in compare.RESULT_COLUMNS}
    rows = [dict(base, vented_t="1", eval_seed="101"), dict(base, vented_t="2", eval_seed="102")]
    compare.write_results_csv(path, rows)
    compare.write_results_csv(path, rows)
    with path.open(encoding="utf-8", newline="") as handle:
        assert [row["eval_seed"] for row in csv.DictReader(handle)] == ["101", "102"]
    with pytest.raises(FileExistsError, match="different content"):
        compare.write_results_csv(path, [dict(base, vented_t="9", eval_seed="101")])


def test_generate_demos_normalizes_npz_path_before_collision_check(tmp_path):
    requested = tmp_path / "shared-cache"
    actual = tmp_path / "shared-cache.npz"
    actual.write_bytes(b"do-not-overwrite")
    args = compare.parse_args(
        [
            "generate-demos",
            "--demo-cache",
            str(requested),
            "--demo-seeds",
            "1",
        ]
    )

    with (
        patch.object(compare, "collect_mpc_demonstrations") as collect,
        patch.object(compare, "save_demonstrations") as save,
        pytest.raises(FileExistsError, match="shared-cache.npz"),
    ):
        compare.generate_demos(args)

    collect.assert_not_called()
    save.assert_not_called()
    assert actual.read_bytes() == b"do-not-overwrite"


def _complete_run_manifest():
    return {
        "kind": "forecast_encoder_training_run",
        "demo_cache_path": "shared.npz",
        "demo_cache_sha256": "same-cache-sha",
        "git_commit": "abc123",
        "environment": {"schema": 1},
        "demo_seeds": [1, 2],
        "eval_seeds": [101],
        "bc": {"epochs": 20, "batch_size": 256, "learning_rate": 1e-3},
        "ppo": {"timesteps": 100_000, "n_steps": 512, "batch_size": 64},
        "kickstart": {"coefficient": 1.0, "decay": "linear"},
        "device_request": "auto",
    }


def _write_report_run(tmp_path, variant, vented, manifest):
    result_path = tmp_path / f"results_{variant}_seed0.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_paired_row(variant, vented)))
        writer.writeheader()
        writer.writerow(_paired_row(variant, vented))
    (tmp_path / f"run_{variant}_seed0.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return result_path


@pytest.mark.parametrize(
    "field, changed_value",
    [
        ("kind", "different_run_kind"),
        ("demo_cache_path", "other.npz"),
        ("demo_cache_sha256", "different-cache-sha"),
        ("git_commit", "different-commit"),
        ("environment", {"schema": 2}),
        ("demo_seeds", [3, 4]),
        ("eval_seeds", [102]),
        ("bc", {"epochs": 10, "batch_size": 256, "learning_rate": 1e-3}),
        ("ppo", {"timesteps": 200_000, "n_steps": 512, "batch_size": 64}),
        ("kickstart", {"coefficient": 0.5, "decay": "linear"}),
        ("device_request", "cpu"),
    ],
)
def test_report_rejects_scientifically_incompatible_run_manifests(
    tmp_path, field, changed_value
):
    baseline = _complete_run_manifest()
    for variant, vented in (("state", 10.0), ("state_mode", 7.0), ("tcn", 12.0), ("tcn_mode", 11.0)):
        manifest = copy.deepcopy(baseline)
        if variant == "tcn":
            manifest[field] = changed_value
        _write_report_run(tmp_path, variant, vented, manifest)

    with pytest.raises(ValueError, match=field):
        compare.report(SimpleNamespace(out_dir=str(tmp_path)))


def test_report_manifest_validation_allows_run_identity_and_diagnostics_to_differ(tmp_path):
    paths = []
    for index, variant in enumerate(("state", "state_mode", "tcn", "tcn_mode")):
        manifest = _complete_run_manifest()
        manifest.update(
            {
                "variant": variant,
                "model_seed": index,
                "policy": {"name": f"policy-{variant}"},
                "checkpoints": {"bc": f"bc-{variant}", "ppo": f"ppo-{variant}"},
                "results_csv": f"results-{variant}",
                "trainable_parameters": index + 1,
                "demonstration_accuracy": {"exact": index / 10},
                "verbose": index,
                "progress_bar": bool(index % 2),
            }
        )
        paths.append(_write_report_run(tmp_path, variant, 10.0 + index, manifest))

    compare._validate_report_manifests(paths)
