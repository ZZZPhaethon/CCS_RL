import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scripts import compare_forecast_encoders_rl as compare
from sim.environment.forecast import current_state_feature_names, forecast_channel_names
from sim.environment.forecast_encoder import TCNForecastExtractor
from sim.metrics import EpisodeMetrics


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

    assert {train.command, demos.command, report.command} == {"train", "generate-demos", "report"}
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
    assert train.kickstart_coef == 1.0
    assert train.eval_seeds == [101, 102, 103, 104, 105]
    assert train.model_seed == 0


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


def test_metadata_is_derived_from_environment_helpers_without_schema_drift(tmp_path):
    args = _train_args(tmp_path)
    factory = compare.ExperimentEnvFactory(args)
    metadata = factory.metadata()
    env = factory()

    assert metadata["forecast_channels"] == list(forecast_channel_names(env))
    assert metadata["forecast_shape"] == [168, 9]
    assert metadata["state_feature_names"] == list(current_state_feature_names(env))
    assert metadata["state_size"] == len(current_state_feature_names(env)) == 51
    assert metadata["action_dimensions"] == [*env.vessel_action_dims, *env.well_rate_action_dims]
    assert metadata["weather_mode"] == "block"
    assert metadata["weather_observation_layout"] == "global"
    assert metadata["reward"]["mode"] == "vent_first"
    assert metadata["partial_load_dispatch"] is True
    assert metadata["warm_start"] is True
    assert metadata["scenario_context_hours"] == 169
    assert metadata["emitter_buffer_capacity_t"]["yara_sluiskil"] == 15_000.0


def test_policy_mapping_uses_custom_extractor_only_for_tcn():
    assert compare.model_policy_config("state") == ("MlpPolicy", {})
    assert compare.model_policy_config("flat") == ("MlpPolicy", {})
    policy, kwargs = compare.model_policy_config("tcn")
    assert policy == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is TCNForecastExtractor
    assert kwargs["features_extractor_kwargs"] == {
        "state_features": 64,
        "forecast_features": 64,
    }


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


def _paired_row(variant: str, vented: float, *, seed: int = 101):
    return {
        "policy": f"learned_{variant}",
        "family": "learned",
        "variant": variant,
        "stage": "ppo",
        "deterministic": "true",
        "model_seed": "0",
        "eval_seed": str(seed),
        "vented_t": str(vented),
    }


def test_paired_report_computes_flat_and_tcn_deltas(tmp_path):
    rows = [_paired_row("state", 10.0), _paired_row("flat", 7.0), _paired_row("tcn", 12.0)]
    summary_path, markdown_path = compare.write_paired_report(rows, tmp_path)

    with summary_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames.index("vented_t_mean") < reader.fieldnames.index(
            "paired_vented_delta_vs_state_mean"
        )
        summary = {row["variant"]: row for row in reader}
    assert float(summary["flat"]["paired_vented_delta_vs_state_mean"]) == -3.0
    assert float(summary["tcn"]["paired_vented_delta_vs_state_mean"]) == 2.0
    assert "Lower venting is the primary outcome" in markdown_path.read_text(encoding="utf-8")
    assert "statistical significance" not in markdown_path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    "rows, message",
    [
        ([_paired_row("state", 10.0), _paired_row("flat", 7.0)], "missing paired keys"),
        (
            [
                _paired_row("state", 10.0),
                _paired_row("state", 11.0),
                _paired_row("flat", 7.0),
                _paired_row("tcn", 12.0),
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


def test_report_rejects_runs_from_different_cache_manifests(tmp_path):
    for variant, vented in (("state", 10.0), ("flat", 7.0), ("tcn", 12.0)):
        result_path = tmp_path / f"results_{variant}_seed0.csv"
        with result_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_paired_row(variant, vented)))
            writer.writeheader()
            writer.writerow(_paired_row(variant, vented))
        manifest = {
            "demo_cache_path": "shared.npz",
            "demo_cache_sha256": "same" if variant != "tcn" else "different",
            "git_commit": "abc123",
            "environment": {"schema": 1},
        }
        (tmp_path / f"run_{variant}_seed0.manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="demo_cache_sha256"):
        compare.report(SimpleNamespace(out_dir=str(tmp_path)))
