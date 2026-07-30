from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

from experiments import smoke_test_paper_controllers as smoke
from sim.control.rolling_milp import (
    RollingMilpController,
    _materialize_cplex_actions,
    greedy_warm_start_actions,
)
from sim.environment import CCSEnv, CCSEnvConfig
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import (
    TOY_TWO_SOURCE_LOCATIONS,
    make_toy_two_source_network,
)


def _toy_env(hours=4, scenario_hours=None):
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(
            ScenarioConfig(
                episode_hours=(
                    hours if scenario_hours is None else scenario_hours
                )
            )
        ),
        config=CCSEnvConfig(
            episode_hours=hours,
            well_control_mode="automatic_max",
        ),
    )


def test_smoke_defaults_use_validation_seed_and_short_solver_limits(tmp_path):
    args = smoke.parse_args(["--out-dir", str(tmp_path / "smoke")])

    assert args.seed == 8_100_001
    assert args.rolling_time_limit_seconds == 5.0
    assert args.full_milp_time_limit_seconds == 30.0
    assert args.rolling_planning_horizon_hours == 48
    assert args.full_milp_horizon_hours == 48


def test_greedy_warm_start_is_complete_and_replay_valid():
    env = _toy_env()
    env.reset(seed=1)

    actions = greedy_warm_start_actions(env, 4)

    assert len(actions) == 4
    assert all("vessels" in action for action in actions)
    assert all("wells" not in action for action in actions)
    assert env.simulator_step_usage().calls == 8


def test_materialized_milp_actions_omit_wells_in_automatic_mode():
    env = _toy_env(hours=2)
    env.reset(seed=1)

    actions = _materialize_cplex_actions(
        env,
        [{"vessels": [0] * len(env.vessel_ids)} for _hour in range(2)],
    )

    assert len(actions) == 2
    assert all("wells" not in action for action in actions)


def test_rolling_smoke_locks_greedy_only_without_fallback(tmp_path):
    env = _toy_env()
    fake_controller = SimpleNamespace(
        replan_count=0,
        status_counts={},
        replan_diagnostics=[],
        warm_start_mode="greedy",
        shifted_milp_warm_start=False,
    )
    fake_record = SimpleNamespace(
        as_dict=lambda: {
            "controller": "rolling_milp",
            "seed": 8_100_001,
            "total_cost": 1.0,
        }
    )
    args = Namespace(
        online_episode_hours=4,
        forecast_context_hours=168,
        rolling_replan_hours=2,
        rolling_planning_horizon_hours=4,
        rolling_time_limit_seconds=1.0,
        mip_gap_relative=None,
        seed=8_100_001,
    )

    with (
        patch.object(smoke, "make_env", return_value=env),
        patch.object(
            smoke,
            "RollingMilpController",
            return_value=fake_controller,
        ) as controller_type,
        patch.object(
            smoke,
            "run_recorded_episode",
            return_value=fake_record,
        ),
    ):
        row, _diagnostics = smoke._run_rolling_milp(args)

    kwargs = controller_type.call_args.kwargs
    assert kwargs["warm_start_mode"] == "greedy"
    assert kwargs["shifted_milp_warm_start"] is False
    assert row["fallback_used"] is False
    assert row["run_status"] == "completed"


def test_rolling_milp_plans_into_read_only_forecast_context():
    env = _toy_env(hours=2, scenario_hours=5)
    env.reset(seed=1)
    actions = [
        {"vessels": [0] * len(env.vessel_ids)}
        for _hour in range(4)
    ]
    plan = SimpleNamespace(
        status="Optimal",
        is_valid=True,
        solver_is_valid=True,
        validation_error="",
        native_actions_by_hour=actions,
        replay_is_exact=True,
        replay_mismatches=(),
        vented_t=0.0,
        shortfall_t=0.0,
    )

    with patch(
        "sim.control.rolling_milp._plan_native_cplex_actions",
        return_value=plan,
    ) as planner:
        RollingMilpController(
            env,
            replan_every=2,
            planning_horizon_h=4,
        ).policy(env)

    assert planner.call_args.args[1] == 4


def test_full_milp_plans_context_but_scores_only_evaluation_prefix():
    planning_env = _toy_env(hours=5, scenario_hours=5)
    replay_env = _toy_env(hours=2, scenario_hours=5)
    actions = [
        {"vessels": [0] * len(planning_env.vessel_ids)}
        for _hour in range(5)
    ]
    result = SimpleNamespace(
        horizon_h=5,
        stage_diagnostics=(),
        native_actions_by_hour=actions,
        status="Optimal",
        is_valid=True,
        validation_error="",
    )
    args = Namespace(
        full_milp_horizon_hours=2,
        forecast_context_hours=3,
        full_milp_time_limit_seconds=1.0,
        mip_gap_relative=None,
        seed=8_100_001,
    )

    with (
        patch.object(
            smoke,
            "make_env",
            side_effect=(planning_env, replay_env),
        ),
        patch.object(
            smoke,
            "greedy_warm_start_actions",
            return_value=actions,
        ),
        patch.object(
            smoke,
            "solve_full_scenario_with_cplex",
            return_value=result,
        ) as solve,
        patch.object(smoke, "_cleanup_cost", return_value=0.0),
    ):
        row, _diagnostics = smoke._run_full_milp(args)

    assert solve.call_args.kwargs["horizon_h"] == 5
    assert row["planning_horizon_hours"] == 5
    assert row["evaluation_horizon_hours"] == 2
    assert row["run_status"] == "completed"
    assert replay_env.t == 2
