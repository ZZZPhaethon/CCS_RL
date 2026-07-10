from types import SimpleNamespace
from dataclasses import replace

from sim.scenario_generation import ScenarioConfig


def test_weekly_weather_probability_defaults_to_one_when_unspecified():
    from experiments import window_stress_headroom as window_stress

    args = SimpleNamespace(
        hours=1,
        weather_window_mode="weekly",
        capture_noise_std=0.10,
        capture_outage_rate_per_week=0.0,
        capture_outage_mean_hours=12.0,
        capture_high_output_mean_hours=48.0,
        capture_multiplier_min=1.25,
        capture_multiplier_max=1.75,
        weather_window_mean_hours=48.0,
        weather_speed_min=0.6,
        weather_speed_max=0.8,
        well_maintenance_rate_per_week=0.0,
        well_maintenance_mean_hours=24.0,
        weather_weekly_probability=None,
        weather_weekly_duration_min_hours=None,
        weather_weekly_duration_max_hours=None,
        yara_buffer_t=7500.0,
        terminal_buffer_t=7500.0,
        storage_reward_eur_per_t=0.0,
        reward_mode="vent_first",
        vent_first_vent_eur_per_t=10_000.0,
        overflow_risk_eur_per_t=100.0,
        overflow_risk_lookahead_h=24.0,
        operating_cost_weight=1.0,
    )

    env = window_stress.make_env(
        args,
        window_stress.EconomicParameters(),
        capture_high_output_rate_per_week=0.5,
        weather_rate_per_week=0.3,
    )

    assert env.scenario_generator.weekly_probability == 1.0
    assert env.config.reward_mode == "vent_first"
    assert env.config.vent_first_vent_eur_per_t == 10_000.0
    assert env.config.overflow_risk_eur_per_t == 100.0
    assert env.config.operating_cost_weight == 1.0


def test_default_scenario_profile_matches_scenario_config_except_episode_hours():
    from experiments import window_stress_headroom as window_stress

    args = SimpleNamespace(
        hours=720,
        weather_window_mode="weekly",
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        capture_outage_mean_hours=1.0,
        capture_high_output_mean_hours=1.0,
        capture_multiplier_min=1.0,
        capture_multiplier_max=1.0,
        weather_window_mean_hours=1.0,
        weather_speed_min=1.0,
        weather_speed_max=1.0,
        well_maintenance_rate_per_week=0.0,
        well_maintenance_mean_hours=1.0,
        capture_high_output_rates=None,
        weather_rates=None,
    )

    window_stress.apply_scenario_config_defaults(args)
    defaults = ScenarioConfig()
    config = window_stress.make_config(
        args,
        defaults.capture_high_output_rate_per_week,
        defaults.weather_window_rate_per_week,
    )

    assert config == replace(defaults, episode_hours=720)
    assert args.weather_window_mode == "hourly"
    assert args.capture_high_output_rates == "0.5"
    assert args.weather_rates == "0.3"


def test_native_action_oracle_dispatches_to_full_scenario_solver(monkeypatch):
    from experiments import window_stress_headroom as window_stress

    expected = object()
    captured = {}

    def fake_solver(env, **kwargs):
        captured["env"] = env
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(window_stress.cplex_milp, "solve_full_scenario_with_cplex", fake_solver)
    args = SimpleNamespace(
        oracle_model="native_action",
        hours=360,
        storage_reward_eur_per_t=0.0,
        cplex_time_limit_s=300.0,
        cplex_mip_gap_rel=0.02,
        cplex_threads=1,
        cplex_msg=False,
    )
    env = object()
    economics = object()
    warm_start = [{"vessels": [0], "wells": [0]}]

    result = window_stress.solve_oracle(args, env, economics, warm_start)

    assert result is expected
    assert captured["env"] is env
    assert captured["horizon_h"] == 360
    assert captured["warm_start_native_actions_by_hour"] == warm_start


def test_headroom_summary_requires_exact_replays():
    from experiments import window_stress_headroom as window_stress

    rows = [
        {
            "vent_penalty_eur_per_t": 80.0,
            "weather_window_mode": "hourly",
            "rate_per_week": 0.5,
            "weather_rate_per_week": 0.3,
            "seed": 1,
            "replay_is_executable": True,
            "replay_is_exact": False,
            "greedy_stored_t": 10.0,
            "optimized_stored_t": 20.0,
            "vented_reduction_t": 5.0,
        }
    ]

    summary = window_stress.summarize(rows)

    assert summary[0]["all_replay_executable"] is True
    assert summary[0]["all_replay_exact"] is False
    assert summary[0]["greedy_stored_t_mean"] == 10.0
    assert "optimized_stored_t_mean" not in summary[0]
    assert "vented_reduction_t_mean" not in summary[0]
