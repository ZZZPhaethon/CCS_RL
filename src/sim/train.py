"""Train a centralized RL policy on the CCS network and score it vs baselines.

The Gym adapter presents vessel destinations plus discrete well injection-rate
indices as one flat ``MultiDiscrete`` action, so ``sb3_contrib.MaskablePPO`` can
consume the per-dimension action mask.

Run as a script:
    PYTHONPATH=src python -m sim.train --timesteps 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .economics import CostModel, EconomicParameters
from .control.baselines import greedy_shuttle_policy, idle_policy
from .environment import CCSEnvConfig, build_phase1_env
from .environment.gym_adapter import CCSGymEnv, make_ppo_policy
from .metrics import evaluate
from .scenario_generation import ScenarioConfig


def make_native_env(
    episode_hours: int = 168,
    storage_target_rate: float = 0.9,
    warm_start: bool = True,
    storage_shortfall_penalty: float = 0.0,
    injection_reward_eur_per_t: float = 0.0,
    include_weather_obs: bool = False,
    store_reward_eur_per_t: float | None = None,
    vent_penalty_weight: float = 1.0,
    operating_cost_weight: float = 1.0,
    reward_mode: str = "economic",
    vent_first_vent_eur_per_t: float = 10_000.0,
    overflow_risk_eur_per_t: float = 100.0,
    overflow_risk_lookahead_h: float = 24.0,
    carbon_price_eur_per_t: float | None = None,
    enforce_full_load_dispatch: bool = False,
    scenario: str = "northern_lights_phase1",
    include_goal_obs: bool = False,
    capture_noise_std: float = 0.30,
    initial_inventory_fill_max: float = 0.5,
    leg_wave_slowdown_multiplier: float = 1.0,
    leg_wave_speed_factor_floor: float = 0.0,
    weather_mode: str = "window",
    wave_height_nc_paths: str | Path | list[str | Path] | None = None,
    lstm_prediction_csv: str | Path | None = None,
):
    """A native CCSEnv on the real Phase 1 network configured for RL.

    ``storage_shortfall_penalty`` is passed through for experiments that
    explicitly price storage shortfall; the default leaves it as a KPI only.
    ``injection_reward_eur_per_t`` adds a dense per-step reward for injected
    CO2 (0.0 = off); a positive value fixes the short-horizon objective, which
    otherwise rewards idling until the delayed venting penalty kicks in.
    ``include_weather_obs`` exposes weather speed factors + seasonality in the
    observation so the policy can react to rough weather.
    ``carbon_price_eur_per_t`` is the single, economically-faithful knob: it sets
    both the venting carbon tax and (by default) the stored-CO2 credit to the
    same value, so storing and avoiding a vent are worth the same (symmetric).
    """
    econ_kwargs = {"storage_shortfall_eur_per_t": storage_shortfall_penalty}
    if carbon_price_eur_per_t is not None:
        econ_kwargs["carbon_price_eur_per_t"] = carbon_price_eur_per_t
        if store_reward_eur_per_t is None:
            store_reward_eur_per_t = carbon_price_eur_per_t  # symmetric credit = tax
    cost_model = CostModel(EconomicParameters(**econ_kwargs))
    return build_phase1_env(
        scenario=scenario,
        weather_mode=weather_mode,
        wave_height_nc_paths=wave_height_nc_paths,
        lstm_prediction_csv=lstm_prediction_csv,
        cost_model=cost_model,
        config=CCSEnvConfig(
            episode_hours=episode_hours,
            storage_target_rate=storage_target_rate,
            injection_reward_eur_per_t=injection_reward_eur_per_t,
            include_weather_obs=include_weather_obs,
            store_reward_eur_per_t=store_reward_eur_per_t,
            vent_penalty_weight=vent_penalty_weight,
            operating_cost_weight=operating_cost_weight,
            reward_mode=reward_mode,
            vent_first_vent_eur_per_t=vent_first_vent_eur_per_t,
            overflow_risk_eur_per_t=overflow_risk_eur_per_t,
            overflow_risk_lookahead_h=overflow_risk_lookahead_h,
            enforce_full_load_dispatch=enforce_full_load_dispatch,
            include_goal_obs=include_goal_obs,
        ),
        scenario_config=ScenarioConfig(
            episode_hours=episode_hours,
            warm_start=warm_start,
            capture_noise_std=capture_noise_std,
            emitter_initial_fill_range=(0.0, initial_inventory_fill_max),
            terminal_initial_fill_range=(0.0, initial_inventory_fill_max),
            reservoir_initial_pressure_fill_range=(0.0, initial_inventory_fill_max),
            leg_wave_slowdown_multiplier=leg_wave_slowdown_multiplier,
            leg_wave_speed_factor_floor=leg_wave_speed_factor_floor,
        ),
    )


def train_ppo(
    total_timesteps: int = 200_000,
    seed: int = 0,
    gamma: float = 0.999,
    episode_hours: int = 168,
    warm_start: bool = True,
    storage_shortfall_penalty: float = 0.0,
    injection_reward_eur_per_t: float = 0.0,
    verbose: int = 1,
    n_steps: int = 128,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device: str = "auto",
    progress_bar: bool = False,
):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("train_ppo requires sb3-contrib. Install with `pip install sb3-contrib`.") from exc

    native_env = make_native_env(
        episode_hours=episode_hours,
        warm_start=warm_start,
        storage_shortfall_penalty=storage_shortfall_penalty,
        injection_reward_eur_per_t=injection_reward_eur_per_t,
    )
    gym_env = CCSGymEnv(native_env)
    model = MaskablePPO(
        "MlpPolicy",
        gym_env,
        seed=seed,
        gamma=gamma,
        verbose=verbose,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    if verbose:
        print(f"[train_ppo] policy device = {model.policy.device}", flush=True)
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
    return model


def compare(model, seeds: list[int], episode_hours: int = 168, warm_start: bool = False):
    """Score idle / greedy_shuttle / PPO on the same scenarios."""
    policies = {
        "idle": idle_policy,
        "greedy_shuttle": greedy_shuttle_policy,
        "ppo_stochastic": make_ppo_policy(model, deterministic=False),
        "ppo_deterministic": make_ppo_policy(model, deterministic=True),
    }
    rows = {}
    for name, policy in policies.items():
        env = make_native_env(episode_hours=episode_hours, warm_start=warm_start)
        _episodes, summary = evaluate(env, policy, seeds=seeds)
        rows[name] = summary
    return rows


def _summary_mean(summary: dict, key: str) -> float:
    return float(summary.get(key, {}).get("mean", float("nan")))


def _format_comparison(rows: dict) -> str:
    header = (
        f"{'policy':18} {'storage%':>9} {'loss%':>7} {'stored t':>10} "
        f"{'vented t':>10} {'op EUR':>12} {'vent EUR':>12} "
        f"{'total EUR':>12} {'op/t':>8} {'total/t':>9}"
    )
    lines = [header, "-" * len(header)]
    for name, s in rows.items():
        lines.append(
            f"{name:18} {_summary_mean(s, 'storage_rate') * 100:8.1f}% "
            f"{_summary_mean(s, 'loss_rate') * 100:6.1f}% "
            f"{_summary_mean(s, 'stored_t'):10,.0f} "
            f"{_summary_mean(s, 'vented_t'):10,.0f} "
            f"{_summary_mean(s, 'operating_cost'):12,.0f} "
            f"{_summary_mean(s, 'vent_penalty'):12,.0f} "
            f"{_summary_mean(s, 'total_cost'):12,.0f} "
            f"{_summary_mean(s, 'cost_per_stored_t'):8,.1f} "
            f"{_summary_mean(s, 'total_cost_per_stored_t'):9,.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid-action PPO policy on the CCS network.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--episode-hours", type=int, default=168)
    parser.add_argument(
        "--injection-reward-eur-per-t",
        type=float,
        default=0.0,
        help="Dense per-step reward per tonne injected (0 = off; try 80 to match the carbon price).",
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    args = parser.parse_args()

    model = train_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        gamma=args.gamma,
        episode_hours=args.episode_hours,
        injection_reward_eur_per_t=args.injection_reward_eur_per_t,
    )
    rows = compare(model, seeds=args.eval_seeds, episode_hours=args.episode_hours)
    print("\n=== PPO vs baselines (Phase 1, evaluation seeds) ===")
    print(_format_comparison(rows))


if __name__ == "__main__":
    main()
