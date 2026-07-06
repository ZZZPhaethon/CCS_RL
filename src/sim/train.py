"""Train a centralized RL policy on the CCS network and score it vs baselines.

The Gym adapter presents vessel destinations plus discrete well injection-rate
indices as one flat ``MultiDiscrete`` action, so ``sb3_contrib.MaskablePPO`` can
consume the per-dimension action mask.

Run as a script:
    PYTHONPATH=src python -m sim.train --timesteps 200000
"""

from __future__ import annotations

import argparse

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
):
    """A native CCSEnv on the real Phase 1 network configured for RL.

    ``storage_shortfall_penalty`` is passed through for experiments that
    explicitly price storage shortfall; the default leaves it as a KPI only.
    """
    cost_model = CostModel(EconomicParameters(storage_shortfall_eur_per_t=storage_shortfall_penalty))
    return build_phase1_env(
        cost_model=cost_model,
        config=CCSEnvConfig(episode_hours=episode_hours, storage_target_rate=storage_target_rate),
        scenario_config=ScenarioConfig(episode_hours=episode_hours, warm_start=warm_start),
    )


def train_ppo(
    total_timesteps: int = 200_000,
    seed: int = 0,
    gamma: float = 0.999,
    episode_hours: int = 168,
    warm_start: bool = True,
    storage_shortfall_penalty: float = 0.0,
    verbose: int = 1,
    n_steps: int = 128,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("train_ppo requires sb3-contrib. Install with `pip install sb3-contrib`.") from exc

    native_env = make_native_env(
        episode_hours=episode_hours,
        warm_start=warm_start,
        storage_shortfall_penalty=storage_shortfall_penalty,
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
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def compare(model, seeds: list[int], episode_hours: int = 168, warm_start: bool = False):
    """Score idle / greedy_shuttle / PPO on the same scenarios."""
    policies = {
        "idle": idle_policy,
        "greedy_shuttle": greedy_shuttle_policy,
        "ppo": make_ppo_policy(model),
    }
    rows = {}
    for name, policy in policies.items():
        env = make_native_env(episode_hours=episode_hours, warm_start=warm_start)
        _episodes, summary = evaluate(env, policy, seeds=seeds)
        rows[name] = summary
    return rows


def _format_comparison(rows: dict) -> str:
    header = f"{'policy':16} {'storage%':>9} {'loss%':>7} {'shortfall EUR':>14} {'total EUR':>14}"
    lines = [header, "-" * len(header)]
    for name, s in rows.items():
        lines.append(
            f"{name:16} {s['storage_rate']['mean'] * 100:8.1f}% "
            f"{s['loss_rate']['mean'] * 100:6.1f}% "
            f"{s['storage_shortfall_penalty']['mean']:14,.0f} "
            f"{s['total_cost']['mean']:14,.0f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid-action PPO policy on the CCS network.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--episode-hours", type=int, default=168)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    args = parser.parse_args()

    model = train_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        gamma=args.gamma,
        episode_hours=args.episode_hours,
    )
    rows = compare(model, seeds=args.eval_seeds, episode_hours=args.episode_hours)
    print("\n=== PPO vs baselines (Phase 1, evaluation seeds) ===")
    print(_format_comparison(rows))


if __name__ == "__main__":
    main()
