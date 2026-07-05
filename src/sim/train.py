"""Train a centralized RL policy on the CCS network and score it vs baselines.

The environment now has a hybrid action space: discrete vessel destinations plus
continuous well injection rates. The old flat ``MaskablePPO`` trainer cannot
represent that distribution, so this module keeps evaluation helpers and fails
fast if the legacy trainer is called.

Run as a script:
    PYTHONPATH=src python -m sim.train --timesteps 200000
"""

from __future__ import annotations

import argparse

from .economics import CostModel, EconomicParameters
from .control.baselines import greedy_shuttle_policy, idle_policy
from .environment import CCSEnvConfig, build_phase1_env
from .environment.gym_adapter import make_ppo_policy
from .metrics import evaluate
from .scenario_generation import ScenarioConfig


def make_native_env(
    episode_hours: int = 168,
    storage_target_rate: float = 0.9,
    warm_start: bool = True,
    storage_shortfall_penalty: float = 100.0,
):
    """A native CCSEnv on the real Phase 1 network configured for RL.

    ``storage_shortfall_penalty`` is the common storage-obligation weight used by
    RL rewards and both MILP objectives.
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
    storage_shortfall_penalty: float = 100.0,
    verbose: int = 1,
):
    raise NotImplementedError(
        "The CCS env now uses hybrid actions (discrete vessels + continuous wells). "
        "sb3_contrib.MaskablePPO only handled the removed flat discrete action space; "
        "use a hybrid PPO policy/distribution before training PPO here."
    )


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
