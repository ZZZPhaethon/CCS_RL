"""Run comparable physical rollouts for algorithm-layer controllers.

为算法层控制器运行可比的物理回放。

All controllers are evaluated by stepping the same ``CCSEnv`` interface.  This
prevents comparing an optimiser's planned flow with an RL or rule controller's
realised physical flow.

所有控制器均通过同一 ``CCSEnv`` 接口评估。这避免把优化器的计划流量与 RL
或规则控制器的实际物理流量直接比较。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping

from sim.environment import CCSEnv

from .contracts import ActionExecutor, DispatchGoal


@dataclass(frozen=True)
class RolloutMetrics:
    """Realised metrics from one physically executed controller rollout.

    单次控制器在物理实执回放中的实际指标。
    """

    elapsed_hours: float
    captured_t: float
    stored_t: float
    vented_t: float
    operating_cost: float
    total_cost: float
    total_reward: float
    unit_storage_cost_eur_per_t: float | None
    storage_rate: float
    vent_rate: float
    wall_clock_seconds: float
    violation_counts: Mapping[str, int]


def evaluate_executor(
    env: CCSEnv,
    executor: ActionExecutor,
    goal: DispatchGoal,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> RolloutMetrics:
    """Execute an algorithm through one environment and collect realised metrics.

    在一个环境中执行算法，并收集实际发生的指标。

    ``env`` must be a fresh environment, or one whose previous episode is no
    longer needed.  Use a new environment for each controller to guarantee a
    fair comparison.

    ``env`` 必须是新环境，或者不再需保留旧回合的环境。每个控制器应使用新环境，以保证
    比较公平。
    """
    started_at = perf_counter()
    env.reset(seed=seed)
    total_reward = 0.0
    violations: Counter[str] = Counter()
    limit = env.n_steps if max_steps is None else min(env.n_steps, max(0, max_steps))

    for _ in range(limit):
        action = executor.propose_action(goal, {"env": env})
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        violations.update(_violation_code(value) for value in info.get("violations", []))
        if terminated or truncated:
            break

    captured_t = float(env.cumulative_captured_t)
    stored_t = float(env.cumulative_stored_t)
    vented_t = float(env.ledger.vented_t)
    total_cost = float(env.ledger.total_cost)
    return RolloutMetrics(
        elapsed_hours=float(env.simulator.state.time_h),
        captured_t=captured_t,
        stored_t=stored_t,
        vented_t=vented_t,
        operating_cost=float(env.ledger.operating_cost),
        total_cost=total_cost,
        total_reward=total_reward,
        unit_storage_cost_eur_per_t=(
            total_cost / stored_t if stored_t > 1e-9 else None
        ),
        storage_rate=stored_t / captured_t if captured_t > 1e-9 else 0.0,
        vent_rate=vented_t / captured_t if captured_t > 1e-9 else 0.0,
        wall_clock_seconds=perf_counter() - started_at,
        violation_counts=dict(sorted(violations.items())),
    )


def compare_executors(
    environment_factory: Callable[[], CCSEnv],
    executors: Mapping[str, ActionExecutor],
    goal: DispatchGoal,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> dict[str, RolloutMetrics]:
    """Evaluate named executors on independent but identical environment resets.

    在独立但完全一致的环境重置上评估带名控制器。
    """
    return {
        name: evaluate_executor(
            environment_factory(),
            executor,
            goal,
            seed=seed,
            max_steps=max_steps,
        )
        for name, executor in executors.items()
    }


def _violation_code(value: object) -> str:
    """Extract a stable violation code from a simulator diagnostic.

    从仿真诊断中提取稳定的违规代码。
    """
    code = getattr(value, "code", None)
    return str(code) if code is not None else str(value)

