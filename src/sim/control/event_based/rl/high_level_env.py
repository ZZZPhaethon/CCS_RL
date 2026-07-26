"""Event-triggered semi-MDP environment for high-level CCS dispatch RL.

用于高层 CCS 调度 RL 的事件触发半 MDP 环境。

One high-level action becomes a :class:`DispatchGoal`.  A low-level executor
then applies legal native actions at each physical hour.  The returned reward
and transition metrics are aggregated from realised simulator outcomes.

一个高层动作被转为 :class:`DispatchGoal`。底层执行器随后在每个物理小时使用合法的原生动作。返回的
奖励和转移指标均由仿真器的实际结果聚合而成。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim.environment import CCSEnv
from sim.environment.vessel_mode import vessel_operation_modes

from ..contracts import ActionExecutor
from ..hybrid import GoalAwareRuleExecutor
from .action_codec import HighLevelActionCodec
from .observation_encoder import high_level_observation, high_level_observation_size
from .reward import HighLevelRewardConfig, high_level_reward


@dataclass(frozen=True)
class HighLevelEnvConfig:
    """Configure the decision interval and reward for high-level RL.

    配置高层 RL 的决策间隔与奖励。
    """

    decision_interval_h: float = 24.0
    event_triggered: bool = True
    emitter_fill_event_thresholds: tuple[float, ...] = (0.8, 0.95)
    minimum_operable_speed_factor: float = 0.75
    reward: HighLevelRewardConfig = HighLevelRewardConfig()

    def __post_init__(self) -> None:
        """Validate time and event thresholds before an episode starts.

        在回合开始前校验时间与事件阈值。
        """
        if self.decision_interval_h <= 0.0:
            raise ValueError("decision_interval_h must be positive.")
        thresholds = self.emitter_fill_event_thresholds
        if tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError(
                "emitter_fill_event_thresholds must be unique and increasing."
            )
        if any(not 0.0 < threshold < 1.0 for threshold in thresholds):
            raise ValueError("Emitter fill thresholds must be inside (0, 1).")
        if self.minimum_operable_speed_factor < 0.0:
            raise ValueError("minimum_operable_speed_factor must be non-negative.")


@dataclass(frozen=True)
class _DecisionEventSnapshot:
    """Store only discrete physical changes that can justify replanning.

    仅保存能够触发重规划的离散物理变化。
    """

    vessel_modes: tuple[str, ...]
    emitter_fill_bands: tuple[int, ...]
    well_available: tuple[bool, ...]
    weather_operable: tuple[bool, ...]
    overflow_active: bool


class HighLevelDispatchEnv:
    """Native semi-MDP that aggregates hourly CCS control into sparse decisions.

    将每小时 CCS 控制聚合为稀疏决策的原生半 MDP。
    """

    def __init__(
        self,
        env: CCSEnv,
        *,
        config: HighLevelEnvConfig | None = None,
        executor_factory: Callable[[], ActionExecutor] = GoalAwareRuleExecutor,
    ) -> None:
        """Wrap a physical environment and choose its low-level executor.

        包装一个物理环境并选择底层执行器。
        """
        self.env = env
        self.config = config or HighLevelEnvConfig()
        self.executor_factory = executor_factory
        steps = self.config.decision_interval_h / self.env.network.time_step_hours
        if steps <= 0.0 or abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                "decision_interval_h must be a positive multiple of the physical time step."
            )
        self._decision_steps = int(round(steps))
        self.codec = HighLevelActionCodec(
            self.env.emitter_ids,
            self.env.vessel_ids,
            self.env.well_ids,
        )
        self.executor: ActionExecutor | None = None

    @property
    def action_count(self) -> int:
        """Return the number of discrete high-level goals.

        返回离散高层目标的数量。
        """
        return self.codec.action_count

    @property
    def observation_size(self) -> int:
        """Return the fixed encoded observation size.

        返回固定的编码观测维度。
        """
        return high_level_observation_size(self.env)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the physical episode and return the first sparse observation.

        重置物理回合并返回首个稀疏观测。
        """
        self.env.reset(seed=seed)
        self.executor = self.executor_factory()
        return high_level_observation(self.env)

    def step(
        self,
        action_index: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one high-level goal through hourly physical control.

        通过逐小时物理控制执行一个高层目标。
        """
        if self.env.simulator is None or self.executor is None:
            raise RuntimeError("Call reset() before step().")
        goal = self.codec.decode(
            int(action_index),
            replan_after_h=self.config.decision_interval_h,
        )
        started_at_h = self.env.simulator.state.time_h
        start_stored_t = self.env.cumulative_stored_t
        start_vented_t = self.env.ledger.vented_t
        start_operating_cost = self.env.ledger.operating_cost
        start_total_cost = self.env.ledger.total_cost
        start_captured_t = self.env.cumulative_captured_t
        overflow_risk_t_hours = 0.0
        native_reward = 0.0
        violations: Counter[str] = Counter()
        terminated = False
        truncated = False
        native_steps = 0
        event_snapshot = _decision_event_snapshot(self.env, self.config)
        decision_trigger = "maximum_interval"

        for _ in range(self._decision_steps):
            action = self.executor.propose_action(goal, {"env": self.env})
            _obs, reward, terminated, truncated, info = self.env.step(action)
            native_steps += 1
            native_reward += float(reward)
            overflow_risk_t_hours += (
                float(info.get("overflow_risk_t", 0.0))
                * self.env.network.time_step_hours
            )
            violations.update(str(code) for code in info.get("violations", []))
            if terminated or truncated:
                decision_trigger = "episode_end"
                break
            next_snapshot = _decision_event_snapshot(self.env, self.config)
            event_reason = _replan_event_reason(
                event_snapshot,
                next_snapshot,
                self.env.vessel_ids,
                self.env.emitter_ids,
                self.env.well_ids,
            )
            event_snapshot = next_snapshot
            if self.config.event_triggered and event_reason is not None:
                decision_trigger = event_reason
                break

        stored_t = self.env.cumulative_stored_t - start_stored_t
        vented_t = self.env.ledger.vented_t - start_vented_t
        operating_cost = self.env.ledger.operating_cost - start_operating_cost
        total_cost = self.env.ledger.total_cost - start_total_cost
        captured_t = self.env.cumulative_captured_t - start_captured_t
        reward, reward_breakdown = high_level_reward(
            stored_t=stored_t,
            captured_t=captured_t,
            vented_t=vented_t,
            operating_cost=operating_cost,
            overflow_risk_t_hours=overflow_risk_t_hours,
            violation_counts=violations,
            config=self.config.reward,
        )
        elapsed_hours = self.env.simulator.state.time_h - started_at_h
        info = {
            "action_label": self.codec.label(int(action_index)),
            "dispatch_goal": {
                "emitter_to_vessel": dict(goal.emitter_to_vessel),
                "vessel_to_emitter": dict(goal.vessel_to_emitter),
                "well_rate_indices": dict(goal.well_rate_indices),
            },
            "elapsed_hours": elapsed_hours,
            "native_steps": native_steps,
            "decision_trigger": decision_trigger,
            "stored_t": stored_t,
            "vented_t": vented_t,
            "captured_t": captured_t,
            "operating_cost": operating_cost,
            "total_cost": total_cost,
            "overflow_risk_t_hours": overflow_risk_t_hours,
            "violation_counts": dict(sorted(violations.items())),
            "high_level_reward": reward_breakdown,
            "native_hourly_reward": native_reward,
            "cumulative_stored_t": self.env.cumulative_stored_t,
            "cumulative_vented_t": self.env.ledger.vented_t,
            "cumulative_total_cost": self.env.ledger.total_cost,
        }
        return high_level_observation(self.env), reward, terminated, truncated, info


def _decision_event_snapshot(
    env: CCSEnv,
    config: HighLevelEnvConfig,
) -> _DecisionEventSnapshot:
    """Encode physical event bands without reacting to continuous noise.

    将物理状态编码为事件区间，避免连续扰动引起频繁重规划。
    """
    assert env.simulator is not None
    state = env.simulator.state
    fill_bands: list[int] = []
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        inventory_t = state.entity_inventory_t.get(emitter_id, 0.0)
        fill = inventory_t / max(1e-9, emitter.buffer_capacity_t)
        fill_bands.append(
            sum(fill >= threshold for threshold in config.emitter_fill_event_thresholds)
        )
    speed_operable = tuple(
        state.vessel_speed_factor.get(vessel_id, 1.0)
        >= config.minimum_operable_speed_factor
        for vessel_id in env.vessel_ids
    )
    return _DecisionEventSnapshot(
        vessel_modes=vessel_operation_modes(env),
        emitter_fill_bands=tuple(fill_bands),
        well_available=tuple(
            bool(state.well_available.get(well_id, True))
            for well_id in env.well_ids
        ),
        weather_operable=speed_operable,
        overflow_active=env._overflow_risk_t() > 1e-9,
    )


def _replan_event_reason(
    previous: _DecisionEventSnapshot,
    current: _DecisionEventSnapshot,
    vessel_ids: list[str],
    emitter_ids: list[str],
    well_ids: list[str],
) -> str | None:
    """Return the first operational event that warrants a new high-level goal.

    返回首个需要新高层目标的运行事件。
    """
    for position, vessel_id in enumerate(vessel_ids):
        before = previous.vessel_modes[position]
        after = current.vessel_modes[position]
        if before == "sailing" and after != "sailing":
            return f"vessel_arrival:{vessel_id}"
        if before in {"loading", "unloading"} and after not in {
            before,
            "sailing",
        }:
            return f"{before}_completed:{vessel_id}"
    for position, emitter_id in enumerate(emitter_ids):
        if current.emitter_fill_bands[position] > previous.emitter_fill_bands[position]:
            return f"emitter_fill_threshold:{emitter_id}"
    for position, well_id in enumerate(well_ids):
        if current.well_available[position] != previous.well_available[position]:
            return f"well_availability_changed:{well_id}"
    for position, vessel_id in enumerate(vessel_ids):
        if current.weather_operable[position] != previous.weather_operable[position]:
            return f"weather_threshold:{vessel_id}"
    if current.overflow_active and not previous.overflow_active:
        return "overflow_risk_started"
    return None

