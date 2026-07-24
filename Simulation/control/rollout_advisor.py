"""Online context and event triggers from the native rollout-rule planner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..environment import CCSEnv, VESSEL_WAIT
from ..environment.forecast import (
    current_state_observation,
    future_forecast_observation,
    replan_phase_observation,
)
from ..environment.vessel_mode import (
    vessel_operation_mode_observation,
    vessel_sailing_destination_observation,
)
from .native_mpc import RollingNativeMpcController, native_mpc_candidate_names


@dataclass
class RollingRolloutAdvisor:
    """Expose a selected rollout rule as legitimate, online plan context.

    The advisor is reset with each episode.  It retains the rule schedule for
    the 24-hour execution window, but replans early if a learner action makes
    that trace infeasible.  It deliberately exposes only the selected-rule
    one-hot vector (whose length is read from the scenario), not the rule's
    current native action, so BC cannot copy its action label from the
    observation.
    """

    replan_every: int = 24
    planning_horizon_h: int = 168
    _controller: RollingNativeMpcController | None = None
    _candidate_names: tuple[str, ...] | None = None
    unscheduled_replans: int = 0

    def reset(self, env: CCSEnv) -> None:
        self._candidate_names = native_mpc_candidate_names(env)
        self._controller = RollingNativeMpcController(
            env,
            replan_every=self.replan_every,
            planning_horizon_h=self.planning_horizon_h,
        )
        self.unscheduled_replans = 0

    def policy(self, env: CCSEnv) -> dict[str, list]:
        if self._controller is None:
            self.reset(env)
        assert self._controller is not None
        try:
            return self._controller.policy(env)
        except RuntimeError as error:
            # A lower-level PPO action may differ from the plan.  Do not keep
            # feeding an invalid trace; receding-horizon planning replans here.
            if "trace" not in str(error) and "infeasible" not in str(error):
                raise
            self._controller._plan_origin_h = -1e9  # reset only this stale plan
            self.unscheduled_replans += 1
            return self._controller.policy(env)

    def trigger(self, env: CCSEnv) -> tuple[str, ...]:
        """Wake the executor exactly when the retained rollout plan dispatches."""
        action = self.policy(env)
        return tuple(
            vessel_id
            for vessel_id, choice in zip(env.vessel_ids, action["vessels"])
            if int(choice) != VESSEL_WAIT
        )

    def candidate_context(self, env: CCSEnv) -> np.ndarray:
        self.policy(env)
        assert self._controller is not None
        assert self._candidate_names is not None
        try:
            index = self._candidate_names.index(self._controller.last_candidate_name)
        except ValueError as error:  # defensive schema check
            raise RuntimeError("rollout planner selected an unknown candidate") from error
        context = np.zeros(len(self._candidate_names), dtype=np.float32)
        context[index] = 1.0
        return context

    def event_observation(self, event_env) -> np.ndarray:
        """Forecast-v4 + vessel state + replan phase + online rollout context."""
        env = event_env.env
        state = np.asarray(current_state_observation(env), dtype=np.float32)
        mode = np.asarray(vessel_operation_mode_observation(env), dtype=np.float32)
        destination = np.asarray(vessel_sailing_destination_observation(env), dtype=np.float32)
        phase = np.asarray(replan_phase_observation(env.simulator.state.time_h), dtype=np.float32)
        forecast = np.asarray(future_forecast_observation(env), dtype=np.float32)
        return np.concatenate(
            (
                state,
                mode.reshape(-1),
                destination.reshape(-1),
                phase,
                self.candidate_context(env),
                forecast.reshape(-1),
            )
        ).astype(np.float32, copy=False)

    def event_observation_size(self, env: CCSEnv) -> int:
        return int(
            len(current_state_observation(env))
            + np.asarray(vessel_operation_mode_observation(env)).size
            + np.asarray(vessel_sailing_destination_observation(env)).size
            + 2
            + len(native_mpc_candidate_names(env))
            + np.asarray(future_forecast_observation(env)).size
        )
