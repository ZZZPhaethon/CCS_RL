"""Sparse vessel-dispatch decisions over the hourly CCS physical simulator.

The physical model still advances one hour at a time.  This wrapper changes
only the policy clock: it returns a :class:`DecisionEvent` when a vessel enters
a new dispatch stage and automatically advances forced periods in between.

The design follows MARO's ``DecisionEvent -> action -> simulation`` boundary,
while retaining the project's existing ``CCSEnv`` and its hourly physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from .env import CCSEnv, MIN_WELL_RATE_INDEX, OFF_WELL_RATE_INDEX, VESSEL_WAIT


class VesselDecisionType(str, Enum):
    """A vessel stage with a non-forced dispatch decision."""

    IDLE = "idle_dispatch"
    LOADING = "loading_dispatch"
    FULL_LOAD = "full_load_dispatch"
    QUEUED = "queued_dispatch"
    PLANNER_REPLAN = "planner_replan"


@dataclass(frozen=True)
class DecisionEvent:
    """The dispatch choices that are live at one physical simulation time."""

    time_h: float
    vessel_ids: tuple[str, ...]
    vessel_types: dict[str, VesselDecisionType]
    vessel_action_masks: dict[str, tuple[bool, ...]]


@dataclass(frozen=True)
class HourlyTransition:
    """One hidden hourly physics transition inside an event-level step."""

    reward: float
    info: dict


@dataclass
class EventTransition:
    """Hourly rewards accumulated until the next policy-facing event."""

    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    decision_event: DecisionEvent | None
    elapsed_hours: float
    hourly_transitions: tuple[HourlyTransition, ...]


# A planner may request an early re-evaluation for selected vessels.  This is
# the hook through which rolling-MPC plan changes or predicted overflow risk can
# create an event while a ship is still loading.
ReplanTrigger = Callable[[CCSEnv], Iterable[str] | None]
ReplanResetHook = Callable[[CCSEnv], None]


class EventDrivenCCSEnv:
    """Policy wrapper that event-sparsifies vessel dispatches.

    The public action retains the native ``{"vessels": ..., "wells": ...}``
    format.  At an event only listed vessels may take a non-WAIT action; all
    other vessel dimensions are forced to WAIT.  Well-rate commands are held
    across automatic hourly transitions, so the wrapper does not silently
    change the injection controller's command.

    A vessel is asked once on entering ``IDLE``, ``LOADING``, or ``FULL_LOAD``.
    Choosing WAIT at a loading event therefore means "continue loading until a
    new stage or replan event", rather than creating another hourly WAIT label.
    """

    _EPS = 1e-9

    def __init__(
        self,
        env: CCSEnv,
        *,
        replan_trigger: ReplanTrigger | None = None,
        replan_reset_hook: ReplanResetHook | None = None,
    ) -> None:
        self.env = env
        self.replan_trigger = replan_trigger
        self.replan_reset_hook = replan_reset_hook
        self._acknowledged_stages: dict[str, str] = {}
        self._last_well_actions: list[int] = []
        self._current_event: DecisionEvent | None = None

    @property
    def current_event(self) -> DecisionEvent | None:
        return self._current_event

    def reset(self, seed: int | None = None) -> EventTransition:
        """Reset the hourly environment and expose its first decision event."""
        observation = self.env.reset(seed=seed)
        if self.replan_reset_hook is not None:
            self.replan_reset_hook(self.env)
        self._acknowledged_stages = {}
        self._last_well_actions = [self._default_well_action(well_id) for well_id in self.env.well_ids]
        self._current_event = None
        return self._advance_until_event(observation, 0.0, 0.0, {}, hourly_transitions=())

    def step(self, action: dict[str, list]) -> EventTransition:
        """Act at the current event, then advance through forced hourly steps."""
        if self._current_event is None:
            raise RuntimeError("No active DecisionEvent. Call reset() before step().")
        normalized = self._event_action(action, self._current_event)
        self._last_well_actions = list(normalized["wells"])
        observation, reward, terminated, truncated, info = self.env.step(normalized)
        self._current_event = None
        return self._advance_until_event(
            observation,
            float(reward),
            float(self.env.network.time_step_hours),
            info,
            terminated=terminated,
            truncated=truncated,
            hourly_transitions=(HourlyTransition(float(reward), dict(info)),),
        )

    def _advance_until_event(
        self,
        observation: list[float],
        reward: float,
        elapsed_hours: float,
        info: dict,
        *,
        terminated: bool = False,
        truncated: bool = False,
        hourly_transitions: tuple[HourlyTransition, ...],
    ) -> EventTransition:
        while not (terminated or truncated):
            event = self._next_event()
            if event is not None:
                self._current_event = event
                return self._transition(
                    observation, reward, terminated, truncated, info, event, elapsed_hours, hourly_transitions
                )

            automatic_action = {
                "vessels": [VESSEL_WAIT] * len(self.env.vessel_ids),
                "wells": list(self._last_well_actions),
            }
            observation, step_reward, terminated, truncated, info = self.env.step(automatic_action)
            reward += float(step_reward)
            elapsed_hours += float(self.env.network.time_step_hours)
            hourly_transitions += (HourlyTransition(float(step_reward), dict(info)),)

        self._current_event = None
        return self._transition(
            observation, reward, terminated, truncated, info, None, elapsed_hours, hourly_transitions
        )

    def _next_event(self) -> DecisionEvent | None:
        requested_replans = set(self.replan_trigger(self.env) or ()) if self.replan_trigger else set()
        vessel_types: dict[str, VesselDecisionType] = {}
        masks: dict[str, tuple[bool, ...]] = {}

        for vessel_id, mask in zip(self.env.vessel_ids, self.env.vessel_action_mask()):
            stage = self._decision_stage(vessel_id, mask)
            if stage is None:
                continue
            is_new_stage = self._acknowledged_stages.get(vessel_id) != stage.value
            if not is_new_stage and vessel_id not in requested_replans:
                continue
            self._acknowledged_stages[vessel_id] = stage.value
            vessel_types[vessel_id] = (
                VesselDecisionType.PLANNER_REPLAN if vessel_id in requested_replans else stage
            )
            masks[vessel_id] = tuple(mask)

        if not vessel_types:
            return None
        return DecisionEvent(
            time_h=float(self.env.simulator.state.time_h),
            vessel_ids=tuple(vessel_types),
            vessel_types=vessel_types,
            vessel_action_masks=masks,
        )

    def _decision_stage(self, vessel_id: str, mask: list[bool]) -> VesselDecisionType | None:
        # A lone WAIT is a forced physical period, not a learning sample.
        if sum(bool(allowed) for allowed in mask) <= 1:
            return None
        vessel_state = self.env.simulator.vessel_states[vessel_id]
        if vessel_state["mode"] != "berthed":
            return None
        berth = str(vessel_state["berth"])
        cargo_t = self.env.simulator.state.entity_inventory_t.get(vessel_id, 0.0)
        vessel = self.env.network.entities[vessel_id]
        if berth in self.env.emitter_ids:
            if cargo_t <= self._EPS:
                return VesselDecisionType.IDLE
            if cargo_t < vessel.capacity_t - self._EPS:
                return VesselDecisionType.LOADING
            return VesselDecisionType.FULL_LOAD
        # With the default terminal constraint, a loaded vessel has only WAIT
        # and no event is emitted. QUEUED is available for variants that permit
        # terminal departure before unloading finishes.
        if berth in self.env.terminal_ids and cargo_t > self._EPS:
            return VesselDecisionType.QUEUED
        return VesselDecisionType.IDLE

    def _event_action(self, action: dict[str, list], event: DecisionEvent) -> dict[str, list]:
        normalized = self.env._normalize_action(action)
        event_vessels = set(event.vessel_ids)
        for index, vessel_id in enumerate(self.env.vessel_ids):
            if vessel_id not in event_vessels:
                normalized["vessels"][index] = VESSEL_WAIT
                continue
            choice = normalized["vessels"][index]
            mask = event.vessel_action_masks[vessel_id]
            if choice < 0 or choice >= len(mask) or not mask[choice]:
                raise ValueError(f"Illegal event action {choice} for vessel {vessel_id}.")
        return normalized

    def _default_well_action(self, well_id: str) -> int:
        mask = self.env._well_rate_action_mask(well_id)
        return MIN_WELL_RATE_INDEX if mask[MIN_WELL_RATE_INDEX] else OFF_WELL_RATE_INDEX

    @staticmethod
    def _transition(
        observation: list[float],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        event: DecisionEvent | None,
        elapsed_hours: float,
        hourly_transitions: tuple[HourlyTransition, ...],
    ) -> EventTransition:
        result_info = dict(info)
        result_info["event_driven"] = {
            "elapsed_hours": elapsed_hours,
            "decision_event": None
            if event is None
            else {
                "time_h": event.time_h,
                "vessel_ids": list(event.vessel_ids),
                "vessel_types": {vessel_id: kind.value for vessel_id, kind in event.vessel_types.items()},
            },
        }
        return EventTransition(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=result_info,
            decision_event=event,
            elapsed_hours=elapsed_hours,
            hourly_transitions=hourly_transitions,
        )
