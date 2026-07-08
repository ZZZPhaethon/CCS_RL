from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionProposal:
    """One agent's requested action for one physical entity."""

    agent_id: str
    entity_id: str
    verb: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionFrame:
    """All action proposals submitted for one simulation time step."""

    time_h: float
    proposals: list[ActionProposal] = field(default_factory=list)


@dataclass(frozen=True)
class ActionDecision:
    proposal: ActionProposal
    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class CommittedActionFrame:
    """Resolved actions allowed to enter the physical step for one time step."""

    time_h: float
    actions: dict[str, dict[str, Any]]
    decisions: list[ActionDecision] = field(default_factory=list)
