"""Encode masked residual actions including adaptive greedy mode.

编码包含 adaptive greedy 模式的掩码残差动作。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskedResidualIntervention:
    """Describe one v2 residual intervention.

    描述一个 v2 残差干预。
    """

    kind: str
    emitter_id: str | None = None

    def validate(self) -> None:
        """Validate kind and emitter requirements.

        校验动作类型与排放源要求。
        """
        valid = {
            "keep_default",
            "adaptive_greedy",
            "prioritise",
            "add_one",
        }
        if self.kind not in valid:
            raise ValueError(f"Unknown intervention kind: {self.kind!r}.")
        requires_emitter = self.kind in {"prioritise", "add_one"}
        if requires_emitter and not self.emitter_id:
            raise ValueError(f"{self.kind} requires an emitter_id.")
        if not requires_emitter and self.emitter_id is not None:
            raise ValueError(
                f"{self.kind} must not specify an emitter_id."
            )


class MaskedResidualActionCodec:
    """Map ``2 + 2 * n_emitters`` actions to v2 interventions.

    将 ``2 + 2 * 排放源数量`` 个动作映射为 v2 干预。
    """

    def __init__(self, emitter_ids: list[str] | tuple[str, ...]) -> None:
        """Store a stable emitter order.

        保存稳定的排放源顺序。
        """
        if not emitter_ids:
            raise ValueError("At least one emitter is required.")
        if len(set(emitter_ids)) != len(emitter_ids):
            raise ValueError("Emitter IDs must be unique.")
        self.emitter_ids = tuple(emitter_ids)

    @property
    def action_count(self) -> int:
        """Return the total v2 action count.

        返回 v2 动作总数。
        """
        return 2 + 2 * len(self.emitter_ids)

    def decode(self, action_index: int) -> MaskedResidualIntervention:
        """Decode one policy index.

        解码一个策略索引。
        """
        index = int(action_index)
        if not 0 <= index < self.action_count:
            raise ValueError(
                f"Action {index} is outside [0, {self.action_count})."
            )
        if index == 0:
            return MaskedResidualIntervention("keep_default")
        if index == 1:
            return MaskedResidualIntervention("adaptive_greedy")
        emitter_count = len(self.emitter_ids)
        if index < 2 + emitter_count:
            return MaskedResidualIntervention(
                "prioritise",
                self.emitter_ids[index - 2],
            )
        return MaskedResidualIntervention(
            "add_one",
            self.emitter_ids[index - 2 - emitter_count],
        )

    def label(self, action_index: int) -> str:
        """Return one readable action label.

        返回一个易读动作标签。
        """
        intervention = self.decode(action_index)
        if intervention.kind == "keep_default":
            return "keep_rule_default"
        if intervention.kind == "adaptive_greedy":
            return "use_adaptive_greedy"
        return f"{intervention.kind}:{intervention.emitter_id}"

    def labels(self) -> tuple[str, ...]:
        """Return all labels in policy-index order.

        按策略索引顺序返回全部标签。
        """
        return tuple(self.label(index) for index in range(self.action_count))

