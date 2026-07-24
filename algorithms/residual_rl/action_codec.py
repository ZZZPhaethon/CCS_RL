"""Encode compact residual interventions over a safe rule baseline.

在安全规则基线上编码紧凑的残差干预动作。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualIntervention:
    """Describe one residual change to the rule executor.

    描述对规则执行器的一次残差修正。

    ``keep_default`` delegates all vessel decisions to the rule baseline.
    ``prioritise`` continuously redirects currently dispatchable empty vessels
    toward one emitter until the next decision. ``add_one`` redirects at most
    one suitable vessel during the current decision interval.

    ``keep_default`` 完全沿用规则基线；``prioritise`` 在当前决策区间持续将
    可调度空船引导至指定排放源；``add_one`` 在当前决策区间最多增派一艘
    合适船舶。
    """

    kind: str
    emitter_id: str | None = None

    def validate(self) -> None:
        """Validate the intervention schema.

        校验干预动作的数据结构。
        """
        valid_kinds = {"keep_default", "prioritise", "add_one"}
        if self.kind not in valid_kinds:
            raise ValueError(f"Unknown residual intervention kind: {self.kind!r}.")
        if self.kind == "keep_default" and self.emitter_id is not None:
            raise ValueError("keep_default must not name an emitter.")
        if self.kind != "keep_default" and not self.emitter_id:
            raise ValueError(f"{self.kind} requires a non-empty emitter_id.")


class ResidualActionCodec:
    """Map ``1 + 2 * n_emitters`` discrete actions to interventions.

    将 ``1 + 2 * 排放源数量`` 个离散动作映射为残差干预。
    """

    def __init__(self, emitter_ids: list[str] | tuple[str, ...]) -> None:
        """Store a stable emitter order used by training and evaluation.

        保存训练与评估共同使用的稳定排放源顺序。
        """
        if not emitter_ids:
            raise ValueError("ResidualActionCodec requires at least one emitter.")
        if len(set(emitter_ids)) != len(emitter_ids):
            raise ValueError("Emitter IDs must be unique.")
        self.emitter_ids = tuple(emitter_ids)

    @property
    def action_count(self) -> int:
        """Return the compact residual action count.

        返回紧凑残差动作数量。
        """
        return 1 + 2 * len(self.emitter_ids)

    def decode(self, action_index: int) -> ResidualIntervention:
        """Decode one discrete action.

        解码一个离散动作。
        """
        index = int(action_index)
        if not 0 <= index < self.action_count:
            raise ValueError(
                f"Residual action {index} is outside [0, {self.action_count})."
            )
        if index == 0:
            return ResidualIntervention("keep_default")
        emitter_count = len(self.emitter_ids)
        if index <= emitter_count:
            return ResidualIntervention(
                "prioritise",
                self.emitter_ids[index - 1],
            )
        return ResidualIntervention(
            "add_one",
            self.emitter_ids[index - emitter_count - 1],
        )

    def label(self, action_index: int) -> str:
        """Return a readable action label.

        返回便于日志阅读的动作标签。
        """
        intervention = self.decode(action_index)
        if intervention.kind == "keep_default":
            return "keep_rule_default"
        return f"{intervention.kind}:{intervention.emitter_id}"

    def labels(self) -> tuple[str, ...]:
        """Return labels in policy-index order.

        按策略索引顺序返回全部动作标签。
        """
        return tuple(self.label(index) for index in range(self.action_count))
