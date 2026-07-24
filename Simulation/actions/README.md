# `Simulation.actions`

本目录定义物理仿真时间步的动作协议。策略、规则控制器或 MPC 不应直接修改
`PhysicalState`；它们先提交动作提议，再由解析器检查实体类型、动词和参数。

| 文件 | 职责 |
| --- | --- |
| `action.py` | 定义 `ActionProposal`、`ActionFrame`、`ActionDecision` 和 `CommittedActionFrame` 等不可变动作记录。 |
| `resolver.py` | 定义各实体允许的动作规格，并将提议解析为可提交、拒绝或裁剪后的动作。 |

```text
policy/controller
      │ ActionProposal
      ▼
ActionFrame ─► ActionResolver ─► CommittedActionFrame ─► PhysicalNetwork.step()
```

`ActionProposal` 包含发起者 ID、目标实体 ID、动词、参数和元数据。`ActionFrame`
汇集一个时间步内全部提议。解析阶段会将无法识别、不适用于目标实体或参数不完整的
动作记录为决策结果，从而使物理层只接收结构化、经过校验的操作请求。

新增动作时，应同时更新实体类型的 `ActionSpec`、解析逻辑、环境动作编码和测试。
