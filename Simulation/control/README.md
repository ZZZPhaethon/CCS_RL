# `Simulation.control`

本目录收集用于 CCS 调度的非学习控制方法和训练辅助工具。它们既可作为 RL 的比较
基线，也可用于生成示范、分析策略行为或给出具有完全/有限预见信息的优化上界。

| 文件 | 主要职责 |
| --- | --- |
| `baselines.py` | 提供 `idle`、`greedy_shuttle` 等简单环境策略。 |
| `rule_based.py` | 提供基于库存、泊位和船舶状态的规则调度器。 |
| `objective.py` | 统一控制目标的成本、放空、封存奖励和库存溢出风险权重。 |
| `milp.py` | 提供固定时域、完全信息的 MILP 基准。 |
| `cplex_milp.py` | 通过外部 IBM CPLEX 可执行程序求解全场景 MILP oracle。 |
| `trip_milp.py` | 提供以航次为单位的 CPLEX MILP 调度 oracle。 |
| `rolling_milp.py` | 在当前状态上反复重规划的滚动时域 MILP 控制器。 |
| `native_mpc.py` | 在环境原生动作空间上进行回放验证的 MPC。 |
| `rollout_advisor.py` | 从原生 rollout 规则规划器提取在线上下文和事件触发条件。 |
| `plan_context.py` | 保存 24 小时重规划周期的已学习计划上下文。 |
| `replay.py` | 保存求解器无关的回放快照，并验证计划在物理仿真中是否可执行。 |
| `demonstrations.py` | 采集、缓存和验证 MPC 示例轨迹。 |
| `imitation.py` | 用示范动作对混合 MaskablePPO 策略进行行为克隆。 |
| `vessel_diagnostics.py` | 分析船舶 WAIT 与发船决策的条件和后果。 |

## 方法定位

```text
规则基线 ───────────────► 快速、可解释的下限参考
固定时域 MILP / CPLEX ──► 完全信息、可能乐观的性能上界
滚动 MILP / 原生 MPC ───► 在线重规划、物理可回放验证
示范 + 模仿学习 ─────────► 改善 RL 的早期探索
RL 策略 ─────────────────► 在随机场景中追求长期鲁棒表现
```

MILP 通常对终端、船舶旅行和未来信息做了不同程度的抽象，因此其目标值不一定能按
时间步直接在物理仿真器中执行。需要比较方法时，优先使用 `replay.py` 的回放验证，
并在相同种子、相同经济参数和相同场景信息假设下报告指标。

含 CPLEX 的模块依赖项目外部安装的 IBM CPLEX；其余模块不应隐式要求该依赖。
