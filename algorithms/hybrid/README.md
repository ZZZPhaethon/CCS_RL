# `algorithms.hybrid`

此目录放置“高层目标 + 底层安全执行器”的组合控制器。

This directory contains controllers that combine a high-level goal with a
low-level safe executor.

| File | Role / 作用 |
| --- | --- |
| `rule_executor.py` | `GoalAwareRuleExecutor`: translates `DispatchGoal` into the native masked action using the existing cluster-shuttle rule. / 将 `DispatchGoal` 借助已有 cluster-shuttle 规则转换为带掩码的原生动作。 |
| `mpc_executor.py` | `GoalAwareNativeMpcExecutor`: evaluates the goal-induced candidate against native MPC candidates in copied physical rollouts. / 在复制的物理回放中，把目标候选方案与原生 MPC 候选方案共同评估。 |
| `milp_executor.py` | `RollingMilpExecutor`: adapts the existing replay-checked rolling MILP as a comparable optimisation baseline. / 将现有、带回放校验的滚动 MILP 适配为可比的优化基线。 |

The executor is deliberately conservative: unknown scenario IDs are rejected;
infeasible well-rate requests fall back to the baseline's feasible selection;
and the physical network remains the final authority.

执行器采取保守策略：未知场景 ID 会被拒绝；不可行的注入率请求回退到基线可行选择；
物理网络仍是最终裁决者。MPC 中的高层目标只作为候选偏好，在渢出或可行性更差时不会强行覆盖其他候选方案。
Rolling MILP 目前是独立优化基线：它记录高层目标并使用其重规划周期，但不将船舶分配惄然设为硬约束。
