# `algorithms`

本目录是 CCS 项目的算法层，位于 `Simulation/` 之外。它负责“在什么目标下
运行”，而 `Simulation/` 仍是唯一的物理事实来源，负责“这个目标是否可以执行、
实际发生了什么、产生了什么成本和排放”。

This directory is the algorithm layer, intentionally outside `Simulation/`.
It decides *which operating goal to pursue*. `Simulation/` remains the single
source of physical truth: it determines feasibility, realised flows, cost, and
emissions.

## Recommended control split / 推荐控制分层

```text
Sparse high-level decision / 稀疏高层决策 (RL, heuristic, or LLM)
    ↓ DispatchGoal: emitter↔vessel preference, well-rate target, replan horizon
Low-level executor / 底层执行器 (rule, MPC, rolling MILP)
    ↓ native action: {vessels: [...], wells: [...]}
Simulation.environment → simulator → physical constraints and reward
```

`contracts.py` defines this boundary without tying it to Stable-Baselines3,
CPLEX, or a particular MPC implementation.  This keeps experiments comparable
and lets the same physical scenario be run with rules, MPC, RL, or hybrids.

`contracts.py` 用不依赖 Stable-Baselines3、CPLEX 或特定 MPC 实现的方式定义该
边界，从而可以在相同物理场景上公平比较规则、MPC、RL 和混合方法。

## Why this split / 为什么这样划分

- The simulator still advances at its physical time step (currently usually one
  hour), so capture, vessel travel, loading, pressure, and annual pipeline
  limits remain accurate.
- The high-level policy should normally replan every 12–24 simulated hours, or
  immediately after a material event: vessel arrival/availability, berth or
  well outage, buffer-risk alert, or forecast revision.
- The executor must respect the current action masks and must never treat a
  high-level goal as a physical override. `network.step()` and its validation
  remain the final feasibility check.

- 仿真器仍按物理时间步（当前通常为一小时）推进，因此捕集、航行、装卸、压力和
  管道滚动年能力约束保持准确。
- 高层策略通常每 12–24 个仿真小时重规划一次；船舶到港/可用、泊位或注入井故障、
  库存风险预警或预报更新时应立即重规划。
- 执行器必须遵守当前动作掩码，不能将高层目标当作物理层覆盖。`network.step()` 及其
  校验仍是可行性的最终判定。

## Current contents / 当前内容

| File | Role / 作用 |
| --- | --- |
| `contracts.py` | `DispatchGoal`, replanning schedule, and interfaces for a high-level policy and low-level executor. / 调度目标、重规划日程及高层策略、底层执行器接口。 |
| `evaluation.py` | Physical rollout evaluator for fair controller comparisons. / 用于公平控制器比较的物理回放评估器。 |
| `hybrid/rule_executor.py` | A first goal-aware, mask-respecting rule executor. / 第一个目标感知且遵守掩码的规则执行器。 |
| `hybrid/mpc_executor.py` | A replay-validated native MPC executor that evaluates the high-level goal as a candidate. / 将高层目标作为候选方案评估的、经回放验证的原生 MPC 执行器。 |
| `hybrid/milp_executor.py` | An adapter for the replay-checked rolling MILP optimisation baseline. / 将带回放校验的滚动 MILP 优化基线适配为统一接口。 |
| `rl/` | Sparse 24-hour high-level PPO, its action codec, physical reward, and Gym adapter. / 稀疏的 24 小时高层 PPO、动作编码、物理奖励与 Gym 适配器。 |
| `__init__.py` | Public exports. / 公共导出。 |

## Next implementation order / 后续实现顺序

1. Compare the included rule and native-MPC executors through identical
   physical rollouts; then add a horizon-aware MILP executor when a solver
   budget is available.
2. Add deterministic and rolling-MPC baselines under `algorithms/baselines/`;
   use identical scenarios, seeds, forecasts, and metrics.
3. Add an event-triggered or 12–24 h goal-level RL policy under
   `algorithms/rl/`. Its reward must be computed from realised simulator
   outcomes, not from planned tonnes.
4. If the decision interval varies, train and evaluate as an SMDP: discount a
   transition by `gamma ** elapsed_hours` (or an equivalent time-consistent
   formulation), rather than treating all event intervals as one hour.

1. 通过完全相同的物理回放比较已提供的规则与原生 MPC 执行器；当有可用求解时间预算时，再加入具备预测时域的 MILP 执行器。
2. 在 `algorithms/baselines/` 中加入确定性和滚动 MPC 基线，确保场景、随机种子、
   预报与指标完全一致。
3. 在 `algorithms/rl/` 中加入事件触发或每 12–24 小时决策一次的目标级 RL；奖励必须
   来自实际仿真结果，不能来自计划吨数。
4. 如果决策间隔可变，应按 SMDP 训练与评估：用 `gamma ** elapsed_hours`（或等价的
   时间一致形式）折扣，而不能把所有事件间隔都视为一小时。

## Rolling MILP budget / 滚动 MILP 求解预算

`RollingMilpExecutor` is a valid optimisation baseline only when its solver
returns a replay-checked integer plan. With CBC on the three-vessel scenario,
a 72-hour planning window required about 30 seconds for one successful solve.
Start with one seed and short windows; use a faster solver such as CPLEX before
attempting a 720-hour multi-seed MILP study.

`RollingMilpExecutor` 只有在求解器返回通过回放校验的整数计划时才是有效的优化基线。在三船场景中使用 CBC
时，72 小时预见窗口需要约 30 秒才能完成一次成功求解。应先从单种子和短时域开始；若要尝试 720 小时的多种子
MILP 研究，建议先使用 CPLEX 等更快的求解器。

## Fair comparison / 公平比较

Use `compare_executors()` with a factory that creates the same scenario for
each controller. Report realised `stored_t`, `vented_t`, operating/total cost,
unit storage cost, storage/vent rate, wall-clock time, reward, and violation
counts—not a solver's planned objective alone.

使用 `compare_executors()` 并传入一个为每个控制器创建同一场景的工厂函数。应报告实际的
`stored_t`、`vented_t`、运行/总成本、单位封存成本、封存/放空率、运行时间、奖励和违规次数，
而非仅报告求解器的计划目标值。

## Scope boundary / 边界

Do not add physical capacities, pressure equations, or clipping rules here.
Those belong in `Simulation/entities`, `Simulation/operations`, and
`Simulation/network.py`, where all controllers receive the same treatment.

不要在此处加入物理能力、压力方程或流量裁剪规则；它们应位于
`Simulation/entities`、`Simulation/operations` 和 `Simulation/network.py`，以保证
所有控制器受到同一套物理约束。
