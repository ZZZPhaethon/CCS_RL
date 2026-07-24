# Methods Summary / 方法总结

**CCS_RL — Ship-Based CCS Dispatch: Existing Methods**
**船运 CCS 调度:现有方法总结**

> This document summarises the control and learning methods already implemented
> in this repository. Its core narrative: *on top of a strong operational rule,
> can learning add value — where, and at what cost?*
>
> 本文总结仓库中**已实现**的控制与学习方法。核心叙事:**在强运营规则之上,学习
> 能否/在何处/以多大代价带来增量?**

Last updated / 最近更新: 2026-07-24

---

## A. Shared framework / 整体框架

All methods share the same physics/decision foundation.
所有方法共享同一物理与决策底座。

- **Two-layer separation / 两层解耦**: `Simulation/` is the single source of
  physical truth (entities, operations, and the `network.step()` validation);
  `algorithms/` only decides *which operating goal to pursue*. The two connect
  through the solver-independent `DispatchGoal` / `HighLevelPolicy` /
  `ActionExecutor` contract in [algorithms/contracts.py](algorithms/contracts.py).
  `Simulation/` 是唯一物理事实来源;`algorithms/` 只决定"追求什么目标",经
  求解器无关的接口连接。

- **Decision structure = event-triggered SMDP / 决策结构 = 事件触发 SMDP**: the
  high-level policy decides at most every 24 h, or on an event (vessel arrival,
  buffer-risk alert, well outage, forecast revision). The low-level executor
  translates a `DispatchGoal` into the hourly native action
  `{vessels:[...], wells:[...]}`. Because intervals vary, transitions are
  discounted by `γ^Δt` (event-triggered default `γ=1.0`).
  高层每 ≤24h 或遇事件才决策;底层把目标翻译成每小时原生动作;因间隔不定按
  `γ^Δt` 折扣。

- **Unified reward / 统一奖励** ([algorithms/rl/reward.py](algorithms/rl/reward.py)):
  `stored_credit − vent_penalty(base + excess above tolerance) − operating_cost
  − overflow_risk(tonne·hour) − hard_violation(1e6)`, with `reward_scale=1e-6`.
  Unit storage cost is an evaluation KPI only, not part of the training reward.
  单位封存成本只作评估 KPI,不进训练奖励。

---

## B. Baselines & classical controllers / 基线与经典控制器

Located in [Simulation/control/](Simulation/control/).

| Method / 方法 | File / 文件 | Role / 作用 |
|---|---|---|
| Idle / Greedy shuttle | `baselines.py` | Idle baseline + greedy nearest shuttle (a strong baseline). / 空闲基线 + 贪心就近穿梭(强基线)。 |
| Rule-based (cluster-shuttle / balanced) | `rule_based.py` | Mask-respecting rule dispatch; the "safe default" behind every residual method. / 满足掩码的规则调度,后续残差方法的安全默认。 |
| Static MILP | `milp.py` / `cplex_milp.py` | Perfect-foresight benchmark. / 完美预见基准。 |
| Rolling MILP | `rolling_milp.py` (+ `replay.py`) | Rolling re-planning with replay validation. / 滚动重规划 + 回放校验。 |
| Trip MILP | `trip_milp.py` | Per-trip MILP variant. / 按航次的 MILP 变体。 |
| Native MPC | `native_mpc.py` (+ `rollout_advisor.py`) | Replay-evaluates candidate actions in copied environments. / 在复制环境里回放评估候选动作。 |
| Objective & diagnostics | `objective.py` (`vent_first` / `economic`), `plan_context.py`, `vessel_diagnostics.py`, `demonstrations.py`, `imitation.py` | Shared objective modes and support tooling. / 共享目标模式与支撑工具。 |

---

## C. Hybrid controllers: goal-aware executors / 混合控制器:目标感知执行器

Located in [algorithms/hybrid/](algorithms/hybrid/). These adapt the classical
controllers to the unified *high-level goal → low-level execution* interface so
they can be compared fairly.
把经典控制器适配成统一的"高层目标 → 底层执行"接口,以便公平对比。

- `GoalAwareRuleExecutor` — translates a `DispatchGoal` into the native masked
  action via the existing rule. / 用规则把目标转成带掩码原生动作。
- `GoalAwareNativeMpcExecutor` — evaluates the goal-induced candidate against
  native MPC candidates in copied physical rollouts. / 在复制的物理 rollout 里把
  目标候选与 MPC 候选一起评估。
- `RollingMilpExecutor` — adapts the replay-checked rolling MILP as a comparable
  optimisation baseline (≈30 s per 72 h solve with CBC). / 将回放校验的滚动 MILP
  适配为可比优化基线(CBC 下 72h 窗口约 30s/次)。
- Conservative by design: unknown scenarios rejected, infeasible well rates fall
  back, and `network.step()` is the final authority. / 保守原则:未知场景拒绝、
  不可行注入率回退、`network.step()` 最终裁决。

---

## D. Reinforcement learning methods / 强化学习方法(四代演进)

The core contribution — four generations, each fixing the previous one's flaw.
核心贡献,四代演进,每一版修上一版的缺陷。

| Version / 版本 | Dir / 目录 | Action space / 动作空间 | Key mechanisms / 关键机制 | Problem solved / result — 解决的问题 / 结论 |
|---|---|---|---|---|
| **High-level PPO / 高层 PPO** | `algorithms/rl` | `Discrete(192)` = per-vessel service preference × 3 injection modes / 每船服务偏好 × 3 注入模式 | Event-triggered; 79-dim obs (state + 24/72 h forecast); stable realised-outcome reward; `ent_coef=0.01`. / 事件触发;79 维观测;稳定实际结果奖励。 | Lets RL make only sparse high-level decisions instead of controlling every hour. / 让 RL 只做稀疏高层决策。 |
| **Residual v1 / 残差 v1** | `algorithms/residual_rl` | 7 actions: keep / prioritise×3 / add_one×3 | Rule produces a safe default; PPO only decides whether to intervene; 103-dim obs; 30% hard-scenario mix + fixed validation seeds. / 规则出安全默认,PPO 只决定是否小幅干预。 | **Problem / 问题**: often chose an `add_one` that could not change the native action → matched the rule but ~0.5% more cost. / 常选无法改变原生动作的 `add_one`,追平规则却多 ~0.5% 成本。 |
| **Masked Residual v2 / 掩码残差 v2** | `algorithms/residual_rl_v2` | 8 actions (+`use_adaptive_greedy`) | ① `MaskablePPO` dynamic masks (expose only interventions that truly change the native action); ② **persistent rule-counterfactual reward** (parallel shadow rule env, `reward = actual − rule_shadow`); ③ layered logging; ④ pre-training per-action full-episode screening; ⑤ **curriculum learning** (hard-scenario prob. 0→15→30→50%). / ①动态掩码;②持续规则反事实奖励;③分层日志;④训练前逐动作筛选;⑤课程学习。 | Fixes v1's ineffective interventions; the counterfactual pulls back the delayed credit-assignment (venting shows up dozens of hours later). / 修复无效干预,并处理延迟信用分配。 |
| **Risk-gated v3 / 风险门控 v3** | `algorithms/residual_rl_v3` | Frozen v2 policy + risk gate / 冻结 v2 策略 + 风险门控 | `risk_gate.py` hand-tuned thresholds (hours-to-overflow ≤48, fill ≥0.80, weather+fill…) allow adaptive-greedy only under physical risk; `sweep_risk_gate.py` scans gate configs. / 手调阈值只在物理风险时才允许 adaptive-greedy;扫描门控。 | Lets adaptive-greedy act only when it should (pure adaptive hurts some seeds). / 让 adaptive-greedy 只在该出手时出手。 |

---

## E. Training & evaluation methodology / 训练与评估方法论

Applies across all methods. 贯穿各方法。

- **Hard scenarios / 困难场景**: 70% normal + 30% hard (higher capture peaks,
  sustained bad weather, well maintenance, initial inventory pressure) — no
  physical constraint is changed. / 70% 普通 + 30% 困难,不改物理约束。
- **Strict validation isolation / 严格验证隔离**: training seeds
  `100000–999999`, fixed validation seeds `2000001–2000008`, checked disjoint
  before training starts. / 训练/验证 seed 强制不重叠。
- **Tail-aware model selection / 尾部感知模型选择**:
  `selection_loss = mean_total_cost + tail_vent_penalty·CVaR_vent
  + hard_violation_penalty`, where CVaR = mean vent of the worst 25% scenarios;
  both `final` and `best-validation` models are saved. / CVaR = 最差 25% 场景的
  平均放空;同时保存 final 与 best 两个模型。
- **Fair-comparison harness / 公平比较框架**
  ([experiments/](experiments/) `compare_shared_*`): sample one `720 h + 168 h`
  trajectory per seed, deep-copy it into every controller, run only the first
  720 h, and **assert identical cumulative capture**. Rewards are not comparable
  across families; compare only stored_t / vented_t / total cost / unit storage
  cost / violations. / 每 seed 只采一次轨迹深拷贝给各控制器,断言累计捕集量一致;
  reward 跨族不可比,只比物理指标。

---

## Method genealogy / 方法谱系(一句话总结)

A complete lineage from classical control (rule / MILP / MPC) → a unified
goal–executor interface → four generations of residual / high-level RL, wrapped
in a tail-aware (CVaR), scenario-isolated, paired-trajectory evaluation
methodology.

一条从"经典控制(规则/MILP/MPC)→ 统一目标-执行器接口 → 四代残差/高层 RL"的完整
谱系,配套尾部感知(CVaR)、场景隔离、同轨迹配对的评估方法论。

---

## Related documentation / 相关文档

- [algorithms/README.md](algorithms/README.md) — algorithm-layer contract / 算法层接口
- [algorithms/hybrid/README.md](algorithms/hybrid/README.md)
- [algorithms/rl/README.md](algorithms/rl/README.md)
- [algorithms/residual_rl/README.md](algorithms/residual_rl/README.md)
- [algorithms/residual_rl_v2/README.md](algorithms/residual_rl_v2/README.md)
- [experiments/README.md](experiments/README.md)
- [Simulation/README.md](Simulation/README.md)
