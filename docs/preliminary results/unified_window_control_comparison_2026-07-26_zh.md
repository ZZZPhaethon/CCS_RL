# 统一 Window 场景下的控制方法初步对比（2026-07-26）

## 目的

在统一物理层、统一场景生成器和相同未见测试 seeds 下，初步比较：

- Greedy；
- Residual PPO v4；
- 不使用未来信息的 Iterative Q；
- 使用 24/72 h 未来摘要的 Iterative Q；
- 远端 `event-based` 分支定义的 Greedy 底 Hybrid RL。

本文件记录当前阶段结果，不作为最终论文定稿。正式结论仍需增加训练随机种子并完成稳定性验证。

## 统一场景协议

所有方法在测试时使用同一套 `unified_window_v1` 协议：

- Episode：720 h；
- 船舶配置：Phase 1，3 vessels；
- Capture 基础曲线：真实 hourly CSV；
- Capture Gaussian noise std：0.30；
- Capture outage：0.5 次/周，平均持续 12 h；
- Capture 高产事件：0.5 次/周，平均持续 48 h，倍率 1.25--1.75；
- 天气：Window，0.5 次/周，平均持续 48 h；
- 天气窗口内船速因子：0.50--0.80，窗口外为 1.0；
- 井维护：0.3 次/周，平均持续 12 h；
- Emitter 初始库存：容量的 0--50%；
- Terminal 初始库存：容量的 0--50%；
- Reservoir 初始压力：0--50% warm start；
- Forecast context：168 h；
- 测试 seeds：8,000,001--8,000,030，共 30 个，均未参与训练或选模。

Iterative Q 和 v4 使用 12 个固定干预窗口，每个窗口最多进行一次 override。
Hybrid RL 按远端原生设计进行事件触发高层决策，最长决策间隔为 24 h，不使用
12-window override 预算。

## 方法与训练

| 方法 | 训练规模 | 当前/未来信息 | 底层控制 |
|---|---:|---|---|
| Greedy | 无训练 | 当前物理状态 | `greedy_shuttle_policy` |
| Residual PPO v4 | 100,352 PPO steps | 当前状态及 24/72 h 摘要 | 规则默认动作上的稀疏残差干预 |
| Iterative Q（无未来） | 4,800 roots，P1--P4 | 当前状态 | Greedy 动作上的候选动作 Q 评估 |
| Iterative Q（有未来） | 4,800 roots，P1--P4 | 当前状态及 24/72 h 摘要 | Greedy 动作上的候选动作 Q 评估 |
| Hybrid RL（Greedy 底） | 50,176 PPO steps | 当前状态及 24/72 h 摘要 | `GoalAwareRuleExecutor` |

这里的 Hybrid RL 严格采用远端 `origin/event-based` 的默认代码链：

```text
High-level PPO
    -> HighLevelDispatchEnv
    -> GoalAwareRuleExecutor
    -> greedy_shuttle_policy
```

高层 PPO 为每艘船选择 adaptive/指定 emitter 偏好，并选择
conservative/balanced/aggressive 注入模式。底层先生成 Greedy 动作，再仅在当前动作
可行时应用高层船舶偏好；本结果不使用优化器执行器。

## 30-seed 初步结果

表中数值为 30 个配对测试 seeds 的均值。

| 方法 | 总成本 (EUR) | 相对 Greedy (EUR) | 运行成本 (EUR) | Vent penalty (EUR) | Vent (t) | Stored (t) | 单位成本 (EUR/t) | 胜 Greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 2,059,907 | -- | 1,429,261 | 630,646 | 7,883.1 | 100,959.4 | 21.01 | -- |
| Residual PPO v4 | 1,942,032 | -117,876 | 1,520,987 | 421,045 | 5,263.1 | 103,421.5 | 18.87 | 13/30 |
| Iterative Q（无未来） | 1,699,864 | -360,043 | 1,563,506 | 136,358 | 1,704.5 | 108,989.6 | 15.68 | 23/30 |
| **Iterative Q（24/72 h 未来）** | **1,633,631** | **-426,276** | 1,567,943 | **65,688** | **821.1** | **109,242.1** | **14.97** | **25/30** |
| Hybrid RL（Greedy 底） | 3,134,909 | +1,075,002 | **1,295,680** | 1,839,229 | 22,990.4 | 87,728.0 | 36.05 | 2/30 |

## 配对统计

差异定义为“方法减去 Greedy”；负值表示方法成本更低。

| 对比 | 平均总成本差异 (EUR) | 95% bootstrap CI (EUR) | 方法胜/负 |
|---|---:|---:|---:|
| Residual PPO v4 - Greedy | -117,876 | [-309,118, +47,944] | 13/17 |
| Iterative Q（无未来）- Greedy | -360,043 | [-522,435, -215,588] | 23/7 |
| Iterative Q（有未来）- Greedy | -426,276 | [-599,466, -281,035] | 25/5 |
| Hybrid RL（Greedy 底）- Greedy | +1,075,002 | [+882,930, +1,253,813] | 2/28 |

未来信息消融的配对差异：

| 对比 | 平均总成本差异 (EUR) | 95% bootstrap CI (EUR) | 前者胜/负 |
|---|---:|---:|---:|
| Iterative Q（有未来）- Iterative Q（无未来） | -66,233 | [-119,717, -15,833] | 18/12 |

## 初步结论

1. 当前统一协议下，使用 24/72 h 未来摘要的 Iterative Q 表现最好。其平均总成本比
   Greedy 低 EUR 426,276，并在 25/30 个测试 seeds 上胜出。
2. 不使用未来信息的 Iterative Q 仍明显优于 Greedy，说明改进并不完全依赖未来信息；
   但加入未来摘要后，平均总成本进一步下降 EUR 66,233，配对置信区间不跨 0。
3. v4 的平均成本低于 Greedy，但仅在 13/30 个 seeds 上获胜，且总成本差异的
   95% CI 跨 0，因此当前不能称为稳定优于 Greedy。
4. Greedy 底 Hybrid RL 的运行成本最低，但放空量升至 22,990.4 t，导致总成本达到
   EUR 3.135M。它在 28/30 个 seeds 上劣于 Greedy，当前训练结果不能作为有竞争力的
   v4 替代方案。
5. Hybrid RL 的主要问题不是动作不可行或物理违规，而是高层策略频繁改变 Greedy
   偏好后牺牲了封存量。后续若继续研究，应优先检查奖励与总成本口径的一致性、
   高层动作保持时间，以及保留纯 Greedy 的安全回退机制。

## 当前限制

- 每种学习方法当前只有一套正式训练结果；30 个测试 seeds 的置信区间只反映测试场景
  变化，不包含训练随机性。
- Q、v4 和 Hybrid RL 的训练预算、动作空间及决策频率不同，因此本表是最终控制策略的
  物理效果比较，不是样本效率比较。
- 有未来的 Iterative Q 和 Hybrid RL 使用仿真中已实现的 24/72 h 实际未来摘要；
  正式论文需明确这是 perfect-forecast 实验，或进一步加入预测误差测试。
- Hybrid RL 平均约进行 80.2 次事件触发高层决策；其决策次数不能与 Q/v4 的平均
  override 次数直接比较。
- 本结果仅适用于当前统一 Window 分布，不能直接外推到旧 Block、Normal 或 Hard 协议。

## 复现产物

- Greedy、Iterative Q、v4 逐 seed 对比：
  `output/unified_window12_20260726/comparison/per_seed.csv`
- Greedy、Iterative Q、v4 汇总：
  `output/unified_window12_20260726/comparison/summary.json`
- 无未来 Iterative Q：
  `output/unified_window12_20260726/iterative_q_state/eval/evaluation.csv`
- Hybrid RL 逐 seed：
  `output/unified_window12_20260726/hybrid_rl/evaluation/evaluation_30.csv`
- Hybrid RL 汇总：
  `output/unified_window12_20260726/hybrid_rl/evaluation/evaluation_30.json`
- 有未来 Iterative Q checkpoint：
  `output/unified_window12_20260726/iterative_q/p4/iterative_action_q.pt`
- 无未来 Iterative Q checkpoint：
  `output/unified_window12_20260726/iterative_q_state/p4/iterative_action_q.pt`
- Hybrid RL checkpoint：
  `output/unified_window12_20260726/hybrid_rl/ppo_high_level_final.zip`

