# Rolling CPLEX MILP 与八策略 MPC：720 h 初步对比

## 1. 实验目的

在相同场景、相同扰动、相同预测窗口和滚动频率下，对比可执行的 rolling MILP 与当前八策略 MPC。本文只报告 seed 1 的初步结果，不将单 seed 结果解释为统计结论或 MILP 理论最优上界。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 场景 | `northern_lights_phase1_3vessels` |
| 扰动 | block profile |
| Scenario seed | 1 |
| Episode | 720 h |
| 单次预测窗口 | 最多 168 h |
| 执行/重规划间隔 | 24 h |
| 重规划次数 | 30 |
| MPC 候选 | 8 种原生策略，共评价 240 个候选窗口 |
| MILP 求解器 | native CPLEX |
| MILP 时间限制 | 每个求解阶段 60 s |
| 目标优先级 | 最小化 vented CO2 → 最小化窗口末未储存 CO2 → 最小化 operating cost |

两种控制器均执行完整的 720 个小时级原生动作。保存的 MPC 和 rolling MILP 动作轨迹均通过 `CCSEnv` 的 720 h 精确重放。

## 3. 720 h 结果

| 指标 | 八策略 MPC | Rolling CPLEX MILP | MILP − MPC |
|---|---:|---:|---:|
| Vented CO2 (t) | **295.200** | 6,701.394 | +6,406.193 |
| Episode 末未储存 CO2 (t) | **25,621.641** | 30,953.595 | +5,331.953 |
| Stored CO2 (t) | **111,684.980** | 99,946.833 | −11,738.147 |
| Total cost (EUR) | **1,614,149.03** | 1,990,033.30 | +375,884.28 |
| Unit cost (EUR/t stored) | **14.453** | 19.911 | +5.458 |
| Controller wall time (s) | **49.853** | 1,684.349 | +1,634.496 |
| 720 h 精确重放 | 通过 | 通过 | — |

在这个 seed 上：

- rolling MILP 的 venting 是 MPC 的 **22.70 倍**；
- rolling MILP 的 stored CO2 比 MPC 低 **10.51%**；
- rolling MILP 的 total cost 比 MPC 高 **23.29%**；
- rolling MILP 的单位储存成本比 MPC 高 **37.77%**；
- rolling MILP 的控制器计算时间约为 MPC 的 **33.79 倍**。

## 4. 目标一致性与解释边界

两者的目标优先级一致，但当前还不是完全相同数学模型之间的比较：

1. 八策略 MPC 直接复制 `CCSEnv`，在真实环境动力学中对八种策略进行 168 h rollout，并按 lexicographic objective 选择候选。
2. Rolling MILP 使用线性化优化模型。为保证接下来 24 h 的动作可以在环境中执行，还加入了保守的 safe-execution restriction。
3. 30 次 MILP 重规划都记录到优化模型的全窗口预测与 `CCSEnv` 重放不完全一致；因此表中 KPI 全部采用真实环境执行/精确重放结果，而不是 MILP 模型内部的目标值。
4. CPLEX 使用有限求解时间，且当前可执行性约束缩小了 MILP 的决策空间。因此该结果代表当前可执行 rolling CPLEX 控制器，不能视为 MILP 的全局最优性能或 MPC 的理论上界。

当前结果说明，性能差距更可能来自 MILP 与环境之间的状态转移/服务过程偏差以及保守执行约束，而不是来自两者目标排序不同。后续若要做严格的一致性比较，应先使 MILP 的装卸、泊位、航行和自动流量动态与 `CCSEnv` 完全一致，再进行多 seed 配对评估。

## 5. 验证与结果文件

- 全套测试：`624 passed, 41 subtests passed`。
- [原始指标 CSV](../../output/rolling_native_cplex_vs_mpc_720h_seed1/by_seed.csv)
- [720 h 原生动作轨迹](../../output/rolling_native_cplex_vs_mpc_720h_seed1/seed_1_native_actions.json)
- [运行配置](../../output/rolling_native_cplex_vs_mpc_720h_seed1/run_config.json)
- [输出目录中的对比报告](../../output/rolling_native_cplex_vs_mpc_720h_seed1/comparison_report.md)
