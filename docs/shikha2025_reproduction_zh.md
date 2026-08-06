# Shikha et al. (2025) 方法复现与同 case 对比

## 复现范围

本文复现 Shikha et al. (2025) 针对允许 milk-run 的 Case 2 主方法：

1. 按船舶进行空间拉格朗日分解，每艘船对应一个子问题；
2. 长时域子问题采用 shrinking horizon；
3. 将子问题给出的航行/等待路线固定到全空间模型，重新优化装载、卸载和注入流量，以恢复物理可行解；
4. 根据共享码头服务约束的残差，用论文式 (35) 的投影次梯度形式更新乘子；
5. 保留迭代中成本最低的全空间可行解。

官方来源：

- 论文：[Computational Strategies for RTN Model for Supply Logistics of Carbon Dioxide for Carbon Capture and Storage](https://doi.org/10.1021/acs.iecr.5c00695)
- 完整 RTN 补充公式：[Supporting Information](https://acs.figshare.com/articles/journal_contribution/Computational_Strategies_for_RTN_Model_for_Supply_Logistics_of_Carbon_Dioxide_for_Carbon_Capture_and_Storage/29181525)

实现入口为 `src/sim/control/shikha2025.py`，比较入口为
`experiments/smoke_test_paper_controllers.py` 中的 `shikha2025` controller。

## 与论文算法的对应

| 论文步骤 | 本仓库实现 |
|---|---|
| 每艘船一个子问题 `P_j` | 深复制同一个初始环境，仅保留目标船舶的路线变量 |
| 非船舶资源分配因子之和为 1 | emitter capture/buffer、terminal、pipeline、manifold、well 按船舶数等份分配 |
| emitter/terminal jetty 耦合约束拉格朗日化 | 对每个 node-hour 的 loading/unloading service binary 加乘子价格 |
| `alpha_k = 1` | 默认 `step_size=1.0` |
| 收敛阈值 2% | 默认 `tolerance_rel=0.02`，同时检查共享服务冲突 |
| 120 h active window | 默认 `active_window_h=120` |
| 每次固定 60 h | 默认 `fix_window_h=60` |
| 固定 Sail/Bunker/Idle/activeness 后求原问题 | 固定原生 action MILP 的 vessel route arcs，连续装卸、库存和注入重新求解 |

对 240 h，shrinking-horizon 阶段为 `(0,120)`、`(60,180)`、`(120,240)`，共 3 次；对
360 h 为 5 次，与论文第 7 节的设置一致。后半时域的 binary 在当前阶段放松为 `[0,1]`，下一阶段再转回整数，并固定已经接受的前缀路线。

## 为什么不是作者数值表的逐项复刻

论文和 Supporting Information 公布了模型与算法，但没有公开 Table 5-9 所用的完整实例参数和 GAMS 工程。因此这里复现的是算法结构，并把它应用到仓库现有、可审计的同一物理 case：

- scenario：`northern_lights_phase1_3vessels`；
- protocol：`unified_window_v1`；
- 时间步：1 h；
- 同一 seed、同一 capture/weather/well 扰动；
- 同一成本参数、terminal cleanup 规则和物理回放评分。

仓库模型没有论文中的有限 bunkering fuel resource；燃油在这里是航行成本，不构成耦合资源。CO2、pipeline 和 well 资源在子问题间显式等份分配，码头服务资源使用乘子协调。因此日志中的
`surrogate_dual_objective` 是算法诊断量，不应当作为经过证明的全模型对偶界。若共享服务冲突已经归零但 surrogate gap 尚未达到 2%，实现会以
`decomposition_stopping_reason=service_coupling_consistent` 停止重复求解，同时保持
`decomposition_converged=false`；只有 surrogate gap 达标才标记为收敛。最终报告的成本和 KPI 始终来自全模型可行恢复及 simulator 回放。

## 运行同 case 对比

短时冒烟测试：

```powershell
$env:PYTHONPATH = "src"
python -m experiments.smoke_test_paper_controllers `
  --out-dir output\shikha2025_smoke `
  --controllers greedy rolling_milp full_milp shikha2025 `
  --seed 8100001 `
  --online-episode-hours 24 `
  --forecast-context-hours 168 `
  --full-milp-horizon-hours 24 `
  --shikha-max-iterations 3 `
  --shikha-subproblem-time-limit-seconds 30 `
  --shikha-repair-time-limit-seconds 30
```

720 h 正式 case：

```powershell
$env:PYTHONPATH = "src"
python -m experiments.smoke_test_paper_controllers `
  --out-dir output\shikha2025_seed_9000031 `
  --controllers shikha2025 `
  --seed 9000031 `
  --online-episode-hours 720 `
  --forecast-context-hours 168 `
  --full-milp-horizon-hours 720 `
  --shikha-active-window-hours 120 `
  --shikha-fix-window-hours 60 `
  --shikha-max-iterations 18 `
  --shikha-tolerance-relative 0.02 `
  --shikha-step-size 1.0 `
  --shikha-subproblem-time-limit-seconds 600 `
  --shikha-repair-time-limit-seconds 600 `
  --solver-threads 4
```

720 h 下，每个外层迭代最多包含 `3 vessels x 11 shrinking stages`，再加一次全空间可行恢复，计算量仍然很大。建议先用 1-3 个外层迭代和较短 solver budget 做资源估算，再锁定正式预算。

## 输出与比较口径

`per_controller.csv` 中优先比较：

- `total_cost`、`operating_cost`；
- `stored_t`、`vented_t`；
- `total_cost_per_stored_t`；
- `wall_clock_seconds`、`solver_solve_wall_seconds`；
- `decomposition_iterations`、`subproblem_solve_count`；
- `decomposition_converged`、`decomposition_stopping_reason`。

`smoke_summary.json` 的 `diagnostics.shikha2025_iterations` 还保存：

- 各船舶子问题状态和 shrinking-stage 数量；
- 全空间恢复是否可行；
- 共享服务最大冲突；
- 乘子范数；
- surrogate gap。

`shikha2025` 与 `full_milp` 一样读取整个规划时域的完美信息，因此属于
`offline_reference`。它可以与现有方法在相同物理轨迹上比较结果质量和计算时间，但不能把它描述成与只读 168 h forecast 的在线控制器具有相同信息集。
