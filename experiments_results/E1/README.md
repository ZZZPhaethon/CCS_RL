# E1 结果目录

## 命名规则

顶层结果目录统一使用：

`<stage>_<method>[_<key-settings>][_seeds_<start>-<end>]_runNN`

- `stage`：`formal`、`training` 或 `ablation`。
- 方法名使用小写 snake_case。
- 正式测试目录必须写明 seed 范围。
- 仅保留会改变实验含义的关键设置，例如 Rolling MILP 的 horizon、replan interval、time limit 和 solver 版本。
- `runNN` 取代不统一的 `attemptN`；历史日期只在训练批次需要区分时保留。
- `models/` 保存冻结 checkpoint，不与训练日志或正式测试结果混放。

结果 JSON 中记录的旧绝对路径和旧运行目录名属于 provenance，不回写修改。

## 当前目录

| 目录 | 内容 |
|---|---|
| `formal_fixed_assignment_seeds_9000031-9000060_run01/` | Fixed-Assignment 正式测试 |
| `formal_greedy_seeds_9000031-9000060_run01/` | Greedy 正式测试 |
| `formal_ppo_hourly_seeds_9000031-9000060_run02/` | Hourly Centralized Maskable PPO 正式测试 |
| `formal_ppo_high_level_seeds_9000031-9000060_run01/` | High-level Centralized Maskable PPO 正式测试 |
| `formal_ppo_event_residual_seeds_9000031-9000060_run01/` | Event-Residual PPO 正式测试 |
| `formal_iterative_action_q_g60_p4_seeds_9000031-9000060_run01/` | Iterative Action-Q G60-P4 正式测试；model seed 0/1/2 |
| `formal_rolling_milp_h168_r24_t600s_cplex222_seeds_9000031-9000060_run03/` | Rolling MILP 当前正式测试 |
| `superseded_formal_rolling_milp_h168_r24_t300s_cplex222_seeds_9000031-9000060_run02/` | Rolling MILP 已替代的 300 s 历史结果 |
| `formal_comparison/` | 七种算法的统一 episode、model-seed 和 algorithm 级数据集 |
| `training_ppo_hourly_20260728_run01/` | Hourly PPO 训练与 validation 产物 |
| `models/` | 冻结模型；三种 PPO 分别使用 `ppo_hourly/`、`ppo_high_level/` 和 `ppo_event_residual/` |

## 2026-07-29 完整性核对

当前统一整理的七个在线控制器为 Fixed-Assignment、Greedy、Hourly Centralized
Maskable PPO、High-level Centralized Maskable PPO、Event-Residual PPO、
Iterative Action-Q 和 Rolling MILP。

| 所需算法 | 9000031–9000060 覆盖 | 核对结果 |
|---|---:|---|
| Fixed-Assignment Heuristic | 30/30 | 完整；30 个 episode 均完成并包含 terminal cleanup |
| Greedy | 30/30 | 完整；30 个 episode 均完成并包含 terminal cleanup |
| Hourly Centralized Maskable PPO | 3 × 30/30 | 完整；model seed 0/1/2 各 30 条 |
| High-level Centralized Maskable PPO | 3 × 30/30 | 完整；model seed 0/1/2 各 30 条 |
| Event-Residual PPO | 3 × 30/30 | 完整；model seed 0/1/2 各 30 条 |
| Iterative Action-Q G60-P4 | 3 × 30/30 | 完整；model seed 0/1/2 各 30 条；seed 0 曾参与模型采用判断，不能再把该集合称为未访问 holdout |
| Rolling MILP | 30/30 | 结果完整；无 solver failure、无 fallback，所有执行 replay 有效 |

三种 PPO 使用统一且可区分的短名称：`ppo_hourly` 表示逐小时直接动作 PPO，
`ppo_high_level` 表示 24 h 高层 PPO，`ppo_event_residual` 表示事件残差 PPO。

Rolling MILP 的 900 次重规划中有 4 次 `model_replay_is_exact=false`，位于
seed 9000056/hour 264、9000058/hour 120、9000059/hour 552 和
9000060/hour 240。它们是小量数值偏差；对应 solver 与最终执行 replay 仍有效。

## 详细成本字段覆盖

| 算法结果 | Episode fuel/conditioning/reconditioning/loading/unloading | vent 与 storage shortfall penalty | Terminal cleanup 分项 |
|---|---:|---:|---:|
| Fixed-Assignment | 有 | 有 | 无；仅 cleanup 总额 |
| Greedy | 有 | 有 | 无；仅 cleanup 总额 |
| Rolling MILP | 有 | 有 | 无；仅 cleanup 总额 |
| Iterative Action-Q model seed 0/1/2 | 有 | 有 | 无；仅 cleanup 总额 |
| 三种 PPO model seed 0/1/2 | 有 | 有 | 无；仅 cleanup 总额 |

当前七种算法均已具备 episode 级 fuel、conditioning、reconditioning、
loading、unloading、vent penalty 和 storage-shortfall penalty，可用于统一的
episode 成本分项比较。Cleanup 因核算方法和模式不同，统一保留为单独总额，
不参与上述分项拆分。机器可读的完整核对结果见
`cost_field_coverage_audit_20260729.json`。
