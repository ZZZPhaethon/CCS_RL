# `experiments`

本目录放置可复现的对比实验脚本，而不放置仿真物理逻辑。脚本应使用同一场景、同一随机种子和同一物理环境，以保证不同控制方法之间的比较公平。

This directory contains reproducible comparison scripts, not physical-simulation
logic. Every comparison should use the same scenario, random seed, and physical
environment for every controller.

## `compare_hybrid_controllers.py`

比较目标感知规则执行器与目标感知原生 MPC 执行器。它报告实际的封存量、放空量、总成本、单位封存成本、封存/放空率、运行时间和违规统计。

By default it compares the goal-aware rule executor and goal-aware native MPC
executor. Add `rolling_milp` explicitly when a solver budget is available. It
reports realised stored/vented tonnes, total and unit storage cost, storage/vent
rates, execution time, and diagnostics.

The script defaults to `--objective-mode vent_first`: it is the appropriate
shared objective for a storage-first benchmark and ensures the rolling MILP
does not select idling merely because a short look-ahead has not yet vented.
Use `--objective-mode economic` only when that cost-only objective is the
explicit experiment being studied.

脚本默认使用 `--objective-mode vent_first`：这是以封存为主的基准比较所需的共享目标，可避免
滚动 MILP 仅因短期预见内尚无放空而选择空闲。只有在研究明确的纯成本目标时，才使用
`--objective-mode economic`。

```powershell
python experiments\compare_hybrid_controllers.py `
  --scenario northern_lights_phase1_3vessels `
  --seeds 1 2 3 4 5 `
  --episode-hours 168 `
  --planning-horizon-hours 72
```

若需加入滚动 MILP，例如设置每次重规划最多求解 30 秒：

```powershell
python experiments\compare_hybrid_controllers.py `
  --scenario northern_lights_phase1_3vessels `
  --seeds 1 2 3 4 5 `
  --episode-hours 720 `
  --controllers rule native_mpc rolling_milp `
  --planning-horizon-hours 168 `
  --milp-time-limit-seconds 30
```

默认结果位于 `output/hybrid_controller_comparison/`：

- `comparison_raw.csv`：每个控制器、每个种子的实际指标。
- `comparison_summary.csv`：每个控制器的均值和样本标准差。
- `comparison_metadata.json`：命令参数和初始高层调度目标。

脚本默认拒绝覆盖已有结果。只有在确定覆盖时才使用 `--overwrite`。

## `compare_shared_scenario_controllers.py`

该脚本用于规则执行器、PPO 和 rollout MPC 的严格配对比较。对于每个 seed，
脚本只生成一次 `720 h + 168 h` 扰动轨迹，再将其深拷贝给三个控制器；三个
控制器都只执行前 720 h。这样可以保证捕集量、天气、井可用性和船速轨迹完全
一致。每个 seed 结束后，脚本还会断言三个控制器的累计捕集量相同，防止误把
不同场景上的结果进行比较。

This script performs a strictly paired comparison of the rule executor, PPO,
and rollout MPC. For each seed it samples one `720 h + 168 h` disturbance
trajectory, deep-copies it into the three controller environments, and executes
only the first 720 hours. It also asserts equal cumulative captured CO2 for all
controllers before accepting a seed result.

```powershell
python experiments\compare_shared_scenario_controllers.py `
  --scenario northern_lights_phase1_3vessels `
  --seeds 1 2 3 4 5 `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --controllers rule ppo rollout_mpc `
  --ppo-run-dir logs\high_level_rl\northern_lights_phase1_3vessels__720h__decision24h__seed0__20260722_222828 `
  --output-dir output\fair_controller_comparison\northern_lights_phase1_3vessels__720h__context168h__seeds1-5__ppo50k
```

输出文件如下：

- `comparison_raw.csv`：各 seed、各控制器的原始结果。
- `comparison_summary.csv`：均值和样本标准差。
- `comparison_metadata.json`：共享场景长度、seed、PPO 模型和配对检查信息。

注意：`episode_reward` 只保留在原始结果中作为控制器内部诊断。PPO 使用高层
塑形奖励，而规则和 rollout MPC 使用仿真器奖励，因此三者的 reward 数值不能
直接横向比较；公平评价应使用封存量、放空量、总成本、单位封存成本和物理违规。
