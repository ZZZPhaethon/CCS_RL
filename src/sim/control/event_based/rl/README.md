# `sim.control.event_based.rl`

本目录实现高层、稀疏决策的 RL，而不是让 RL 在每个仿真小时直接决定所有船舶与注入细节。

This directory implements high-level sparse-decision RL. It does not ask an RL
policy to control every vessel and injection detail at every simulation hour.

```text
Operational event or maximum 24 h
    ↓ 64 actions: per-vessel service preferences only
DispatchGoal
    ↓ masked GoalAwareRuleExecutor, every physical hour
CCSEnv / physical simulator
    ↓ realised operating cost + vent cost
objective-aligned MaskablePPO reward
```

## Files / 文件

| File | Role / 作用 |
| --- | --- |
| `action_codec.py` | Maps one `Discrete(64)` action to independent vessel-service preferences; wells use automatic-max control. / 将一个 `Discrete(64)` 动作映射为各船独立服务偏好；井使用自动最大可行注入。 |
| `observation_encoder.py` | State, vessel mode/destination, and the shared 168 h forecast summary. / 状态、船舶模式/目的地与统一的 168 小时预测摘要。 |
| `reward.py` | Stable realised reward; unit cost stays an evaluation KPI. / 稳定的实际结果奖励；单位成本保留为评估 KPI。 |
| `high_level_env.py` | Advances to an operational event or the maximum 24 h interval. / 推进到运行事件或最长 24 小时间隔。 |
| `gym_env.py` | Gymnasium `Discrete` adapter with the MaskablePPO action-mask interface. / 带 MaskablePPO 动作掩码接口的 Gymnasium `Discrete` 适配器。 |
| `train_high_level_ppo.py` | Objective-aligned MaskablePPO training entry point. / 目标对齐的 MaskablePPO 训练入口。 |
| `evaluate_high_level_ppo.py` | Held-out physical evaluation with total-cost metrics. / 使用总成本指标进行留出种子物理评估。 |

## Train / 训练

```powershell
python -m sim.control.event_based.rl.train_high_level_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 `
  --decision-interval-h 24 `
  --event-triggered `
  --ent-coef 0.01 `
  --timesteps 50000 `
  --seed 0 `
  --progress-mode lines
```

After training, evaluate seeds 1–5 with the same total-cost definition used by
the rule/MPC comparison:

```powershell
python -m sim.control.event_based.rl.evaluate_high_level_ppo `
  --run-dir logs\high_level_rl\YOUR_RUN_DIRECTORY `
  --seeds 1 2 3 4 5
```

训练完成后，使用上述命令在种子 1–5 上评估；其总成本和单位封存总成本口径与
规则/MPC 对比一致，结果保存在该训练目录的 `evaluation/` 下。

All training artifacts are written under `logs/high_level_rl/` by default. Each
run receives a configuration-labelled timestamped directory containing:

- `config.json`: training, scenario, action, and observation settings;
- `status.json`: live/completed state, timestep progress, speed, and latest
  PPO metrics;
- `training_metrics.csv`: periodic optimisation metrics;
- `monitor.csv`: realised episode rewards and lengths;
- `checkpoints/`: periodic PPO checkpoints;
- `ppo_high_level_final.zip`: final trained model;
- `training_complete.json`: final model location and completion marker.

所有训练产物默认写入 `logs/high_level_rl/`。每次运行会获得一个带配置和时间戳的目录，其中包含配置、实时
状态、训练指标、每回合记录、检查点和最终模型。训练过程默认以易读的单行状态记录显示进度。

Console progress defaults to `--progress-mode lines`: one complete status line
is printed about every 5%, avoiding the concatenated redraw artefact sometimes
seen in PowerShell. Use `--progress-mode bar` only when your terminal renders
dynamic `tqdm` bars correctly. `--status-every-steps` still controls how often
`status.json` and `training_metrics.csv` are updated.

终端进度默认采用 `--progress-mode lines`：大约每完成 5% 输出一条完整状态行，
避免 PowerShell 中可能出现的进度条重绘拼接。仅当终端能正确渲染动态 `tqdm`
进度条时，才使用 `--progress-mode bar`。`--status-every-steps` 仍控制
`status.json` 与 `training_metrics.csv` 的写入频率。

Formal training fixes `gamma=1.0`. The default high-level reward is exactly
`-1e-6 × realised total cost`; it contains no stored-CO2 credit, excess-vent
shaping, or overflow-risk shaping.

`--max-simulator-hour-steps` is a hard limit on bottom-level physical advances,
not PPO decision steps. The environment checks it before every 1 h simulator
advance, so a sparse high-level transition cannot overshoot `B_4800`. Omit the
option for smoke tests until the 4,800-root budget has been measured.

PPO uses `ent_coef=0.01` by default to slow premature collapse to one static
action. Train with the fast rule executor. Use native MPC later for evaluation,
demonstrations, or teacher data—not inside the PPO training loop.

Models trained with the previous 18-action or 192-action interfaces are
historical only and are not shape-compatible with this 64-action environment.
Start a new run after this interface change.
MLP PPO defaults to `cpu`; using CUDA for this small policy usually adds
overhead without useful GPU utilisation.

正式训练固定 `gamma=1.0`。高层奖励严格等于 `-1e-6 × 实际总成本`，不包含封存量奖励、
额外放空塑形或溢出风险塑形。PPO 默认 `ent_coef=0.01`，用于减缓策略过早收敛到单一静态动作。
`--max-simulator-hour-steps` 限制的是底层 1 h 物理仿真推进次数；环境在每次推进前检查，
因此高层稀疏决策不会越过 \(B_{4800}\)。
MLP PPO 默认使用 `cpu`；原生 MPC 仅用于评估、示范或教师数据，不放入 PPO 训练内环。

旧版 18 动作和 192 动作模型仅作为历史结果保留，与新版 64 动作环境不兼容；
本次接口升级后需要启动新的训练。
