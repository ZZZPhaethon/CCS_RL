# Residual RL / 残差强化学习

本目录是一套独立的新实现，不会覆盖或改变原有
`src/sim/control/event_based/rl` 中的 192 动作高层 PPO。原模型仍可用于复现实验。

This directory is a separate implementation. It does not overwrite or alter
the original 192-action PPO under `src/sim/control/event_based/rl`.

## 设计目标 / Design goal

规则执行器先生成一个满足物理约束的默认安全动作，PPO 只决定是否进行少量船舶
调度干预。所有动作仍交给原物理环境执行和检查。

The rule executor first produces a physically safe default action. PPO only
decides whether a small vessel-dispatch intervention is needed. The original
physical environment remains the authoritative constraint checker.

```text
事件触发 / event trigger
          ↓
平衡规则执行器生成默认动作 / balanced rule default
          ↓
PPO: 保持默认或干预一个排放源 / keep or intervene
          ↓
底层选择可行船舶 + 最高可行注入率
          ↓
原物理环境执行并校验
```

## 动作空间 / Action space

对于 Northern Lights 三排放源场景，动作空间从 192 降至 7：

| 索引 | 动作 | 含义 |
|---:|---|---|
| 0 | `keep_rule_default` | 完全保持平衡规则执行器 |
| 1 | `prioritise:brevik` | 当前决策窗口持续优先 Brevik |
| 2 | `prioritise:celsio` | 当前决策窗口持续优先 Celsio |
| 3 | `prioritise:yara_sluiskil` | 当前决策窗口持续优先 Yara |
| 4 | `add_one:brevik` | 最多向 Brevik 增派一艘空载可出发船 |
| 5 | `add_one:celsio` | 最多向 Celsio 增派一艘空载可出发船 |
| 6 | `add_one:yara_sluiskil` | 最多向 Yara 增派一艘空载可出发船 |

`prioritise` 会在整个决策区间影响新出现的可调度空船；`add_one` 在该区间最多
选择一次，并按照当前天气下的预计航行时间选择最近的合法船舶。

注入档位不属于 RL 动作。每个物理小时都调用
`highest_feasible_well_rate_index`，由底层使用当前最高可行注入率。

## 新增观测 / Added observations

残差观测保留原高层状态和 24/72 小时预测，并增加：

- 决策事件 one-hot：初始、最大间隔、船舶到达、装载完成、卸载完成、
  排放源库存阈值、溢出风险开始、天气阈值和井可用性变化；
- 每个排放源预计距离溢出的小时数；
- 每艘船到每个排放源的当前预计航行时间；
- 可立即调度的空载船比例；
- 终端卸载队列长度；
- 距离上一次决策的小时数。

三排放源、三船配置下，当前观测维度为 103。

## 困难场景与验证 / Hard scenarios and validation

默认训练使用 70% 普通场景和 30% 困难场景。困难场景提高捕集高峰、连续恶劣
天气、井维护和初始库存压力，但不会改变任何物理约束。

- 训练 episode seeds：默认 `100000–999999`；
- 固定验证 seeds：默认 `2000001–2000008`；
- 两者在启动训练前强制检查不得重叠；
- 默认每 5,000 steps 验证；
- 同时保存最终模型和验证集最优模型；
- 最优模型按以下最小化指标选择：

```text
selection_loss =
    mean_total_cost
    + tail_vent_penalty × CVaR_vent
    + hard_violation_penalty
```

其中 `CVaR_vent` 是验证集中最差 25% 场景的平均放空量。

## 正式训练 / Training

PowerShell：

```powershell
python -m sim.control.event_based.residual_rl.train_residual_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --decision-interval-h 24 `
  --timesteps 100000 `
  --num-envs 4 `
  --vec-env subproc `
  --hard-scenario-probability 0.30 `
  --validation-every-steps 5000 `
  --seed 0 `
  --device cpu
```

训练结果保存在 `logs/residual_rl/`。主要产物：

- `ppo_residual_best_validation.zip`：固定验证集上最优模型；
- `ppo_residual_final.zip`：最后一次参数更新后的模型；
- `validation/metrics.csv`：验证集均值、最差放空和 CVaR；
- `validation/best.json`：最优模型对应的验证指标；
- `training_metrics.csv` 和 `status.json`：训练过程与状态；
- `config.json`：完整动作、观测、seed 和超参数元数据。

## 独立评估 / Standalone evaluation

在原始普通场景 seeds 1–5 上评估验证集最优模型：

```powershell
python -m sim.control.event_based.residual_rl.evaluate_residual_ppo `
  --run-dir logs\residual_rl\<run_name> `
  --model best `
  --seeds 1 2 3 4 5 `
  --hard-scenario-probability 0
```

## 严格共享场景比较 / Strict shared-scenario comparison

```powershell
python experiments\compare_shared_residual_controllers.py `
  --residual-run-dir logs\residual_rl\<run_name> `
  --residual-model best `
  --scenario northern_lights_phase1_3vessels `
  --seeds 1 2 3 4 5 `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --replan-hours 24 `
  --planning-horizon-hours 168 `
  --controllers rule residual_ppo rollout_mpc `
  --output-dir output\fair_controller_comparison\<experiment_name>
```

该脚本为每个 seed 只生成一次 `720 h + 168 h` 场景，再深拷贝给所有控制器，
并断言累计捕集量完全一致。

## 文件作用 / Files

- `action_codec.py`：7 动作残差编码；
- `executor.py`：平衡规则默认动作、船舶干预和最高可行注入；
- `observation.py`：事件原因、溢出风险、航行时间和队列观测；
- `scenario.py`：可复现的普通/困难场景混合；
- `env.py`：事件触发残差半 MDP；
- `gym_env.py`：Gymnasium 接口和训练 seed 隔离；
- `factory.py`：原生/Gym 环境构建；
- `evaluation.py`：逐 seed 评估、最差场景和 CVaR；
- `train_residual_ppo.py`：4 环境训练、日志、验证和最优模型保存；
- `evaluate_residual_ppo.py`：独立模型评估。
