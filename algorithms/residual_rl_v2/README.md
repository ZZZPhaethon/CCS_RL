# Masked Residual RL v2 / 掩码残差强化学习 v2

本目录是独立的新实现，不覆盖：

- 原始 192 动作高层 PPO：`algorithms/rl`；
- 第一版 residual PPO：`algorithms/residual_rl`。

## v2 解决的问题

第一版 residual PPO 经常选择无法改变原生动作的 `add_one`，最终在 seeds 1–5
上获得与规则执行器相同的封存和放空结果，却增加约 0.5% 总成本。

v2 增加以下机制：

1. `MaskablePPO` 动态动作掩码；
2. 只有存在空载、靠泊、可合法出发且规则未派往目标的船时，干预才合法；
3. `prioritise` 至少需要两艘可改派船，避免与 `add_one` 重复；
4. 增加 `use_adaptive_greedy`，让 PPO 学习何时临时切换到全局贪心；
5. 真实系统旁边维护一个从相同初始场景出发的纯规则影子环境；
6. 训练奖励是实际系统相对规则影子环境的增量收益；
7. 记录动作被选择、动作可行、原生动作改变和最终物理收益四个层级；
8. 提供训练前的逐动作全回合筛选。

## 动态动作掩码

动作 0 `keep_rule_default` 始终合法。

动作 1 `use_adaptive_greedy` 仅在全局 greedy 动作与平衡规则动作不同时合法。
纯 adaptive 在 seed 4 表现很好、在部分其他 seed 上变差，因此它适合作为由 PPO
按状态选择的干预，而不是永久默认规则。

`add_one:<emitter>` 只有满足以下条件才合法：

- 至少一艘船空载；
- 船舶当前靠泊；
- 原生物理掩码允许驶向目标排放源；
- 规则执行器当前没有把该船派往同一目标。

`prioritise:<emitter>` 还要求至少存在两艘符合条件的船，否则它与
`add_one` 没有区别，会被掩码。

## 持续规则反事实

每个 episode reset 时创建两个完全相同的物理环境：

```text
实际环境：规则默认 + PPO 干预
规则影子：始终执行平衡规则
```

两个环境使用相同捕集、天气、井可用性和船速轨迹，并同步推进相同物理小时数。
每次训练奖励为：

```text
incremental_reward =
    actual_transition_reward
    - rule_shadow_transition_reward
```

因此一次改道在几十小时后造成的放空变化，仍会通过后续 advantage 变化反映出来。
如果整回合始终选择动作 0，则实际环境和规则影子完全一致，总增量奖励严格为 0。

## 分层日志

每个决策的 `info` 包含：

- `intervention_selected`：PPO 是否选择了非默认动作；
- `intervention_feasible_at_decision`：是否找到合法候选船；
- `eligible_vessels_at_decision`：候选船列表；
- `native_action_changed`：原生动作是否真的不同于规则；
- `changed_native_steps`：改变了多少个物理小时的动作；
- `overridden_vessels`：实际被改派的船；
- `incremental_stored_t`：相对规则的封存增量；
- `avoided_vent_t`：相对规则减少的放空；
- `total_cost_saving_eur`：相对规则节省的总成本；
- `actual_reward`、`counterfactual_reward` 和 `incremental_reward`。

## 训练前动作筛选

以下命令沿 seed 1、4 的规则轨迹，枚举所有未被掩码的动作。每个候选动作只执行
一次，随后恢复动作 0，并一直追踪到 720h 终点：

```powershell
python -m algorithms.residual_rl_v2.validate_interventions `
  --seeds 1 4 `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --decision-interval-h 24 `
  --output-dir output\residual_action_validation_v2\<experiment_name>
```

最终 8 动作接口的筛选结果表明：

- seed 1、4 共检查 140 个未被掩码的候选；
- 所有候选都真实改变了原生动作；
- 普通 `add_one` 没有减少最终放空，很多重派反而使原分配源失去服务；
- `use_adaptive_greedy` 有 18 次可行机会；
- seed 4 初始时使用一次 adaptive，最终减少约 324.6 t 放空、增加约
  324.6 t 封存，并节省约 EUR 23,275；
- 纯 adaptive 全回合在 seed 4 可把放空从约 15,843 t 降至 6,957 t，
  但它在其他 seeds 上可能变差。

因此 v2 已经存在可学习的正向动作，但建议先训练 10,000–20,000 steps 并检查
固定验证集，再决定是否扩展到 100,000 steps。

## MaskablePPO 试训练命令

建议先运行 20,000 steps 的试训练：

```powershell
python -m algorithms.residual_rl_v2.train_masked_residual_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --decision-interval-h 24 `
  --timesteps 20000 `
  --num-envs 4 `
  --vec-env subproc `
  --hard-scenario-probability 0.30 `
  --validation-every-steps 2000 `
  --seed 0 `
  --device cpu
```

日志保存在 `logs/residual_rl_v2/`。

## Curriculum learning / 课程学习

普通训练入口从第一步到最后一步都使用固定的困难场景概率，例如
`hard_scenario_probability=0.30`。这是静态混合采样，不是课程学习。

课程训练使用独立入口，不修改普通训练脚本。默认课程为：

| 训练进度 | 困难场景概率 |
|---:|---:|
| 0%–20% | 0% |
| 20%–40% | 15% |
| 40%–70% | 30% |
| 70%–100% | 50% |

阶段切换会同步更新全部并行环境，只影响下一次 episode reset。这样不会在一个
episode 中途更换天气或捕集轨迹。

建议使用 40,000 steps 运行第一组课程实验：

```powershell
python -m algorithms.residual_rl_v2.train_curriculum_masked_residual_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --decision-interval-h 24 `
  --timesteps 40000 `
  --curriculum-stages 0.00:0.00 0.20:0.15 0.40:0.30 0.70:0.50 `
  --num-envs 4 `
  --vec-env subproc `
  --validation-every-steps 5000 `
  --seed 0 `
  --device cpu
```

课程训练使用两套互不重叠的固定验证集：

- 普通验证：`hard_scenario_probability=0`；
- 困难验证：`hard_scenario_probability=1`。

最优模型依据两套验证集 selection loss 的均值保存，而不是只看训练奖励。
阶段记录保存在 `curriculum/transitions.csv`，普通/困难验证结果保存在
`validation/metrics.csv`。

## 评估

```powershell
python -m algorithms.residual_rl_v2.evaluate_masked_residual_ppo `
  --run-dir logs\residual_rl_v2\<run_name> `
  --model best `
  --seeds 1 2 3 4 5 `
  --hard-scenario-probability 0
```

## 同场景公平比较

评估结束后，使用以下命令让规则、v2 和 rollout MPC 分别复制同一份
`720h + 168h` 场景，并且只执行前 720h：

```powershell
python experiments\compare_shared_masked_residual_v2.py `
  --run-dir logs\residual_rl_v2\<run_name> `
  --model best `
  --scenario northern_lights_phase1_3vessels `
  --seeds 1 2 3 4 5 `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --replan-hours 24 `
  --planning-horizon-hours 168 `
  --controllers rule masked_residual_v2 rollout_mpc `
  --output-dir output\fair_controller_comparison\<experiment_name>
```

脚本会检查每个 seed 下三个控制器的捕集量是否完全一致，从而确认它们使用了
相同的捕集、天气、井可用性和船速轨迹。

## 文件

- `executor.py`：动态掩码、合法候选船和真实原生动作覆盖；
- `action_codec.py`：8 动作接口与 adaptive greedy 模式；
- `env.py`：持续规则影子环境和增量奖励；
- `gym_env.py`：MaskablePPO 的 `action_masks()` 接口；
- `factory.py`：环境构建；
- `evaluation.py`：掩码推理、物理指标和干预诊断；
- `validate_interventions.py`：训练前逐动作全回合筛选；
- `train_masked_residual_ppo.py`：4 环境训练、验证和最优模型保存；
- `curriculum.py`：动态难度环境、课程阶段解析和并行环境难度更新；
- `train_curriculum_masked_residual_ppo.py`：课程训练和普通/困难双验证；
- `evaluate_masked_residual_ppo.py`：固定 seeds 独立评估。
- `experiments/compare_shared_masked_residual_v2.py`：同场景公平比较入口。
