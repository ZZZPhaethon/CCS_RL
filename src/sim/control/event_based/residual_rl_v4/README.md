# Residual RL v4：单策略尾部风险训练

本目录是独立实现，不覆盖 `residual_rl_v3`。v4 不使用 ensemble，也不调用
MPC；运行时仍然只有一个 masked residual PPO 策略。

## 两个训练入口

- `train_tail_robust_ppo.py` 保留原 v4 的 tail curriculum、failure replay、
  shaped reward、训练默认值和 checkpoint 命名；已有 v4 实验继续使用这个入口。
- `train_objective_aligned_ppo.py` 是论文 E1 的独立入口。它复用 v4 的 residual
  action、Greedy default、event trigger、intervention windows、action masks 和
  risk gate，但从随机初始化使用 objective-aligned reward 训练，不启用 curriculum
  或 failure replay，也不会覆盖原 v4 checkpoint。

E1 reward 使用实际总成本：

```text
reward = scale × (Greedy counterfactual total cost − actual total cost)
```

Greedy 反事实项与当前策略动作无关，因此该 residual reward 与最小化实际总成本
目标一致。训练预算同时计入实际轨迹和 Greedy 反事实轨迹的所有底层 simulator-hour
calls。`--max-simulator-hour-steps` 应在正式训练时设为测得的 `B_4800`；若剩余预算
不足以完成一对实际/反事实推进，训练会提前停止且不会超支。validation/test 的
simulator calls 不计入训练预算。

```powershell
python -m sim.control.event_based.residual_rl_v4.train_objective_aligned_ppo `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --future-summary-windows-h 24 72 `
  --decision-interval-h 24 `
  --max-simulator-hour-steps <B_4800> `
  --seed 0 `
  --device cpu
```

## 设计目标

v3 seed0 在普通场景中表现稳定，但少数困难场景仍有较高放空。v4 的目标不是
提高动作多样性，而是让单个策略在保持普通场景性能的同时，更频繁地学习自身
处理失败的训练场景。

## 核心变化

### 1. 尾部风险课程

默认困难场景比例为：

| 训练进度 | 困难场景概率 |
|---:|---:|
| 0% | 10% |
| 20% | 25% |
| 40% | 40% |
| 60% | 55% |
| 80% | 40% |

最终保持 40%，避免训练末期重新遗忘困难场景。

### 2. 训练失败 replay

每个向量 worker 保存本地放空量最高的 20 个训练 episode。回放池至少包含
4 个场景后，后续 episode 有 30% 概率从回放池采样，并按放空量加权。

回放只允许使用训练 seed 范围 `100000–999999`。验证 seeds、锁定测试 seeds
不会进入回放池。重放同时保存原始 `normal/hard` 难度，避免课程概率变化后
同一个 seed 被生成成不同难度。

### 3. 受约束的 checkpoint 选择

checkpoint 使用严格的字典序排名：

1. 硬物理违规为 0；
2. 在以下两个性能约束中，失败数量尽可能少：
   - 普通场景平均放空不超过 v3 seed0 参考值的 110%；
   - 困难场景最坏放空不高于 v3 seed0；
3. 满足相同约束时，选择尾部风险损失最低的模型。

尾部风险损失为：

```text
0.5 × 普通平均成本
+ 0.5 × 困难平均成本
+ 37.5 × 普通 CVaR 放空
+ 112.5 × 困难 CVaR 放空
+ 100 × 困难最坏放空
+ 10,000,000 × 硬物理违规
```

如果训练期间没有 checkpoint 同时满足全部参考约束，仍会保存约束失败数最少
的候选模型，并在 `validation/best.json` 中标记 `"qualified": false`。

## 正式训练

v4 会自动查找验证 seeds 完全一致的最新 v3 seed0 训练作为参考。也可以通过
`--reference-run-dir` 显式指定。

```powershell
python -m sim.control.event_based.residual_rl_v4.train_tail_robust_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --decision-interval-h 24 `
  --timesteps 100000 `
  --num-envs 4 `
  --vec-env subproc `
  --validation-every-steps 10000 `
  --replay-probability 0.30 `
  --replay-capacity 20 `
  --minimum-replay-pool 4 `
  --seed 0 `
  --device cpu
```

分别运行 seed 0、1、2。最终根据固定验证集选择一个模型，不在运行时组合三个
策略。

## 冒烟测试

小验证集只用于检查代码是否能够运行，不可用于正式选模：

```powershell
python -m sim.control.event_based.residual_rl_v4.train_tail_robust_ppo `
  --timesteps 512 `
  --num-envs 1 `
  --vec-env dummy `
  --n-steps 128 `
  --validation-every-steps 256 `
  --normal-validation-seeds 2000001 2000002 `
  --hard-validation-seeds 3000001 3000002 `
  --no-reference-constraints `
  --log-dir logs\residual_rl_v4\smoke
```

## 独立测试

正式训练后，普通测试使用锁定 seeds `6000001–6000020`：

```powershell
python -m sim.control.event_based.residual_rl_v4.evaluate_ppo `
  --run-dir logs\residual_rl_v4\<run_name> `
  --model best `
  --seeds 6000001 6000002 6000003 6000004 6000005 `
  --hard-scenario-probability 0
```

困难测试使用 `7000001–7000020`，并设置
`--hard-scenario-probability 1`。这些测试 seeds 必须保持锁定，不能用于训练、
回放、调参或 checkpoint 选择。

## 输出

- `config.json`：训练、回放、参考模型和选模配置；
- `curriculum/`：课程阶段切换记录；
- `validation/metrics.csv`：每次普通/困难验证指标；
- `validation/best.json`：最佳模型的约束状态；
- `replay_summary.json`：各 worker 的回放池及重放比例；
- `maskable_residual_v4_best_validation.zip`：验证集最佳模型；
- `maskable_residual_v4_final.zip`：训练结束模型；
- `training_complete.json`：训练完成状态。

## 文件作用

- `scenario.py`：保持 seed 原始难度的可重放场景生成器；
- `replay_env.py`：top-failure 回放池和训练 episode 采样；
- `factory.py`：v4 原生及 Gym 环境工厂；
- `model_selection.py`：参考约束和尾部风险评分；
- `train_tail_robust_ppo.py`：原 v4 尾部风险训练入口；
- `train_objective_aligned_ppo.py`：论文 E1 目标对齐训练入口；
- `evaluate_ppo.py`：锁定测试集评估入口；
- `evaluate_greedy.py`：在相同物理配置和 seeds 上评估 raw
  `greedy_shuttle_policy`。
