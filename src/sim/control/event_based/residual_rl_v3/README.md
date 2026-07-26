# Residual RL v3 / 第三版残差强化学习

本目录是独立实现，不覆盖 `src/sim/control/event_based/residual_rl_v2` 或已有模型。

## 为什么需要 v3

固定 30% 难度的 v2 在标准 seeds 上优于 curriculum 40k；curriculum 40k
降低了困难场景平均放空和成本，但显著增加了 adaptive greedy 使用率，并使标准
场景和困难尾部风险变差。

v3 包含：

1. adaptive 风险信号和门控诊断；
2. 硬门控阈值扫描工具；
3. 默认使用软门控奖励，而不是强制禁止预防性调度；
4. `0% → 15% → 30% → 50% → 25%` 回落式课程；
5. 普通和困难验证集各 20 个固定 seeds；
6. 将困难最差 seed 放空加入 best-model 选择；
7. 默认使用全新的测试 seed 范围。

## 门控实验结论

对 curriculum 40k 冻结模型扫描后：

- 24–96h 硬门控会过晚开放干预，普通和困难场景都变差；
- 144h 门控改善标准场景，但明显损害困难场景；
- 恶劣天气提前解锁仍不足以恢复困难场景表现；
- 综合鲁棒指标下，无硬门控仍然最好。

因此正式 v3 默认使用 `gate_mode=soft`。低风险干预仍可执行，但训练奖励会扣除
一个小的即时惩罚；如果它确实能减少未来放空，长期增量收益仍可抵消该惩罚。

## 单个 policy seed 训练

```powershell
python -m sim.control.event_based.residual_rl_v3.train_curriculum_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --decision-interval-h 24 `
  --timesteps 50000 `
  --curriculum-stages `
    0.00:0.00 0.20:0.15 0.40:0.30 0.65:0.50 0.85:0.25 `
  --gate-mode soft `
  --risk-hours-threshold-h 144 `
  --risk-fill-threshold 0.70 `
  --outside-risk-penalty 0.02 `
  --num-envs 4 `
  --vec-env subproc `
  --validation-every-steps 10000 `
  --seed 0 `
  --device cpu
```

需要分别训练 `--seed 0`、`--seed 1` 和 `--seed 2`。

## 独立测试

标准测试使用 seeds `4,000,001–4,000,020`：

```powershell
python -m sim.control.event_based.residual_rl_v3.evaluate_ppo `
  --run-dir logs\residual_rl_v3\<run_name> `
  --model best `
  --seeds `
    4000001 4000002 4000003 4000004 4000005 `
    4000006 4000007 4000008 4000009 4000010 `
    4000011 4000012 4000013 4000014 4000015 `
    4000016 4000017 4000018 4000019 4000020 `
  --hard-scenario-probability 0
```

困难测试使用互不重叠的 seeds `5,000,001–5,000,020`，并设置
`--hard-scenario-probability 1`。

## 纯 RL 风险 ensemble

`ensemble_executor.py` 提供一个不依赖 MPC 的三策略 ensemble：

1. seed0、seed1 和 seed2 在同一状态与动态动作掩码下分别预测；
2. 三者一致时直接执行共同动作；
3. 三者不一致且状态达到高风险阈值时，使用 seed1 的动作；
4. 其余情况使用整体表现最稳定的 seed0；
5. 如果选中动作失效，则回退到 residual action 0，即规则执行器默认动作。

seed2 目前仅作为分歧检测器，不直接承担高风险控制。该结构不会调用 rolling
MILP/MPC，因此仍然是“规则默认动作 + 纯 RL 残差干预”。

先在固定验证集上扫描风险阈值：

```powershell
python -m sim.control.event_based.residual_rl_v3.tune_ensemble_thresholds `
  --seed0-run logs\residual_rl_v3\<seed0_run> `
  --seed1-run logs\residual_rl_v3\<seed1_run> `
  --seed2-run logs\residual_rl_v3\<seed2_run> `
  --output-dir output\residual_rl_v3_ensemble\threshold_tuning `
  --hours-values 72 96 120 `
  --fill-values 0.8 `
  --score-values 2 3
```

当前验证集选择的阈值为：预计溢出时间 `72 h`、库存率 `0.8`、高风险分数
`2`。评估命令示例：

```powershell
python -m sim.control.event_based.residual_rl_v3.evaluate_ensemble `
  --seed0-run logs\residual_rl_v3\<seed0_run> `
  --seed1-run logs\residual_rl_v3\<seed1_run> `
  --seed2-run logs\residual_rl_v3\<seed2_run> `
  --output-dir output\residual_rl_v3_ensemble\<evaluation_name> `
  --seeds 6000001 6000002 6000003 6000004 6000005 `
  --hard-scenario-probability 0 `
  --risk-hours-h 72 `
  --risk-fill-ratio 0.8 `
  --high-risk-score 2
```

在独立测试 seeds 上，当前阈值 ensemble 尚未稳定超过 seed0 单策略：

- 普通场景：平均封存减少 `523 t`，平均放空增加 `324 t`；
- 困难场景：平均封存增加 `934 t`，但平均放空增加 `487 t`；
- 困难场景最坏放空由 seed0 的 `28,484 t` 增至 `35,984 t`；
- 所有测试均保持硬物理违规为 0。

这表明“发生分歧且风险较高就切换 seed1”仍不够精确。它能改善部分困难
seed，但会伤害另一些 seed，因此当前应保留为实验控制器，不应替换 seed0
作为默认部署策略。下一步应训练一个基于验证数据的轻量 selector，或者让不同
策略输出 value/风险估计后再选择，而不是继续手工增加阈值规则。

## 文件

- `risk_gate.py`：风险信号、阈值和门控原因；
- `env.py`：硬门控、软门控奖励和诊断日志；
- `factory.py`：原生、Gym 和课程训练环境；
- `sweep_risk_gate.py`：冻结模型阈值扫描；
- `train_curriculum_ppo.py`：五阶段课程和鲁棒模型选择；
- `evaluate_ppo.py`：全新标准/困难测试集评估；
- `ensemble_executor.py`：三策略风险 ensemble 执行器；
- `evaluate_ensemble.py`：ensemble 的逐 seed 评估与诊断；
- `tune_ensemble_thresholds.py`：只使用验证集的 ensemble 阈值扫描。
