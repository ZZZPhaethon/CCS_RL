# Gated Past24 BC-only 单 seed 快速消融（2026-07-15）

## 目的

验证完整 24 h 历史信息能否通过 gate 改善现有 `Small state MLP + FutureMLP`，并快速检查 100 个 MPC 训练场景是否偏少。

本轮按快速协议只训练 `model_seed=0`。所有闭环结果均使用同一组、未参与训练的 20 个场景（seed 101--120），每个场景 720 h。

## 模型与数据

- 当前信息：78 维 state，经 MLP 编码为 64 维。
- 未来信息：168 x 9 forecast，经 FutureMLP 编码为 64 维。
- 完整 past：严格因果的前 24 h；每小时包含 78 维策略 state、4 维归一化动作和 1 维有效位，共 `24 x 83`。
- PastMLP：`1992 -> 35 -> 64`。
- 融合：PastMLP 产生 128 维 correction，并通过逐特征 `tanh(gate)` 加到原 128 维 state+future 表示上。
- gate 参数初始化为 0，所以初始化时表示严格等于原 FutureMLP；训练后才允许 past 路径打开。
- BC：decision-only、50 epochs、batch size 256、learning rate 1e-3；不进行 PPO。
- 100-case：MPC seed 0--99。
- 120-case：MPC seed 0--99 加 121--140。闭环评估 seed 101--120 未进入训练。

## 闭环结果

数值为 20 个 evaluation seed 的均值。Venting、stored 单位为 t；total cost 单位为 EUR。

| 模型 | MPC cases | 推理 | venting | stored | total cost |
|---|---:|---|---:|---:|---:|
| 原 FutureMLP | 100 | deterministic | 2,402.7 | 108,272.8 | 1,767,963 |
| Direct-concat Past24 | 100 | deterministic | 4,710.1 | 104,489.2 | 1,973,984 |
| Gated Past24 | 100 | deterministic | 4,734.3 | 103,797.0 | 1,998,344 |
| Gated Past24 | 120 | deterministic | 6,307.6 | 102,793.4 | 2,075,852 |
| 原 FutureMLP | 100 | stochastic | 11,029.7 | 95,827.8 | 2,429,786 |
| Direct-concat Past24 | 100 | stochastic | 7,098.7 | 100,677.5 | 2,163,492 |
| Gated Past24 | 100 | stochastic | 9,853.1 | 98,232.5 | 2,367,011 |
| Gated Past24 | 120 | stochastic | 5,049.9 | 103,805.6 | 2,027,631 |

100 -> 120 cases 的配对场景差异：

- deterministic venting：`+1,573 t`，95% CI `[-1,512, +4,658]`；20 个场景中 9 个改善，方向不可靠且均值变差。
- stochastic venting：`-4,803 t`，95% CI `[-7,533, -2,073]`；20 个场景中 16 个改善，改善清晰。
- 相对原 FutureMLP，120-case gated stochastic venting 为 `-5,980 t`，95% CI `[-8,702, -3,258]`；但 deterministic 为 `+3,905 t`，95% CI `[+1,113, +6,697]`。

## Gate 与 past 使用审计

100-case checkpoint 训练后的 gate：

- mean `|tanh(gate)| = 0.120`
- max `|tanh(gate)| = 0.334`
- 仅 3.1% 的 gate 绝对值小于 1e-3

在独立 MPC held-out seed 121--140 上，把 past 跨场景、同小时打乱：

- active-action accuracy：90.26% -> 77.26%
- destination-action accuracy：60.74% -> 54.64%
- 14.62% 的 observation 至少有一个船舶 argmax 动作改变
- 平均动作分布 total variation：0.0555

因此 gate 确实打开并使用了 past；100-case deterministic 退化不能归因于 past 分支没有学到信号。

## 当前判断

1. 完整 past 对 BC 学到的动作概率分布有用。最强证据是 120-case stochastic venting 相对 100-case 和原 FutureMLP 都显著下降。
2. 100 个 MPC cases 很可能不足以稳定学习带 past 的更高容量模型，但“数据少”不是唯一问题。增加到 120 后 stochastic 明显改善，deterministic argmax 却继续退化。
3. 当前主要问题更像是概率校准/argmax 决策边界，而不是 past 无信息。只看默认 deterministic rollout 会错过 past 对分布质量的收益，但在解决 argmax 退化前也不能宣布 gated past 已优于基线。
4. 本轮只有一个 model seed，场景配对置信区间只衡量 evaluation-scenario 变异，不衡量训练随机性。120-case 还比 100-case 多 20% 的每 epoch 梯度更新，因此这是快速方向检查，不是最终数据规模曲线。

## 产物

- 100-case：`output/rl_forecast/gated_past24_mlp_bc_seed0_100/`
- 120-case：`output/rl_forecast/gated_past24_mlp_bc_seed0_120/`
- 120-case 合并缓存：`output/rl_forecast/corrected_forecast_cache/destination_mask_train_120_gated_scale_v4.npz`
