# 当前状态 MLP + 未来信息 MLP 的 BC 对照（2026-07-14）

**结果日期：2026-07-14（Europe/London）**  
**结果性质：preliminary results；BC-only，无 PPO/RL 更新。**

## 1. 研究问题

本实验只替换未来信息 encoder，检验当前状态和未来信息都使用 MLP 时的闭环效果。其他设置与正式 Small MLP + TCN 对照保持一致：

- 当前状态：基础状态 + operation mode + sailing destination，共 78 维；
- 当前状态 encoder：Small MLP，`78 → 64`；
- 未来输入：修复排放量和时间对齐后的 v4 forecast，`168 h × 9 channels`；
- Future-MLP：展平后 `1,512 → 35 → 64`；
- Future-MLP 激活与归一化：SiLU + non-affine LayerNorm；
- 融合：当前状态 64 维与未来信息 64 维直接拼接；
- BC objective：decision-only；
- 训练：50 epochs，model seeds 0–4；
- demonstrations：train seeds 0–99，held-out seeds 121–140；
- 闭环评估：eval seeds 101–120，每个模型同时评估 deterministic 和 stochastic；
- 所有指标均先对每个 model seed 的 20 个 eval seeds 求均值，再报告 5 个 model seeds 的均值 ± 样本标准差。

Future-MLP 的 future 分支有 55,259 个可训练参数，FixedScale TCN future 分支有 54,848 个，仅相差 411 个参数（0.75%）。完整 policy 参数量分别为 86,577 和 86,166，相差 0.48%。因此该实验基本排除了参数容量差异。

## 2. Deterministic 主结果

| Future encoder | Vent (t) ↓ | Stored (t) ↑ | Operating cost ↓ | Total cost ↓ | Operating €/t ↓ | Total €/t ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original TCN | **2,940.9 ± 479.6** | **107,051.9 ± 954.0** | €1.5973M ± €0.0104M | €1.8326M ± €0.0286M | **€14.94 ± €0.16** | **€17.19 ± €0.40** |
| FixedScale TCN | 3,867.2 ± 1,217.6 | 105,394.9 ± 1,647.0 | €1.5974M ± €0.0262M | €1.9068M ± €0.0749M | €15.18 ± €0.18 | €18.24 ± €1.08 |
| **Future-MLP** | 3,004.2 ± 536.3 | 106,246.9 ± 1,460.7 | **€1.5860M ± €0.0100M** | **€1.8264M ± €0.0447M** | €14.95 ± €0.23 | €17.28 ± €0.64 |

Future-MLP 的平均 venting 只比 Original TCN 高 63.3 t（2.2%），但平均 operating cost 低约 €11.3k、total cost 低约 €6.2k。相对 Original TCN 的这些差异区间均跨零，因此不能宣称 Future-MLP 更好或更差。

### 每个 model seed 的 deterministic venting

| Future encoder | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---:|---:|---:|---:|---:|
| Original TCN | 2,627.0 | 2,303.1 | 3,337.5 | 3,445.8 | 2,991.1 |
| FixedScale TCN | 2,342.1 | 4,336.9 | 3,998.2 | 5,541.7 | 3,117.3 |
| **Future-MLP** | 2,402.7 | 2,849.6 | 3,641.2 | 3,483.2 | 2,644.3 |

## 3. 配对差值

差值均为 `Future-MLP − 对照`；vent 和成本为负表示 Future-MLP 更好。95% 区间使用 5 个 model seeds 的配对差值和 t 区间。

| Deterministic 对照 | Vent 差值 | Stored 差值 | Operating cost 差值 | Total cost 差值 | Operating €/t 差值 | Total €/t 差值 |
|---|---:|---:|---:|---:|---:|---:|
| vs Original TCN | +63.3 t `[-394.1, +520.7]` | −805.0 t `[−2,143.6, +533.6]` | −€11.3k `[−€24.6k, +€2.1k]` | −€6.2k `[−€48.5k, +€36.1k]` | +€0.01 `[−€0.29, +€0.32]` | +€0.08 `[−€0.42, +€0.59]` |
| vs FixedScale TCN | −863.0 t `[−1,952.6, +226.5]` | +852.0 t `[−785.1, +2,489.1]` | −€11.4k `[−€48.0k, +€25.2k]` | **−€80.4k `[−€141.9k, −€19.0k]`** | −€0.23 `[−€0.60, +€0.15]` | **−€0.96 `[−€1.91, −€0.02]`** |

Future-MLP 相对 FixedScale TCN 的 deterministic total cost 和 total unit cost 更低，区间不跨零；但主要 venting 指标的区间仍跨零。

## 4. Stochastic 次要结果

| Future encoder | Vent (t) ↓ | Stored (t) ↑ | Operating cost ↓ | Total cost ↓ | Operating €/t ↓ | Total €/t ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original TCN | 12,671.5 ± 2,616.9 | 93,751.2 ± 3,380.5 | €1.5646M ± €0.0105M | €2.5783M ± €0.2029M | €16.76 ± €0.55 | €28.00 ± €3.19 |
| FixedScale TCN | **8,690.7 ± 3,417.1** | **98,768.5 ± 5,097.9** | €1.5841M ± €0.0133M | **€2.2793M ± €0.2631M** | **€16.11 ± €0.76** | **€23.51 ± €4.03** |
| **Future-MLP** | 11,908.0 ± 1,750.3 | 94,210.8 ± 2,157.1 | **€1.5646M ± €0.0194M** | €2.5173M ± €0.1278M | €16.66 ± €0.32 | €27.06 ± €2.06 |

Future-MLP 相对 Original TCN 的 stochastic venting 平均低 763.5 t，但 95% 区间为 `[−5,102.2, +3,575.2]`。相对 FixedScale TCN 则平均高 3,217.3 t，95% 区间为 `[−2,276.9, +8,711.5]`。两项区间均跨零。

## 5. Future 使用审计

| Future encoder | 活跃 model seeds | Forecast feature L2 | Input-gradient L2 | Shuffle TV | Shuffle argmax change |
|---|---:|---:|---:|---:|---:|
| Original TCN | 1/5 | 0.483 | 1.196e−4 | 0.0057 | 1.21% |
| FixedScale TCN | 5/5 | 3.923 | 1.164e−3 | 0.0295 | 6.62% |
| **Future-MLP** | **5/5** | 1.935 | 1.065e−4 | 0.0150 | 3.02% |

Future-MLP 在 5 个 model seeds 上都保持非零 forecast embedding 和非零输入梯度，没有出现 Original TCN 的精确全零分支。打乱未来信息会改变约 3.02% 的样本 argmax，说明策略确实使用未来；其使用强度高于 Original TCN，但低于 FixedScale TCN。

Future-MLP 的 held-out active-decision accuracy 为 89.56% ± 1.31%，destination accuracy 为 64.88% ± 2.74%。

## 6. 完整性检查

- 5 个正式 checkpoints，model seeds 0–4；
- 5 个 evaluation CSV，每个 40 行，共 200 行；
- 200 个唯一 `(variant, model_seed, eval_seed, deterministic)` 键；
- 主指标非有限值数量：0；
- 所有模型均为 50 epochs、decision-only、BC-only；
- train cache SHA：`52eacbf34eaefd37568e406232838889182af57aa96426ed8e9463c084913a54`；
- held-out cache SHA：`4761bba0cda9fe69b80fadce3e1491b30dcbc72c8c1225ffd15b04d5b261b784`；
- 两个 SHA 与正式六版本完全一致；
- forecast schema version 4，范围为当前小时到未来 167 小时；
- focused tests：153 passed；全量回归：607 passed，41 subtests passed。

## 7. Preliminary verdict

1. **Future-MLP 是可用架构，但没有成为新的 venting 最佳模型。**
2. Deterministic venting 与 Original TCN 基本相当：3,004.2 t vs 2,940.9 t，配对区间跨零。
3. Future-MLP 的 deterministic total cost 数值最低，但相对 Original TCN 的差异不显著；相对 FixedScale TCN 则显著更低。
4. Future-MLP 成功避免未来分支精确死亡，而且比 Original TCN 更稳定地使用未来信息。
5. FixedScale TCN 仍有最好的 stochastic 平均结果，但 deterministic 结果不稳定。
6. 若主要目标是 deterministic venting，当前 baseline 仍保留 Small MLP + Original TCN；Future-MLP 可以作为一个梯度稳定、性能接近的替代分支进入后续小规模 PPO 或融合实验，但现有证据不支持直接替换 baseline。

## 8. 原始产物

- [训练 checkpoints 与 manifests](../../output/rl_forecast/future_mlp_bc/)
- [101–120 闭环评估](../../output/rl_forecast/future_mlp_bc/eval_101_120/)
- [forecast-use audit](../../output/rl_forecast/future_mlp_bc/forecast_use_audit.json)
- [Future-MLP encoder 实现](../../src/sim/environment/forecast_encoder.py)
