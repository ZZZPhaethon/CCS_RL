# GNN 归因 3×2 BC 正式对照（2026-07-14）

**结果日期：2026-07-14（Europe/London）**
**结果性质：preliminary results；全部为 BC-only，没有 PPO/RL 更新。**

## 1. 研究问题

本实验使用完整 3×2 因子设计，区分以下三个可能混杂的因素：

1. 当前状态表示是否应从小 MLP 换成 GNN；
2. GNN 的差异究竟来自图结构还是更多参数；
3. 修复未来 TCN 梯度死亡后，GNN 是否能更有效地利用正确的未来信息。

当前状态 encoder 包含三个水平：

- Small MLP：当前正式 baseline；
- Large MLP：与 Edge-GNN 参数匹配的无图结构容量对照；
- Edge-GNN：在 vessel-location 边上显式表示当前 travel time、at-location 和 sailing destination。

未来 encoder 包含两个水平：

- Original TCN：原始 ReLU TCN，forecast 分支可能随 model seed 死亡；
- FixedScale TCN：SiLU + non-affine LayerNorm，避免 forecast embedding 精确归零。

## 2. 固定实验协议

六组实验除当前状态 encoder 和未来 encoder 外完全一致：

- forecast schema v4：正确逐小时 `capture_rate_tph_at × availability`；
- forecast 时间范围：`t ... t+167`，与 MPC 的 168 个 simulator steps 对齐；
- 当前 observation：51 维基础状态 + 15 维 operation mode + 12 维 current destination；
- BC objective：decision-only；
- non-WAIT action-dimension weight：10；
- forced vessel-action weight：0；
- BC epochs：50；batch size：256；learning rate：0.001；
- demonstration seeds：0--99，共 72,000 个小时样本；
- held-out demonstration seeds：121--140，共 14,400 个小时样本；
- model seeds：0--4；
- closed-loop evaluation seeds：101--120；
- episode：720 h；
- 每组 100 条 deterministic rollout 和 100 条 stochastic rollout。

统计单位为 model seed。表中 `mean ± SD` 是先在每个 model seed 的 20 个 evaluation scenarios 上求均值，再对五个 model-seed 均值计算 mean 和 sample SD。配对 95% 区间同样以五个 model-seed 配对差值为独立单位，不把 100 条 rollout 视为 100 个独立训练重复。

成本定义为：

\[
\text{Total cost}=\text{Operating cost}+80\,€ / \mathrm{t}\times\text{Vented CO}_2.
\]

运营单位成本和总单位成本均先逐 rollout 除以该 rollout 的 `stored_t`，再跨 seeds 求平均。

## 3. Deterministic 主结果

| 当前状态 | 未来 | 参数量 | Vent (t) ↓ | Stored (t) ↑ | 运营成本 ↓ | 总成本 ↓ | 运营单位成本 ↓ | 总单位成本 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Small MLP | Original TCN | 86,166 | **2,940.9 ± 479.6** | **107,051.9 ± 954.0** | €1.5973M ± €0.0104M | **€1.8326M ± €0.0286M** | **€14.94 ± €0.16/t** | **€17.19 ± €0.41/t** |
| Small MLP | FixedScale | 86,166 | 3,867.2 ± 1,217.6 | 105,394.9 ± 1,647.0 | €1.5974M ± €0.0262M | €1.9068M ± €0.0749M | €15.18 ± €0.18/t | €18.24 ± €1.08/t |
| Large MLP | Original TCN | 118,640 | 5,270.4 ± 1,251.2 | 103,130.5 ± 1,842.6 | €1.5796M ± €0.0198M | €2.0013M ± €0.0899M | €15.34 ± €0.22/t | €19.60 ± €1.29/t |
| Large MLP | FixedScale | 118,640 | 6,027.6 ± 2,047.5 | 101,952.0 ± 2,724.1 | €1.5761M ± €0.0295M | €2.0583M ± €0.1376M | €15.49 ± €0.19/t | €20.38 ± €2.00/t |
| Edge-GNN | Original TCN | 118,614 | 9,060.0 ± 4,193.6 | 96,500.5 ± 5,338.6 | €1.5521M ± €0.0412M | €2.2769M ± €0.2996M | €16.17 ± €0.56/t | €24.22 ± €4.87/t |
| Edge-GNN | FixedScale | 118,614 | 6,889.8 ± 2,250.4 | 99,592.0 ± 2,000.2 | €1.5799M ± €0.0161M | €2.1311M ± €0.1690M | €15.91 ± €0.20/t | €21.67 ± €2.27/t |

当前 deterministic 最佳仍为 **Small MLP + Original TCN**。其平均 venting 为 2,940.9 t，平均 stored 为 107,051.9 t，平均总成本为 €1.833M，平均总单位成本为 €17.19/t。

### 3.1 逐 model-seed deterministic venting

| 方法 | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| Small MLP + Original | 2,627.0 | 2,303.1 | 3,337.5 | 3,445.8 | 2,991.1 |
| Small MLP + FixedScale | 2,342.1 | 4,336.9 | 3,998.2 | 5,541.7 | 3,117.3 |
| Large MLP + Original | 4,724.1 | 7,355.9 | 5,122.4 | 5,133.8 | 4,015.8 |
| Large MLP + FixedScale | 3,304.1 | 6,160.5 | 8,876.2 | 6,668.5 | 5,128.7 |
| Edge-GNN + Original | 6,257.7 | 6,201.7 | 9,511.8 | 7,161.5 | 16,167.4 |
| Edge-GNN + FixedScale | 6,142.8 | 7,966.0 | 5,021.8 | 10,292.3 | 5,026.0 |

Edge-GNN 的跨 model-seed 方差明显更大。特别是 Original TCN seed 4 的 16,167.4 t 对组均值影响较大，因此所有归因结论均以五个 seed 的配对差值及其区间为准，而不是只比较总均值。

## 4. 参数容量与图结构归因

Large MLP 和 Edge-GNN 的总参数量分别为 118,640 和 118,614，仅差 26 个参数（0.022%），可作为参数匹配对照。

差值均为左侧方法减右侧方法；venting 越低越好。

| 对比 | Deterministic vent 差值 | 95% CI | 解释 |
|---|---:|---:|---|
| Large MLP − Small MLP，Original | +2,329.5 t | `[+378.0, +4,281.0]` | 更大 MLP 显著变差 |
| Large MLP − Small MLP，FixedScale | +2,160.4 t | `[+194.7, +4,126.1]` | 更大 MLP 显著变差 |
| Edge-GNN − Large MLP，Original | +3,789.6 t | `[−2,508.6, +10,087.9]` | 不支持图结构改善 |
| Edge-GNN − Large MLP，FixedScale | +862.2 t | `[−2,842.1, +4,566.4]` | 不支持图结构改善 |

首先，单纯增加当前状态 encoder 容量会显著恶化 deterministic BC，因此不能把 Edge-GNN 与 Small MLP 的差异直接归因于图结构。

其次，在参数匹配的 Large MLP 对照下，Edge-GNN 在 Original 和 FixedScale 两列中的平均 venting 都更高。两个图结构差值区间均跨零，所以当前证据不能证明 Edge-GNN 必然更差；但可以明确拒绝“当前 Edge-GNN 稳定改善 BC”的主张。

## 5. Future encoder 效果

| 当前状态 encoder | FixedScale − Original vent | 95% CI | FixedScale 改善的 model-seed 对 |
|---|---:|---:|---:|
| Small MLP | +926.3 t | `[−429.9, +2,282.6]` | 1/5 |
| Large MLP | +757.2 t | `[−1,896.0, +3,410.4]` | 2/5 |
| Edge-GNN | −2,170.3 t | `[−9,348.0, +5,007.5]` | 3/5 |

FixedScale 没有在任何当前状态 encoder 上形成稳定的 deterministic 改善。Edge-GNN 列的平均改善具有很高的 seed 方差，区间宽且跨零，不能作为 GNN 与 future encoder 存在可靠组合收益的证据。

图结构与 future 修复对 venting 的交互项为 −2,927.5 t，95% CI 为 `[−11,446.4, +5,591.5]`，同样没有可靠交互证据。

## 6. Stochastic 次要结果

| 当前状态 | 未来 | Vent (t) ↓ | Stored (t) ↑ | 总成本 ↓ | 总单位成本 ↓ |
|---|---|---:|---:|---:|---:|
| Small MLP | Original | 12,671.5 ± 2,616.9 | 93,751.2 ± 3,380.5 | €2.5783M ± €0.2029M | €28.00 ± €3.19/t |
| Small MLP | FixedScale | **8,690.7 ± 3,417.1** | 98,768.5 ± 5,097.9 | €2.2793M ± €0.2631M | €23.51 ± €4.03/t |
| Large MLP | Original | **7,940.5 ± 1,649.1** | **99,805.6 ± 2,415.6** | **€2.2168M ± €0.1190M** | **€22.58 ± €1.73/t** |
| Large MLP | FixedScale | 8,890.1 ± 1,494.3 | 98,643.3 ± 1,654.2 | €2.2875M ± €0.1156M | €23.42 ± €1.51/t |
| Edge-GNN | Original | 19,036.7 ± 4,195.5 | 85,166.2 ± 4,866.8 | €3.0436M ± €0.3071M | €36.83 ± €6.63/t |
| Edge-GNN | FixedScale | 20,673.9 ± 5,071.3 | 83,084.2 ± 5,651.4 | €3.1641M ± €0.3700M | €39.36 ± €7.22/t |

Small MLP 上，FixedScale 将 stochastic venting 降低 3,980.8 t，95% CI `[−7,341.1, −620.5]`，复现了其对采样策略分布的改善。然而该收益没有泛化到 Large MLP 或 Edge-GNN。

相对参数匹配 Large MLP，Edge-GNN 的 stochastic venting 在 Original 和 FixedScale 下分别高 11,096.2 t 和 11,783.8 t；两者 95% 区间均完全大于零。因此当前 Edge-GNN 不仅没有 stochastic 收益，反而稳定恶化采样策略。

## 7. Forecast 使用诊断

“活跃”要求 forecast embedding 和 forecast-input gradient 均严格非零。Shuffle intervention 使用另一 held-out scenario 在相同小时的 forecast，保持当前状态和 action mask 不变。

| 方法 | Future 活跃 seeds | Feature L2 | Input-gradient L2 | Shuffle TV | Shuffle argmax change |
|---|---:|---:|---:|---:|---:|
| Small MLP + Original | 1/5 | 0.483 | 1.196e−4 | 0.0057 | 1.21% |
| Small MLP + FixedScale | 5/5 | 3.923 | 1.164e−3 | 0.0295 | 6.62% |
| Large MLP + Original | 2/5 | 0.775 | 2.549e−4 | 0.0094 | 2.28% |
| Large MLP + FixedScale | 5/5 | 3.902 | 6.584e−4 | 0.0163 | 4.07% |
| Edge-GNN + Original | 1/5 | 1.580 | 1.056e−3 | 0.0033 | 0.45% |
| Edge-GNN + FixedScale | 5/5 | 2.400 | 5.474e−5 | 0.0028 | 0.40% |

FixedScale 在三种当前状态 encoder 上都消除了精确全零分支。但是 Edge-GNN + FixedScale 的平均 shuffle TV 和 argmax change 仍然很小，甚至略低于 Edge-GNN + Original。由此可见：非零梯度证明数值通路没有完全死亡，但不等于策略强烈或有效地利用未来。

## 8. Held-out imitation 与闭环分离

| 方法 | Held-out active accuracy | Held-out destination accuracy |
|---|---:|---:|
| Small MLP + Original | 90.64% | 62.47% |
| Small MLP + FixedScale | 88.86% | 59.71% |
| Large MLP + Original | 89.23% | 59.58% |
| Large MLP + FixedScale | 88.60% | 58.91% |
| Edge-GNN + Original | **93.13%** | 69.15% |
| Edge-GNN + FixedScale | 87.84% | **72.41%** |

Edge-GNN 的 held-out destination accuracy 最好，但 deterministic 和 stochastic 闭环都更差。该结果再次说明一步 imitation accuracy 不能替代闭环控制评价。更可能的剩余问题包括 dispatch 时机、argmax/概率校准，以及每 24 h 重规划一次的 MPC 隐藏计划状态。

## 9. 完整性检查

- 30 个正式 checkpoints：6 variants × 5 model seeds；
- 30 个结果 CSV；
- 1,200 条 rollout rows；
- 1,200 个唯一 `(variant, model_seed, eval_seed, deterministic)` 键；
- 所有主指标非有限值数量：0；
- train cache SHA 数量：1；held-out cache SHA 数量：1；
- 所有训练均为 50 epochs、decision-only；
- 新组合 focused tests：148 passed；
- 全量回归：601 passed，40 subtests passed；
- `git diff --check` 无错误。

## 10. Preliminary verdict

1. 当前 deterministic 最佳仍是 **Small MLP + Original TCN**，平均 venting 为 2,940.9 t。
2. 更大的无结构状态 encoder 显著恶化 deterministic BC，说明增加容量本身不是改进方向。
3. 在参数匹配 Large MLP 对照下，当前 Edge-GNN 没有通过 GNN attribution gate。
4. FixedScale 确实修复了精确梯度死亡，但没有形成稳定 deterministic 收益；其 stochastic 收益只在 Small MLP 上成立。
5. Edge-GNN 的一步 destination imitation 更好，但闭环更差，不能据此进入完整 PPO sweep。
6. 下一步若继续研究表示，应优先处理隐藏 24 h MPC plan、dispatch 时序和闭环校准，而不是继续扩大当前状态 encoder。

## 11. 原始产物

- [完整英文/机器生成汇总](gnn_attribution_3x2/summary.md)
- [六组 aggregate metrics](gnn_attribution_3x2/aggregate_metrics.csv)
- [配对差值与 95% 区间](gnn_attribution_3x2/paired_contrasts.csv)
- [forecast 梯度与 shuffle 审计](gnn_attribution_3x2/forecast_use_audit.json)
- [详细中文运行报告](gnn_attribution_3x2/summary_zh.md)
