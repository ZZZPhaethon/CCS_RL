# GNN 归因 3×2 正式 BC 对照

## 协议

- 当前状态编码器：Small MLP、参数匹配 Large MLP、Edge-GNN。
- 未来编码器：原始 ReLU TCN、FixedScale TCN。
- 六组均使用 forecast schema v4、正确逐小时排放量和 `t...t+167` 对齐。
- BC：decision-only、non-WAIT weight 10、50 epochs、无 PPO/RL 更新。
- 训练 demonstration seeds：0--99；held-out seeds：121--140。
- model seeds：0--4；闭环 eval seeds：101--120。
- 每组包含 100 条 deterministic 和 100 条 stochastic 720 h rollout。
- 总成本为运营成本加 `80 EUR/t × vented_t`；单位成本先逐 rollout 除以 `stored_t`，再跨 seeds 求平均。

## Deterministic 主结果

数值为五个 model-seed 均值的 mean ± sample SD。

| 当前状态 | 未来 | Vent (t) ↓ | Stored (t) ↑ | 运营成本 | 总成本 | 运营单位成本 | 总单位成本 |
|---|---|---:|---:|---:|---:|---:|---:|
| Small MLP | Original TCN | **2,940.9 ± 479.6** | **107,051.9 ± 954.0** | €1.5973M | **€1.8326M** | **€14.94/t** | **€17.19/t** |
| Small MLP | FixedScale | 3,867.2 ± 1,217.6 | 105,394.9 ± 1,647.0 | €1.5974M | €1.9068M | €15.18/t | €18.24/t |
| Large MLP | Original TCN | 5,270.4 ± 1,251.2 | 103,130.5 ± 1,842.6 | €1.5796M | €2.0013M | €15.34/t | €19.60/t |
| Large MLP | FixedScale | 6,027.6 ± 2,047.5 | 101,952.0 ± 2,724.1 | €1.5761M | €2.0583M | €15.49/t | €20.38/t |
| Edge-GNN | Original TCN | 9,060.0 ± 4,193.6 | 96,500.5 ± 5,338.6 | €1.5521M | €2.2769M | €16.17/t | €24.22/t |
| Edge-GNN | FixedScale | 6,889.8 ± 2,250.4 | 99,592.0 ± 2,000.2 | €1.5799M | €2.1311M | €15.91/t | €21.67/t |

## 参数量与图结构归因

Large MLP 和 Edge-GNN 的总参数量分别为 118,640 和 118,614，仅差 26。
差值为左侧方法减右侧方法，区间为五个配对 model-seed 均值的双侧 95% t 区间。

| 对比 | Deterministic vent 差值 | 95% CI | 结论 |
|---|---:|---:|---|
| Large MLP − Small MLP，Original | +2,329.5 t | [+378.0, +4,281.0] | 增加 MLP 容量显著变差 |
| Large MLP − Small MLP，FixedScale | +2,160.4 t | [+194.7, +4,126.1] | 增加 MLP 容量显著变差 |
| Edge-GNN − Large MLP，Original | +3,789.6 t | [−2,508.6, +10,087.9] | 不支持图结构改善 |
| Edge-GNN − Large MLP，FixedScale | +862.2 t | [−2,842.1, +4,566.4] | 不支持图结构改善 |

参数匹配后，Edge-GNN 在两种未来编码器下的均值都比 Large MLP 差；区间较宽并跨零，因此不能声称 GNN 必然更差，但可以明确拒绝“当前 Edge-GNN 稳定改善 BC”的主张。

## Future encoder 效果

| 当前状态 | FixedScale − Original vent | 95% CI | 改善 model-seed 对 |
|---|---:|---:|---:|
| Small MLP | +926.3 t | [−429.9, +2,282.6] | 1/5 |
| Large MLP | +757.2 t | [−1,896.0, +3,410.4] | 2/5 |
| Edge-GNN | −2,170.3 t | [−9,348.0, +5,007.5] | 3/5 |

FixedScale 没有在任何当前状态 encoder 上给出稳定的 deterministic 改善。Edge-GNN 的均值改善主要受高方差影响，区间很宽，不能作为组合收益。

## Stochastic 次要结果

| 当前状态 | Original TCN vent | FixedScale vent | FixedScale − Original |
|---|---:|---:|---:|
| Small MLP | 12,671.5 t | **8,690.7 t** | −3,980.8 t，95% CI [−7,341.1, −620.5] |
| Large MLP | **7,940.5 t** | 8,890.1 t | +949.6 t，区间跨零 |
| Edge-GNN | **19,036.7 t** | 20,673.9 t | +1,637.2 t，区间跨零 |

Small MLP 上的 FixedScale stochastic 收益可以复现，但没有泛化到 Large MLP 或 Edge-GNN。相对参数匹配 Large MLP，Edge-GNN 的 stochastic vent 在 Original 和 FixedScale 下分别高 11,096.2 t 和 11,783.8 t，两个 95% 区间都完全大于零。

## Future 使用与 held-out imitation

| 方法 | Future 活跃 seeds | Shuffle TV | Shuffle argmax change | Held-out active | Held-out destination |
|---|---:|---:|---:|---:|---:|
| Small MLP + Original | 1/5 | 0.0057 | 1.21% | 90.64% | 62.47% |
| Small MLP + FixedScale | 5/5 | 0.0295 | 6.62% | 88.86% | 59.71% |
| Large MLP + Original | 2/5 | 0.0094 | 2.28% | 89.23% | 59.58% |
| Large MLP + FixedScale | 5/5 | 0.0163 | 4.07% | 88.60% | 58.91% |
| Edge-GNN + Original | 1/5 | 0.0033 | 0.45% | **93.13%** | 69.15% |
| Edge-GNN + FixedScale | 5/5 | 0.0028 | 0.40% | 87.84% | **72.41%** |

FixedScale 在三种当前状态 encoder 上都消除了精确全零分支，但 Edge-GNN + FixedScale 的平均 shuffle TV 和 argmax change 仍很小。因此“梯度非零”只证明通路没有数值死亡，不代表策略强烈或有效地使用未来。

Edge-GNN 的 held-out destination accuracy 最好，但 deterministic 和 stochastic 闭环都更差。这再次说明一步 imitation accuracy 不能替代闭环控制评价；问题更可能位于 dispatch 时机、动作校准和 24 h MPC 隐藏计划状态。

## 结论

1. 当前 deterministic 最佳仍是 **Small MLP + Original TCN：2,940.9 t vent**。
2. 更大的状态 encoder 本身显著恶化 deterministic BC，因此不能把 GNN 与小 MLP 的差异直接归因于图结构。
3. 在参数匹配 Large MLP 对照下，当前 Edge-GNN 没有通过 GNN attribution gate。
4. FixedScale 确实修复梯度死亡，但没有形成稳定 deterministic 收益；其 stochastic 收益只在 Small MLP 上成立。
5. 当前证据不支持从这六组中的 GNN checkpoint 启动完整 PPO sweep。
