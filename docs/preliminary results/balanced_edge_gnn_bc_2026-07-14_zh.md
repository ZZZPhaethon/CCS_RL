# Balanced Edge-GNN BC 初步结果（2026-07-14）

## 结论

给 Edge-GNN 当前状态分支增加无可学习参数的 LayerNorm，成功消除了此前由训练形成的状态/未来表征尺度严重失衡，并明显增强了策略对未来信息的敏感度。不过，该修改只带来方向一致但不显著的 deterministic 闭环改善，仍明显落后于 Small MLP + Original TCN 基线。因此，这项实验支持“尺度失衡是 Edge-GNN 表现差的一个原因”，但也表明它不是唯一原因。

当前最好的 BC 基线仍为 **Small MLP current state + Original TCN future**：deterministic venting 为 **2,940.9 ± 479.6 t**。Balanced Edge-GNN 对应结果为 **6,181.9 ± 2,435.6 t**。

## 改动与假设

实验只改变一个因素：在 Edge-GNN 当前状态编码器输出后增加

```text
Edge-GNN -> LayerNorm(elementwise_affine=False) -> SiLU
```

未来分支仍使用 FixedScale TCN，其余 BC 数据、任务表示、训练轮数、随机种子和评估设置保持不变。LayerNorm 不含可学习仿射参数，因此没有增加模型容量；模型仍为 118,614 个可训练参数。

待验证假设是：Edge-GNN 在 BC 中把当前状态特征放大，压制了 TCN 未来特征；固定归一化当前状态分支后，未来分支应获得更合理的相对权重，并改善闭环控制。

## 实验设置

- variant：`balanced_edge_gnn_mode_destination`
- policy：BC only；未使用 PPO
- current state：Edge-GNN + non-affine LayerNorm + SiLU
- future：FixedScale TCN
- output：mode + current destination，decision-only
- forecast：已修复未来排放量的 v4 数据
- 训练场景种子：0–99
- held-out 场景种子：121–140
- 模型种子：0–4
- 闭环评估种子：101–120
- 每个模型种子：50 epochs
- deterministic/stochastic：各 5 × 20 = 100 episodes
- 运行设备：本地 NVIDIA RTX 3080

训练数据指纹：

- train SHA-256：`52eacbf34eaefd37568e406232838889182af57aa96426ed8e9463c084913a54`
- held-out SHA-256：`4761bba0cda9fe69b80fadce3e1491b30dcbc72c8c1225ffd15b04d5b261b784`

## 表征尺度与未来信息使用

下表均为 5 个模型种子的均值；`future/state contribution` 是 policy 第一层中两个分支的有效输入贡献比。

| 指标 | Edge-GNN + FixedScale TCN | Balanced Edge-GNN | 变化 |
|---|---:|---:|---:|
| current-state feature L2 | 28.604 | 6.779 | 降低 76.3% |
| future feature L2 | 2.400 | 2.066 | 基本同量级 |
| future/state feature ratio | 0.088 | 0.305 | 3.47× |
| future/state effective contribution | 0.057 | 0.300 | 5.27× |
| future shuffle probability TV | 0.0028 | 0.00538 | 约 1.9× |
| future shuffle argmax change | 0.40% | 1.02% | 约 2.5× |
| held-out active accuracy | 87.84% | 92.29% | +4.45 pp |
| held-out destination accuracy | 72.41% | 74.30% | +1.89 pp |

归一化后的 current-state L2 在五个种子间为 **6.779 ± 0.122**，说明它确实稳定了状态分支尺度。未来输入梯度 L2 为 **1.44e-4 ± 1.80e-4**，5/5 模型都存在非零未来梯度。

但未来使用仍明显弱于 Small MLP + FixedScale TCN：后者 future shuffle argmax change 为 6.62%，而 Balanced Edge-GNN 只有 1.02%。这意味着尺度修复只是部分解决了 current-state shortcut。

## 闭环结果

数值为五个模型种子的 episode-mean 再取均值 ± SD；每个模型种子包含 20 个相同评估场景。

| 模式 | Vent (t) | Stored (t) | Operating cost (€) | Total cost (€) | Operating €/t | Total €/t |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic | 6,181.9 ± 2,435.6 | 100,884.4 ± 3,648.4 | 1,586,716 ± 29,867 | 2,081,269 ± 180,384 | 15.769 ± 0.504 | 20.877 ± 2.714 |
| Stochastic | 11,829.8 ± 4,430.5 | 93,592.9 ± 4,693.6 | 1,568,832 ± 33,892 | 2,515,218 ± 320,581 | 16.841 ± 0.514 | 27.440 ± 5.145 |

### 每个模型种子的 venting

| Model seed | Deterministic vent (t) | Stochastic vent (t) |
|---:|---:|---:|
| 0 | 10,230.3 | 11,123.9 |
| 1 | 4,853.9 | 14,449.9 |
| 2 | 6,714.7 | 17,970.8 |
| 3 | 4,565.1 | 7,094.5 |
| 4 | 4,545.5 | 8,510.0 |

seed 0 的 deterministic 结果明显偏差，表明 Edge-GNN 仍存在较强的训练种子不稳定性。

## 与现有版本对比

### 对 Edge-GNN + FixedScale TCN 的直接归因

配对单位为相同的 5 个模型种子；括号内为 seed-level paired difference 的 95% CI。负 vent/cost 表示 Balanced 更好。

| 模式 | 指标 | Balanced − Edge Fixed | 95% CI | 判断 |
|---|---|---:|---:|---|
| Deterministic | Vent | −707.8 t | [−5,508.2, 4,092.5] | 均值改善，但不显著 |
| Deterministic | Stored | +1,292.4 t | [−4,402.3, 6,987.1] | 不显著 |
| Deterministic | Total cost | −€49,810 | [−€417,164, €317,544] | 不显著 |
| Deterministic | Total €/t | −0.788 | [−5.970, 4.394] | 不显著 |
| Stochastic | Vent | −8,844.1 t | [−18,955.4, 1,267.2] | 大幅方向性改善，CI 仍跨 0 |
| Stochastic | Stored | +10,508.7 t | [−578.8, 21,596.3] | 大幅方向性改善，CI 仍跨 0 |
| Stochastic | Total cost | −€648,854 | [−€1,387,118, €89,410] | 大幅方向性改善，CI 仍跨 0 |

因此，尺度归一化对 stochastic Edge-GNN 的均值改善很大，对 deterministic 的改善较小；但五个模型种子不足以在当前高方差下给出显著结论。

### 对当前最佳 BC 基线

| Deterministic 指标 | Balanced Edge-GNN | Small MLP + Original TCN | Balanced − baseline（95% CI） |
|---|---:|---:|---:|
| Vent (t) | 6,181.9 | 2,940.9 | +3,241.0 [22.8, 6,459.3] |
| Stored (t) | 100,884.4 | 107,051.9 | −6,167.5 [−11,250.7, −1,084.3] |
| Total cost (€) | 2,081,269 | 1,832,552 | +248,717 [17,682, 479,753] |
| Operating €/t | 15.769 | 14.937 | +0.832 [0.131, 1.532] |
| Total €/t | 20.877 | 17.191 | +3.686 [0.149, 7.223] |

Balanced Edge-GNN 在 deterministic 闭环中仍显著差于当前基线，不能替代 Small MLP + Original TCN。

## 解释与下一步

本实验将 Edge-GNN 的 current-state feature L2 从 28.6 压到稳定的 6.8，并将未来/当前状态的有效贡献比从 0.057 提升到 0.300，因此“状态分支尺度压制未来分支”这个诊断得到支持。

仍未解决的部分包括：

1. BC 仍可依赖当前状态形成 shortcut；归一化不能强制策略真正使用未来轨迹。
2. Edge-GNN 的容量大于 Small MLP，更容易拟合专家数据中的隐藏计划或场景特征，从而在闭环分布偏移时失效。
3. 现有图编码器共享异质节点嵌入并使用固定拓扑 flatten readout，可能没有形成合适的设备级归纳偏置。
4. seed 0 的明显退化说明优化稳定性仍不足。

如果继续做最小归因，优先顺序应是：先缩小 Edge-GNN 容量并保持这次归一化，再比较；随后才考虑双分支对称归一化或 gated fusion。若目标是直接提升闭环性能，补充专家计划上下文或采用 DAgger 比继续堆叠图层更有希望。

## 产物与验证

- 训练与 checkpoint：`output/rl_forecast/balanced_edge_gnn_bc/`
- 评估结果：`output/rl_forecast/balanced_edge_gnn_bc/eval_101_120/`
- 未来信息审计：`output/rl_forecast/balanced_edge_gnn_bc/forecast_use_audit.json`
- 完整测试：614 passed；另有 41 个 subtests passed
- focused tests：158 passed
- 结果完整性：5 checkpoints、5 training manifests、5 evaluation CSVs、200 evaluation rows、5 audit rows；无非有限数值
