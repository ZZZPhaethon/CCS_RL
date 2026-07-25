# Balanced Edge-GNN + Future-MLP BC 初步结果（2026-07-14）

## 结论

将 Balanced Edge-GNN 的未来分支从 FixedScale TCN 换成 Future-MLP 后，五个模型种子的未来输入梯度都保持非零，没有出现 Original TCN 的精确分支死亡。Deterministic venting 从 **6,181.9 ± 2,435.6 t** 降至 **4,995.4 ± 3,340.7 t**，均值改善 1,186.5 t，但配对 95% CI 为 `[−6,728.8, 4,355.7]`，不能认定改善显著。

该组合仍落后于 Small MLP + Future-MLP 的 **3,004.2 ± 536.3 t**，也没有超过当前最佳 Small MLP + Original TCN 的 **2,940.9 ± 479.6 t**。因此，Original TCN 的梯度死亡确实可以由 Future-MLP 避免，但 Edge-GNN 的主要闭环问题并不只来自未来编码器。

## 实验设置

- variant：`balanced_edge_gnn_future_mlp_mode_destination`
- current state：Edge-GNN + non-affine LayerNorm + SiLU
- future：展平后的参数匹配 Future-MLP，`1,512 → 35 → 64`
- future normalization：non-affine LayerNorm + SiLU
- 输出：mode + current destination，decision-only
- policy：BC only；无 PPO
- forecast：已修复排放量和时间对齐的 v4 forecast
- demonstrations：train seeds 0–99；held-out seeds 121–140
- model seeds：0–4
- eval seeds：101–120
- 训练：每个模型 50 epochs，本地 NVIDIA RTX 3080
- 闭环：deterministic/stochastic 各 100 episodes
- policy 可训练参数：119,025；只比 Balanced Edge-GNN + FixedScale TCN 多 411 个

缓存指纹：

- train SHA-256：`52eacbf34eaefd37568e406232838889182af57aa96426ed8e9463c084913a54`
- held-out SHA-256：`4761bba0cda9fe69b80fadce3e1491b30dcbc72c8c1225ffd15b04d5b261b784`

## 闭环结果

数值为每个模型种子先对 20 个 eval seeds 求均值，再报告五个模型种子的均值 ± 样本 SD。

| 模式 | Vent (t) ↓ | Stored (t) ↑ | Operating cost (€) ↓ | Total cost (€) ↓ | Operating €/t ↓ | Total €/t ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic | 4,995.4 ± 3,340.7 | 102,664.6 ± 5,053.1 | 1,590,485 ± 37,558 | 1,990,117 ± 235,675 | 15.539 ± 0.496 | 19.718 ± 3.592 |
| Stochastic | 10,865.9 ± 3,310.9 | 94,757.1 ± 3,800.1 | 1,579,732 ± 20,537 | 2,449,006 ± 245,377 | 16.741 ± 0.523 | 26.391 ± 4.203 |

### 每个模型种子的 venting

| Model seed | Deterministic vent (t) | Stochastic vent (t) |
|---:|---:|---:|
| 0 | 4,217.6 | 7,746.9 |
| 1 | 3,632.9 | 10,729.9 |
| 2 | 4,645.8 | 9,259.3 |
| 3 | 1,827.0 | 10,158.0 |
| 4 | 10,653.8 | 16,435.4 |

seed 4 出现明显退化。去掉 seed 4 会极大改变均值，因此不能把 4,995.4 t 当作稳定性能；高模型种子方差仍是该 GNN 架构的核心问题。

## 与同一 GNN + FixedScale TCN 对比

差值为 `GNN + Future-MLP − GNN + FixedScale TCN`，使用五个相同模型种子的 seed-level paired t interval。

| Deterministic 指标 | 差值 | 95% CI | 判断 |
|---|---:|---:|---|
| Vent | −1,186.5 t | [−6,728.8, 4,355.7] | 均值改善，不显著 |
| Stored | +1,780.2 t | [−5,874.7, 9,435.0] | 均值改善，不显著 |
| Operating cost | +€3,769 | [−€72,601, €80,139] | 不显著 |
| Total cost | −€91,152 | [−€484,523, €302,218] | 均值改善，不显著 |
| Operating €/t | −0.230 | [−1.089, 0.629] | 不显著 |
| Total €/t | −1.159 | [−7.092, 4.773] | 不显著 |

Stochastic venting 也从 11,829.8 t 降至 10,865.9 t，差值 −963.9 t，95% CI `[−9,027.1, 7,099.3]`。两种评估模式都呈方向性改善，但当前证据不足以排除模型种子随机性。

## 与 MLP 当前状态版本对比

| Deterministic 版本 | Vent (t) | Stored (t) | Total cost (€) | Operating €/t | Total €/t |
|---|---:|---:|---:|---:|---:|
| Small MLP + Original TCN | **2,940.9 ± 479.6** | **107,051.9 ± 954.0** | **1,832,552 ± 28,621** | **14.937 ± 0.163** | **17.191 ± 0.405** |
| Small MLP + Future-MLP | 3,004.2 ± 536.3 | 106,246.9 ± 1,460.7 | 1,826,361 ± 44,710 | 14.948 ± 0.233 | 17.275 ± 0.642 |
| Balanced Edge-GNN + Future-MLP | 4,995.4 ± 3,340.7 | 102,664.6 ± 5,053.1 | 1,990,117 ± 235,675 | 15.539 ± 0.496 | 19.718 ± 3.592 |

相对 Small MLP + Future-MLP，GNN + Future-MLP 的 deterministic vent 平均高 1,991.2 t，95% CI `[−2,485.1, 6,467.5]`。相对 Small MLP + Original TCN，vent 平均高 2,054.5 t，95% CI `[−2,167.2, 6,276.2]`。高 seed 方差使两项 vent 区间都跨零，但相对 Original TCN 的 operating €/t 高 0.602，95% CI `[0.056, 1.147]`，区间不跨零。

## 未来信息梯度与实际使用

| 指标 | GNN + FixedScale TCN | GNN + Future-MLP | Small MLP + Future-MLP |
|---|---:|---:|---:|
| Active future seeds | 5/5 | **5/5** | 5/5 |
| State feature L2 | 6.779 | 6.653 | — |
| Future feature L2 | 2.066 | 2.341 | 1.935 |
| Future/state feature ratio | 0.305 | 0.352 | — |
| Future/state effective contribution | 0.300 | 0.341 | — |
| Future input-gradient L2 | 1.44e−4 | 3.28e−5 | 1.07e−4 |
| Shuffle probability TV | 0.00538 | 0.00497 | 0.0150 |
| Shuffle argmax change | 1.02% | 0.83% | 3.02% |

GNN + Future-MLP 五个种子的 future input-gradient L2 分别为 `3.89e−5`、`6.21e−6`、`3.20e−5`、`3.85e−5`、`4.82e−5`，全部大于零。因此它没有 Original TCN 那种精确全零未来分支。

然而，梯度存在不等于策略强烈使用未来。打乱未来轨迹只改变 0.83% 的 argmax，甚至略低于 GNN + FixedScale TCN 的 1.02%，远低于 Small MLP + Future-MLP 的 3.02%。Future-MLP 修复了梯度死亡，但没有消除 GNN 的 current-state shortcut。

Held-out active-decision accuracy 为 92.06% ± 2.75%，destination accuracy 为 73.85% ± 1.83%。离线准确率较高而闭环不稳定，再次指向 BC covariate shift，而不是简单的监督拟合不足。

## 判断

1. **Original TCN 梯度死亡问题在该组合中已避免。** Future-MLP 为 5/5 非零未来梯度。
2. **换 Future-MLP 对 GNN 有方向性帮助，但没有可靠地解决闭环问题。** Deterministic vent 均值下降约 19%，但 CI 很宽。
3. **GNN 仍不如 Small MLP。** 相同 Future-MLP 下，GNN 的平均 vent 更高、seed 方差大约六倍。
4. **瓶颈主要转向 GNN 当前状态表示及 BC shortcut/covariate shift。** 未来分支活着，但策略对 future shuffle 仍不敏感。
5. 当前最佳 BC 不变：Small MLP + Original TCN；如果希望完全规避未来分支死亡，Small MLP + Future-MLP 是性能接近且更稳定的替代方案。

下一步若继续做最小归因，应缩小 Edge-GNN 容量，同时保留 state LayerNorm 和 Future-MLP；如果目标是提升闭环而非继续归因，应优先考虑 DAgger 或增加专家计划上下文。

## 完整性与产物

- 5 个正式 checkpoints 和 5 个 training manifests
- 5 个 evaluation CSV，每个 40 行，共 200 行
- 200 个唯一 `(variant, model_seed, eval_seed, deterministic)` 键
- 六个主指标非有限值数量：0
- focused tests：163 passed
- 全量回归：623 passed，另有 41 subtests passed
- 训练与评估：`output/rl_forecast/balanced_edge_gnn_future_mlp_bc/`
- 未来信息审计：`output/rl_forecast/balanced_edge_gnn_future_mlp_bc/forecast_use_audit.json`
- 机器可读汇总：`output/rl_forecast/balanced_edge_gnn_future_mlp_bc/summary.json`
