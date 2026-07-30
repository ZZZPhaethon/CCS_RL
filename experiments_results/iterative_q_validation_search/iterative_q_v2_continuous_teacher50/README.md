# Iterative Q v2：continuous-teacher ablation

本实验测试以下训练期 previous-policy teacher 权重：

| G3 exact 改善 | Teacher 权重 |
|---:|---:|
| €0 | 100% |
| €10,000 | 80% |
| €20,000 | 60% |
| €30,000 | 40% |
| €40,000 | 20% |
| ≥€50,000 | 0% |

实现为单一线性函数：

`weight = max(0, 1 - exact_improvement_eur / 50000)`

其他条件均与选定的 hard-anchor Iterative Q v2 相同：model seed 0、
P3 初始化、累计 G0+G1+G2+G3、anchor coefficient 1.0、4/5 heads、
€40,000 inference margin、12 个决策窗口，以及相同的 20 个
controller-validation seeds（8100001–8100020）。没有访问 formal test
seeds。

## Exact-loop 结果

| 模型 | 平均总成本 € | 相对 Greedy | 胜率 | Vent t | Stored t | 单位成本 €/t | 成本标准差 € |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform P3 | 1,853,090 | −10.65% | 16/20 | 751.0 | 109,326.8 | 17.007 | 172,372 |
| Uniform P4 | 1,825,688 | −11.97% | 17/20 | 728.1 | 109,924.9 | 16.624 | 179,027 |
| **Hard-anchor v2** | **1,790,875** | **−13.65%** | **18/20** | **330.3** | **110,919.5** | **16.157** | **120,329** |
| Continuous teacher | 1,814,583 | −12.51% | 18/20 | 619.0 | 110,832.1 | 16.381 | 180,651 |

Continuous teacher 相对 hard-anchor v2：

- 平均成本增加 €23,709；
- 7/20 seeds 改善，13/20 seeds 退化；
- paired bootstrap 95% CI 为 [−€13,870, +€69,710]；
- mean Vent 增加 288.7 t；
- 成本标准差从 €120,329 增至 €180,651。

Continuous teacher 相对 P3 的平均成本降低 €38,506，但 paired
bootstrap 95% CI 为 [−€102,033, +€17,926]，且仅在 15/20 seeds
上更好。相对 P4 的平均成本降低 €11,105，95% CI 为
[−€41,206, +€14,200]。

## Previous-policy 保留效果

G3 train 中 933 个 roots 获得正权重，267 个权重为零；总有效权重
为 749.55，平均权重 62.46%。G3 validation 中分别为 194/46，
总有效权重 148.76。

| 模型 | G3 validation teacher agreement | 相对 P3 最大单-seed 回退 |
|---|---:|---:|
| Hard-anchor v2 | 88.59% | +€78,811（8100014） |
| Continuous teacher | 86.07%（weighted） | +€234,951（8100011） |

关键退化：

| Seed | P3 € | Hard v2 € | Continuous € | Continuous 相对 P3 | Continuous 相对 Hard |
|---:|---:|---:|---:|---:|---:|
| 8100011 | 2,219,845 | 2,131,436 | 2,454,797 | +€234,951 | +€323,361 |
| 8100017 | 1,879,468 | 1,791,498 | 2,030,737 | +€151,269 | +€239,239 |

Continuous teacher 在 8100002 和 8100008 分别比 hard v2 改善
€52,723 和 €143,241，但这些收益不足以抵消重新出现的尾部遗忘。

## 结论

不选择本连续版本，继续保留 hard-anchor λ=1.0 为 Iterative Q v2。

原因不是线性公式无法工作，而是本曲线对所有 €0–€40k 的非零改善都
比 hard anchor 更弱。共享网络参数会把这些局部小改善放大成后续
rollout 的策略漂移。连续版的 checkpoint 选择发生在 epoch 19，
而 hard v2 在 epoch 1；额外漂移降低了 teacher agreement，并重新
引入 8100011/8100017 的大幅回退。

若继续尝试连续形式，更合理的方向应是先保留一个 100% 权重平台，
再在较高 exact 改善区间衰减，而不是从 €0 立即线性下降。

## 作业与产物

- 环境检查：33126
- 训练：33127
- 20-seed evaluation：33128
- 所有作业退出码均为 0，stderr 均为空。
- Checkpoint：`linear_050/iterative_action_q.pt`
- SHA-256：
  `79f483c4687ef69440971acace51fff4c5c62f6a08760f514a789cffbeec5734`
