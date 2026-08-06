# 删除 hour_of_week 的预算匹配 one-shot Iterative-Q

日期：2026-07-30

## 实验问题

训练一个不进行 policy iteration 的 Iterative Action-Q：

- roll-in 数据仅来自 Greedy policy；
- 只训练一次 P1；
- 模型从随机初始化开始，`initial_checkpoint=None`；
- 删除 observation `[0] hour_of_week`；
- 与正式 G60-P4 iterative-Q 使用近似相同的训练物理模拟预算；
- model seeds 为 0、1、2；
- 最终在正式 30 个场景 seeds 9000031–9000060 上配对评估。

## 物理模拟预算

| 模型 | train simulator calls | train roots |
|---|---:|---:|
| 正式 iterative-Q G60-P4 | 9,526,297 | 3,599 |
| Greedy-only one-shot | 9,511,567 | 3,611 |
| 差异 | −14,730（−0.1546%） | +12 |

该口径与项目正式 protocol 一致，指训练数据生成的 simulator calls。若额外计入 checkpoint-selection validation 数据，正式 iterative-Q 为 11,607,560 calls，one-shot 为 10,784,595 calls；后者低 7.09%。validation 不参与梯度训练。

## 与正式 iterative-Q 的结果

`delta = one-shot no-hour − formal iterative-Q`，正数表示 one-shot 更差。

| model seed | 成本 delta | 相对变化 | vented delta | one-shot 胜/负场景 |
|---|---:|---:|---:|---:|
| 0 | +€89,979 | +4.812% | +1,073.4 t | 7 / 23 |
| 1 | +€25,702 | +1.375% | +278.5 t | 12 / 18 |
| 2 | +€34,146 | +1.791% | +209.3 t | 10 / 20 |
| pooled | +€49,942 | +2.654% | +520.4 t | 29 / 61 |

不确定性：

- 在每个场景先跨三个 model seeds 求均值、再重采样场景：95% bootstrap CI 为 `[−€3,774, +€104,418]`；
- 同时重采样 model seed 与场景：95% bootstrap CI 为 `[−€23,819, +€115,549]`。

三个 model seed 的均值方向一致，均为 one-shot no-hour 成本更高；但场景方差较大，bootstrap 区间仍跨 0。

## hour_of_week 在 one-shot 中的影响

同预算、同 Greedy-only 数据下，将新模型与原始 full-feature one-shot 对比：

| model seed | 删除 hour 后的成本变化 |
|---|---:|
| 0 | +5.093% |
| 1 | −1.063% |
| 2 | +3.445% |
| pooled | +€46,414（+2.462%） |

场景 bootstrap CI 为 `[+€13,015, +€84,411]`，但同时重采样 model seed 与场景的区间为 `[−€28,863, +€120,988]`。这说明在仅含 Greedy roots 的 one-shot 分布中，删除 `hour_of_week` 的平均结果偏向有害，但 model-seed 稳定性仍不足。

原始 full-feature one-shot 与正式 iterative-Q 的 pooled 成本差只有 `+€3,529（+0.188%）`。因此本次 one-shot no-hour 的 `+2.654%` 差距主要不是“没有迭代”单独造成，而更像是删除 hour 与 one-shot 数据分布/训练随机性共同造成。

## 结论

1. 不能把“在自洽 iterative 重训中 hour 没有稳定作用”推广为“在所有训练分布中 hour 都没用”。
2. 在 Greedy-only one-shot 下，删除 hour 后三个 seed 相对正式 iterative-Q 均变差，pooled 成本高 2.65%。
3. 与 full-feature one-shot 的对照也偏向删除 hour 有害。因此如果决定采用 one-shot 方案，应暂时保留 `hour_of_week`。
4. 当前结果不支持用“删 hour + 不迭代”的模型替换正式 iterative-Q。

## 完整性核验

- Borg jobs：环境检查 35419，训练数组 35420，正式评估 35421–35423；
- 所有任务均 `COMPLETED 0:0`；
- 三个 checkpoint 均为 94 维 source schema、93 维模型输入；
- 三个 checkpoint 均记录 `excluded_state_feature_names=['hour_of_week']`；
- 三个 checkpoint 均记录 `initial_checkpoint=None`；
- 7 个 `.err` 日志全部为空；
- 本地相关测试：29 passed。
