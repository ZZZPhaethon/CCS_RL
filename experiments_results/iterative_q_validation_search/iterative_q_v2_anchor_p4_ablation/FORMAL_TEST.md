# Iterative Q v2：一次性 formal test

## 锁定配置

- 模型：Iterative Q v2 hard-anchor
- Checkpoint：`anchor_100/iterative_action_q.pt`
- SHA-256：
  `ea82fa0aec2d312cefac9e4c52a5bdc03e1a21c11a5b21aae70e37013ad4053b`
- Model seed：0
- Previous-policy anchor：hard，coefficient 1.0，€40,000 release margin
- Inference gate：4/5 heads，€40,000 margin，最多12次干预
- Scenario protocol：`unified_window_v1`，normal/Medium
- Formal test seeds：9000001–9000030，共30个 paired seeds
- Continuous-teacher 模型未在 formal test 上运行
- Test 结果不用于重新选择 checkpoint、gate 或超参数

完整访问快照见 `formal_test_lock.txt`。

## Formal test 结果

| 指标 | Greedy | Iterative Q v2 | v2 − Greedy |
|---|---:|---:|---:|
| 平均总成本 € | 2,023,880 | **1,842,876** | **−181,004（−8.94%）** |
| 平均 Vent t | 4,226.5 | **1,184.6** | −3,041.9 |
| 平均 Stored t | 104,459.7 | **108,571.1** | +4,111.4 |
| 平均单位成本 €/t | 19.481 | **17.015** | −2.466 |

- 胜/负：23/7，无平局；
- median paired cost difference：−€188,348；
- paired mean-difference 95% CI：
  [−€243,115, −€122,204]；
- 30 episodes 合计节省 €5,430,129；
- 18/30 episodes 为零 vent；
- 26/30 episodes 的 vent 低于 Greedy；
- 23/30 episodes 的 stored mass 高于 Greedy；
- 平均干预次数：9.13/episode。

## 失败 seeds

| Seed | v2 − Greedy € |
|---:|---:|
| 9000006 | +6,972 |
| 9000007 | +68,169 |
| 9000009 | +23,663 |
| 9000012 | +91,408 |
| 9000013 | +12,889 |
| 9000021 | +18,266 |
| 9000022 | +16,253 |

最坏 formal-test 回退为 seed 9000012 的 +€91,408。结果支持 v2
在总体均值上优于 Greedy，但不能声称它在每个场景上都占优。

## 与 validation 的关系

| Split | Seeds | 平均 v2 − Greedy € | 胜率 |
|---|---:|---:|---:|
| Controller validation | 20 | −283,134 | 18/20 |
| Formal test | 30 | −181,004 | 23/30 |

Formal-test 优势比 validation 收窄约 €102,130，但 formal-test paired
95% CI 仍完全低于零。由于 formal test 已访问，不再根据这些结果修改
模型、teacher 权重、gate 或 margin。

## 作业与产物

- 环境检查：33131
- Formal-test evaluation：33132
- 两个作业退出码均为0，stderr 均为空。
- 逐 seed 数据：
  `eval/iterative_q_v2_formal_test/evaluation.csv`
- 原始汇总：
  `eval/iterative_q_v2_formal_test/summary.json`
