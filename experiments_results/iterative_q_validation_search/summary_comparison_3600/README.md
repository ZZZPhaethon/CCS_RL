# Iterative Q exact closed-loop validation search

## 最终建议

保留已选定的 single-168 P3：

- G0/G1/G2 = 2,400/480/720 nominal roots；
- G0 是固定、策略无关的 Greedy root bank；
- G1、G2 分别由对应 model seed 的 P1、P2 策略重新生成；
- 5-head ensemble，至少 4 heads 同意；
- Q margin = 0.40（€40,000），12 个干预窗口，最多 12 次干预；
- `shared_future_summary` 只读单一 168 h summary，不含 `valid_fraction`；
- 正式参考 checkpoint 保持预先指定的 model seed 0。

seed-0 的实际训练预算为：

| 阶段 | nominal roots | 实际 simulator calls |
|---|---:|---:|
| G0 | 2,400 | 6,328,003 |
| G1 | 480 | 1,270,670 |
| G2 | 720 | 1,906,646 |
| **P3 / B_selected** | **3,600** | **9,505,319** |

`B_selected` 只统计入选 checkpoint 原始 training shards 中记录的底层 1 h simulator step calls。controller validation、formal test、SGD、纯特征计算和未入选开发分支不计入。

## Exact P3 与 P4

P4 不是 fixed-data 训练。对于 model seeds 1/2，分别使用其自身 exact P3 策略生成新的 G3；seed 0 使用原始完整 closed-loop P1–P4 链。P4 的 root 分配为 G0/G1/G2/G3 = 2,400/480/720/1,200。

| 阶段 | model seed | 总成本 € | 相对 Greedy 节省 | 胜/20 | 单位成本 €/t | vent t | stored t | 实际 calls | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 | 0 | 1,837,041 | 11.43% | 17 | 16.7740 | 615.78 | 109,873.76 | 9,505,319 | 21 |
| P3 | 1 | 1,869,260 | 9.87% | 16 | 17.1140 | 1,015.29 | 109,414.99 | 9,563,967 | 26 |
| P3 | 2 | 1,872,186 | 9.73% | 14 | 17.3484 | 1,113.70 | 108,077.54 | 9,458,126 | 24 |
| P4 | 0 | 1,865,713 | 10.04% | 17 | 17.1079 | 1,151.38 | 109,137.82 | 12,754,167 | 25 |
| P4 | 1 | 1,876,455 | 9.53% | 15 | 17.3900 | 1,290.47 | 108,065.14 | 12,803,898 | 40 |
| P4 | 2 | 1,841,038 | 11.23% | 16 | 16.9454 | 850.98 | 108,978.85 | 12,692,258 | 18 |

三 model seed 聚合：

| 指标 | P3 | P4 | P4 − P3 |
|---|---:|---:|---:|
| 总成本均值 € | 1,859,496 | 1,861,069 | +1,573 |
| model-seed 间总成本 SD € | 19,501 | 18,160 | −1,341 |
| 单位成本 €/t | 17.0788 | 17.1478 | +0.0689 |
| 平均胜 Greedy /20 | 15.67 | 16.00 | +0.33 |
| vent t | 914.92 | 1,097.61 | +182.69 |
| stored t | 109,122.10 | 108,727.27 | −394.83 |
| 实际 simulator calls 均值 | 9,509,137 | 12,750,108 | +3,240,970（+34.08%） |

同一 model seed、同一 20 个 validation scenarios 的 P4−P3 成本差为：

- seed 0：+€28,672；
- seed 1：+€7,196；
- seed 2：−€31,148；
- 三-seed 平均：+€1,573；
- hierarchical bootstrap 95% CI：[-€62,870, +€67,325]；
- 60 个 model-seed × scenario 配对中，P4 更好 32 次、较差 28 次。

结论：P4 的部署成本 SD 比 P3 低 6.9%，但差异很小；其成本变化跨 model seed 方向相反，训练早停轮数为 18/25/40，反而比 P3 的 21/24/26 分散。P4 没有验证出平均成本优势，却多用约 34% simulator calls。因此不能称为整体更稳定，也不值得替换 P3。

## single 168 与 24/72

所有表示比较也只采用 exact closed-loop 结果：

| 表示 | 三-seed 总成本均值 € | seed 间 SD € | 单位成本 €/t | 平均胜 Greedy /20 | vent t | stored t |
|---|---:|---:|---:|---:|---:|---:|
| single 168 | 1,859,496 | 19,501 | 17.0788 | 15.67 | 914.92 | 109,122.10 |
| 24/72 | 1,857,181 | 25,203 | 16.9750 | 17.00 | 913.91 | 109,685.02 |

24/72−single168 的均值为 −€2,314，hierarchical 95% CI 为 [-€52,872, +€44,652]，model seeds 0/1/2 的差值分别为 +€10,683、+€16,486、−€34,111。方向不一致，故保留更简单且 seed 间 SD 更低的 single 168。

## 审计

- 所有阶段选择和 P3/P4 比较只使用 controller-validation seeds 8100001–8100020。
- 未访问 formal test seeds。
- 每个 episode 校验 `total cost = 720 h episode cost + compact terminal cleanup`。
- 888 h 场景、前 720 h 执行计分、末尾 168 h 只读 summary 和 cleanup 口径保持不变。
- 本地权威目录和远端实验输出中均已删除本轮 fixed-data 训练结果；manifest 和汇总只包含 exact closed-loop。
- seed 1 P4 在 Borg 的 3 分钟 backfill 实例异常慢并超时，没有采用部分 checkpoint；随后从 P3 在本地 CPU 用相同数据、超参数和 model seed 完整训练 40 epochs 并完成 validation。seed 2 Borg 训练完整完成 18 epochs。
- 没有训练 PPO，没有创建 git commit。

## 文件

- `analysis/`：single-168 与 24/72 的 exact closed-loop 汇总；
- `exact_p4/analysis/`：exact P3/P4 的逐 seed、聚合和配对结果；
- `exact_p4/audits/`：P4 三 model seeds 的 simulator-call 审计；
- `recommendation.json`：最终推荐和 `B_selected`；
- `selected/iterative_action_q.pt`：保留的 P3 seed-0 checkpoint；
- `raw/`：validation JSON/CSV 和训练 summary 原始产物。
