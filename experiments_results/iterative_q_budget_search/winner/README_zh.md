# Iterative Q 预算搜索优胜配置：g50_p2

本目录保存 2026-07-29 完成的验证集预算搜索中，唯一同时满足全部硬约束的配置。

## 选择规则

- 训练数据生成预算目标：9,525,119 个 physical simulator steps。
- 预算容差：±8%。
- 只使用锁定验证 seeds `8100001–8100020`；未访问 formal test。
- 最终模型相对 Greedy 不允许出现单 seed 大于 €100,000 的回退。
- 最终模型相对任一前序 checkpoint 不允许出现单 seed 大于 €100,000 的回退。
- 在满足以上约束的配置中，按验证集平均总成本、worst-4 CVaR、最坏单 seed 回退依次排序。

## 配置

- 配置名：`g50_p2`
- 最终 checkpoint：`p2/iterative_action_q.pt`
- 模型随机种子：0
- 阶段：G0 → P1 → G1 → P2
- G0：1,799 train roots，4,730,960 simulator steps
- G1：1,800 train roots，4,775,936 simulator steps
- 合计：3,599 train roots，9,506,896 simulator steps
- 相对预算误差：−0.1913%
- G0 实际预算占比：49.7635%
- residual margin：所有轮次统一为 0.40
- economic margin：所有轮次统一为 €40,000
- observation input：`shared_future_summary`
- teacher/behavior anchor：未启用

## 锁定验证集结果

| 指标 | P1 | P2（入选） |
|---|---:|---:|
| 平均总成本 | €1,882,355.68 | €1,821,170.76 |
| 相对 Greedy 平均改善 | 9.2407% | 12.1908% |
| 胜率 | 17/20 | 16/20 |
| 相对 Greedy 最坏单 seed 回退 | €237,638.58 | €93,406.50 |
| 相对 Greedy worst-4 CVaR | €103,672.93 | €41,956.92 |

P2 相对 P1 的平均成本降低 €61,184.92；最坏回退为 seed `8100009` 的 €93,051.09，未超过 €100,000 阈值。

P2 的其他平均指标：

- 单位成本：€16.5845/t（Greedy：€19.9448/t）
- vent：368.70 t（Greedy：4,852.71 t）
- stored：110,129.53 t（Greedy：104,675.64 t）

## Checkpoint 校验

- P1 SHA-256：`8da2c4e5e00aa40a454d2d1d78c7b4ad0f2dcdaaf9c13ca15e541d6c45bb0933`
- P2 SHA-256：`4fe8a108f9fbf33bdbea75cf1068b78679bec04ccde023966af499384a804889`

完整 20 配置排行榜位于上级目录的 `validation_analysis.csv` 和 `validation_analysis.json`。
