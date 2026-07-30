# P4-only Plateau-20/50 teacher 快速筛选

## 结论

Plateau-20/50 基本恢复了 hard teacher 的表现，并明显优于从 €0 立即衰减的 Linear-0/50；但它没有超过 hard teacher。
Plateau 的成本标准差介于 hard 与 linear 之间，且最差 seed 相对 Greedy 的回退小于另外两种权重。

## 20-seed validation

| 模型 | 平均总成本 € | 相对 Greedy | 胜/平/负 | Vent t | Stored t | Overrides |
|---|---:|---:|---:|---:|---:|---:|
| Hard-40 | 1,790,875 | −13.65% | 18/0/2 | 330.3 | 110,919.5 | 8.70 |
| Linear-0/50 | 1,814,583 | −12.51% | 18/0/2 | 619.0 | 110,832.1 | 10.35 |
| Plateau-20/50 | 1,793,611 | −13.52% | 18/0/2 | 267.2 | 109,872.9 | 10.50 |

## Paired comparisons

负差值表示 Plateau 更便宜。

| 对照 | Plateau − 对照 € | Bootstrap 95% CI € | Plateau 胜/平/负 |
|---|---:|---:|---:|
| Hard-40 | 2,736 | [-19,695, 28,775] | 11/0/9 |
| Linear-0/50 | -20,973 | [-54,373, 1,109] | 12/1/7 |

## 协议核验

- Jobs 34133–34135 全部 COMPLETED，ExitCode 0:0；stderr 为空。
- 同一个 P3 checkpoint、同一份 G0–G3 数据、model seed 0。
- 仅使用 validation seeds 8100001–8100020，未访问 formal-test。
- Paired bootstrap：200,000 次，seed 20260729。
