# E1 PPO 正式测试报告

测试范围为 seeds 9000031–9000060。两个算法均固定使用 validation 阶段选出的 best checkpoint，model seeds 为 0/1/2；未根据测试结果重新选模型。总成本包含 terminal cleanup。

## 总体结果

每个 PPO 方法包含 3 个 model seeds × 30 个 test seeds，共 90 个配对评估结果。Greedy 为 30 个唯一 test seeds 的基线。

| 方法 | 总成本 EUR ↓ | 单位成本 EUR/t ↓ | Vent t ↓ | Stored t ↑ | 相对 Greedy EUR | 胜/平/负 | 胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 2,254,212 | 22.40 | 7,296.8 | 102,984.2 | 0 | — | — |
| Centralized Maskable PPO | **2,187,244** | **20.61** | **3,803.5** | **107,406.2** | **-66,967** | 48/0/42 | 53.3% |
| Event-Residual PPO | 2,239,850 | 22.01 | 5,963.2 | 102,490.0 | -14,362 | 47/0/43 | 52.2% |

Centralized Maskable PPO 在正式测试均值上优于 Event-Residual PPO，并同时降低总成本、单位成本和 vent，提高 stored。

## Hierarchical bootstrap 95% CI

| 方法 | 平均成本差 vs Greedy EUR | Hierarchical 95% CI EUR | 跨 model-seed 成本 SD EUR |
|---|---:|---:|---:|
| Centralized Maskable PPO | -66,967 | [-168,907, +36,305] | 26,224 |
| Event-Residual PPO | -14,362 | [-117,626, +95,347] | 80,553 |

两种方法的 hierarchical CI 都跨 0，因此正式测试不支持“总体成本显著低于 Greedy”的强结论。Centralized 的均值改善更大，且跨 model seed 波动明显更小。

## 各 model seed 配对结果

| 方法 | Model seed | 总成本 EUR | 相对 Greedy EUR | Paired 95% CI EUR | 胜/平/负 | 胜率 |
|---|---:|---:|---:|---:|---:|---:|
| Centralized Maskable PPO | 0 | 2,184,256 | -69,956 | [-238,861, +95,096] | 17/0/13 | 56.7% |
| Centralized Maskable PPO | 1 | 2,162,642 | -91,569 | [-265,311, +74,302] | 15/0/15 | 50.0% |
| Centralized Maskable PPO | 2 | 2,214,834 | -39,377 | [-231,909, +146,638] | 16/0/14 | 53.3% |
| Event-Residual PPO | 0 | 2,212,818 | -41,393 | [-189,874, +95,751] | 18/0/12 | 60.0% |
| Event-Residual PPO | 1 | 2,330,442 | +76,230 | [-57,058, +203,876] | 12/0/18 | 40.0% |
| Event-Residual PPO | 2 | 2,176,290 | -77,922 | [-217,010, +51,416] | 17/0/13 | 56.7% |

Event-Residual PPO 的 model seed 1 再次表现较弱，但其 paired CI 仍跨 0。六个 model seed 的 paired CI 均跨 0。

## 协议与执行审计

- Smoke 作业 34178 使用 validation seed 8100001，两算法均通过模型加载和 cleanup 恒等式检查。
- 正式 array 34179 的六个任务全部 `COMPLETED`、exit code `0:0`，错误日志为空。
- 六个任务各包含恰好 30 个指定 test seeds，共 180 个 PPO episode。
- 180 个 PPO 记录和 30 个 Greedy 记录全部满足：
  `total cost = episode total cost + terminal cleanup operating cost`。
- 所有配置均为 720 h 执行、168 h 只读上下文和单一 168 h future summary。
- checkpoint SHA-256、SLURM accounting、执行脚本和原始日志见 `../provenance/`。
