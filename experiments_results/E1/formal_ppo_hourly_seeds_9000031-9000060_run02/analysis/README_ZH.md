# E1 PPO 正式测试汇总

固定使用 validation 选出的 best checkpoint；测试 seeds 为
9000031–9000060。总成本包含 terminal cleanup。

| 方法 | 总成本 EUR | 单位成本 EUR/t | Vent t | Stored t | 相对 Greedy EUR | 胜率 |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 2,254,212 | 22.40 | 7,296.8 | 102,984.2 | 0 | — |
| Hourly Centralized Maskable PPO | 5,659,632 | 136.05 | 49,756.8 | 44,082.9 | +3,405,421 | 0.0% |

Paired 和 hierarchical bootstrap 95% CI 见 `per_model_seed.csv` 和
`aggregate.csv`。
