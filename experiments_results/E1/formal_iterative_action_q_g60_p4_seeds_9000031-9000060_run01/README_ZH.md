# Iterative Action-Q G60-P4：正式测试

本目录保存 3 个独立模型 seed（`0/1/2`）在同一组 30 个正式测试 seeds
`9000031–9000060` 上的结果。每个 `model_seed_N/` 都包含逐测试 seed 的
`evaluation.csv` 和评估汇总 `summary.json`。

| 模型 seed | 平均总成本 | 相对 Greedy 平均差值 | 胜/平/负 | 详细 episode 成本分项 |
|---:|---:|---:|---:|---|
| 0 | €1,870,036.72 | −€384,174.84 | 27/0/3 | 是 |
| 1 | €1,868,639.41 | −€385,572.15 | 28/0/2 | 是 |
| 2 | €1,906,400.13 | −€347,811.44 | 28/0/2 | 是 |
| 三模型合计（90 条） | €1,881,692.09 | −€372,519.48 | 83/0/7 | — |

模型 seed 0/1/2 均已于 2026-07-29 使用冻结 checkpoint 重新运行或补充正式
评估，新增以下 episode 级成本字段：

- `episode_vessel_fuel_eur`
- `episode_conditioning_eur`
- `episode_reconditioning_eur`
- `episode_loading_eur`
- `episode_unloading_eur`
- `episode_vent_penalty_eur`
- `episode_storage_shortfall_penalty_eur`

每个字段同时保存 `greedy_` 和 `delta_` 版本。90 条结果全部通过成本恒等式
核对。`terminal_cleanup_operating_cost_eur` 仍是 cleanup 总额，没有继续拆分
成 fuel、conditioning、loading 等分项。

三模型汇总见 `analysis/`；checkpoint 见
`../models/iterative_q/g60_p4_model_seed_0/`、`g60_p4_model_seed_1/` 和
`g60_p4_model_seed_2/`；SLURM job、日志、评估代码快照和协议见
`provenance/`。

模型 seed 0 的原结果曾参与 G60-P4 的模型采用判断，因此该测试集合不能再称为
seed 0 的未访问选择 holdout。seed 1/2 使用相同正式测试集合，用于独立训练重复
的稳健性比较，也不应再用于后续模型选择。
