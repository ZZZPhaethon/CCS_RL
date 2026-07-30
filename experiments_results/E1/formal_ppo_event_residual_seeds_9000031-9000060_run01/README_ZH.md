# Event-Residual PPO 正式测试

目录短名为 `ppo_event_residual`，表示基于事件残差动作的 PPO。
三个独立训练 model seed `0/1/2` 各评估正式 seeds
`9000031–9000060`，共 90 个 episode。

## 汇总结果

- 平均总成本：EUR 2,239,850
- 相对 Greedy：EUR -14,362
- 胜/平/负：47/0/43
- Hierarchical bootstrap 95% CI：EUR [-118,314, 94,923]

每个 model seed 均精确覆盖 30 个正式 seed；独立汇总位于 `analysis/`。
本算法的正式结果、smoke 结果和相关 SLURM provenance 均与
High-level Centralized Maskable PPO 分开保存。

## 详细成本补评估

2026-07-29 使用同一组冻结 checkpoint 和测试 seeds 补充运行了 episode 级
成本分项评估：共享 smoke job `34442`，正式 array tasks `34443_3/4/5`，
全部 `COMPLETED 0:0` 且 stderr 为空。

新的 `results.csv/json` 增加 fuel、conditioning、reconditioning、loading、
unloading、vent penalty 和 storage-shortfall penalty。逐 seed 对比旧结果时，
90 条记录的全部公共字段完全一致；新增成本分项的求和恒等式全部通过。
Cleanup 仍只保留总额。本次运行及旧结果归档见
`provenance/cost_fields_rerun_20260729/`。
