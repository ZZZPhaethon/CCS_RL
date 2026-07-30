# High-level Centralized Maskable PPO 正式测试

目录短名为 `ppo_high_level`，表示每 24 h 进行一次高层决策的 PPO。
三个独立训练 model seed `0/1/2` 各评估正式 seeds
`9000031–9000060`，共 90 个 episode。

## 汇总结果

- 平均总成本：EUR 2,187,244
- 相对 Greedy：EUR -66,967
- 胜/平/负：48/0/42
- Hierarchical bootstrap 95% CI：EUR [-168,907, 36,305]

每个 model seed 均精确覆盖 30 个正式 seed；独立汇总位于 `analysis/`。
本算法的正式结果、smoke 结果和相关 SLURM provenance 均与
Event-Residual PPO 分开保存。

原双算法目录的联合分析与完整 provenance 仅为历史追溯保存在
`provenance/original_combined_analysis/` 和
`provenance/original_combined_provenance/`，不属于当前独立汇总。

## 详细成本补评估

2026-07-29 使用同一组冻结 checkpoint 和测试 seeds 补充运行了 episode 级
成本分项评估：共享 smoke job `34442`，正式 array tasks `34443_0/1/2`，
全部 `COMPLETED 0:0` 且 stderr 为空。

新的 `results.csv/json` 增加 fuel、conditioning、reconditioning、loading、
unloading、vent penalty 和 storage-shortfall penalty。逐 seed 对比旧结果时，
90 条记录的全部公共字段完全一致；新增成本分项的求和恒等式全部通过。
Cleanup 仍只保留总额。本次运行及旧结果归档见
`provenance/cost_fields_rerun_20260729/`。
