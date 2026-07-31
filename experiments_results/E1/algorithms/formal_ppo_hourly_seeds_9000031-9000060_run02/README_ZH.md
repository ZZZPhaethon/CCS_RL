# Hourly Centralized Maskable PPO 正式测试

本目录保存 validation-best checkpoint 在正式测试 seeds
`9000031–9000060` 上的确定性评估结果。三个独立训练 model seed
`0/1/2` 各评估 30 个场景，共 90 个 episode。

## 汇总结果

| Model seed | 平均总成本 EUR | 平均 Vent t | 相对 Greedy | 胜/平/负 |
|---:|---:|---:|---:|---:|
| 0 | 5,948,887 | 53,409.4 | +3,694,675 | 0/0/30 |
| 1 | 5,466,416 | 47,229.4 | +3,212,205 | 0/0/30 |
| 2 | 5,563,593 | 48,631.5 | +3,309,382 | 0/0/30 |
| 三个模型汇总 | 5,659,632 | 49,756.8 | +3,405,421 | 0/0/90 |

Greedy 的 30-seed 平均总成本为 EUR 2,254,212。Hourly PPO 相对
Greedy 的 hierarchical bootstrap 95% CI 为
`[+3,159,058, +3,684,811] EUR`，全部 90 个配对 episode 均劣于
Greedy。

## 完整性

- 每个 model seed 精确覆盖 `9000031–9000060`，无缺失或重复。
- 每个 episode 均进行 720 次直接逐小时决策并推进 720 h。
- 90 个 episode 均满足
  `total_cost = episode_total_cost + terminal_cleanup_operating_cost`。
- 三个 checkpoint 的 SHA-256 均与 `models/` 中冻结副本一致。
- Smoke job `34211` 和正式 array `34212_0/1/2` 均
  `COMPLETED 0:0`；正式任务的 stderr 均为空。

## 详细成本补评估

2026-07-29 使用同一组冻结 checkpoint 和测试 seeds 补充运行了 episode 级
成本分项评估：smoke job `34440`，正式 array `34441_0/1/2`，全部
`COMPLETED 0:0` 且 stderr 为空。

新的 `results.csv/json` 增加 fuel、conditioning、reconditioning、loading、
unloading、vent penalty 和 storage-shortfall penalty。逐 seed 对比旧结果时，
90 条记录的全部公共字段完全一致；新增成本分项的求和恒等式也全部通过。
Cleanup 仍只记录 `terminal_cleanup_operating_cost_eur` 总额，不做分项拆分。
旧结果和本次运行的完整副本、代码及日志保存在
`provenance/cost_fields_rerun_20260729/`。

## 命名和目录

每个 `model_seed_N/` 使用相同文件名：

- `checkpoint_best_validation.zip`
- `source_checkpoint.sha256`
- `config.json`
- `training_complete.json`
- `best_validation.json`
- `results.csv`
- `results.json`
- `audit.json`

跨 model-seed 汇总位于 `analysis/`，SLURM 日志和实际运行脚本位于
`provenance/`。

第一次 smoke job `34209` 因环境中没有非必要的 `pytest` 而在模型评估前
失败；依赖 array `34210` 未运行并已取消，没有产生正式结果。移除该非必要
依赖后，以 `run02` 完成了上述正式评估。
