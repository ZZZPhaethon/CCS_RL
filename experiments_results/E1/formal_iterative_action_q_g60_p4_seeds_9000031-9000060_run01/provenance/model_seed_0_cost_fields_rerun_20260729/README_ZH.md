# Model seed 0 详细成本补评估 provenance

- 正式评估 job：`34444`
- 模型 seed：`0`
- 正式测试 seeds：`9000031–9000060`
- 结果：`COMPLETED 0:0`，stderr 为空
- 新旧公共数值字段：30 条记录、0 个差异
- 元数据差异：`gate` 标签从旧运行名变为本次输出名，共 30 条
- 新增成本分项恒等式：30 条记录、0 个失败
- Cleanup：仅保留总额，不拆分

`previous_results/` 保存更新前的正式结果；本目录根部的
`evaluation.csv` 和 `summary.json` 是 Borg 原始新结果，另保存代码及日志。
