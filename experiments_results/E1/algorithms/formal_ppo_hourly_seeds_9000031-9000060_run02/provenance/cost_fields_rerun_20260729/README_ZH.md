# 详细成本补评估 provenance

- Smoke job：`34440`
- 正式 array：`34441_0/1/2`
- 模型 seeds：`0/1/2`
- 正式测试 seeds：`9000031–9000060`
- 结果：全部 `COMPLETED 0:0`，stderr 为空
- 新旧公共字段：90 条记录、0 个差异
- 新增成本分项恒等式：90 条记录、0 个失败
- Cleanup：仅保留总额，不拆分

`previous_results/` 保存更新前的正式结果；带
`formal_ppo_hourly_cost_fields_...` 名称的目录保存 Borg 原始新结果；
`logs/` 和代码快照用于复现。
