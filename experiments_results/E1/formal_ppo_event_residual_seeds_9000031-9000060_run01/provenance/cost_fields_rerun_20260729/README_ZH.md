# 详细成本补评估 provenance

- 共享 smoke job：`34442`
- Event-Residual 正式 array tasks：`34443_3/4/5`
- 模型 seeds：`0/1/2`
- 正式测试 seeds：`9000031–9000060`
- 结果：全部 `COMPLETED 0:0`，stderr 为空
- 新旧公共字段：90 条记录、0 个差异
- 新增成本分项恒等式：90 条记录、0 个失败
- Cleanup：仅保留总额，不拆分

`previous_results/` 保存更新前的正式结果；`ppo_event_residual/` 保存 Borg
原始新结果；`logs/` 和代码快照用于复现。
