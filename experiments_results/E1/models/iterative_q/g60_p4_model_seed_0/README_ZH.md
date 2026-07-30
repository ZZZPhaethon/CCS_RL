# E1 当前 Iterative Action-Q：G60-P4

本目录保存 E1 当前采用的无 teacher Iterative Action-Q 模型。

- 模型：G60-P4，model seed 0，最终阶段 P4
- checkpoint：`iterative_action_q.pt`
- SHA-256：`e529eb06038a4842f58eb97a912ad0f72de9000f03d39d9bc8839e8690febedc`
- teacher：未使用
- 训练 margin：所有轮次统一 residual margin 0.40，即 €40,000
- 推理 gate：5 个 heads 中至少 4 个同意，margin 0.40，最多 12 次干预
- 训练预算：3,599 个有效训练 roots，9,526,297 次物理 simulator hour-step calls

`G60` 表示 G0 预算配置，不表示 margin 0.60。具体训练配置、预算和来源见
`source_training_summary.json`、`budget.json` 和 `source_protocol_lock.txt`。

对应的 seeds 9000031–9000060 测试结果保存在
`experiments_results/E1/formal_iterative_action_q_g60_p4_seeds_9000031-9000060_run01`。

## 协议说明

该模型是在查看上述 30-seed 测试结果之后被指定为 E1 当前模型。因此，
9000031–9000060 对本次采用决定不再是未访问的模型选择 holdout；新的
`current.json` 和 `model_manifest.json` 已对此如实记录。
