# 三模型 seed 汇总

`per_model_seed.csv` 是每个训练模型 seed 的 30-episode 汇总；
`aggregate.json` 是三模型 seed、共 90 条结果的总体汇总；
`protocol_audit.json` 记录 seed 覆盖和详细成本字段可用性。

总体均值把每个“模型 seed × 测试 seed”视为一条观测。模型间标准差先计算每个
模型 seed 的 30-episode 平均总成本，再对三个均值计算样本标准差。

模型 seed 0 已用同一冻结 checkpoint 补充评估；当前三个模型 seed 的 CSV
均包含 episode fuel、conditioning、loading 等成本分项。
