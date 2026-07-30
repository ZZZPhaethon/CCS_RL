# Iterative Q seed-block 敏感性评估

本实验固定使用 `reallocated B / P3 / margin50-cap12 / model seed 0`
checkpoint（SHA-256
`c97f018e63b336fe15c34b4546ea7908a2c271dccb7cc23b32c3aad8f44d1ed4`），
比较 Iterative Q 与 Greedy 在相同场景 seed 上的 720 h 配对总成本。总成本
包含 common compact terminal cleanup。

四个额外且互不重叠的 30-seed 区间在查看新结果前一起固定并提交。随后使用
完全相同的 checkpoint 和 gate 补跑 protocol 原 test seeds
`9000001–9000030`。manifest v4 自 2026-07-29 起将原区间及其结果标记为
弃用于后续主比较但保留作 provenance，并将未访问测试集 `9000031–9000060` 固定为后续
frozen-controller comparison set；该集合不得再用于任何选择。

合并结果的 bootstrap CI 使用固定 seed `20260728` 和 200,000 次配对重采样。
Iterative Q 的平均 vent 为 1,291.9 t，Greedy 为 6,406.9 t；平均 stored
分别为 109,272.5 t 和 102,209.3 t。五个区间的成本优势方向一致，但成本降幅
从 8.56% 到 16.20%、胜率从 24/30 到 29/30 波动，因此不能只选表现最好的
区间作为主结果。

SLURM array job `33149` 的四个额外区间任务及原 test block job `33176`
均为 `COMPLETED 0:0`，所有 stderr 为空。每个子目录包含逐-seed
`evaluation.csv` 和原始 `summary.json`；`logs/` 保存提交日志，
`analysis_summary.json` 保存机器可读汇总。
