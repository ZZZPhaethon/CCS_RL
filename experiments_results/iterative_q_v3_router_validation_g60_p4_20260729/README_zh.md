# Iterative Q v3 Router 可行性验证（G60-P4）

## 结论

当前不应把 checkpoint router 作为 v3 的正式策略。锁定的
controller-validation seeds `8100001–8100020` 上，所有预注册、无需
test label 的 router 都未超过单独使用 P4。

P4 的平均总成本为 €1,821,812.14。最接近的 `p3_p4_pooled` 为
€1,841,011.24，相对 P4 平均增加 €19,199.10；配对 bootstrap 95% CI
为 `[−€24,372.52, €63,792.73]`。其 worst-4 CVaR 相对 Greedy 从 P4
的 €9,223.98 恶化至 €54,983.48，最坏单 seed 相对 P4 回退
€291,265.85。

前十个 development seeds 和后十个 confirmation seeds 的方向相反：
`p3_p4_pooled` 在 development 上平均回退 €52,906.35，在
confirmation 上平均改善 €14,508.15。这不足以证明一个稳定的 router。

## Oracle 上限

如果事后知道完整 episode 成本，并在 P1–P4 中逐 seed 选择最佳
checkpoint，平均成本可降至 €1,803,936.67，相对 P4 改善 €17,875.46
（0.981%）。20 个 seeds 中，P1/P2/P3/P4 分别成为最优
4/1/2/13 次。这个 oracle 使用了 episode 结果，不能部署，只用于衡量
router 的理论空间。

## 验证的 router

- `p4_reference`：只使用 P4，复现原 G60-P4 validation 结果。
- `p3_p4_confidence`、`all_confidence`：选择 lower-confidence
  advantage 最大的 checkpoint。
- `p3_p4_lcb`、`all_lcb`：在 confidence router 中加入一个标准差的
  uncertainty penalty。
- `p3_p4_pooled`、`all_pooled`：合并不同 checkpoint 的 ensemble
  heads，再应用统一安全门。

最大置信度方案会系统性偏向较早 checkpoint：四代 confidence router
的被选次数为 P1=4939、P2=559、P3=112、P4=300。这表明不同迭代代次
的 Q margin 没有跨模型校准，不能直接用于路由。

## 数据协议

- 模型：v1 G60-P4 的 P1–P4 checkpoints，model seed 0。
- 场景协议：`unified_window_v1`。
- validation seeds：`8100001–8100020`。
- development/confirmation：前 10 / 后 10 seeds。
- 每 episode 最多 12 次 override，每个固定 48 h window 最多一次。
- formal test 未访问。

完整聚合结果见 `analysis.json` 和 `analysis.csv`；各 router 的逐 seed
结果位于 `routers/`，原 P1–P4 逐 seed 结果位于
`checkpoint_evaluations/`，checkpoint 校验值见
`checkpoint_sha256.csv`。
