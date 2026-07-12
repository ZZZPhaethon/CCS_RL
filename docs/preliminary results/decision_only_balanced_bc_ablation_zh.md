# Decision-only loss 与平衡采样：BC 消融实验

## 1. 结论摘要

本实验比较三种保持小时级动作空间不变的 MPC behavior cloning（BC）方法：

- A（current）：按原始行均匀采样，包含 action mask 已强制的船舶动作维度，dispatch 动作维度权重为 10；
- B（decision-only）：强制船舶动作维度的 loss 权重改为 0，其余与 A 相同；
- C（decision-balanced）：只训练真正可决策的船舶动作，将主动 WAIT 与 dispatch 决策对按 50:50 平衡采样；船舶动作权重统一为 1，well 目标仍按原始行均匀训练。

预注册主指标是 deterministic 720 h 闭环 rollout。结果不支持“B 后再加入 C 能稳定改善确定性 BC”这一主张：

- B 相对 A 的 deterministic venting 平均下降，但两个编码器的五 model-seed 95% 区间都跨 0；
- C 大幅提高了训练示范上的 dispatch recall 和 destination accuracy，但 deterministic venting 反而上升；
- C 的 stochastic rollout 明显改善，表明平衡采样改变了策略概率分布，但这种改善没有转化为确定性 argmax 策略的收益。

因此，当前证据说明问题不只是“dispatch 样本太少所以标签没有学会”。C 的 TCN 已在训练 cache 上达到 100% exact match，却仍在 held-out 闭环确定性测试中恶化，说明 covariate shift、过度重复少量 dispatch 样本、概率校准和 dispatch 时机错误仍是主要瓶颈。

## 2. 实验协议

| 项目 | 设置 |
|---|---|
| Git commit | `479659e` |
| MPC demonstrations | seeds 0–99，100 × 720 h = 72,000 状态—动作对 |
| Demonstration cache SHA256 | `8eb6708d954bb4cf0c0de895d800a88343753bb93236bd3995e63c4694cf2584` |
| BC | 50 epochs，batch size 256 |
| PPO | 不运行；所有 manifest 均记录 `explicitly_skipped=true` |
| 正式 variants | `state_mode`、`tcn_mode` |
| Model seeds | 0–4 |
| Evaluation seeds | 101–120 |
| Episode | 720 h |
| 主指标 | deterministic rollout；stochastic 仅作诊断 |
| B 正式 Borg job | `24903`，10/10 tasks `COMPLETED 0:0` |
| C smoke jobs | `24915` 因漏传提交号在训练前退出；修正配置后的 `24917` 为 2/2 `COMPLETED 0:0` |
| C 正式 Borg job | `24919`，10/10 tasks `COMPLETED 0:0` |

C 的 sampler audit 在全部 10 个正式 run 中一致：

| 目标 | 数量/epoch |
|---|---:|
| 原始主动 WAIT decision pairs | 65,785 |
| 原始 dispatch decision pairs | 3,829 |
| 采样后 WAIT pairs | 65,785 |
| 采样后 dispatch pairs | 65,785 |
| 均匀 well targets | 72,000 |
| 总训练 targets | 203,570 |

B 和 C 各产生 10 个 BC checkpoint、10 个 results CSV、10 个 demonstration diagnostics、10 个 rollout diagnostics 和 10 个 manifest。C 的 400 条结果记录全部为 finite，10 份正式 stderr 均为空。

## 3. 720 h 主结果

表中的“±”是先在每个 model seed 内对 20 个 evaluation seeds 求均值，再对五个 model-seed 均值计算的 95% t 区间半宽。

### 3.1 Deterministic 主指标

| 方法 | Variant | Vented t | Stored t | Total cost EUR |
|---|---|---:|---:|---:|
| Rolling MPC | reference | 514 | 109,583 | 1,637,715 |
| Greedy | reference | 8,014 | 101,341 | 2,121,699 |
| A current | state_mode | 6,134 ± 1,530 | 103,282 | 2,034,470 |
| B decision-only | state_mode | **5,892 ± 1,045** | 103,483 | **2,007,975** |
| C balanced | state_mode | 7,321 ± 882 | 100,723 | 2,153,205 |
| A current | TCN_mode | 6,632 ± 1,776 | 102,566 | 2,075,003 |
| B decision-only | TCN_mode | **6,176 ± 1,842** | 103,377 | **2,046,334** |
| C balanced | TCN_mode | 8,912 ± 1,210 | 100,128 | 2,209,156 |

B 是两个编码器中 deterministic 均值最好的 BC，但证据只支持“方向性改善”，不支持“稳定改善”：

| 配对差值 | Mean delta vented t | 95% CI | 判断 |
|---|---:|---:|---|
| B − A，state_mode | −241 | [−1,209, +726] | 跨 0 |
| B − A，TCN_mode | −456 | [−2,453, +1,541] | 跨 0 |
| C − A，state_mode | +1,187 | [−748, +3,122] | 跨 0 |
| C − A，TCN_mode | +2,280 | [+988, +3,572] | C 显著更差 |
| C − B，state_mode | +1,429 | [+276, +2,581] | C 显著更差 |
| C − B，TCN_mode | +2,736 | [+533, +4,940] | C 显著更差 |

B-state_mode 比 Greedy 少 vent 26.5%，B-TCN_mode 少 22.9%；但两者仍分别约为 rolling MPC venting 的 11.5 倍和 12.0 倍。C-state_mode 仍比 Greedy 好 8.6%，C-TCN_mode 则比 Greedy 多 vent 11.2%。

### 3.2 Stochastic 诊断

| 方法 | state_mode vented t | TCN_mode vented t |
|---|---:|---:|
| A current | 16,242 | 14,385 |
| B decision-only | 16,208 | 14,210 |
| C balanced | **12,250** | **8,672** |

C 相对 A 的严格配对 stochastic 差值为：

- state_mode：−3,992 t，95% CI [−7,589, −395]；
- TCN_mode：−5,713 t，95% CI [−8,041, −3,386]。

这说明 C 使随机动作采样明显更稳定，但 stochastic 是预注册诊断指标，不能覆盖 deterministic 主结果。两者反向也说明策略概率和 argmax 决策之间存在校准问题。

## 4. 示范动作是否学得更好

以下指标在训练 demonstration cache 上计算。

| 方法 | Variant | Voluntary-WAIT acc. | Dispatch recall | Destination acc. | Loading dispatch recall |
|---|---|---:|---:|---:|---:|
| A | state_mode | 96.44% | 82.20% | 73.29% | 36.44% |
| B | state_mode | 96.47% | 82.10% | 73.37% | 35.76% |
| C | state_mode | 95.38% | **94.37%** | **92.55%** | **84.46%** |
| A | TCN_mode | 95.88% | 86.70% | 79.69% | 57.18% |
| B | TCN_mode | 95.94% | 86.88% | 80.22% | 56.33% |
| C | TCN_mode | **100%** | **100%** | **100%** | **100%** |

B 几乎没有改变示范诊断，符合其 deterministic 收益较小且不确定的结果。C 确实解决了训练 cache 上的 dispatch recall，尤其是部分装载时的发船标签；因此不能再简单归因于“网络没有能力学会 MPC 标签”。

但 C 每个 epoch 将 3,829 个 dispatch pairs 重复采样到 65,785 个，约为原始数量的 17.2 倍。TCN 的训练 cache 100% 指标与 held-out deterministic 恶化同时出现，更符合记忆、决策边界过度移动或闭环分布偏移，而不是更好的控制泛化。

## 5. 闭环行为变化

以下为 deterministic 100 episodes/variant 的每 episode 均值。

| 方法 | Variant | Dispatches | Partial-load departures | Milk runs | Longest berthed no-dispatch h |
|---|---|---:|---:|---:|---:|
| A | state_mode | 35.51 | 24.76 | 6.31 | 133.86 |
| B | state_mode | 35.11 | 24.59 | 5.89 | 137.65 |
| C | state_mode | 40.34 | 28.43 | 12.09 | 100.68 |
| A | TCN_mode | 36.72 | 25.76 | 7.64 | 122.86 |
| B | TCN_mode | 36.47 | 25.57 | 7.03 | 124.26 |
| C | TCN_mode | 35.09 | 23.94 | 5.37 | 118.99 |

C-state_mode 更频繁 dispatch、部分装载出发和 milk run，虽然缩短了靠泊不发船时间，但总 venting 更差，说明它学到的是“更多发船”，未必是“在正确时机发船”。C-TCN_mode 在训练 cache 上完美识别 dispatch，但闭环中发船反而更少，进一步证明 learner rollout 访问到的状态与 MPC demonstration cache 不同。

## 6. Result-to-claim 判断

**Verdict：`no`，confidence：medium。**

数据支持：

1. B 去掉 forced-action loss 后，在两个编码器上都有小幅平均改善，但五个 model seed 还不足以确认稳定收益；
2. C 能显著改变 dispatch 分类和策略概率，并大幅提高训练 cache 上的 action accuracy；
3. C 能改善 stochastic rollout，但不能改善预注册的 deterministic 主指标。

数据不支持：

1. “decision-only loss 已稳定解决 forced WAIT 淹没”；
2. “50:50 平衡采样能改善确定性 BC”；
3. “示范 dispatch recall 越高，闭环控制一定越好”；
4. “当前 BC 已接近 MPC teacher”。

更准确的结论是：

> 在动作空间和小时级决策保持不变的条件下，去除 forced vessel-action loss 带来小幅但统计上不确定的确定性改善。将主动 WAIT 与 dispatch 按 50:50 平衡采样显著提高了训练示范上的 dispatch/destination 准确率，也改善了随机采样策略，但没有改善确定性策略；TCN 相对 A、两个编码器相对 B 的 deterministic venting 均显著恶化。

## 7. 下一步建议

在暂不引入 dispatch gate + destination 分层动作的前提下，下一步优先做：

1. 建立完全 held-out 的 MPC demonstration cache，报告 dispatch precision、recall、specificity、destination accuracy 和 calibration，避免只看训练 cache recall；
2. 在 learner 闭环实际访问到的状态上做 mode/load-conditioned audit，区分 premature dispatch、missed dispatch、wrong destination 和不必要 milk run；
3. 测试较温和的平衡比例，例如 WAIT:dispatch = 70:30、60:40，并匹配总 optimizer steps；
4. 增加 matched-exposure 对照，避免把“类别比例”与“dispatch 样本重复约 17 倍”混为同一变量；
5. 用 held-out validation 做 logit margin、temperature 和 deterministic/stochastic disagreement 检查。

不建议直接采用当前 50:50 C 作为后续 PPO warm-start。若继续推进，B 是更安全的初始化候选，但在增加 model seeds 或 held-out 决策验证前，应将其表述为方向性结果。

## 8. 结果文件

- `output/rl_forecast/bc_objective_decision_only_formal/`
- `output/rl_forecast/bc_objective_decision_balanced_formal/`
- `output/rl_forecast/bc_objective_ablation_report/bc_objective_summary.csv`
- `output/rl_forecast/bc_objective_ablation_report/bc_objective_paired_deltas.csv`
- `output/rl_forecast/bc_objective_ablation_report/bc_objective_demo_summary.csv`
- `output/rl_forecast/bc_objective_ablation_report/bc_objective_rollout_summary.csv`
