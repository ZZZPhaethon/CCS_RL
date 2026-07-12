# 船舶运行状态观测对 MPC-BC/PPO 的影响：720 h 后续实验

> Forced-action decision-only loss 与 WAIT/dispatch 平衡采样的后续消融见 [Decision-only loss 与平衡采样：BC 消融实验](decision_only_balanced_bc_ablation_zh.md)。

## 1. 结论摘要

加入每艘船的五类 one-hot 运行状态 `sailing/loading/unloading/queued/idle` 后，两种编码器的平均 venting 均下降，并且确定性 rollout 中最长靠泊不发船时间缩短。状态观测确实帮助策略区分“航行或作业中必须 WAIT”和“可主动选择 WAIT/dispatch”，但尚未彻底解决 dispatch 稀缺与长期信用分配问题。

确定性 PPO 的平均改进为：

- `state_mode - state`：venting 减少 1,207 t（17.2%）；
- `tcn_mode - tcn`：venting 减少 893 t（13.5%）。

不过，两项五模型种子配对差值的 95% t 区间都跨越 0。因此当前证据支持“平均表现改善、动作语义更清楚”，但不支持“对不同训练 seed 已稳定改善”的强结论。

## 2. 实验协议与完整性

| 项目 | 设置 |
|---|---|
| Git commit | `41ad6d4` |
| 正式训练 job | Borg `24871`，20/20 tasks `COMPLETED 0:0` |
| Demonstration jobs | `24860`（10 shards）和 `24870`（merge），全部成功 |
| MPC demonstrations | seeds 0–99，100 × 720 h = 72,000 状态—动作对 |
| Demonstration cache SHA256 | `8eb6708d954bb4cf0c0de895d800a88343753bb93236bd3995e63c4694cf2584` |
| BC | 50 epochs，non-WAIT action-dimension weight = 10 |
| PPO | 100,000 steps，其他 reward/action/mask 设置保持不变 |
| Model seeds | 0–4 |
| Evaluation seeds | 101–120 |
| 正式 variants | `state`、`state_mode`、`tcn`、`tcn_mode` |
| 主结果 | deterministic；stochastic 作为诊断 |
| 引用基线 | Idle、Greedy、`RollingNativeMpcController` |

所有 20 个训练 stderr 为空。共获得 40 个 BC/PPO checkpoints、20 个结果 CSV、20 个 demonstration diagnostics、20 个 rollout diagnostics 和 20 个 run manifests。20 个结果 CSV 共 1,660 行，所有数值均为 finite。

## 3. 确定性主结果

表中学习策略为五个 model-seed 均值；“±”是基于五个 model-seed 均值计算的 95% t 区间半宽。引用基线只运行一次策略配置、共享 20 个 evaluation seeds，因此不报告 model-seed 区间。

| Policy | Stage | Vented t | Stored t | Total cost EUR |
|---|---|---:|---:|---:|
| Rolling MPC | reference | 514 | 109,583 | 1,637,715 |
| Greedy | reference | 8,014 | 101,341 | 2,121,699 |
| Idle | reference | 82,983 | 2,036 | 6,816,105 |
| state | BC | 6,533 ± 2,213 | 102,987 | 2,055,163 |
| state_mode | BC | 6,134 ± 1,530 | 103,282 | 2,034,470 |
| state | PPO | 7,007 ± 1,793 | 101,338 | 2,069,102 |
| state_mode | PPO | **5,800 ± 690** | 103,268 | 1,998,207 |
| TCN | BC | 7,950 ± 1,044 | 100,683 | 2,157,206 |
| TCN_mode | BC | 6,632 ± 1,776 | 102,566 | 2,075,003 |
| TCN | PPO | 6,606 ± 1,395 | 103,013 | 2,058,647 |
| TCN_mode | PPO | **5,713 ± 1,072** | 102,937 | 2,006,828 |

所有学习策略的确定性均值都优于 Greedy，但仍明显落后于 rolling MPC。加入 mode 后，两个 PPO 编码器的平均 venting 和总成本均下降。

## 4. 严格配对的 mode effect

差值定义为 mode variant 减去相同编码器的 base variant；负值表示加入船舶状态后 venting 降低。区间先在每个 model seed 内对 20 个 evaluation seeds 求均值，再跨五个 model seeds 使用样本标准差和 `df=4` 的 t 临界值计算。

| Stage | Evaluation | Pair | Mean delta t | Relative delta | 95% CI t | 判断 |
|---|---|---|---:|---:|---:|---|
| BC | stochastic | state_mode − state | -3,771 | -18.8% | [-6,078, -1,464] | 区间低于 0 |
| BC | stochastic | TCN_mode − TCN | -4,049 | -22.0% | [-7,267, -831] | 区间低于 0 |
| BC | deterministic | state_mode − state | -399 | -6.1% | [-2,484, 1,685] | 跨 0 |
| BC | deterministic | TCN_mode − TCN | -1,319 | -16.6% | [-3,616, 979] | 跨 0 |
| PPO | stochastic | state_mode − state | -4,284 | -24.2% | [-9,352, 784] | 跨 0 |
| PPO | stochastic | TCN_mode − TCN | -3,672 | -20.2% | [-7,531, 188] | 跨 0 |
| PPO | deterministic | state_mode − state | -1,207 | -17.2% | [-3,062, 647] | 跨 0 |
| PPO | deterministic | TCN_mode − TCN | -893 | -13.5% | [-2,456, 670] | 跨 0 |

确定性 PPO 的逐模型配对差值仍不一致：`state_mode` 在五个 seeds 中有 3/5 改善，`TCN_mode` 有 4/5 改善。这解释了平均值较好但区间仍跨 0 的现象。

## 5. WAIT 为什么看起来仍然占绝大多数

100-seed MPC demonstrations 中共有 216,000 个逐船小时动作：

| 动作类别 | 数量 | 占全部动作 |
|---|---:|---:|
| Forced WAIT | 146,386 | 67.8% |
| Voluntary WAIT | 65,785 | 30.5% |
| Dispatch | 3,829 | 1.77% |

因此，原始动作标签中约三分之二的 WAIT 是航行状态等 action mask 强制产生的，并不是真正的“等待还是发船”决策。排除 forced WAIT 后，dispatch 占可决策动作约 5.5%，仍然是稀有动作。加入 mode 能解释 WAIT 的物理语义，但不会自动消除剩余的类别不平衡。

## 6. Demonstration 上的动作诊断

以下为 PPO 结束后，跨五个 model seeds 的全船汇总：

| Variant | Voluntary-WAIT accuracy | Dispatch recall | Destination accuracy | Mean P(WAIT) |
|---|---:|---:|---:|---:|
| state | 96.7% | 79.4% | 70.3% | 95.3% |
| state_mode | **97.2%** | **82.7%** | **74.5%** | 95.6% |
| TCN | 95.3% | 79.3% | 68.8% | 94.7% |
| TCN_mode | **95.6%** | **80.8%** | **71.2%** | 95.1% |

mode variants 没有简单地把所有状态都推向 dispatch。更细的状态分解显示：

- `unloading`：mode variants 的 voluntary-WAIT accuracy 达到 100%，`P(WAIT)` 为 0.999（state_mode）和 0.994（TCN_mode），能明确学到“继续卸载”；
- `idle`：dispatch recall 从 0.839 提升到 0.876（state pair），从 0.840 提升到 0.857（TCN pair）；
- `loading`：dispatch recall 仍然最低，并略有下降，state pair 为 0.359 → 0.350，TCN pair 为 0.334 → 0.328。

这说明新增状态主要改善了“作业中应该继续 WAIT”和“idle 时应该 dispatch”的区分；部分装载后提前出发仍然是最难学习的决策。

## 7. 闭环 rollout 行为

以下是 deterministic rollout 的每个 720 h episode 均值，共 5 × 20 = 100 episodes/variant/stage：

| Variant | Stage | Dispatches | Partial-load departures | Milk runs | Longest berthed no-dispatch streak h |
|---|---|---:|---:|---:|---:|
| state | BC | 35.29 | 25.07 | 6.21 | 140.95 |
| state_mode | BC | 35.51 | 24.76 | 6.31 | **133.86** |
| state | PPO | 34.08 | 23.71 | 5.53 | 151.67 |
| state_mode | PPO | 34.45 | 23.91 | 5.21 | **136.97** |
| TCN | BC | 35.64 | 25.98 | 7.00 | 140.17 |
| TCN_mode | BC | 36.72 | 25.76 | 7.64 | **122.86** |
| TCN | PPO | 34.95 | 26.61 | 5.71 | 135.33 |
| TCN_mode | PPO | 36.25 | 25.71 | 6.87 | **124.11** |

mode variants 的总发船次数没有剧烈增加，但最长靠泊不发船连续时长降低约 8–12%。这与 venting 改善一起表明，收益更可能来自 dispatch 时机和状态条件化，而不是单纯增加发船频率。模型仍然学到了大量部分装载出发和 emitter-to-emitter milk run；新增状态没有把策略退化成“只等满载再发船”。

## 8. 对研究问题的回答

**Result-to-claim verdict：partial（中等置信度）。** 数据支持“显式运行状态改善平均表现并让 WAIT/dispatch 语义更可辨认”；数据不支持“状态观测已经稳定解决 dispatch 被 WAIT 淹没”或“对所有训练 seed 都显著优于 base encoder”。

1. **加入船舶状态有用。** 它使 forced WAIT、作业 WAIT 和主动 WAIT 在观测中可区分，提高总体 dispatch recall 和目的地准确率，并减少长时间靠泊不发船。
2. **depart 没有完全摆脱 WAIT 淹没。** Dispatch 在去除 forced WAIT 后仍只占可决策动作的约 5.5%；尤其是 `loading` 中的部分装载 departure recall 仍低。
3. **当前最稳妥的结论是平均改善而非稳定胜出。** 两种编码器的确定性 PPO 平均 venting 都下降，但五模型种子配对 95% 区间跨 0。
4. **下一步应针对真正的 decision states。** 建议只在非 forced-WAIT 样本上增加辅助 dispatch loss 或分层动作头，并用 DAgger/on-policy MPC queries 补充 learner 实际访问到的 `loading/idle` 临界状态。保持小时级环境和部分装载发船动作不变，以便确认改进来自学习问题而非改写业务约束。

## 9. 结果文件

- `output/rl_forecast/operation_mode_formal/forecast_encoder_summary.csv`
- `output/rl_forecast/operation_mode_formal/forecast_encoder_episode_summary.csv`
- `output/rl_forecast/operation_mode_formal/forecast_encoder_summary.md`
- `output/rl_forecast/operation_mode_formal/results_<variant>_seed<seed>.csv`
- `output/rl_forecast/operation_mode_formal/demo_mode_diagnostics_<variant>_seed<seed>.csv`
- `output/rl_forecast/operation_mode_formal/rollout_mode_diagnostics_<variant>_seed<seed>.csv`
- `output/rl_forecast/operation_mode_formal/run_<variant>_seed<seed>.manifest.json`
