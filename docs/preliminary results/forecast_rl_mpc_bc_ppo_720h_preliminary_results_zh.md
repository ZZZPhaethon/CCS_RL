# 基于 MPC 示范的 Forecast-RL：720 h 初步结果

> 船舶 `sailing/loading/unloading/queued/idle` 状态观测的 100-seed MPC、5-model-seed 后续实验见 [船舶运行状态观测对 MPC-BC/PPO 的影响：720 h 后续实验](forecast_rl_operation_mode_followup_zh.md)。

## 1. 报告范围

本文汇总 720 h CCS 船运—终端—注入系统中以下控制器的初步比较：

1. 非学习基线：Idle、Greedy 和 `RollingNativeMpcController`；
2. 仅行为克隆（BC）：BC-state、BC-flat 和 BC-TCN；
3. 以 MPC-BC warm-start 并进行 kickstarting 的 PPO：PPO-state、PPO-flat 和 PPO-TCN。

Venting 是首要指标，期末未处理库存和运营成本是次级指标。表格按控制器类型组织，不按成本或 venting 排名。

本文结果仍属于 preliminary results，不应据此宣称 forecast encoder 已稳定优于仅使用当前状态的策略。

## 2. 实验与复现信息

| 项目 | 设置 |
|---|---|
| Git 分支 | `codex/mpc-forecast-rl` |
| Git commit | `8193db17da760056b76d464c9034212ab52af157` |
| 单 episode 时长 | 720 h |
| MPC 重规划间隔 | 24 h |
| MPC 预测窗口 | 168 h |
| MPC 示范数据 | 30 个 scenario seed × 720 h = 21,600 条 |
| MPC cache job | Borg job `24645`，成功完成 |
| 正式训练 | `(state, flat, TCN) × model seeds (0,1,2)` |
| BC | 20 epochs，共享同一份 MPC 示范 cache |
| PPO | 100,000 steps，kickstart coefficient = 1.0 |
| 测试场景 | eval seeds 101–110 |
| 正式训练 job | Borg array `24653`，9/9 tasks 成功完成 |
| 数值与产物检查 | stderr 均为空；未发现 NaN/Inf；18 个 checkpoint 均可重新加载且参数有限 |

非学习基线在 10 个测试 scenario 上评估。每个学习控制器的汇总包含 3 个独立 model seed，每个 model seed 在同一组 10 个测试 scenario 上评估，因此共有 30 个 rollout。

## 3. 控制器与观察空间

### 3.1 非学习控制器

- **Idle**：始终不进行有效调度，用于识别策略是否退化为“不行动”。
- **Greedy**：现有手工启发式船舶调度规则，不使用未来 168 h 预测。
- **MPC**：每 24 h 使用未来 168 h 信息重新优化，目标按优先级依次为最小化 venting、降低预测窗口末端未处理库存、降低运营成本。MPC 输出与 RL 环境相同的原生离散动作，并作为 BC teacher。

### 3.2 `state`、`flat` 和 `TCN`

| 变体 | 输入 | 未来信息 | 说明 |
|---|---|---|---|
| state | 当前 51 维状态 | 无 | 只观察当前物理与运营状态 |
| flat | 当前状态 + 展平的 `[168,9]` | 有 | 将 1,512 个 forecast 数值直接拼接到 MLP 输入 |
| TCN | 当前状态 + 结构化 `[168,9]` | 有 | 用 temporal convolution encoder 压缩未来序列，再与当前状态融合 |

`state` 包括当前 emitter 库存、capture/availability，船舶位置、载量、泊位和航行状态，terminal 库存，well availability/injectivity/压力/注入状态，以及当前天气和航行速度因子；它看不到未来 `t+1...t+168 h` 的 capture、outage 或天气。

Forecast 的 9 个通道为：

- 3 个 emitter capture；
- 3 个 emitter availability；
- 1 个 well availability；
- 1 个 injectivity；
- 1 个全域天气速度因子。

天气采用全域共享的 24 h block 更新方式。Forecast 分支读取未来各 block 对应的天气速度因子，因此能够表示 168 h 内的天气变化；TCN 保留时间顺序，而 flat 只保留数值位置。

## 4. 主结果：deterministic/argmax 执行

实际部署通常选择 argmax 动作，因此 deterministic 结果作为主比较。所有数值均为 720 h episode 的均值。

| 控制器类型 | 控制器 | 未来信息 | Venting ↓ | 期末库存 ↓ | 运营成本 | 运营单位成本 ↓ | 总成本 ↓ | 总单位成本 ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 基线 | Idle | 无 | 81,115 t | 52,500 t | €0.177M | €197.62/t | €6.667M | €7,241.86/t |
| 基线 | Greedy | 无 | 8,019 t | 27,203 t | €1.451M | €14.50/t | €2.092M | €21.14/t |
| 基线/Teacher | MPC | 完整 168 h 预测 | **430 t** | 27,976 t | €1.569M | €14.65/t | **€1.603M** | **€14.97/t** |
| BC-only | BC-state | 无 | 27,906 t | 27,590 t | €1.155M | €14.38/t | €3.388M | €44.37/t |
| BC-only | BC-flat | 展平 `[168,9]` | 81,115 t | 52,500 t | €0.177M | €197.62/t | €6.667M | €7,241.86/t |
| BC-only | BC-TCN | TCN 编码 `[168,9]` | 44,843 t | 36,496 t | €0.854M | €74.54/t | €4.442M | €2,383.41/t |
| BC+PPO | PPO-state | 无 | 14,516 t | **25,312 t** | €1.395M | €14.54/t | €2.557M | €28.06/t |
| BC+PPO | PPO-flat | 展平 `[168,9]` | 81,115 t | 52,500 t | €0.177M | €197.62/t | €6.667M | €7,241.86/t |
| BC+PPO | PPO-TCN | TCN 编码 `[168,9]` | 20,431 t | 27,048 t | **€1.301M** | €14.75/t | €2.935M | €35.59/t |

### 4.1 成本口径

本实验中：

\[
\text{总成本}=\text{运营成本}+80\,€/\mathrm{t}\times\text{Venting}.
\]

运营单位成本为运营成本除以实际储存量；总单位成本为总成本除以实际储存量。表中单位成本先在每个 episode 内计算，再跨 episode 取平均。因此，当某些 episode 的实际储存量接近零时，平均单位成本会非常大。BC-TCN、flat 和 Idle 的高单位成本反映了闭环失效或接近零储存，并非简单的格式或除法错误。

当前总成本没有单独加入期末未处理库存 penalty。因此，期末库存仍需作为独立 KPI 判断，不能仅依靠总成本代表所有边界效应。

## 5. Stochastic 执行结果

Stochastic 模式从策略分布采样动作，主要用于诊断策略是否仅在 argmax 下发生坍缩，不作为默认部署结果。

| 训练阶段 | 观察变体 | Venting（stochastic）↓ | Venting（deterministic）↓ |
|---|---|---:|---:|
| BC-only | state | 31,270 t | 27,906 t |
| BC-only | flat | 72,257 t | 81,115 t |
| BC-only | TCN | 33,200 t | 44,843 t |
| BC+PPO | state | 28,034 t | 14,516 t |
| BC+PPO | flat | 71,627 t | 81,115 t |
| BC+PPO | TCN | **24,284 t** | 20,431 t |

在 stochastic PPO 下，TCN 平均比 state 少 vent 3,750 t；但在 deterministic PPO 下，TCN 平均比 state 多 vent 5,915 t。因此，是否使用 forecast encoder 的结论会随执行方式改变。

## 6. Model-seed 稳定性

以下每个 model-seed 数值先对相同的 10 个 eval seed 求平均。`n=3` 很小，95% t 区间仅作描述性参考。

| 执行方式 | PPO 变体 | seed 0 | seed 1 | seed 2 | 均值 ± model-seed SD | 描述性 95% CI |
|---|---|---:|---:|---:|---:|---:|
| stochastic | state | 28,175 t | 28,148 t | 27,780 t | 28,034 ± 220 t | [27,487, 28,582] t |
| stochastic | flat | 71,474 t | 71,405 t | 72,002 t | 71,627 ± 327 t | [70,815, 72,439] t |
| stochastic | TCN | 21,950 t | 22,560 t | 28,342 t | 24,284 ± 3,528 t | [15,521, 33,047] t |
| deterministic | state | 12,576 t | 12,571 t | 18,402 t | 14,516 ± 3,365 t | [6,157, 22,875] t |
| deterministic | flat | 81,115 t | 81,115 t | 81,115 t | 81,115 ± 0 t | [81,115, 81,115] t |
| deterministic | TCN | 20,869 t | 13,270 t | 27,154 t | 20,431 ± 6,952 t | [3,161, 37,702] t |

配对比较显示：

- stochastic PPO 中 TCN 在 3 个 model seed 中有 2 个优于 state，平均差为 `−3,750 t`，但区间跨过 0；
- deterministic PPO 中 TCN 在 3 个 model seed 中全部劣于 state，平均差为 `+5,915 t`；
- flat 在所有训练阶段、执行方式和 model seed 下都劣于 state；deterministic flat 与 Idle 完全一致。

因此，目前只能说 TCN 在 stochastic 模式下存在潜在平均收益，不能说其收益具有稳健性。

## 7. 主要观察与分析

### 7.1 MPC teacher 显著优于现有学习策略

**观察：** MPC 平均 venting 为 430 t，总成本为 €1.603M；Greedy 为 8,019 t 和 €2.092M。当前最好的 deterministic RL 是 PPO-state，但仍有 14,516 t venting 和 €2.557M 总成本。

**解释：** MPC 每 24 h 使用完整 168 h forecast 重新规划，并在环境真实状态上闭环求解。BC/PPO 只通过有限示范和策略更新近似该决策过程。

**影响：** 当前瓶颈不是 teacher 的质量，而是 teacher policy 的闭环蒸馏质量。

### 7.2 PPO 改善了 state 和 TCN 的 BC 初始化，但仍未超过 Greedy

| 编码器 | BC venting | PPO venting | PPO 相对 BC 改善 | BC 总成本 | PPO 总成本 | PPO 相对 BC 改善 |
|---|---:|---:|---:|---:|---:|---:|
| state | 27,906 t | 14,516 t | 48.0% | €3.388M | €2.557M | 24.5% |
| flat | 81,115 t | 81,115 t | 0.0% | €6.667M | €6.667M | 0.0% |
| TCN | 44,843 t | 20,431 t | 54.4% | €4.442M | €2.935M | 33.9% |

**解释：** PPO 能修正部分 BC 误差，但没有解决关键调度动作覆盖不足和长时序误差累积。flat 已经进入严重的闭环退化区域，100k PPO steps 未能将其恢复。

### 7.3 Flat forecast 输入明确失败

**观察：** flat 拥有最多参数（209,942）且 BC demonstration exact-match 最高（95.06%），但 deterministic BC-flat 和 PPO-flat 均与 Idle 完全一致。

**解释：** 直接展平 `[168,9]` 会产生高维、强相关且时间尺度混合的输入；同时总体 exact-match 容易被 WAIT 等高频动作主导，不能反映少数关键 dispatch 动作是否学会。

**影响：** 离线 BC accuracy 不能作为闭环控制性能的代理指标。后续不应继续扩大 flat MLP，除非作为失败对照。

### 7.4 TCN 尚未稳定利用 forecast

**观察：** TCN 在 stochastic PPO 下平均优于 state，但 deterministic 下三个 model seed 全部更差，且 model-seed SD 最大。

**解释：** TCN 可能学习到部分有用的未来风险信号，但输出分布的 argmax 决策不稳定；也可能由于 expert data 中关键 forecast-conditioned actions 太少，encoder 学到相关性却没有稳定映射到调度动作。

**影响：** 在证明 teacher distillation 有效之前，单纯增加 TCN 宽度或复杂度不太可能解决核心问题。

### 7.5 成本必须与 venting 和储存量共同解释

**观察：** Idle/flat 的运营成本最低，但总成本和总单位成本最高。PPO-TCN 的运营成本低于 PPO-state，但由于 venting 更高，其总成本反而高约 €0.379M。

**解释：** 不运行设备会降低直接运营费用，却会造成大量 venting 和极低储存量。

**影响：** 报告中必须同时保留 venting、期末库存、运营成本、总成本和单位成本，不能按单一成本指标选择策略。

## 8. 当前结果支持和不支持的结论

### 8.1 当前支持

1. 在当前训练协议下，直接展平 `[168,9]` 明显劣于 state。
2. 当前 deterministic 部署结果中，PPO-state 是表现最好的 RL 变体。
3. TCN 在 stochastic PPO 中可能带来平均改善，但结果对 model seed 敏感。
4. 一次性 MPC-BC warm-start 加 100k PPO kickstarting 尚未成功蒸馏 MPC teacher。
5. Greedy 仍优于所有已训练 RL，MPC 则显著优于 Greedy 和 RL。

### 8.2 当前不支持

1. 不能声称“加入 168 h forecast 一定优于只看 state”。
2. 不能声称 TCN 的 stochastic 优势具有统计稳健性。
3. 不能用 BC demonstration exact-match 证明策略具有良好闭环性能。
4. 不能把较低运营成本解释为整体经济性更好。

## 9. 可能的改进与建议实验

### 优先级 P0：先解决 MPC 蒸馏和闭环分布偏移

#### 9.1 On-policy MPC dataset aggregation / DAgger

让 BC 或 PPO 策略在训练场景中实际 rollout，在 learner 访问到的状态上查询 MPC 动作，将新样本加入示范集并重复训练。建议至少进行 2–3 轮，并固定 MPC query 数量以便公平比较。

该实验直接检验当前失败是否主要来自 covariate shift 和 720 h 内的复合误差。第一轮只需要比较 state 与 TCN；flat 保留为失败对照即可。

#### 9.2 按动作类别诊断 BC，而非只看 overall exact-match

需要新增以下诊断：

- WAIT、dispatch、load/unload、well-rate 等动作类别的 precision、recall 和 confusion matrix；
- 每个动作维度的 rare-action recall；
- learner rollout 中的动作频率与 MPC 示范动作频率；
- deterministic top-1/top-2 margin、action entropy 和无效动作 mask 触发情况；
- 首次偏离 MPC 的时间，以及偏离后 venting/库存误差如何累计。

只有确认具体失效类别后，再测试 class-balanced sampling、关键 dispatch 样本过采样或加权 BC loss。

#### 9.3 单独诊断 deterministic argmax collapse

state 在 deterministic PPO 下明显优于 stochastic，而 TCN 的优势方向相反。应记录动作分布、entropy、logit margin 和每个 model seed 的闭环轨迹，判断 TCN 是否把有效动作概率分散在多个候选上，导致 argmax 选中错误的高频动作。

### 优先级 P1：在蒸馏改善后再优化 forecast 表示

#### 9.4 Forecast horizon 与多尺度编码消融

在相同 DAgger 数据和训练预算下比较：

- 24 h、72 h、168 h forecast horizon；
- TCN-32、TCN-64；
- 多尺度 forecast：前 24 h 保留逐小时信息，24–168 h 使用 6 h 或 12 h pooling；
- 分通道归一化，以及 outage 二值通道与连续 capture/weather 通道分开编码。

这能判断 168 h 全分辨率是否包含过多远期噪声，同时保留近期细粒度风险。

#### 9.5 增强稀有扰动覆盖

检查 MPC 示范数据中 emitter outage、well outage、高 capture 和恶劣天气组合事件的覆盖率。若关键组合过少，应采用分层 scenario sampling，而不是仅增加普通随机 seed。

### 优先级 P2：奖励函数作为独立消融

当前主实验保持 vent-first reward 不变是必要的，否则无法区分提升来自 observation/encoder 还是 reward。后续可以单独比较：

1. 当前 vent-first reward；
2. vent-first + terminal inventory penalty；
3. vent-first + 基于库存风险的 potential shaping；
4. 上述奖励在 state 和 TCN 上的交叉实验。

期末库存项确实可以缓解 720 h 截断产生的边界问题，但权重必须保证不会牺牲首要的 venting 目标。建议先报告独立 KPI，再通过消融确定 penalty 是否改善跨边界行为。

### 优先级 P3：提高结论的统计可信度

- 将独立 model seed 从 3 增加到至少 5；
- 保持完全配对的 eval seeds；
- 同时报告 model-seed 均值、SD、配对差值和置信区间；
- 增加 stress test，包括 outage 密集、高 capture 和连续恶劣天气；
- 所有策略统一报告 deterministic 与 stochastic，但预先指定 deterministic 为主结果。

## 10. 建议的下一轮最小实验矩阵

| 实验 | 观察空间 | 示范方式 | PPO/reward | 目的 |
|---|---|---|---|---|
| A | state | 当前 one-shot BC | 当前 PPO | 复现实验基线 |
| B | TCN-168 | 当前 one-shot BC | 当前 PPO | 复现实验基线 |
| C | state | DAgger 2–3 rounds | 当前 PPO | 检验 covariate shift |
| D | TCN-168 | DAgger 2–3 rounds | 当前 PPO | 检验 forecast 是否在更好蒸馏下有效 |
| E | TCN 多尺度 | 与 D 相同 | 当前 PPO | 检验 forecast 压缩方式 |

建议在 C/D 达到稳定改善后，再启动 terminal-inventory reward ablation。下一阶段的最低成功标准可以设为：deterministic RL 在不少于 4/5 个 model seed 上优于 Greedy，且不出现与 Idle 相同的坍缩策略。

## 11. 原始产物

- 汇总 CSV：`output/rl_forecast/borg/formal_8193db1/forecast_encoder_summary.csv`
- 汇总 Markdown：`output/rl_forecast/borg/formal_8193db1/forecast_encoder_summary.md`
- Model-seed 分析：`output/rl_forecast/borg/formal_8193db1/formal_model_seed_analysis.md`
- 最终 Borg 实验报告：`.superpowers/sdd/task-8-final-report.md`
- Checkpoints、manifests 和单任务结果：`output/rl_forecast/borg/formal_8193db1/`

## 12. 补充实验：terminal 业务规则与航行目的地观察

### 12.1 实验动机与设置

代码核查发现，旧环境允许仍载有货物的船舶从 terminal 离港；同时，航行中的船舶只有 `sailing` mode 和航行进度，没有当前目的地。这会让“相同 mode、相同进度、但驶向不同地点”的状态在观察空间中发生混叠。

本轮做了两项最小修正：

1. terminal 中仍有货物的船必须继续卸载，只允许 `WAIT`；emitter 中部分装载的船仍可主动离港，保留 milk run 决策；
2. 新增每艘航行船的 `current_destination` one-hot，目的地槽位为 terminal 加 3 个 emitters；船舶靠泊时该向量全为 0。

正式比较固定使用：100 个 MPC demonstration seeds（0–99）、每个 seed 720 h、共 72,000 个样本、decision-only BC loss、50 epochs、BC-only、5 个 model seeds 和 20 个完全配对的 rollout seeds（101–120）。另用 seeds 121–140 的 14,400 个 MPC 样本做独立 held-out 诊断。deterministic/argmax 为预先指定的主结果。

缓存审计结果如下：

- 训练缓存形状：state `[72000,51]`、forecast `[72000,168,9]`、mode `[72000,3,5]`、destination `[72000,3,4]`、mask `[72000,21]`；
- 17,356 个“terminal 有货”训练船时全部只有 `WAIT` 合法，mask 违规数为 0；
- 47,413 个“emitter 部分装载”训练船时全部仍允许驶向 terminal；
- held-out 缓存有 14,400 行、20 个独立 seeds，并通过相同规则审计。

### 12.2 闭环结果

为避免歧义，本节两组学习策略均使用 `decision-only` loss，训练阶段均为 **BC-only（没有 PPO，也没有 balanced sampling）**；两组之间唯一的观察空间差异是是否加入 `current_destination`。

| 变体 | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | model-seed 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|
| 修正 mask + TCN + mode + decision-only BC-only | 6,698 t | 6,455 t | 5,572 t | 9,283 t | 4,999 t | 6,602 ± 1,647 t |
| 修正 mask + TCN + mode + destination + decision-only BC-only | 2,948 t | 3,636 t | 1,915 t | 4,300 t | 5,832 t | **3,726 ± 1,471 t** |
| 配对差值（destination − 无 destination） | −3,750 t | −2,820 t | −3,657 t | −4,983 t | +833 t | **−2,875 ± 2,212 t** |

加入 destination 后，deterministic venting 从 6,602 t 降到 3,726 t，平均减少 2,875 t，即 43.6%。5 个 model seeds 中 4 个改善；以 model seed 为独立单位的配对差值 95% t 区间为 `[−5,622, −129] t`，`p=0.0438`。若把全部 100 个 rollout 对视为独立样本，配对 t 检验为 `p=1.84e-6`，但该检验忽略了同一 model seed 内的相关性，只应作为补充。

| 指标 | 无 destination（decision-only BC-only） | 有 destination（decision-only BC-only） | 差值 |
|---|---:|---:|---:|
| Deterministic venting | 6,602 t | **3,726 t** | −2,875 t |
| Deterministic storage rate | 81.47% | **82.69%** | +1.22 pp |
| Deterministic total cost | €2.085M | **€1.883M** | −€0.202M |
| Deterministic dispatch 数 | 36.40 | **37.86** | +1.46 |
| 最长靠泊未 dispatch streak | 124.45 h | **109.13 h** | −15.32 h |
| Stochastic venting | 14,687 t | **10,655 t** | −4,033 t |

stochastic 平均值也改善 27.5%，但按 5 个 model seeds 聚合后的差值 95% 区间为 `[−11,067, +3,002] t`，跨过 0，因此 stochastic 改善尚不稳健。

作为历史上下文，terminal mask 修正前的 `tcn_mode + decision-only` deterministic venting 为 6,176 t；仅修正 mask 后为 6,602 t。因此不能把本轮提升归因于业务规则修正本身，主要增量来自 destination observation 与修正后状态定义的组合。

### 12.3 离线 BC 诊断

| Held-out 指标 | 修正 mask + mode | 修正 mask + mode + destination | 变化 |
|---|---:|---:|---:|
| WAIT specificity | 92.12% | **93.21%** | +1.09 pp |
| Dispatch precision | 43.16% | **44.28%** | +1.12 pp |
| Dispatch recall | **75.89%** | 69.87% | −6.02 pp |
| Conditional destination accuracy | **62.10%** | 58.01% | −4.09 pp |
| Mean argmax margin | 0.776 | **0.830** | +0.054 |

训练集 joint exact-match 从 95.21% 提高到 96.93%，但 held-out dispatch recall 和 conditional destination accuracy 反而下降，且不同 model seed 的变化较大。这说明闭环改善不能解释为“模型全面学得更像 MPC”。更符合结果的解释是：destination 减少了舰队状态混叠，使策略的 WAIT/dispatch 边界和闭环协调更稳定；它没有解决全部 MPC 蒸馏误差。

值得注意的是，一艘船靠泊并可以决策时，它自己的 destination one-hot 为全 0；该特征主要告诉它其他正在航行的船正驶向哪里。因此当前结果支持“舰队上下文有价值”，但还没有通过因果消融证明具体改善机制。

### 12.4 与 Greedy/MPC 的关系和结论边界

修正规则后的 Greedy、Rolling MPC 和 Idle 已在相同 seeds 101–120 上重新计算；三者的 venting 和 total cost 均与原基线逐 seed 完全一致，最大绝对差值为 0。Greedy 平均 venting 为 8,014 t，Rolling MPC 为 514 t。因此新 destination BC 的平均 deterministic venting 比 Greedy 低 4,287 t，但仍比 MPC 高 3,213 t。

本轮结果支持的保守结论是：**在当前单一 Northern Lights 3-vessel、720 h、decision-only BC 协议下，加入航行目的地观察能显著改善 deterministic 闭环表现，并使 BC-only 超过 Greedy；但它仍未接近 MPC，也不能证明 MPC imitation fidelity 全面改善。**

当前不支持以下更强结论：

1. destination 在其他网络、训练预算或 PPO 阶段一定有效；
2. destination 已经解决 WAIT 样本主导或长时序 credit assignment；
3. held-out MPC 动作预测已经充分正确；
4. 可以跳过 covariate-shift/DAgger 诊断。

下一步按最小增量顺序应为：先做 destination 的 fleet-context 消融（仅观察其他船目的地 vs 完整目的地），再检查 learner 首次偏离 MPC 的状态；只有确认收益来自减少状态混叠后，再启动小规模 DAgger。当前不建议重新引入 50:50 平衡采样。

### 12.5 本轮原始产物

- 正式 checkpoints、results、manifests 和诊断：`output/rl_forecast/destination_bc_formal_cpu/`
- Pilot：`output/rl_forecast/destination_bc_pilot_cpu/`
- 训练 demonstration：`output/rl_forecast/demos/destination_mask_train_0_99.npz`
- Held-out demonstration：`output/rl_forecast/demos/destination_mask_heldout_121_140.npz`
- 修正规则后的 reference baselines：`output/rl_forecast/corrected_references/`
- 实验代码提交：`0b10c9b`
