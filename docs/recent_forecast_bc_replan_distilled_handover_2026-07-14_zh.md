# Forecast-BC 与 Replan Context 最近问题、修改及结果蒸馏（2026-07-14）

> 用途：供后续 AI 或研究者在不重跑全部探索过程的前提下，理解最近几轮修改解决了什么问题、为什么这样修改、哪些假设被支持或否定、结果文件中有哪些关键证据，以及下一步应从哪里继续。
>
> 本文给出的是可审计的“问题 → 假设 → 修改 → 实验 → 证据 → 决策”记录，不包含不可验证的私有思维过程。所有判断尽量绑定到代码、提交、结果文件或可复现实验。

## 1. 一页结论

当前最重要的结论不是“找到一个更大的网络”，而是逐层排除了三个混杂因素：

1. **Forecast 语义和时间范围必须先修正。** v4 forecast 使用当前小时到未来 167 小时（`t..t+167`），与 MPC 实际 168-step rollout 完全对齐；旧版本的排放量定义、裁剪和偏移均存在问题。
2. **原始 TCN 存在 future branch 精确死亡。** FixedScale TCN 和 FutureMLP 都能让 5/5 seeds 稳定使用 forecast，但“使用更多 future”不自动转化为更好的 deterministic 闭环控制。
3. **更大的状态编码器或 Edge-GNN 不是当前瓶颈。** 3×2 attribution 表明 Large MLP 已显著劣于 Small MLP，Edge-GNN 在 stochastic 闭环中又显著劣于匹配容量的 Large MLP。
4. **单纯添加 replan phase 或放大 replan action loss 不足以恢复 MPC 行为。** 它能改变策略分布并改善部分 stochastic 指标，但没有稳定改善 deterministic venting。
5. **真正缺失的是 MPC 隐藏的 plan identity。** Oracle candidate context 将 held-out dispatch accuracy 提高 `+14.06 pp`、destination accuracy 提高 `+24.38 pp`，而 WAIT 基本不变，强力支持“老师有计划记忆、学生只看当前观测”的状态别名假设。
6. **learned plan context 当前失败在数据泛化，不是接口本身。** selector 的 held-out accuracy 只有 `30.53%`，低于 `32.67%` majority baseline；因此 learned context 没有带来 BC 改善，并显著损害 WAIT。
7. **`imitation-only` 应保留。** 对只有离线 oracle/learned context、但运行时还不能真实生成该 context 的实验，它阻止语义无效的闭环 rollout，同时保留训练、checkpoint、train/held-out imitation 诊断和 manifest。
8. **FutureMLP 是有效但尚未替代 baseline 的本地分支。** 它避免了 future branch 死亡，deterministic venting 与 Original TCN 基本相当（`3,004.2` vs `2,940.9 t`），但没有统计证据证明更优。

因此，当前 baseline 仍应保留 **Small state MLP + Original TCN**（若主目标是 deterministic venting）；FutureMLP 可作为梯度稳定的备选。后续主线应优先解决 **可泛化的 plan-state/context 学习及独立 plan-cycle 数据量**，而不是继续无目标地扩大 encoder。

## 2. 当前 Git 与产物状态

### 2.1 已推送到 `origin/dev-refactor`

| Commit | 内容 |
|---|---|
| `2d095af` | Align forecast horizon with MPC rollout steps |
| `62481c7` | Add fixed-scale MLP and Edge-GNN encoders |
| `1225ebe` | Add GNN attribution 3x2 experiment results |
| `6119442` | Add replan-aware plan context BC experiments |
| `f906d87` | Separate experiment entry points from training scripts |
| `ccebee1` | Document progressive behavior cloning results |

提交后的 clean-HEAD 回归结果为 **602 passed，40 subtests passed**。远端 `dev-refactor` 已同步到 `ccebee1`。

### 2.2 仍在本地、尚未提交

- FutureMLP encoder、注册/测试与 [HPC 提交脚本](../hpc/submit_future_mlp_bc.sh)；
- [FutureMLP 正式 5-seed 报告](preliminary%20results/future_mlp_bc_2026-07-14_zh.md)及 `output/rl_forecast/future_mlp_bc/` 原始产物；
- 根目录研究记录：`HANDOVER_REPLAN_CONTEXT_BC.md`、`findings.md`、`progress.md`、`task_plan.md`；
- 若干用户已有的 `output/` 删除，不属于本轮蒸馏文档的修改范围；
- 本文自身。

FutureMLP 加入后的当前本地回归记录为 **607 passed，41 subtests passed**，但这不能替代 clean-HEAD 已提交状态的 602-test 记录。

## 3. 总体因果模型：为什么一步准确率高，闭环仍可能差

MPC 每 24 小时重新规划一次，然后连续执行同一计划约 23 小时。demonstration 中的动作因而不仅取决于当前可见状态和 forecast，还取决于“上一次 replan 选中了哪个候选计划”。

可以把老师的真实决策写成：

```text
action_t = teacher(state_t, forecast_t, replan_phase_t, selected_plan_t)
```

而最初的 memoryless BC 只能学习：

```text
action_t = student(state_t, forecast_t)
```

当两个小时拥有相似的当前状态与 forecast、但属于不同的已选计划时，输入相同而标签不同，形成 **state aliasing / hidden-plan supervision mismatch**。这解释了以下看似矛盾的现象：

- 一步 exact match 可达到约 96–99%，闭环 venting 仍远高于 MPC；
- GNN 的 held-out destination accuracy 可以更高，闭环却更差；
- 强化 future branch 会改变动作分布，但未稳定改善 deterministic 控制；
- oracle plan identity 对 dispatch/destination 的帮助远大于对 WAIT 的帮助。

后续实验按“先修输入语义 → 再审计 encoder → 再测试时间相位 → 最后直接测试隐藏计划”逐层定位问题。

## 4. 问题一：Forecast 的物理语义和 MPC 时间轴不一致

### 4.1 问题演化

Forecast schema 经过了以下修正：

| 版本 | 问题或修改 | 判断 |
|---|---|---|
| 早期版本 | nominal rate 的语义不能代表真实每小时可捕集量 | 作废 |
| v2 | 加入 hourly production profile，但错误地按最大产能裁剪 | 作废 |
| v3 | 去掉错误裁剪，使用 `t+1..t+168` | 物理量修正，但时间偏移仍不匹配 |
| **v4** | 使用 `t..t+167`，完整覆盖 MPC 的 168 个 simulator steps | 当前标准 |

### 4.2 修改

- [forecast.py](../src/sim/environment/forecast.py)：forecast 起止 offset 改为 `0..167`；
- demonstration cache 升级到 schema v4；
- observation/manifest 中记录 forecast schema、shape、channel 和 offset；
- 保证 MPC 看到的当前小时与 policy forecast 第一个时间点一致。

### 4.3 数据完整性

- train cache：72,000 rows，SHA-256 `52eacbf34eaefd37568e406232838889182af57aa96426ed8e9463c084913a54`；
- held-out cache：14,400 rows，SHA-256 `4761bba0cda9fe69b80fadce3e1491b30dcbc72c8c1225ffd15b04d5b261b784`；
- v4 相对 v3 的 state/action/mask/seed/hour/mode/destination 字节级一致，主要变化是 forecast 时间轴；
- 当前 forecast shape 为 `168 × 9`。

### 4.4 结果与决定

来自 [aligned forecast v4 summary](../output/rl_forecast/aligned_forecast_v4_bc/expanded_eval_summary.md)：

| 模型 | Train exact | Held-out active | Held-out destination | Deterministic vent | Stochastic vent |
|---|---:|---:|---:|---:|---:|
| Original TCN | 96.188% ± 0.740 | 90.636% ± 1.674 | 62.467% ± 2.741 | **2,940.9 ± 479.6 t** | 12,671.5 ± 2,616.9 t |
| FixedScale TCN | 97.030% ± 1.858 | 88.857% ± 2.173 | 59.708% ± 4.510 | 3,867.2 ± 1,217.6 t | **8,690.7 ± 3,417.1 t** |

FixedScale 相对 Original 的 deterministic vent 差值为 `+926.3 t`，95% CI `[−429.9, +2,282.6]`；stochastic 差值为 `−3,980.8 t`，95% CI `[−7,341.1, −620.5]`。

**决定：** v4 对齐是后续实验的必要基础，但它本身不解决 hidden-plan mismatch。旧 schema 的模型或结果不能和 v4 结果无条件横向比较。

## 5. 问题二：Original TCN 的 future branch 会精确死亡

### 5.1 观察

较早 checkpoint 中，8 个模型有 7 个出现 forecast embedding 精确为零；在正式 Original TCN 五个 seeds 中只有 1/5 仍是活跃 future branch。这意味着高 imitation accuracy 可能主要来自当前状态，不能证明模型真正使用了未来信息。

### 5.2 修改

- [forecast_encoder.py](../src/sim/environment/forecast_encoder.py) 增加 FixedScale TCN：SiLU + non-affine LayerNorm，避免 affine 参数把整条 future 分支压成精确零；
- 增加 forecast-use audit：feature L2、input-gradient L2、shuffle probability TV、shuffle argmax change；
- 后续加入参数量匹配的 FutureMLP，检验问题是否来自 TCN 的局部/时序归纳偏置。

### 5.3 证据

| Future encoder | 活跃 model seeds | Feature L2 | Input-gradient L2 | Shuffle TV | Argmax change |
|---|---:|---:|---:|---:|---:|
| Original TCN | 1/5 | 0.483 | 1.196e−4 | 0.0057 | 1.21% |
| FixedScale TCN | **5/5** | 3.923 | 1.164e−3 | 0.0295 | 6.62% |
| FutureMLP（本地） | **5/5** | 1.935 | 1.065e−4 | 0.0150 | 3.02% |

**决定：** future branch 死亡是真问题，FixedScale 和 FutureMLP 均修复了“完全不用 forecast”。但 forecast 利用强度是诊断指标，不是闭环性能代理；不能仅凭 embedding/gradient/shuffle 更大就选择模型。

## 6. 问题三：Progressive BC 中哪些观测和规则真正有效

完整内容见 [progressive BC method comparison](preliminary%20results/progressive_bc_method_comparison_zh.md)。统一 720 h 场景的关键结果如下：

| 方法 | Vent (t) ↓ | Stored (t) ↑ | Total cost | Total €/t |
|---|---:|---:|---:|---:|
| Rolling MPC | **514** | **109,583** | **€1.638M** | **€14.96** |
| Greedy | 8,014 | 101,341 | €2.122M | €21.12 |
| TCN standard BC | 7,950 | — | — | — |
| TCN + mode standard BC | 6,632 | — | — | — |
| 旧 TCN + mode + PPO | 5,713 | — | — | — |
| TCN + mode + decision-only | 6,176 | — | — | — |
| + terminal must unload | 6,602 | — | — | — |
| **+ current destination** | **3,726** | **105,118** | **€1.883M** | **€18.01** |

加入 current destination 相对前一版本：vent `−2,875 t`、stored `+1,546 t`、total cost `−€202k`、total unit cost `−€2.30/t`。当前 BC-only 最佳相对 Greedy 减少 vent `4,287 t`（53.5%），但相对 MPC 仍多 vent `3,213 t`、total cost 多约 `€245k`。

**决定：** operation mode、decision-only 和 current destination 是有物理意义的进步；终端强制规则没有单独解决行为差距。旧 PPO 分支缺少 decision-only、terminal rule 和 destination，不能作为“PPO 是否有效”的公平对照。

## 7. 问题四：扩大状态 MLP 或引入 Edge-GNN 能否解决闭环差距

完整报告见 [GNN attribution 3×2](preliminary%20results/gnn_attribution_3x2_bc_2026-07-14_zh.md)，结构化产物位于 [gnn_attribution_3x2](preliminary%20results/gnn_attribution_3x2/)。

### 7.1 实验设计

- 状态 encoder：Small MLP / parameter-matched Large MLP / Edge-GNN；
- future encoder：Original TCN / FixedScale TCN；
- 5 model seeds；eval seeds 101–120；每次 720 h；
- 每组 100 deterministic + 100 stochastic rollouts，总计 1,200 rollouts。

该 3×2 设计用于把“容量增大”和“图结构归纳偏置”分开：`Large − Small` 估计容量效应，`Edge − Large` 估计 GNN 结构的增量效应。

### 7.2 Deterministic 结果

| State encoder | Future encoder | Vent (t) ↓ | Stored (t) ↑ | Total cost | Total €/t |
|---|---|---:|---:|---:|---:|
| Small | Original | **2,940.9 ± 479.6** | **107,051.9 ± 954.0** | **€1.8326M** | **€17.19** |
| Small | FixedScale | 3,867.2 ± 1,217.6 | 105,394.9 ± 1,647.0 | €1.9068M | €18.24 |
| Large | Original | 5,270.4 ± 1,251.2 | 103,130.5 | €2.0013M | €19.60 |
| Large | FixedScale | 6,027.6 ± 2,047.5 | 101,952.0 | €2.0583M | €20.38 |
| Edge-GNN | Original | 9,060.0 ± 4,193.6 | 96,500.5 | €2.2769M | €24.22 |
| Edge-GNN | FixedScale | 6,889.8 ± 2,250.4 | 99,592.0 | €2.1311M | €21.67 |

关键配对差值：

- Large − Small / Original：`+2,329.5 t`，95% CI `[+378.0, +4,281.0]`；
- Large − Small / FixedScale：`+2,160.4 t`，95% CI `[+194.7, +4,126.1]`；
- Edge − Large / Original：`+3,789.6 t`，CI 跨零；
- Edge − Large / FixedScale：`+862.2 t`，CI 跨零。

### 7.3 Stochastic 结果

| State encoder | Original TCN vent | FixedScale TCN vent |
|---|---:|---:|
| Small | 12,671.5 t | **8,690.7 t** |
| Large | **7,940.5 t** | 8,890.1 t |
| Edge-GNN | 19,036.7 t | 20,673.9 t |

Edge-GNN 相对匹配容量 Large MLP 分别恶化 `+11,096.2 t` 和 `+11,783.8 t`，两个 95% CI 均完全大于零。

### 7.4 为什么不能被一步准确率误导

Edge-GNN + Original 的 held-out active/destination accuracy 为 `93.13% / 69.15%`，高于 Small + Original 的 `90.64% / 62.47%`，但闭环 venting 从 `2,940.9` 恶化到 `9,060.0 t`。

**决定：** 当前证据不支持用 Edge-GNN 替换 Small MLP；更大的容量本身已经伤害 deterministic 泛化。后续不能把 destination accuracy 当作闭环控制的充分代理。

## 8. 问题五：Replan phase 与 action weighting 是否足够

### 8.1 修改

- observation 中加入 replan phase；
- [imitation.py](../src/sim/control/imitation.py) 增加 `apply_replan_action_weight`；
- loss 仅对 phase 0 的非 forced vessel action dimensions 放大；
- [replan phase ablation](../experiments/run_replan_phase_ablation.ps1) 比较 1× / 3× / 5×。

### 8.2 结果

来源：[replan phase summary](../output/rl_forecast/replan_phase_ablation/summary.md)。

| 方法 | Train exact | Held-out dispatch | Held-out destination | Deterministic vent | Stochastic vent |
|---|---:|---:|---:|---:|---:|
| Original TCN | 96.188% | 73.528% | 62.467% | **2,940.9 t** | 12,671.5 t |
| FixedScale baseline | 97.030% | 71.432% | 59.708% | 3,867.2 t | 8,690.7 t |
| Phase 1× | 99.198% | 71.459% | 58.355% | 3,899.4 t | **4,780.5 t** |
| Phase 3× | **99.642%** | 66.286% | 52.308% | 3,800.5 t | 4,883.7 t |
| Phase 5× | 96.922% | **76.605%** | **61.963%** | 4,440.4 t | 8,374.8 t |

相对 FixedScale baseline：

- Phase 1× deterministic `+32.2 t`，CI `[−1,952.1, +2,016.5]`；stochastic `−3,910.1 t`，CI `[−7,534.8, −285.5]`；
- Phase 3× deterministic `−66.8 t`，CI `[−2,637.1, +2,503.6]`；stochastic `−3,807.0 t`，CI `[−6,946.3, −667.6]`；
- Phase 5× deterministic `+573.2 t`，CI `[−1,580.3, +2,726.7]`；stochastic `−315.9 t`，CI 跨零。

**决定：** phase 信息和权重能改变学习分布，1×/3× 还显著改善 stochastic venting，但没有支持预期的 deterministic 假设。phase 只说明“计划执行到第几小时”，不能说明“正在执行哪个计划”，因此下一步转向 candidate identity。

## 9. 问题六：Oracle candidate context 能否验证 hidden-plan 假设

### 9.1 修改

- [native_mpc.py](../src/sim/control/native_mpc.py) 暴露稳定的 `native_mpc_candidate_names`；
- [demonstrations.py](../src/sim/control/demonstrations.py) 在 cache 中记录 candidate label/name 和 plan context；
- [plan_context.py](../src/sim/control/plan_context.py) 增加 `CandidatePlanEncoder`；
- [forecast_gym.py](../src/sim/environment/forecast_gym.py) 增加 oracle/learned context observation plumbing；
- [oracle ablation](../experiments/run_candidate_oracle_ablation.ps1) 使用离线真实 candidate identity。

### 9.2 结果

来源：[oracle summary](../output/rl_forecast/replan_candidate_context/oracle_summary.md)。

| 方法 | Train exact | Held-out dispatch | Held-out destination | Held-out WAIT |
|---|---:|---:|---:|---:|
| Matched phase control | 99.369% ± 0.205 | 70.133% ± 2.162 | 55.942% ± 1.318 | 94.170% ± 0.907 |
| Oracle candidate | 98.623% ± 1.453 | **84.191% ± 5.032** | **80.318% ± 5.658** | 94.301% ± 2.059 |
| Oracle − control | −0.747 pp | **+14.058 pp** | **+24.377 pp** | +0.130 pp |

95% CI：dispatch `[+7.429, +20.687] pp`，destination `[+17.254, +31.499] pp`，WAIT `[−2.037, +2.297] pp`。

按 candidate 分解，greedy 和 forecast-urgency 的 destination 分别改善 `+6.55` 和 `+4.10 pp`；各 dedicated candidate 改善约 `+31.64..+47.45 pp`。

**决定：** hidden-plan hypothesis 得到强支持。candidate identity 主要消除 dispatch/destination 标签歧义，而不是简单提高所有类别准确率。Oracle 仍只是诊断上界，因为部署时不能读取老师的真实 candidate。

## 10. 问题七：Learned plan context 为什么没有复现 Oracle 收益

### 10.1 Selector 结果

来源：[learned context summary](../output/rl_forecast/replan_candidate_context/learned_context_summary.md)。

| 指标 | Train | Held-out |
|---|---:|---:|
| Accuracy | 87.967% ± 18.178 | **30.533% ± 0.721** |
| Macro recall | 93.179% ± 11.033 | 27.824% ± 5.267 |
| True-class probability | 80.951% ± 24.029 | 28.565% ± 0.632 |

Held-out majority baseline 为 `32.667%`，8 类随机基线为 `12.5%`。Selector 在 unseen scenario seeds 上没有超过 majority baseline。

### 10.2 下游 BC 结果

| 方法 | Train exact | Held-out dispatch | Held-out destination | Held-out WAIT |
|---|---:|---:|---:|---:|
| Phase control | 99.369% | 70.133% | 55.942% | **94.170%** |
| Learned soft context | 99.191% | 70.212% | 52.361% | 89.128% |
| Oracle candidate | 98.623% | **84.191%** | **80.318%** | 94.301% |

Learned − control：dispatch `+0.080 pp`，CI `[−7.752, +7.911]`；destination `−3.581 pp`，CI `[−13.375, +6.214]`；WAIT `−5.043 pp`，CI `[−8.719, −1.367]`。

### 10.3 根因判断

72,000 个小时级 rows 并不是 72,000 个独立 plan labels。每 24 小时只产生一次 phase-0 plan choice，因此实际只有约 **3,000 个独立 plan cycles**。数据还包含 8 个不平衡类别，最小 candidate 在训练集只有约 184 个 cycles。重复训练同一计划的 24 个小时或增加 epochs 不会补充新的 plan-level 信息。

**决定：** 当前 learned context 路线的主要瓶颈是跨 scenario 的 plan-cycle 样本量与覆盖，不是把 selector 再训练更久。建议至少做约 5× 数据扩展（约 15,000 个独立 plan cycles），每个 dedicated candidate 尽量达到 1,000–2,000 cycles，并始终按 scenario seed 切分。

继续条件：

1. selector 在 unseen seeds 上显著超过 majority baseline，并有可接受的 macro recall；
2. learned context 在 5 个 paired model seeds 上改善 dispatch/destination，且不损失 WAIT；
3. 满足前两项后才进入 deterministic/stochastic 闭环和 PPO。

## 11. `imitation-only` 是否保留：结论是保留

### 11.1 `bc-only` 与 `imitation-only` 的区别

- `--bc-only`：跳过 PPO 更新，但通常仍会运行 learned-policy/reference 的闭环 evaluation；
- `--imitation-only`：完成 BC 训练、保存 checkpoint、计算 train/held-out imitation diagnostics、写 manifest，但跳过闭环 policy/reference rollout。

### 11.2 为什么 Oracle/Learned Context 需要它

Oracle/learned plan context 当前是 demonstration cache 中的离线每周期信息。运行时 wrapper 尚不能在每次 replan 时可靠地产生等价 context：oracle 只能用占位值，learned context 也不能假定真实标签存在。若仍执行闭环，就会用与训练不同或语义错误的 context，生成一个数字上完整但科学上无效的 rollout。

因此 `imitation-only` 的作用不是“少跑一步以省时间”，而是一个 **实验有效性保护栏**：

- 防止把 placeholder-context rollout 误当作部署性能；
- 仍允许验证数据、训练、checkpoint、held-out imitation 和 manifest；
- 在 runtime context provider 真正实现前，把 oracle 实验明确限制为机制诊断。

删除它会增加误用风险。只有当环境能在闭环中按同一协议生成真实可部署的 learned plan state 时，才应关闭该保护栏并进入闭环评估。

## 12. FutureMLP：本地最新结果及其边界

完整报告见 [FutureMLP BC comparison](preliminary%20results/future_mlp_bc_2026-07-14_zh.md)。该实现和结果目前尚未进入 `origin/dev-refactor`。

### 12.1 设计

- 当前状态仍用 Small MLP：`78 → 64`；
- `168 × 9 = 1,512` 维 future 展平后进入 `1,512 → 35 → 64` MLP；
- SiLU + non-affine LayerNorm；
- future 分支 55,259 参数，FixedScale TCN 为 54,848，只差 411（0.75%）；
- 5 model seeds，50 epochs，decision-only，BC-only；
- train seeds 0–99，held-out 121–140，eval 101–120；
- 每个 model seed 评估 20 deterministic + 20 stochastic rollouts，共 200 rows。

### 12.2 主结果

| Future encoder | Deterministic vent | Deterministic total cost | Stochastic vent | Held-out active | Held-out destination |
|---|---:|---:|---:|---:|---:|
| Original TCN | **2,940.9 ± 479.6 t** | €1.8326M | 12,671.5 ± 2,616.9 t | 90.64% | 62.47% |
| FixedScale TCN | 3,867.2 ± 1,217.6 t | €1.9068M | **8,690.7 ± 3,417.1 t** | 88.86% | 59.71% |
| **FutureMLP** | 3,004.2 ± 536.3 t | **€1.8264M** | 11,908.0 ± 1,750.3 t | 89.56% | **64.88%** |

FutureMLP − Original deterministic vent 为 `+63.3 t`，95% CI `[−394.1, +520.7]`；total cost 为 `−€6.2k`，CI `[−€48.5k, +€36.1k]`，均不能宣称显著差异。FutureMLP 相对 FixedScale 的 deterministic total cost 为 `−€80.4k`，CI `[−€141.9k, −€19.0k]`。

每个 model seed 的 deterministic vent：

| 模型 | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---:|---:|---:|---:|---:|
| Original TCN | 2,627.0 | 2,303.1 | 3,337.5 | 3,445.8 | 2,991.1 |
| FixedScale TCN | 2,342.1 | 4,336.9 | 3,998.2 | 5,541.7 | 3,117.3 |
| FutureMLP | 2,402.7 | 2,849.6 | 3,641.2 | 3,483.2 | 2,644.3 |

### 12.3 决定

FutureMLP 成功避免 future branch 精确死亡，且 deterministic venting 接近当前 baseline，但没有成为新的 venting 最佳模型。它适合作为后续小规模 PPO/融合实验的稳定备选，不应仅凭最低平均 total cost 就直接替换 Original TCN。

`future_mlp_smoke/` 中 1 epoch、单 eval seed 的约 `80–84 kt` venting 只是代码通路 smoke，不是模型能力结果；正式判断必须使用上述 50-epoch、5-model-seed、20-eval-seed 产物。

## 13. 文件与复现入口地图

### 13.1 核心实现

- [Forecast construction](../src/sim/environment/forecast.py)
- [Forecast/state encoders](../src/sim/environment/forecast_encoder.py)
- [Gym observations and variants](../src/sim/environment/forecast_gym.py)
- [Demonstration cache generation](../src/sim/control/demonstrations.py)
- [Native MPC candidate names](../src/sim/control/native_mpc.py)
- [Replan action weighting](../src/sim/control/imitation.py)
- [Candidate plan encoder](../src/sim/control/plan_context.py)
- [Main BC comparison entry](../scripts/compare_forecast_encoders_rl.py)
- [Candidate selector training](../scripts/train_replan_candidate_selector.py)
- [Learned plan-context BC](../scripts/train_learned_plan_context_bc.py)

### 13.2 实验入口

- [GNN 3×2 evaluation](../experiments/evaluate_gnn_attribution_3x2.py)
- [GNN forecast-use audit](../experiments/audit_gnn_attribution_3x2.py)
- [GNN summarization](../experiments/summarize_gnn_attribution_3x2.py)
- [Replan phase ablation](../experiments/run_replan_phase_ablation.ps1)
- [Oracle candidate ablation](../experiments/run_candidate_oracle_ablation.ps1)
- [Learned context ablation](../experiments/run_learned_plan_context_ablation.ps1)
- [FutureMLP HPC job](../hpc/submit_future_mlp_bc.sh)（本地未提交）

### 13.3 结果文件

- [Progressive BC report](preliminary%20results/progressive_bc_method_comparison_zh.md)
- [Aligned forecast v4 summary](../output/rl_forecast/aligned_forecast_v4_bc/expanded_eval_summary.md)
- [GNN 3×2 Chinese report](preliminary%20results/gnn_attribution_3x2_bc_2026-07-14_zh.md)
- [GNN aggregate metrics](preliminary%20results/gnn_attribution_3x2/aggregate_metrics.csv)
- [GNN paired contrasts](preliminary%20results/gnn_attribution_3x2/paired_contrasts.csv)
- [GNN forecast-use audit](preliminary%20results/gnn_attribution_3x2/forecast_use_audit.json)
- [Replan phase summary](../output/rl_forecast/replan_phase_ablation/summary.md)
- [Oracle candidate summary](../output/rl_forecast/replan_candidate_context/oracle_summary.md)
- [Learned context summary](../output/rl_forecast/replan_candidate_context/learned_context_summary.md)
- [FutureMLP report](preliminary%20results/future_mlp_bc_2026-07-14_zh.md)（本地未提交）
- [FutureMLP raw outputs](../output/rl_forecast/future_mlp_bc/)（本地/通常被 Git 忽略）

`HANDOVER_REPLAN_CONTEXT_BC.md` 是 oracle/learned 实验完成前的中间 handover，可用于理解最初假设，但结论已被本文及最新 summary 更新，不应优先于本文。

## 14. 后续 AI 应遵守的判断边界

1. **不要混用 forecast schema。** 任何模型比较前先核对 schema version、offset、cache SHA。
2. **不要把一步准确率当闭环性能。** 至少同时报告 paired model-seed 闭环 venting、stored、成本和不确定区间。
3. **不要把小时 rows 当独立 plan samples。** plan selector 的有效样本单位是 phase-0 plan cycle。
4. **不要用 oracle context rollout 宣称可部署性能。** Oracle 只回答“隐藏计划是否重要”。
5. **不要在 selector 未过 held-out gate 时进入 PPO。** 否则 PPO 会把 context 误差、BC 误差和探索误差混在一起。
6. **不要因为 future-use 指标变大就宣布更优。** 它只证明模型依赖 forecast，不证明依赖方式正确。
7. **不要继续扩大 state encoder 作为默认下一步。** Large/Edge attribution 已表明容量与图结构都不是当前首要瓶颈。
8. **区分已提交与本地产物。** 尤其 FutureMLP 和 `output/rl_forecast/replan_*` 原始结果可能不在远端分支中。

## 15. 推荐下一步

优先级从高到低：

1. 扩充独立 plan-cycle 和 scenario diversity，做 selector data-scaling curve；
2. 保持按 scenario seed 的严格 train/held-out 切分，并审计每个 candidate 的 cycle count；
3. 先通过 selector 与 imitation gate，再实现运行时 learned context provider；
4. 只有运行时 context 与训练语义一致后，关闭 `imitation-only`，运行 paired deterministic/stochastic 闭环；
5. 若进入 PPO，先用 Original TCN baseline 与 FutureMLP 各做小规模、相同预算对照，不默认 FixedScale 或 Edge-GNN；
6. 继续报告 5 model seeds 的 paired CI，避免用单 seed 或单 stochastic mean 作架构结论。

当前最简研究结论是：**Forecast 已经被修正并能被网络使用；剩余主要误差更像是老师的隐藏计划状态未被学生观测，而不是 forecast encoder 不够大。**
