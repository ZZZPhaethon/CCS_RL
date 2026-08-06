# 三船 Unified-Window 场景论文实验计划

## 1. 文档目的

本文档用于规划三船 CCS 船运调度案例在 `unified_window_v1` 扰动协议下的正式论文实验。实验围绕以下三个问题组织：

1. **有效性**：Iterative Action-Q 是否能在相同物理环境和测试场景下优于启发式、PPO 类方法和 Rolling MILP？
2. **可解释性与信息作用**：Q 的稀疏干预如何改变运行轨迹，未来信息及其表示方式如何影响性能？
3. **适用边界**：训练好的控制器在扰动分布变化和部署时域显著延长时是否仍能稳定工作？

本文档默认：

- 论文主方法是 **Iterative Action-Q**；
- 固定船服务固定 emitter 的规则策略称为 **Fixed-Assignment Heuristic**；
- Full-horizon MILP 保留为正式实验，但只作为限时、完美信息的离线参考；最终根据可行解、best bound 和 MIP gap 决定放在正文还是 Supplementary；
- E1 比较七种在线控制器，其中四种学习控制器为 Hourly Centralized Maskable PPO、High-level Centralized Maskable PPO、Masked Double DQN 和 Iterative Action-Q G60-P4；
- 四种学习控制器采用约 \(9.5\times10^6\) 次底层 1 h simulator step calls 的近似匹配环境交互预算；Iterative Action-Q G60-P4 实际使用 9,526,297 次，两个 PPO 的训练 hard cap 为 9,505,319 次，Masked Double DQN 实际使用 9,505,312 次；
- Hourly Centralized Maskable PPO 和 High-level Centralized Maskable PPO 均使用与总成本一致的 objective-aligned reward，不使用额外的 stored-CO₂ credit；
- 每个 seed 生成 888 h 外生场景，其中前 720 h 是唯一执行与计分区间，后 168 h 只允许作为只读预测/规划上下文；
- 所有方法只控制船舶调度；井由共享的最大可行注入底层控制器操作，任何方法均不能直接选择注入率；
- Rolling MILP 使用经过 simulator replay 验证的 Greedy trajectory 作为唯一 warm start，不使用额外候选 warm start、shifted previous-plan warm start 或执行 fallback；
- E1 的七种控制器统一使用固定测试 seeds `9000031–9000060`。

机器可读的设计锁与 seed manifest 分别保存在：

- `experiments/protocols/unified_window_v1_paper_protocol.json`；
- `experiments/protocols/unified_window_v1_seed_manifest.json`。

E1 正式测试集固定为 `9,000,031–9,000,060`，七种方法均使用这 30 个 seeds
的现有结果。

本次重编号只改变论文中的正式实验编号，不移动既有结果目录。正式编号与历史产物目录的对应关系为：

| 正式编号 | 实验 | 既有结果目录 |
|---|---|---|
| E0 | 物理仿真层验证 | `experiments_results/E0/` |
| E1 | 在线控制器主比较 | `experiments_results/E1/`、`experiments_results/E1_addendum_masked_double_dqn_20260804/` |
| E6 | Greedy–Iterative Q 运行机制案例 | `experiments_results/E6/`、`experiments_results/E6_zero_vent_seed_9000031/` |
| E4 | Low/Medium/High 扰动鲁棒性 | `experiments_results/E4/` |
| E7 | 冻结策略的时间跨度泛化 | `experiments_results/E7/` |
| E3 | Future-information 消融 | `experiments_results/E3/` |
| E5 | Full-horizon MILP 离线参考 | `experiments_results/E5/` |

原 `experiments_results/E2/` 是已从正式论文计划移除的 iteration-ablation 历史产物，只作归档，不再对应任何正式实验编号。

---

## 2. 核心论文主张与证据

| 拟支持的主张 | 必要证据 | 对应实验 |
|---|---|---|
| Iterative Action-Q 在三船动态扰动场景中具有更低的总成本和 vent | 与强在线控制器的配对比较 | E1 |
| Q 的收益能够从运行轨迹上解释 | Greedy 与 Iterative Q 的机制案例和稀疏干预轨迹 | E6 |
| 方法并非只适用于训练扰动分布 | 冻结模型后的 Low/Medium/High 综合 stress tests | E4 |
| 方法能够扩展到比训练 episode 更长的部署时域 | 30/90/180/365 天冻结策略评估 | E7 |
| 未来信息的作用取决于表示方式，结构化摘要不一定等同于完整序列 | State-only、摘要和完整序列比较 | E3 |
| Full-horizon MILP 能否提供有意义的离线参考 | 限时求解、可行性验证和 MIP gap | E5 |
| 不同方法的训练与在线计算代价透明 | 统一计算资源核算 | 支持性分析 A1 |

---

## 3. 正式比较方法

### 3.1 主在线控制器

| 方法 | 论文名称 | 角色 | 是否训练 | 运行时未来信息 |
|---|---|---|---|---|
| 固定船–emitter 分工 | Fixed-Assignment Heuristic | 静态业务规则基线 | 否 | 不使用 |
| 动态贪心调度 | Greedy | 强实时启发式基线 | 否 | 不使用 |
| 每小时直接决策 PPO | Hourly Centralized Maskable PPO | 目标对齐的最基础 model-free RL 基线 | 是 | 同源 168 h summary |
| 24 h 高层决策 PPO | High-level Centralized Maskable PPO | 目标对齐的高层 model-free RL 基线 | 是 | 同源 168 h summary |
| 每小时 masked value-based RL | Masked Double DQN | 原生联合船舶动作的 value-based RL 基线 | 是 | 同源 168 h summary |
| 当前主方法 | Iterative Action-Q G60-P4 | 论文主方法 | 是 | 同源 168 h summary |
| 滚动优化 | Rolling MILP | 在线优化基线 | 否 | 使用统一 forecast |

### 3.2 信息与训练策略消融

| 方法 | 目的 |
|---|---|
| Iterative Action-Q，State-only | 测试无未来输入时的控制效果 |
| Iterative Action-Q，168 h summary | 测试低维未来摘要的增益 |
| Iterative Action-Q，full 168 h sequence | 利用已有结果说明完整未来序列不一定更优；可放附录 |
| BC–PPO | 可选历史/训练策略消融，不作为主表基线 |

### 3.3 离线参考方法

Full-horizon MILP 作为 **time-limited perfect-foresight MILP reference** 运行。该实验本身保留，但不预设其能够求得 optimal：

- 若得到经过 simulator replay 验证的可行解和有解释价值的 bound/gap，可在正文讨论；
- 若仅得到较差 incumbent、很大的 gap 或没有有效可行解，完整结果放 Supplementary，正文只说明其计算局限；
- 无论结果如何，不能称为 oracle，除非 solver 给出最优性证明。

### 3.4 正式比较前的算法实现调整

#### Hourly Centralized Maskable PPO

该方法是正式比较中最基础的 PPO 基线。旧 PPO checkpoint 和
`sim.control.event_based.rl` 下的高层事件驱动 PPO 均不进入此基线。正式版本必须
通过 `sim.control.hourly_ppo.train_hourly_ppo` 在三船 `unified_window_v1` 环境中
从零训练，并锁定：

- `episode_hours = 720`，物理 simulator step 和 PPO transition 均为 1 h；
- 每个物理小时都重新计算观测、legal-action mask，并由 PPO 输出一次动作；即使船舶航行中只能合法选择 `WAIT`，该小时仍计为一次 PPO transition；
- 观测严格由当前物理状态、显式船舶运行模式/航行目的地和统一的未来 168 h structured summary 构成；summary 不包含 `valid_fraction`；
- 动作为三条船的原生因子化 `MultiDiscrete` 调度动作：每船选择 `WAIT / Terminal / 3 emitters`，在三船场景中对应 `[5, 5, 5]`；动作直接交给 `CCSEnv.step()`；
- 不使用 event trigger、24 h goal、rule/MPC executor、Greedy default、residual action、BC warm start 或 agent-selected well rate；
- `injection_reward_eur_per_t = 0`；
- `store_reward_eur_per_t = 0`；
- `reward_mode = economic`；
- `vent_penalty_weight = 1`；
- `operating_cost_weight = 1`；
- vent 的 80 EUR/t 碳成本保留，因为它属于论文定义的总成本；
- 若 storage-shortfall penalty 非零，其逐步增量也必须进入 reward；
- 使用 legal-action masks；
- 使用非折扣总成本目标，正式候选设置为 `gamma = 1`；
- 720 h 最后一个 transition 加入共同 compact trip cleanup cost，并标记为真正的 MDP terminal，禁止在 cleanup value 之后继续 bootstrap；因训练预算耗尽而提前结束时只标记 truncation，不加入 cleanup；
- E1 固定使用与另外三种学习方法相同的 168 h structured future summary；State-only 仅作为 E3 消融。

因此，除正的 reward scale 外，完整 episode 的 PPO 累计训练回报严格等于
`−(720 h realised economic cost + common compact trip cleanup cost)`。该定义中的
“每小时给动作”指 PPO 直接给原生船舶调度动作，而不是给一个由底层规则执行多小时的高层目标。

#### Masked Double DQN

该方法作为 E1 的第七个在线控制器，使用逐小时原生联合船舶调度动作，并锁定：

- `episode_hours = 720`，每个决策推进 1 h；
- 将三船 `[5, 5, 5]` `MultiDiscrete` 动作枚举为 125 个联合动作，并使用 legal-action masks；
- 观测使用当前物理状态和与其他学习方法同源的 168 h structured summary，不包含 `valid_fraction`；
- 从零训练，不使用 Greedy default、BC warm start、event trigger、goal executor 或 residual action；
- 使用 Double DQN、`gamma = 1` 和与 realised total cost 加共同 terminal cleanup 一致的训练目标；
- model seeds 为 0/1/2，每个 seed 实际使用 9,505,312 次 simulator calls，并按 validation mean total cost 选择 checkpoint；
- 正式测试使用 seeds `9000031–9000060`，共 90 条 model-seed/test-seed 记录。

Masked Double DQN 是在原 E1 测试结果已被访问后追加的 post-hoc baseline。其实现和配置在 DQN 正式测试前锁定，但论文必须明确披露其 post-hoc 属性，不能将其描述为原始预注册比较的一部分。

#### High-level Centralized Maskable PPO

该方法通过 `sim.control.event_based.rl.train_high_level_ppo` 训练，并锁定：

- 以 24 h 为最大高层重规划间隔，并在事件触发时提前重新决策；使用 64 个安全的三船联合偏好动作；
- 使用 dynamic action masks，并由既有高层执行接口把安全意图推进到物理 simulator；
- 观测使用当前物理状态和与其他学习方法相同的单一 168 h structured summary；
- `objective = realised_total_cost`、`stored_credit_eur_per_t = 0`、`vent_penalty_eur_per_t = 80`、`operating_cost_weight = 1`、`gamma = 1`；
- 每个 model seed 的底层 simulator-hour hard cap 为 9,505,319，并按 validation mean total cost 选择 checkpoint。

#### Rolling MILP

正式 Rolling MILP 锁定为：

- `objective_mode = economic`，不使用 vent-first lexicographic objective；
- 每 24 h replan，planning horizon 为 168 h；
- 每次 replan 在当前环境副本上将 Greedy rollout 至 planning horizon；
- Greedy 动作序列经 simulator replay 验证后作为唯一 CPLEX MIP start；
- MILP 只优化船舶调度，不包含可自由选择的井注入动作；planning model 使用与 simulator 相同的最大可行注入规则；
- 不调用额外的多候选 warm-start selector；
- 不使用 shifted previous-MILP warm start；
- 暂不设置执行 fallback；
- 无 replay-valid incumbent 时终止该 episode，并记录 solver failure、termination status 和已经消耗的求解时间，不为该 seed 填补控制成本。

Warm start 只用于初始化 MILP 求解，不等于求解失败后的控制 fallback。

---

## 4. 公平比较协议

### 4.1 统一物理环境

所有方法必须使用完全相同的：

- 三船网络；
- 888 h sampled scenario = 720 h 执行/评估期 + 168 h 只读未来上下文；
- 1 h 物理仿真步长；
- 初始状态生成规则；
- 船舶、terminal、well 和 reservoir 参数；
- 经济参数和总成本定义；
- action feasibility mask 和物理安全约束；
- 共同的 compact trip cleanup 末端价值：720 h 后停止新增 capture、关闭 cleanup 扰动，将剩余 CO₂ 完成封存，并把该 cleanup operating cost 计入所有方法的 terminal return 和报告总成本；
- 相同 seed 下的扰动轨迹。

任何方法不得拥有单独放宽的容量、装卸、航行、注入或压力约束。

所有 controller rollout 必须在 720 h 截止。720–888 h 的外生轨迹只能被学习方法的预测编码器或 Rolling MILP 读取，不得调用 simulator 执行动作，也不得进入 episode cost、vent、stored 或其他表现指标。Full-horizon MILP 只读取并优化前 720 h，在 720 h 状态上加入共同的 cleanup value，不使用 720–888 h 外生信息。

### 4.2 统一底层井控制

井注入不属于在线控制器的动作空间。Fixed-Assignment、Greedy、Hourly Centralized PPO、High-level Centralized PPO、Iterative Q、Rolling MILP 和 Full-horizon MILP 均只决定船舶调度，并共享同一个确定性底层井控制器。

在每个 1 h simulator step，若井处于维护或不可用状态，则 \(q_t^{inj}=0\)；否则：

\[
q_t^{inj}
=
\min\left(
\frac{I_t^{terminal}}{\Delta t},
\bar q^{equipment},
q_t^{injectivity},
q_t^{pressure}
\right),
\]

其中 \(I_t^{terminal}\) 是可供注入的 terminal inventory，其他三项分别表示设备、当期 injectivity 和压力约束允许的最大注入率。若现有 simulator 已将压力限制合并进 `injectivity`，实现中只保留一个对应上限，不能重复扣减。

该速率使用连续值（t/h），不再向下映射到预设的 Mtpa 离散档位。离散井动作仅保留在旧版 `agent_selected` 兼容接口中，不用于正式实验。

公平性要求：

- Hourly PPO、High-level PPO 和 Iterative Q 的 action space/candidate actions 中移除井注入率；
- Fixed-Assignment 和 Greedy 不再拥有各自的井控制逻辑，而是调用同一底层函数；
- Rolling MILP 和 Full-horizon MILP 不得通过降低或延迟注入获得额外优化自由度，其预测状态转移必须复现同一规则；
- 注入运行成本仍由 simulator 按实际自动注入量计入总成本和成本分解；
- well maintenance、injectivity 和 reservoir pressure 仍是环境状态与扰动，但不是上层控制动作。

该简化建立在以下建模边界上：当前问题不考虑时变注入电价、启停/爬坡约束、主动停井促进压力恢复的控制收益，或随注入率显著变化的非线性边际成本。如果后续模型加入其中任何一项，应重新评估是否需要恢复井控制动作。

E0 必须验证：

- 自动注入量始终满足 inventory、availability、injectivity、equipment 和 pressure constraints；
- terminal inventory 不因注入更新产生负值；
- 各方法在相同物理状态下得到完全相同的井注入量；
- simulator、Rolling MILP 和 Full-horizon MILP 对该规则的预测一致。

### 4.3 锁定 `unified_window_v1`

当前基础协议应在正式实验前冻结并写入配置快照：

| 参数 | 当前基础值 |
|---|---:|
| Sampled scenario length | 888 h |
| Execution and scoring horizon | 720 h |
| Capture Gaussian noise std | 0.30 |
| Capture outage rate | 0.5/week |
| Capture outage mean duration | 12 h |
| High-output event rate | 0.5/week |
| High-output event mean duration | 48 h |
| High-output multiplier | 1.25–1.75 |
| Weather-window rate | 0.5/week |
| Weather-window mean duration | 48 h |
| Weather speed factor | 0.50–0.80 |
| Well-maintenance rate | 0.3/week |
| Well-maintenance mean duration | 12 h |
| Emitter initial fill | 0–50% |
| Terminal initial fill | 0–50% |
| Reservoir pressure warm start | 0–50% |
| Forecast context | 168 h |

> **必须注意：**正式实验启动后不得在查看测试结果后修改这些参数。若修改，应建立新的协议版本并重新运行所有方法。

### 4.4 统一信息权限

公平的原则是“相同原始信息可用”，而不是强迫所有方法采用相同内部表示。

1. 所有方法都观察相同当前物理状态。
2. 所有 forecast-capable 方法访问同一份 168 h forecast 对象，其来源、准确度、更新时间和扰动 realization 一致。
3. Hourly Centralized PPO、High-level Centralized PPO、Masked Double DQN 和 Iterative Q 在 E1 中统一使用由该对象计算的同源 168 h structured summary；不加入 `valid_fraction`，也不在 720 h 处截断摘要。
4. Rolling MILP 使用同一 forecast 对象的逐小时完整序列；这是表示粒度差异，不是 forecast 来源差异。
5. Summary 字段、horizon 和归一化方式只能在 validation 阶段锁定，不能针对四种学习算法分别查看正式测试结果后调整，也不能按测试 seed 动态改变。
6. Fixed-Assignment 和 Greedy 按定义不使用 forecast。
7. 主结果表必须列出每种方法的“forecast available”和“forecast used”。
8. 任何时刻的最大预测终点均不得超过同一样本的 888 h 边界；720 h 后的计划动作不得执行或计分。

如果当前 forecast 直接来自场景真值，论文必须明确称为 **perfect-forecast protocol**。此时结论只能说明理想信息条件下的算法行为，不能直接声称现实部署性能。

> **禁止：**在同一主排名中让 Rolling MILP 使用真实未来，而让其他方法只能访问当前状态，却不披露信息差异。

### 4.5 统一训练、验证和测试划分

- 训练 seeds：仅用于策略训练、root 生成和 curriculum。
- 验证 seeds：用于统一 future-summary 配置、超参数、checkpoint 和门控阈值选择。
- 测试 seeds：E1 固定使用 `9000031–9000060` 的现有正式结果。
- E1 不扩展或替换正式测试 seed 集。
- 正式论文建议至少使用 **3 个独立训练随机种子**；若计算允许，使用 5 个。

正式 Iterative Q 以 E1 冻结模型清单中的 **G60-P4** 为准。该配置共 3,600 nominal roots：
G0/G1/G2/G3 分别为 2,160/288/432/720，对应 180/24/36/60 个训练场景
seeds、每 seed 12 roots。G0 是固定、策略无关的 Greedy root bank；G1–G3
依次使用前一阶段策略进行 learned-policy roll-in，最终使用在 G0+G1+G2+G3
上训练的 P4。三个正式 model seeds 为 0/1/2；部署门控固定为 5 heads、至少
4 heads 同意、Q margin 0.40、12 个干预窗口和最多 12 次干预。

建议执行顺序：

1. 单训练 seed 完成方法筛选和代码检查；
2. 锁定方法、超参数和信息协议；
3. 对最终保留的学习方法训练至少 3 个独立 seeds；
4. 每个训练结果在同一组 30 个测试场景 seeds 上评估。

如果最终只能完成一个训练 seed，必须在 Limitations 中说明：测试场景的不确定性已经测量，但训练随机性尚未充分量化。

### 4.6 统一统计协议

主比较采用相同 seed 的配对统计：

- 每个指标报告 mean、standard deviation 和 median；
- 相对 Greedy 报告配对均值差；
- 使用 paired bootstrap 给出 95% confidence interval；
- 报告相对 Greedy 的胜/负场景数；
- 主结论优先基于总成本的配对差，而不是仅比较均值排名；
- 不把训练期间使用过的 validation episodes 混入 test statistics。

如果存在多个训练 seeds，优先采用分层 bootstrap：

1. 重采样训练 seed；
2. 在每个训练 seed 内重采样测试场景 seed；
3. 计算总体差异区间。

### 4.7 统一训练计算核算

Root 数量不能与 PPO timesteps 直接比较。定义：

\[
B_{\mathrm{Q}}=
\text{3,600-root Iterative Action-Q G60-P4 训练数据生成实际消耗的底层 1 h simulator step calls}
=9{,}526{,}297.
\]

\(B_{\mathrm{Q}}\) 的已审计分解为：

| 数据阶段 | nominal roots | 训练 seeds | 实际可用 roots | 实际 simulator calls |
|---|---:|---:|---:|---:|
| G0，固定 Greedy bank | 2,160 | 180 | 2,159 | 5,639,992 |
| G1，P1 policy roll-in | 288 | 24 | 288 | 760,220 |
| G2，P2 policy roll-in | 432 | 36 | 432 | 1,206,985 |
| G3，P3 policy roll-in | 720 | 60 | 720 | 1,919,100 |
| **合计 / P4** | **3,600** | **300** | **3,599** | **9,526,297** |

\(B_{\mathrm{Q}}\) 包括：

- G0–G3 的训练数据生成；
- Greedy 或 learned-policy baseline rollout；
- policy roll-in 到 root；
- 每个 root 下所有候选动作的后续 rollout；
- 已经执行过 simulator steps 的失败或跳过样本。

\(B_{\mathrm{Q}}\) 不包括：

- validation 和 test rollout；
- 神经网络 forward/backward、重复 SGD epoch；
- 不推进 simulator 的纯 forecast-feature 计算；
- 正式配置锁定前的超参数搜索。超参数搜索成本单独报告。

E1 的四种学习方法使用近似匹配的最大训练环境交互预算：

| 方法 | 最大训练环境交互预算 |
|---|---:|
| Iterative Action-Q G60-P4 | 实际 9,526,297 |
| Hourly Centralized Maskable PPO | hard cap 9,505,319 |
| High-level Centralized Maskable PPO | hard cap 9,505,319 |
| Masked Double DQN | 实际 9,505,312 |

四种学习方法的最大预算差为 20,985 simulator calls，即约 0.22%；
论文统一表述为 approximately matched environment-interaction budget，并逐方法
报告实际 simulator calls。

具体计数规则：

- Hourly Centralized PPO 的每个环境 step 必须恰好推进 1 h，并且直接使用该 step 的 PPO 动作；vectorized training 时所有 workers 的 steps 求和；
- Masked Double DQN 的每个环境 step 必须恰好推进 1 h；vectorized training 时所有 workers 的 steps 求和；
- High-level Centralized PPO 的一个高层 transition 可能跨越多个小时；训练预算累计 transition 内实际推进的全部 physical hours，不能直接使用 SB3 high-level timesteps；
- Iterative Q 按所有真实执行的 roll-in 和 counterfactual rollout 小时求和；
- 每个独立训练 run 均遵守上表对应的预算。

四种方法均可使用相同验证规则在预算上限内选择最佳 checkpoint；不强迫性能已收敛的方法耗尽预算。E1 只比较最终选定模型，不要求为四种学习方法建立统一的 25%/50%/75% 中间 checkpoint。

统一报告：

- 实际 simulator step calls；
- 等价 simulated system hours；
- 数据生成 CPU wall time；
- 神经网络训练时间；
- 峰值内存；
- 最终在线单次决策时间和整段 episode wall time。

近似相同的 simulator budget 不代表相同计算时间，因此 CPU-hours、GPU-hours、并行度和 wall time 仍必须单独报告。只有额外展示性能随累计 simulator calls 的变化时，才讨论 sample efficiency。

---

## 5. 统一评价指标

### 5.1 主结果指标

| 指标 | 作用 | 主表是否展示 |
|---|---|---|
| Total cost (EUR) | 720 h episode cost + common compact trip cleanup operating cost | 是 |
| Paired cost difference vs Greedy | 主要统计结论 | 是 |
| Total cost per captured tonne (EUR/t) | 系统单位成本；分母为全部进入系统的 captured CO₂ | 是 |
| Vented CO₂ (t) / loss rate | 碳损失 | 是 |
| Stored CO₂ (t) / storage rate | 封存效果 | 是 |
| Operating cost (EUR) | 解释成本来源 | 是或成本分解图 |
| Vent penalty (EUR) | 解释总成本来源 | 是或成本分解图 |
| Terminal cleanup operating cost (EUR) | 消除有限时域末端库存偏差 | 是或成本分解图 |
| Episode wall time / decision latency | 在线计算开销 | 简化展示 |

主表与 Figure 3c 的单位成本统一从逐 episode 数据重新计算为 `total_cost_eur / captured_t`。现有汇总文件中的 legacy `unit_total_cost_eur_per_t` 使用 `stored_t` 分母，不用于主表或 Figure 3c。

### 5.2 必要诊断指标

以下指标不必全部进入主表，但必须保存：

- Captured CO₂：防止算法通过降低有效 capture 获得表面优势；
- Emitter、vessel 和 terminal 的 episode-end inventory；
- In-transit inventory growth：识别有限时域末端偏差；
- Override/intervention count；
- Rolling MILP episode completion rate、solver-failure count、timeout count、Greedy warm-start acceptance rate 和 MIP gap；
- 物理违规计数，正式方法应为零。

### 5.3 不作为核心指标

“最长连续 vent 时间”仅在连续 vent 具有独立法规、安全或经济含义时才值得报告。当前实验不将其作为主指标，可保留为内部诊断字段。

---

## 6. 实验清单

## E0. 物理仿真层验证

### 目的

证明后续算法比较建立在可信、守恒且约束一致的物理环境上。

### 实验内容

1. 质量守恒：
   - capture；
   - emitter inventory；
   - vessel cargo；
   - terminal inventory；
   - injected/stored CO₂；
   - vented CO₂。
2. 容量边界：
   - emitter、vessel、terminal 不超容量；
   - reservoir pressure 不超过上限；
   - 自动注入率不超过 inventory、availability、equipment、injectivity 和 pressure 限制；
   - 相同状态下所有控制器调用底层井控制器得到相同注入量。
3. 船舶状态机：
   - 航行中不能改变目的地；
   - 装载、卸载、泊位和排队顺序正确；
   - 航速扰动正确改变 ETA。
4. 简化案例：
   - 无扰动；
   - 单船单 emitter；
   - well maintenance；
   - weather window；
   - capture high-output。
5. 末端核算：
   - 检查 720 h 结束时的在途和未封存库存；
   - 验证共同 compact trip cleanup 对所有控制器读取相同类型的 720 h replay 末状态，并满足既有质量守恒与成本分解误差标准。

### 产出

- **Supplementary Table S1**：验证项目、预期关系、误差容限、是否通过；
- **Supplementary Figure S1**：一个场景的质量平衡和库存变化；
- 自动化测试结果摘要；
- 固定的 simulator version/config hash。

### 完成状态

- [x] E0 已于 2026-07-27 完成，结果保存在 `experiments_results/E0/`；
- 20/20 项验证检查通过，选定的 179/179 项自动化测试通过；
- 720 h Medium 场景的最大全系统质量守恒残差为 \(6.16\times10^{-8}\) t，低于 \(10^{-6}\) t 容限；
- 未发现库存/压力/注入率/船舶状态机硬约束违反，末端 compact cleanup 重复核算误差为 0 EUR；
- E0 不用于选择 Iterative Q 超参数或未来信息形式，相关配置保持未锁定状态。
- E0 结果保留在论文及 Supplementary 的可信性证据中，但当前 PPT 不设置独立 E0 结果页。

### 注意事项

- 验证结果应在训练前完成；
- 不能只用 RL rollout 验证物理层；
- 所有控制器共享同一 simulator build；
- 如果 simulator 与 MILP 的预测模型不完全一致，必须通过 action replay 记录差异。

---

## E1. 七个在线控制器主比较

### 目的

回答：在相同三船 `unified_window_v1` 测试场景下，哪种控制器具有最低的实际执行总成本，并减少 vent、提高 storage？

### 方法

1. Fixed-Assignment Heuristic；
2. Greedy；
3. Hourly Centralized Maskable PPO；
4. High-level Centralized Maskable PPO；
5. Masked Double DQN；
6. Iterative Action-Q G60-P4；
7. Rolling MILP。

### 实验步骤

1. 使用不含 `valid_fraction` 的单一 168 h summary，并锁定 Iterative Q G60-P4 配置和门控；
2. 使用第 4.7 节记录的约 \(9.5\times10^6\) simulator-call 近似匹配预算；
3. 两种 PPO、Masked Double DQN 和 Iterative Q 均使用 validation-best checkpoint；四种学习方法均使用 model seeds 0/1/2；
4. 锁定 Rolling MILP：
   - replan interval；
   - planning horizon；
   - 单次固定 time limit；
   - 与 simulator 一致的经济目标；
   - 与 simulator 一致的自动最大可行井注入规则；
   - Greedy-only warm start；
   - 无有效 incumbent 时的失败记录规则；
5. 在完全相同的测试 seeds 上运行所有方法；
6. 保存逐 seed 结果、失败状态和控制轨迹；
7. 进行配对统计。

### Rolling MILP 设置原则

Rolling MILP 不是本文主方法，因此不做完整时间预算扫描。只在 controller-validation seeds `8100001–8100003` 上进行一次配对预算校准：

- replan every 24 h；
- planning horizon 168 h；
- 比较 30 s 与 300 s time limit per replan；
- 每个 CPLEX 进程固定 4 threads，并使用 deterministic parallel mode；
- 与 simulator 相同的 vent、operation 和 terminal accounting；
- 每次重规划先用 Greedy 从当前 simulator 状态滚动到 planning horizon 末端，并将所得完整、合法且 replay-valid 的离散计划作为唯一 MIP start；
- Greedy 决定全部船舶动作；cleanup 模型的辅助变量由同一末状态完整补齐，避免部分 MIP start 被 CPLEX 拒绝，这不引入第二个调度策略；
- 不使用额外的候选选择，也不使用上一次 MILP 计划平移后的 warm start；
- 超时后执行经过 replay 验证的最佳可行 incumbent；
- 若没有有效 incumbent，则将该次求解和该 episode 标记为 solver failure/incomplete，不切换到 Greedy 或其他 fallback。

若 30 s 与 300 s 的 episode completion/failure 状态相同，且 300 s 相对 30 s 的配对中位总成本改善小于 1%，则正式实验锁定 30 s；否则锁定 300 s。若任一预算没有 replay-valid incumbent，优先选择能够完成全部校准 episodes 的预算。两者均失败时停止并记录。

2026-07-30 经作者明确授权，E1 主结果改用后续完成的 600 s/重规划 extended-budget run03；原 300 s 结果保留为 superseded provenance。由于 extended-budget run03 与原 300 s 正式运行的 runner/solver SHA 不同，两者不得解释为同代码下只改变 time limit 的单因素实验。

这里的 Greedy 只用于向求解器提供初始可行解；最终执行的动作必须来自 MILP incumbent。因此，Greedy warm start 不等于在线 Greedy fallback。

### 产出

- **Table 3：Main online-controller comparison**
  - Total cost；
  - Δ cost vs Greedy 和 95% CI；
  - Total cost/captured t；
  - Vent；
  - Stored；
  - Operating cost；
  - Win/loss vs Greedy。
- **Figure 3a：Paired total-cost differences**
  - 以现实中最容易实施的 Fixed-Assignment 为业务基线，展示 Greedy、两种 PPO、Masked Double DQN、Iterative Q 和 Rolling MILP 的配对总成本差和 95% CI；
  - 横向点区间图；每种方法展示 30 个测试场景的配对差值，学习型方法先在同一测试 seed 内对 3 个 model seeds 取均值；
  - 0 EUR 竖线表示与 Fixed-Assignment 持平，负值表示成本更低；95% CI 对测试场景和学习型方法的 model instances 分层重采样。
- **Figure 3b：Cost decomposition**
  - 以 vessel fuel、conditioning、reconditioning、loading、unloading、vent penalty 和 common terminal-cleanup operating cost 七类成本分项进行堆叠展示；
  - 七种方法按平均 total cost 排序；七类分项之和必须与 total cost 一致。
- **Figure 3c：Unit-cost decomposition**
  - 使用与 Figure 3b 相同的七类成本分项和颜色，但将每个 episode 的各项成本先除以该 episode 全部进入系统的 `captured_t`，再跨测试记录取均值；
  - 该分母包括最终 stored、vented 和期末仍在系统内的 CO₂，避免 vent 已计入成本分子后又因缩小 `stored_t` 分母而被重复放大；
  - 七种方法按平均 unit total cost 排序；堆叠分项之和必须与平均 `total_cost_eur / captured_t` 一致，不使用“平均成本除以平均 captured CO₂”的近似值。
  - Iterative Q 的旧归档未直接导出 `captured_t`，因此按相同 test seed 使用配对外生场景的 `captured_t`；脚本必须验证其他控制器已导出的同 seed 数值完全一致。
- Figure 3a、Figure 3b 和 Figure 3c 只导出 PDF 与 300 dpi PNG。
- 逐 seed CSV、统计 JSON 和完整配置快照。

### 注意事项

- Fixed-Assignment 必须是真正的固定服务映射，不能与动态 Greedy 重复；
- 七个在线控制器的上层动作均只包含船舶调度；任何方法都不得直接控制井注入率；
- Hourly Centralized PPO 必须逐小时直接输出原生船舶动作、使用 legal-action masks，并采用第 3.4 节锁定的目标对齐 reward；不得通过高层 goal 或规则执行器代替逐小时 PPO 决策；
- High-level Centralized PPO 使用 24 h 最大重规划间隔、事件触发更新和 dynamic action masks；
- Masked Double DQN 逐小时输出经过 legal-action mask 的原生联合船舶动作，并按 post-hoc baseline 披露；
- E1 的 Iterative Q 只使用最终 G60-P4 模型；P1–P4 仅描述训练数据聚合过程，不再作为独立正式实验比较；
- 所有结果使用 simulator 实际执行后的成本，不能直接使用 MILP 内部预测 objective；
- 若某方法未完成完整 episode，必须单独报告失败率和已完成比例，不能只对完成 episodes 求均值后与其他方法直接排名，也不能用未声明的 fallback 补齐结果；
- E6 机制案例只用于解释 E1 的统计结果，不能替代多 seed 比较。

---

## E6. Greedy–Iterative Q 运行机制案例

### 目的

解释 E1 中 Iterative Action-Q 相对 Greedy 的成本和 vent 改善如何由少量、可定位的调度干预产生。E6 是单案例机制分析，不是新的多 seed 性能比较。

### 案例选择与设置

- 固定使用 E1 的 Iterative Action-Q G60-P4 model seed 0；
- Greedy 与 Iterative Q 使用完全相同的外生场景和底层物理环境；
- 在 model seed 0 的 E1 正式测试记录中，先筛选 `Iterative Q vented_t == 0` 且 `Greedy vented_t >= 5000` 的案例；
- 在满足条件的案例中，选择成本改善百分比最接近 model-seed-0 中位数的 seed；
- 当前规则选择 seed `9000031`；不得人工挑选成本改善最大或轨迹最好看的场景；
- 只允许使用冻结 checkpoint 做 trace-only replay，不改变 E1 的统计结果。

### 实验内容

1. 逐小时重放 Greedy 与 Iterative Q；
2. 记录两种方法的 emitter buffer、terminal inventory、vessel cargo、累计 vent、累计 storage 和成本；
3. 记录每个事件决策、Greedy 建议、Q 选择、head agreement、Q margin 和是否形成实际动作改变；
4. 汇总所有 accepted Q interventions，解释其发生时刻、被改变的船舶动作和干预前系统状态；
5. 对比相同 captured CO₂ 下的总成本、单位成本、vent 和 stored CO₂。

### 产出

- **Figure 4a：System-state mechanism trace**
  - 同图对照 Greedy 与 Iterative Q 的库存、累计 vent 和关键扰动；
  - 标记 accepted Q interventions，展示干预如何改变后续系统状态。
- **Figure 4b：Vessel-action mechanism trace**
  - 分船展示 Greedy 与 Iterative Q 的离散动作序列；
  - 标出实际发生动作改变的 Q 干预。
- `e6_greedy_hourly_trace.csv`、`e6_iterative_q_hourly_trace.csv`、`e6_event_decisions.csv`、`e6_interventions.csv` 和 `e6_outcome_comparison.csv`；
- Figure 4a/4b 导出 SVG、PDF、TIFF 和 PNG。

### 注意事项

- E6 只支持对所选案例中可直接观察到的运行机制进行解释，不能从一个 seed 推广普遍因果机制；
- E6 不进入 E1 的 bootstrap、均值、胜负率或其他正式统计；
- 正文中的总体性能结论仍必须来自 E1 的多 seed 配对比较；
- 历史目录 `experiments_results/E6/` 中的 seed `9000056` 版本仅保留为 provenance；正式 E6 版本以当前选择规则对应的 seed `9000031` 为准。

---

## E4. 冻结模型后的扰动鲁棒性

### 目的

测试只在 Medium stress 下训练和选模的控制器，在整体运行扰动更弱或更强时是否仍能稳定工作。

### 设计原则

- 所有模型权重、门控和超参数完全冻结；
- Medium 等级严格等于当前 `unified_window_v1`，也是训练和主比较分布；
- Low 和 High 使用同一套综合 stress 定义，同时调整天气、high-output 和 well-maintenance 强度；
- 所有方法使用相同 stress-test seeds；
- 同一个 seed 在 Low/Medium/High 中复用相同随机数流，以尽量维持配对关系；
- Rolling MILP 的 forecast 与对应 stress 等级一致，但 horizon、time limit、Greedy-only warm start 和无 fallback 的失败记录规则不变；
- 不在 stress-test 结果上重新选模型。

### Low/Medium/High 候选定义

下表是正式锁定前的候选设置。具体数值应先在验证 seeds 上确认 Low 不过于简单、High 不会使绝大多数方法都陷入不可恢复状态，然后在测试前冻结。

| 参数 | Low | Medium（训练分布） | High |
|---|---:|---:|---:|
| Weather-window mean duration | 24 h | 48 h | 96 h |
| Weather speed factor range | 0.70–0.90 | 0.50–0.80 | 0.40–0.70 |
| High-output mean duration | 24 h | 48 h | 96 h |
| High-output multiplier | 1.10–1.40 | 1.25–1.75 | 1.50–2.00 |
| Well-maintenance mean duration | 6 h | 12 h | 24 h |

为保持 stress 等级的单调含义，建议：

- weather/high-output/maintenance 的 event rate 暂时保持与 Medium 相同，仅改变持续时间或强度；
- capture outage rate 和 duration 保持固定，因为更频繁的 outage 会减少系统负荷，不一定代表更高调度压力；
- 初始库存范围、经济参数和 episode horizon 保持固定；
- 如果验证发现多个参数同时变化使 High 过度困难，应统一降低 High 系数，而不是根据测试结果修改。

### 可选扩展

- 高初始 emitter inventory；
- forecast error；
- 不同 episode horizon；
- 不同船数或 emitter 数。

这些不进入第一版最小实验包。

### 产出

- **Figure 5：Robustness across Low/Medium/High stress**
  - 横轴为 Low、Medium、High；
  - 每条线或每组柱代表一种控制器；
  - 主纵轴使用 total cost，也可附加相对 Medium 的成本变化。
- **Supplementary Table S2**
  - 每个 stress 等级下完整的 Total cost、Vent、Stored、feasible rate 和 CI。

### 注意事项

- Low/Medium/High 是综合 stress，因此只能支持“整体扰动强度鲁棒性”，不能判断天气、capture 或 well 中哪一个是性能下降的唯一原因；
- stress test 的目标是测边界，不要求方法在所有场景都优于 Greedy；
- 失败结果应保留并在 Discussion 中解释；
- Medium 的结果可直接复用 E1；只需新增 Low 和 High 评估；
- Failure trajectory 可从 High stress 中选择预先定义的最差配对 seed，放 Supplementary Figure S3。

---

## E7. 冻结策略的时间跨度泛化

### 目的

测试仅在 720 h episode 设置下训练和选择的 Iterative Action-Q，能否在不重新训练的情况下，以 direct-global episode progress 稳定部署到 90、180 和 365 天。

### 比较方法

1. Fixed-Assignment；
2. Greedy；
3. Iterative-Q direct-global。

Rolling MILP 不进入首轮 E7，因为其逐次优化成本会随部署时域显著增长，且 E7 的核心问题是冻结学习策略的时间泛化。

### 实验设置

- 评估 horizon 固定为 720、2,160、4,320 和 8,760 h，即 30、90、180 和 365 天；
- 使用正式测试 seeds `9000031–9000060`；Iterative Q 使用冻结的 model seeds 0/1/2；
- 每个 test seed 先生成同一条 8,928 h 场景，再截取嵌套前缀，保证不同 horizon 共享相同随机过程前缀；
- 所有 Iterative Q 权重、门控、future summary 和 720 h 基础 intervention windows 完全冻结；基础 windows 每 720 h 重复；
- Iterative-Q direct-global 使用 `t / H` 作为 episode progress；
- 720 h 下的 total cost、vent 和 stored 必须与 E1 中相同 checkpoint 和场景的基准结果一致，作为长时域 adapter 未改变基准策略的校验。

### 统计与指标

- Total cost、vent 和其他累计量统一换算为每 720 h 数值；
- 报告相对 Fixed-Assignment 的配对成本降低及 95% CI；
- 报告 storage rate；
- 对 Iterative-Q direct-global 相对 Fixed-Assignment 和 Greedy 的差异做 model-seed/test-seed 分层 bootstrap。

### 产出

- **Figure 6：Temporal-horizon generalization**
  - 面板 a 展示 30/90/180/365 天相对 Fixed-Assignment 的配对成本降低；
  - 面板 b 展示各 horizon 每 720 h 的 vent。
- E7 完整汇总表、逐 episode CSV、配对统计表和 audit JSON；
- Figure 6 导出 SVG、PDF、TIFF 和 PNG。

### 注意事项

- E7 只检验时间跨度变化，不等同于新的物理网络、船数或 emitter 数量泛化；
- 年度场景仍使用与 E1 相同的 Medium stress 生成机制；不能把 E7 解释为分布外扰动鲁棒性，后者由 E4 检验；
- E7 不比较不同 episode-progress 编码，结论仅适用于 direct-global（`t / H`）部署协议；
- 历史目录中已标记为 invalid protocol 的 E7 runs 不得进入正式 E7 汇总。

---

## E3. Future information 消融

### 目的

回答未来信息是否有用，以及低维物理摘要是否比直接输入完整未来序列更适合当前数据规模。

### 最小正式比较

1. Iterative Q，State-only；
2. Iterative Q，168 h summary。

### 辅助比较

3. Iterative Q，full 168 h sequence；
4. 若论文声称现实部署能力，再加入 noisy/predicted 168 h summary。

完整序列的 MLP、TCN 和 GRU 不需要全部作为正式多 seed 实验。可使用已有筛选结果，在附录中说明为何选择或放弃完整序列。

### 公平性控制

- 使用相同 roots 和 candidate labels；
- 相同训练/验证划分；
- 相同 Q 网络主体和门控；
- summary 必须由同一份 forecast 对象计算；
- 输入维度变化应披露参数量；
- 不能在 test set 上选择 forecast variant；
- perfect forecast 与 predicted/noisy forecast 不能混为同一个设定。

### 产出

- **Table 4：Future-information ablation**
  - information used；
  - representation dimension；
  - parameter count；
  - Total cost；
  - Vent；
  - Stored；
  - Δ cost vs State-only。
- 可选 **Supplementary Figure S2**
  - 不同表示相对 State-only 的配对成本差及 95% CI。

### 注意事项

- “完整未来反而更差”不等于未来没有信息价值；
- 合理解释包括有限 roots、高维冗余、样本效率和 Q-gate 校准，但除非有直接证据，不应将其中任何一个写成已证明机制；
- 如果 168 h summary 使用真实未来，应称为 idealized/perfect-forecast ablation；
- 主结论可以是“结构化摘要在当前数据规模下更有效”，不能泛化成“摘要永远优于序列模型”。

---

## E5. Full-horizon MILP 离线参考

### 目的

在相同三船场景上运行一个使用完整未来信息的 Full-horizon MILP，记录有限计算预算内能够达到的可行解质量和最优性界限。该实验保留，但不预设能够求到 optimal，也不作为在线控制器与 E1 七种方法直接排名。

### 设置

- 使用 perfect foresight；
- 只优化前 720 h，并在 720 h 状态上计算共同 cleanup value；不读取 720–888 h 外生信息；
- 使用与 simulator 一致的经济参数；
- 只优化船舶调度，并在模型中执行与 simulator 相同的最大可行井注入规则；
- 使用与其他方法相同的 compact trip cleanup terminal value；
- Greedy warm start 覆盖全部 720 h 船舶动作，并从其 720 h 末状态完整补齐 cleanup 辅助变量；
- 使用与正式测试集相同的场景 seeds；若全部 seeds 的计算成本不可接受，必须在求解前锁定一个代表性 seed 子集；
- 主结果 time limit 固定为每个 seed 5 h；原 2 h 结果保留为 superseded provenance；
- 每个 CPLEX 进程固定 4 threads，并使用 deterministic parallel mode；
- 不持续求解到 optimal；
- 对每个 seed 报告 incumbent、best bound、MIP gap、termination status 和 solve time；
- 所有 incumbent 必须经过 simulator replay，记录 replay total cost、vent、stored 和 mismatch；
- 没有经过验证的整数可行解时，不报告控制性能，只报告求解状态和 bound。

### 产出

- **Table 5 或 Supplementary Table S3：Time-limited full-horizon MILP reference**
- 是否放正文由结果决定：
  - 若多数 seeds 有有效可行解，且 bound/gap 能提供有意义参照，则在正文 Section 6.7 展示；
  - 若可行率低或 gap 很大，完整结果放 Supplementary，正文用一段话报告计算局限；
  - 即使没有有效可行解，也保留实验记录，不从实验计划中删除。

### 解释规则

- feasible incumbent 是成本最小化问题中当前已找到的可行上界，不等于最优解；
- solver best bound 是下界，只有 gap 足够小时才接近最优；
- Iterative Q 或 Rolling MILP 优于一个超时 incumbent，不代表优于理论最优；
- 若 simulator replay 与 MILP 内部 objective 不一致，以 simulator replay 结果作为执行性能；
- **禁止称为 oracle**，除非相关 seed 全部获得最优性证明。

---

## 支持性分析 A1. 训练与在线计算成本

### 目的

从 E1、E7 和 E3 的训练与评估日志中汇总不同方法在训练阶段和部署阶段的计算差异，不将 Iterative Q 的大量 counterfactual rollouts 隐藏在“roots 数量”之后。本项不需要额外训练模型，也不作为独立性能实验。

### 实验内容

对学习方法报告：

- 训练 seeds；
- environment/simulator step calls；
- simulated system hours；
- root 和 candidate rollout 数；
- 数据生成 CPU-hours；
- 网络训练 CPU/GPU-hours；
- 模型参数量；
- 单次决策 median/P95 latency；
- 完整 720 h episode wall time。

对 Rolling MILP 报告：

- 固定 time limit；
- 实际 mean/P95 solve time；
- feasible incumbent rate；
- timeout rate；
- Greedy warm-start acceptance rate；
- solver-failure count 和 incomplete-episode rate；
- mean MIP gap（仅对具有有效 bound 的求解）。

### 产出

- **Table 6：Training and deployment cost**
- 不单独为 Rolling MILP 绘制预算曲线，除非后续论文主张计算—性能 Pareto。

### 注意事项

- Vectorized PPO 的所有 worker steps 都要计入；
- High-level PPO 按每个高层 transition 内实际推进的 physical hours 计入，不能直接用 high-level timesteps；
- Q 的每个候选动作后续 rollout 都要计入；
- 四种学习方法均需报告实际 simulator calls 和对应预算使用比例；
- CPU-hours、GPU-hours 和 wall time 应分开；
- 不同硬件上的 wall time 不可直接比较，必须同时报告硬件；
- 即使环境交互预算匹配，也只有在展示性能随累计 simulator calls 的变化时才能使用“sample-efficient”表述。

---

## 7. 图表与数据产物总清单

### 正文建议图表

| 编号 | 内容 | 来源 |
|---|---|---|
| Figure 1 | CCS 系统拓扑和运输—terminal—注入耦合 | 物理模型 |
| Figure 2 | Iterative Action-Q 训练与在线执行流程 | 方法 |
| Table 1 | 物理、经济和 disturbance 参数 | Experimental setup |
| Table 2 | 控制器信息权限、决策频率、训练/求解设置 | Fairness protocol |
| Table 3 | 七个在线控制器主结果 | E1 |
| Figure 3 | 配对成本差和成本分解 | E1 |
| Figure 4 | Greedy–Iterative Q 运行机制与船舶动作 | E6 |
| Figure 5 | Low/Medium/High 综合 stress test | E4 |
| Figure 6 | 30/90/180/365 天时间跨度泛化 | E7 |
| Table 4 | Future-information ablation | E3 |
| Table 5 | Full-horizon MILP 结果（若结果足以进入正文） | E5 |
| Table 6 | 训练和部署计算成本 | 支持性分析 A1 |

### Supplementary

| 编号 | 内容 |
|---|---|
| Table S1 / Figure S1 | Simulator verification |
| Figure S2 | Future representation 配对差 |
| Table S2 | Low/Medium/High 完整 robustness 数值 |
| Figure S3 | Failure trajectory |
| Table S3 | Full MILP 完整结果；Table 5 不进入正文时使用 |
| Tables S4–Sx | 所有训练超参数、网络结构、逐 seed 结果 |

### 每次正式运行必须保存

- 逐 seed CSV；
- summary JSON；
- scenario/config snapshot；
- checkpoint 与 metadata；
- training/validation/test seed manifest；
- forecast protocol；
- code commit hash 或归档版本；
- solver status、bound、gap 和 replay validation；
- 完整 stdout/stderr 日志；
- 生成论文图表所用的中间 tidy-data 文件。

---

## 8. 推荐执行顺序

正式实验已按叙事顺序连续编号：

1. **E0 — 物理仿真层验证**：确认守恒、约束、状态机和末端核算；
2. **E1 — 在线控制器主比较**：锁定七种方法、G60-P4 和正式测试集 `9000031–9000060`，生成 Table 3 与 Figure 3；
3. **E6 — 运行机制案例**：按预定义规则使用 seed `9000031`，生成 Greedy–Iterative Q 的 Figure 4a/4b 和干预明细；
4. **E4 — 扰动鲁棒性**：Medium 复用 E1，新增并汇总 Low/High，生成 Figure 5；
5. **E7 — 时间跨度泛化**：使用冻结 E1 策略完成 30/90/180/365 天评估，生成 Figure 6；
6. **E3 — Future-information 消融**：汇总 State-only、168 h summary 和 full-sequence，生成 Table 4 和可选 Figure S2；
7. **E5 — Full-horizon MILP**：使用锁定时限完成离线参考，生成 Table 5 或 Table S3；
8. **A1 — 计算成本汇总**：汇总训练与部署成本，生成 Table 6；
9. 一次性生成正文和 Supplementary 的最终图表与 source data。

---

## 9. 正式实验开始前的锁定清单

- [x] E0 物理仿真层验证已完成，20/20 项检查通过；完整结果位于 `experiments_results/E0/`；
- [x] E1 七种主方法和 baseline 名称已固定；
- [x] Fixed-Assignment 与 Greedy 的行为不重复：三船正式场景的 5 个 validation seeds 中，两者在可行动状态的决策分歧率为 42.76%，Fixed-Assignment 始终保持一船一 emitter，而 Greedy 会跨 emitter 调度；
- [x] 总成本公式、经济参数和 penalty 已在机器可读协议中固定；
- [x] 共同 compact trip cleanup terminal value 已固定；
- [x] 共享的最大可行井注入函数已实现，正式协议下所有控制器的上层动作空间均已移除井注入率；
- [x] Rolling MILP 和 Full-horizon MILP 已使用相同自动注入规则，不包含额外井控制自由度；
- [x] Forecast 来源、误差和可见范围已固定为 perfect-forecast protocol；
- [x] Hourly Centralized PPO、High-level Centralized PPO、Masked Double DQN 与 Iterative Q 已接入同源的 future-summary encoder；算法各自的动作和控制结构不变；
- [x] 四种学习方法已统一为不含 `valid_fraction` 的 168 h summary；每个样本生成 888 h 场景，720 h 后仅作只读预测/规划上下文；
- [x] Rolling MILP 的经济目标、168 h horizon、24 h replan interval、600 s time limit、Greedy-only warm start 和无 fallback 失败规则已固定；原 300 s 结果仅作 provenance；
- [x] Full-horizon MILP 的 5 h time limit、测试 seed 集和 replay 验证规则已固定；原 2 h 结果仅作 provenance；
- [ ] Low/Medium/High stress 参数已在验证集检查并锁定；
- [x] Q 门控、window 和最大 intervention 数已固定：5 heads、至少 4 heads、margin 0.40、12 windows、最多 12 次；
- [x] Hourly Centralized PPO 已实现为直接逐小时接口：当前状态 + 168 h summary、三船原生 `[5,5,5]` `MultiDiscrete` 动作、legal-action mask、自动最大井注入，无 event/goal/executor/Greedy/residual/BC；完整 episode 末端 reward 含共同 cleanup value 且不继续 bootstrap；
- [x] Hourly Centralized PPO 已接入底层 simulator-hour hard cap：每次 1 h 推进前检查预算，训练产物记录实际 calls、simulated hours、预算使用率及耗尽状态；
- [x] High-level Centralized PPO 已按 24 h 最大重规划间隔、事件触发更新、统一 168 h summary、目标对齐 reward 和 9,505,319 hard cap 完成三个 model seeds；
- [x] Masked Double DQN 已按逐小时 125 联合动作、legal-action masks、统一 168 h summary、目标对齐 reward 和每 seed 9,505,312 simulator calls 完成三个 model seeds；
- [x] 3,600-root Iterative Q G60-P4 的 G0/G1/G2/G3=2,160/288/432/720 已锁定，实际训练预算为 9,526,297 simulator calls；
- [x] 训练、验证、legacy-development 和 E1 正式测试集均已写入 manifest v5；
- [ ] 训练模型不能读取测试 seeds；
- [x] E1 正式测试集固定为 `9000031–9000060`，七种方法结果均已完成；
- [x] Simulator step accounting 已实现：在 `PhysicalSimulator.step()` 成功推进后统一累计 calls 与 simulated hours；深拷贝的 Q root/candidate 共享同一计数器，数据集 metadata 和 summary 均记录实际用量；
- [x] E1 七种方法均能输出统一的 per-seed 核心指标；
- [x] E6 案例选择规则已实现，当前规则锁定 seed `9000031`；
- [ ] E6 正式产物已统一归档；历史目录 `experiments_results/E6/` 中的 seed `9000056` 版本仅保留为 provenance；
- [x] E7 已锁定 720/2,160/4,320/8,760 h、相同嵌套场景前缀及 Iterative-Q direct-global 部署协议；
- [x] E7 四个 horizon 的正式结果和汇总已完成；
- [ ] 统计脚本已在 toy data 上验证；
- [ ] 所有输出目录禁止静默覆盖。
