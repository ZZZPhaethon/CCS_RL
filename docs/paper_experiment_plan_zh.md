# 三船 Unified-Window 场景论文实验计划

## 1. 文档目的

本文档用于规划三船 CCS 船运调度案例在 `unified_window_v1` 扰动协议下的正式论文实验。实验围绕以下三个问题组织：

1. **有效性**：Iterative Action-Q 是否能在相同物理环境和测试场景下优于启发式、PPO 类方法和 Rolling MILP？
2. **因果性**：性能提升究竟来自 Q 学习本身、迭代状态分布扩展，还是未来信息？
3. **适用边界**：训练好的控制器在天气、上游产量和下游注入能力发生分布变化时是否仍能稳定工作？

本文档默认：

- 论文主方法是 **Iterative Action-Q**；
- Event-based v4 在论文中统一称为 **Event-Residual PPO**；
- 固定船服务固定 emitter 的规则策略称为 **Fixed-Assignment Heuristic**；
- Native MPC 不参与比较；
- Full-horizon MILP 保留为正式实验，但只作为限时、完美信息的离线参考；最终根据可行解、best bound 和 MIP gap 决定放在正文还是 Supplementary；
- 三种学习控制器采用相同的最大环境交互预算：以 4,800-root Iterative Action-Q 实际消耗的底层 1 h simulator step calls 定义 \(B_{4800}\)；
- Centralized Maskable PPO 和 Event-Residual PPO 均使用与总成本一致的 objective-aligned reward，不使用额外的 stored-CO₂ credit；
- 所有方法只控制船舶调度；井由共享的最大可行注入底层控制器操作，任何方法均不能直接选择注入率；
- Rolling MILP 使用经过 simulator replay 验证的 Greedy trajectory 作为唯一 warm start，不使用 Native MPC 候选 warm start、shifted previous-plan warm start 或执行 fallback；
- 所有正式结论均来自预先锁定的测试 seeds，不能使用测试集选择模型、阈值或超参数。

机器可读的设计锁与 seed manifest 分别保存在：

- `experiments/protocols/unified_window_v1_paper_protocol.json`；
- `experiments/protocols/unified_window_v1_seed_manifest.json`。

其中旧的 `8,000,001–8,000,030` 已登记为 development-only；正式测试预留全新的 `9,000,001–9,000,030`，在所有配置锁定前不得运行。

---

## 2. 核心论文主张与证据

| 拟支持的主张 | 必要证据 | 对应实验 |
|---|---|---|
| Iterative Action-Q 在三船动态扰动场景中具有更低的总成本和 vent | 与强在线控制器的配对比较 | E1 |
| Iterative state aggregation 对性能有独立贡献 | One-shot Q 与 Iterative Q 的预算控制消融 | E2 |
| 未来信息的作用取决于表示方式，结构化摘要不一定等同于完整序列 | State-only、摘要和完整序列比较 | E3 |
| 方法并非只适用于训练扰动分布 | 冻结模型后的 Low/Medium/High 综合 stress tests | E4 |
| Q 的收益能够从运行轨迹上解释 | 主比较后的代表性轨迹 | E1 |
| Full-horizon MILP 能否提供有意义的离线参考 | 限时求解、可行性验证和 MIP gap | E5 |
| 不同方法的训练与在线计算代价透明 | 统一计算资源核算 | 支持性分析 A1 |

---

## 3. 正式比较方法

### 3.1 主在线控制器

| 方法 | 论文名称 | 角色 | 是否训练 | 运行时未来信息 |
|---|---|---|---|---|
| 固定船–emitter 分工 | Fixed-Assignment Heuristic | 静态业务规则基线 | 否 | 不使用 |
| 动态贪心调度 | Greedy | 强实时启发式基线 | 否 | 不使用 |
| 从零训练 PPO | Centralized Maskable PPO | 目标对齐的标准 model-free RL 基线 | 是 | 同源 24/72 h summary |
| Event-based v4 架构 | Event-Residual PPO | 目标对齐的结构化 RL 基线 | 是 | 同源 24/72 h summary |
| 当前主方法 | Iterative Action-Q | 论文主方法 | 是 | 同源 24/72 h summary |
| 滚动优化 | Rolling MILP | 在线优化基线 | 否 | 使用统一 forecast |

### 3.2 消融方法

| 方法 | 目的 |
|---|---|
| One-shot Action-Q，原始预算 | 测试不进行 policy roll-in 时的直接效果 |
| One-shot Action-Q，匹配仿真预算 | 排除 Iterative Q 仅仅使用了更多模拟数据的解释 |
| Iterative Action-Q，State-only | 测试迭代训练在无未来输入时是否仍有效 |
| Iterative Action-Q，24/72 h summary | 测试低维未来摘要的增益 |
| Iterative Action-Q，full 168 h sequence | 利用已有结果说明完整未来序列不一定更优；可放附录 |
| BC–PPO | 可选历史/训练策略消融，不作为主表基线 |

### 3.3 离线参考方法

Full-horizon MILP 作为 **time-limited perfect-foresight MILP reference** 运行。该实验本身保留，但不预设其能够求得 optimal：

- 若得到经过 simulator replay 验证的可行解和有解释价值的 bound/gap，可在正文讨论；
- 若仅得到较差 incumbent、很大的 gap 或没有有效可行解，完整结果放 Supplementary，正文只说明其计算局限；
- 无论结果如何，不能称为 oracle，除非 solver 给出最优性证明。

### 3.4 正式比较前的算法实现调整

#### Centralized Maskable PPO

旧 PPO checkpoint 不进入正式比较。正式版本必须重新在三船 `unified_window_v1` 环境中从零训练，并锁定：

- `episode_hours = 720`，物理 simulator step 为 1 h；
- `injection_reward_eur_per_t = 0`；
- `store_reward_eur_per_t = 0`；
- `reward_mode = economic`；
- `vent_penalty_weight = 1`；
- `operating_cost_weight = 1`；
- vent 的 80 EUR/t 碳成本保留，因为它属于论文定义的总成本；
- 若 storage-shortfall penalty 非零，其逐步增量也必须进入 reward；
- 使用 legal-action masks；
- 使用非折扣总成本目标，正式候选设置为 `gamma = 1`；
- E1 固定使用与另外两种学习方法相同的 24/72 h structured future summary；State-only 仅作为 E3 消融；
- 不使用 BC warm start，否则应另称 BC–PPO。

因此，除正的 reward scale 外，PPO 的累计训练回报应与负的 episode total cost 一致。

#### Event-Residual PPO

保留 v4 的 residual action architecture、Greedy default、event trigger、intervention windows、action masks 和 gate，但正式 E1 版本改为 objective-aligned reward：

- `stored_credit_eur_per_t = 0`；
- `vent_penalty_eur_per_t = 80`；
- `excess_vent_penalty_eur_per_t = 0`；
- `overflow_risk_eur_per_t_hour = 0`；
- `operating_cost_weight = 1`；
- `gamma = 1`；
- 硬物理违规仍由 action masks/constraints 阻止；若保留 hard-violation penalty，只作为不可行保护，不作为正常轨迹 shaping。

由于 reward 已改变，正式 Event-Residual PPO 必须从头训练。原 tail-robust v4 checkpoint 可作为 E4 或 Supplementary 的附加鲁棒版本，但不能替代 E1 的 objective-aligned baseline。

#### Rolling MILP

正式 Rolling MILP 锁定为：

- `objective_mode = economic`，不使用 vent-first lexicographic objective；
- 每 24 h replan，planning horizon 为 168 h；
- 每次 replan 在当前环境副本上将 Greedy rollout 至 planning horizon；
- Greedy 动作序列经 simulator replay 验证后作为唯一 CPLEX MIP start；
- MILP 只优化船舶调度，不包含可自由选择的井注入动作；planning model 使用与 simulator 相同的最大可行注入规则；
- 不调用 Native MPC 多候选 warm-start selector；
- 不使用 shifted previous-MILP warm start；
- 暂不设置执行 fallback；
- 无 replay-valid incumbent 时终止该 episode，并记录 solver failure、termination status 和已经消耗的求解时间，不为该 seed 填补控制成本。

Warm start 只用于初始化 MILP 求解，不等于求解失败后的控制 fallback。

---

## 4. 公平比较协议

### 4.1 统一物理环境

所有方法必须使用完全相同的：

- 三船网络；
- 720 h episode；
- 1 h 物理仿真步长；
- 初始状态生成规则；
- 船舶、terminal、well 和 reservoir 参数；
- 经济参数和总成本定义；
- action feasibility mask 和物理安全约束；
- 共同的 compact trip cleanup 末端价值：720 h 后停止新增 capture、关闭 cleanup 扰动，将剩余 CO₂ 完成封存，并把该 cleanup operating cost 计入所有方法的 terminal return 和报告总成本；
- 相同 seed 下的扰动轨迹。

任何方法不得拥有单独放宽的容量、装卸、航行、注入或压力约束。

### 4.2 统一底层井控制

井注入不属于在线控制器的动作空间。Fixed-Assignment、Greedy、Centralized PPO、Event-Residual PPO、Iterative Q、Rolling MILP 和 Full-horizon MILP 均只决定船舶调度，并共享同一个确定性底层井控制器。

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

- PPO、Event-Residual PPO 和 Iterative Q 的 action space/candidate actions 中移除井注入率；
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
| Episode length | 720 h |
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
3. Centralized PPO、Event-Residual PPO 和 Iterative Q 在 E1 中统一使用由该对象计算的同一组 24/72 h structured summaries。
4. Rolling MILP 使用同一 forecast 对象的逐小时完整序列；这是表示粒度差异，不是 forecast 来源差异。
5. Summary 字段、horizon 和归一化方式只能在 validation 阶段统一锁定，不能针对三种学习算法分别选择，也不能按测试 seed 动态改变。
6. Fixed-Assignment 和 Greedy 按定义不使用 forecast。
7. 主结果表必须列出每种方法的“forecast available”和“forecast used”。

如果当前 forecast 直接来自场景真值，论文必须明确称为 **perfect-forecast protocol**。此时结论只能说明理想信息条件下的算法行为，不能直接声称现实部署性能。

> **禁止：**在同一主排名中让 Rolling MILP 使用真实未来，而让其他方法只能访问当前状态，却不披露信息差异。

### 4.5 统一训练、验证和测试划分

- 训练 seeds：仅用于策略训练、root 生成和 curriculum。
- 验证 seeds：用于统一 future-summary 配置、超参数、checkpoint 和门控阈值选择。
- 测试 seeds：只在所有选择锁定后使用。
- 当前 30 个正式测试 seeds 可继续作为 paired test set，但不得参与任何训练或选择。
- 正式论文建议至少使用 **3 个独立训练随机种子**；若计算允许，使用 5 个。

4,800 roots 及其 G0–G3 分配也属于方法配置，必须在查看正式 test 结果前锁定。若 4,800 是通过比较同一批 test seeds 上的 1,200/2,400/4,800 等配置后选出的，则该批 seeds 已成为 development set；此时应冻结 4,800-root 配置并使用一批全新的未见 test seeds 报告最终结果。

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
B_{4800}=
\text{4,800-root Iterative Action-Q 训练数据生成实际消耗的底层 1 h simulator step calls}.
\]

\(B_{4800}\) 包括：

- G0–G3 的训练数据生成；
- Greedy 或 learned-policy baseline rollout；
- policy roll-in 到 root；
- 每个 root 下所有候选动作的后续 rollout；
- 已经执行过 simulator steps 的失败或跳过样本。

\(B_{4800}\) 不包括：

- validation 和 test rollout；
- 神经网络 forward/backward、重复 SGD epoch；
- 不推进 simulator 的纯 forecast-feature 计算；
- 正式配置锁定前的超参数搜索。超参数搜索成本单独报告。

E1 的三种学习方法使用同一最大训练预算：

| 方法 | 最大训练环境交互预算 |
|---|---:|
| Iterative Action-Q P4 | \(B_{4800}\) |
| Centralized Maskable PPO | \(\le B_{4800}\) |
| Event-Residual PPO | \(\le B_{4800}\) |

具体计数规则：

- Centralized PPO 的每个环境 step 推进 1 h；vectorized training 时所有 workers 的 steps 求和；
- Event-Residual PPO 的一个高层 transition 可能跨越多个小时，因此按每个 transition 内部实际执行的 physical hours 求和，不能直接使用 SB3 high-level timesteps；
- Iterative Q 按所有真实执行的 roll-in 和 counterfactual rollout 小时求和；
- 每个独立训练 run 均使用相同的 \(B_{4800}\) 上限。

三种方法均可使用相同验证规则在预算上限内选择最佳 checkpoint；不强迫性能已收敛的方法耗尽预算。E1 只比较最终选定模型，不要求为 PPO、Event-Residual PPO 和 Iterative Q 建立统一的 25%/50%/75% 中间 checkpoint。

统一报告：

- 实际 simulator step calls；
- 等价 simulated system hours；
- 数据生成 CPU wall time；
- 神经网络训练时间；
- 峰值内存；
- 最终在线单次决策时间和整段 episode wall time。

相同 simulator budget 不代表相同计算时间，因此 CPU-hours、GPU-hours、并行度和 wall time 仍必须单独报告。主实验可以声称“matched environment-interaction budget”，但只有额外展示性能随累计 simulator calls 的变化时，才讨论 sample efficiency。

---

## 5. 统一评价指标

### 5.1 主结果指标

| 指标 | 作用 | 主表是否展示 |
|---|---|---|
| Total cost (EUR) | 720 h episode cost + common compact trip cleanup operating cost | 是 |
| Paired cost difference vs Greedy | 主要统计结论 | 是 |
| Total cost per stored tonne (EUR/t) | 经济效率 | 是 |
| Vented CO₂ (t) / loss rate | 碳损失 | 是 |
| Stored CO₂ (t) / storage rate | 封存效果 | 是 |
| Operating cost (EUR) | 解释成本来源 | 是或成本分解图 |
| Vent penalty (EUR) | 解释总成本来源 | 是或成本分解图 |
| Terminal cleanup operating cost (EUR) | 消除有限时域末端库存偏差 | 是或成本分解图 |
| Episode wall time / decision latency | 在线计算开销 | 简化展示 |

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

### 注意事项

- 验证结果应在训练前完成；
- 不能只用 RL rollout 验证物理层；
- 所有控制器共享同一 simulator build；
- 如果 simulator 与 MILP 的预测模型不完全一致，必须通过 action replay 记录差异。

---

## E1. 六个在线控制器主比较

### 目的

回答：在相同三船 `unified_window_v1` 测试场景下，哪种控制器具有最低的实际执行总成本，并减少 vent、提高 storage？

### 方法

1. Fixed-Assignment Heuristic；
2. Greedy；
3. Centralized Maskable PPO；
4. Event-Residual PPO；
5. Iterative Action-Q；
6. Rolling MILP。

### 实验步骤

1. 在验证集上统一锁定三种学习方法的 24/72 h summary，并锁定 Iterative Q 的 P4 配置和门控；
2. 运行 4,800-root Iterative Q 并由统一计数器测得 \(B_{4800}\)；
3. 使用相同的 \(B_{4800}\) 上限分别训练经过目标对齐的 Centralized PPO 和 Event-Residual PPO，并仅根据 validation 表现选择预算内 checkpoint；
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

Rolling MILP 不是本文主方法，因此不要求做完整时间预算扫描。采用一个预先声明、相对宽裕且可复现的固定预算，例如：

- replan every 24 h；
- planning horizon 168 h；
- 30 s 或 60 s time limit per replan；
- 与 simulator 相同的 vent、operation 和 terminal accounting；
- 每次重规划先用 Greedy 从当前 simulator 状态滚动到 planning horizon 末端，并将所得完整、合法且 replay-valid 的离散计划作为唯一 MIP start；
- 不使用 Native MPC 候选选择，也不使用上一次 MILP 计划平移后的 warm start；
- 超时后执行经过 replay 验证的最佳可行 incumbent；
- 若没有有效 incumbent，则将该次求解和该 episode 标记为 solver failure/incomplete，不切换到 Greedy 或其他 fallback。

最终具体数值必须在 validation 阶段锁定。

这里的 Greedy 只用于向求解器提供初始可行解；最终执行的动作必须来自 MILP incumbent。因此，Greedy warm start 不等于在线 Greedy fallback。

### 产出

- **Table 3：Main online-controller comparison**
  - Total cost；
  - Δ cost vs Greedy 和 95% CI；
  - Total cost/stored t；
  - Vent；
  - Stored；
  - Operating cost；
  - Win/loss vs Greedy。
- **Figure 3a：Paired total-cost differences**
  - 每种方法相对 Greedy 的点估计和 95% CI。
- **Figure 3b：Cost decomposition**
  - 720 h operating cost、vent penalty 与 common terminal-cleanup operating cost 的堆叠图；
  - 详细分项保存 vessel fuel、conditioning、reconditioning、loading 和 unloading。
- **Figure 4：Representative operational trajectory**
  - 直接作为 E1 主比较的解释性结果，紧跟主结果表和 Figure 3；
  - 展示 Greedy 与 Iterative Q 的 emitter/terminal inventory、cumulative vent、扰动区间、船舶模式和 Iterative Q intervention 时刻。
- 逐 seed CSV、统计 JSON 和完整配置快照。

### 代表性轨迹选择规则

Figure 4 使用 Iterative Q 相对 Greedy 的总成本改善最接近中位数的测试 seed。选择规则必须由脚本自动执行，不能人工选择提升最大或“最好看”的 seed。

失败轨迹不占用独立实验编号。可将 Iterative Q 相对 Greedy 表现最差的测试 seed 放在：

- E4 High-stress 结果之后；或
- Supplementary Figure S4。

### 注意事项

- Fixed-Assignment 必须是真正的固定服务映射，不能与动态 Greedy 重复；
- 六个在线控制器的上层动作均只包含船舶调度；任何方法都不得直接控制井注入率；
- Centralized PPO 必须使用 legal-action masks，并采用第 3.4 节锁定的目标对齐 reward：移除 stored-credit 和额外塑形项，保留共同的经济成本；
- Event-Residual PPO 必须从头训练目标对齐版本；原有 tail-robust v4 权重不得直接用于 E1；
- Event-Residual PPO 与 Iterative Q 的干预次数和决策时机可能不同，应披露而不是强行相等；
- E1 的 Iterative Q 只使用最终 P4 模型；P1–P4 的阶段比较属于 E2；
- 所有结果使用 simulator 实际执行后的成本，不能直接使用 MILP 内部预测 objective；
- 若某方法未完成完整 episode，必须单独报告失败率和已完成比例，不能只对完成 episodes 求均值后与其他方法直接排名，也不能用未声明的 fallback 补齐结果；
- 轨迹图只用于解释 E1 的统计结果，不能替代多 seed 比较。

---

## E2. Iterative state aggregation 消融

### 目的

回答 Iterative Action-Q 的收益是否来自策略访问状态的逐轮扩展，而不是简单地来自更多训练数据或同一个 Q 网络。

### 比较组

1. One-shot Q，G0 原始 roots；
2. One-shot Q，使用与 P4 Iterative Q 相同的 \(B_{4800}\) 仿真预算，但所有 roots 仍来自 Greedy roll-in；
3. Iterative Q P1；
4. Iterative Q P2；
5. Iterative Q P3；
6. Iterative Q P4。

P1–P4 是迭代数据聚合阶段，不是人为切成四个等份的训练 checkpoint。当前预设累计 roots 为：

| 阶段 | 累计 roots |
|---|---:|
| P1 | 2,400 |
| P2 | 2,880 |
| P3 | 3,600 |
| P4 | 4,800 |

各阶段新增 roots 和单个 root 的候选 rollout 长度均可能不同，因此实际累计 simulator step calls 也不会按 25% 等距增长。

若完整 P1–P4 都运行成本过高，正文至少保留：

- One-shot Q，预算匹配；
- Iterative Q P4。

### 公平性控制

- 相同 state/action representation；
- 相同 candidate-action generator；
- 相同 Q-target；
- 相同网络结构、优化器和 early stopping；
- 相同 future variant；
- 相同门控和 intervention budget；
- One-shot matched 使用与 P4 完全相同的 \(B_{4800}\) 上限和计数口径；
- P1–P4 保持各自真实的数据聚合过程，不为了形成 25%/50%/75% checkpoint 而改变 root 分配。

### 产出

- **Table 4：Iteration ablation**
  - 数据来源；
  - 阶段和累计 roots；
  - candidate rollouts；
  - 实际累计 simulator calls；
  - Total cost；
  - Vent；
  - Stored；
  - Win/loss vs Greedy。
- **Supplementary Figure S2**
  - 横轴优先使用实际累计 simulator calls，并按真实数值进行非等距放置；
  - 纵轴为 validation total cost；锁定 P4 后可叠加各阶段一次性 test 评估作为描述性结果，但不得据此选阶段；
  - 该图用于说明 Iterative Q 内部的数据聚合过程，不要求 PPO 和 Event-Residual PPO 提供对应的 P1–P4 checkpoint。

### 注意事项

- 不能只比较 G0-small 与 P4 后断言“iteration有效”，因为两者数据量不同；
- Test seeds 不得用于选择 P1–P4；
- P1–P4 的根数不是各占 25%，不应称为 25%/50%/75%/100% 训练进度；
- 如果 matched-budget One-shot 仍明显较差，才能较强地支持“状态分布扩展”解释；
- 若差异不明显，应将 iteration 描述为训练稳定性或覆盖机制，而不是必然的性能增益。

---

## E3. Future information 消融

### 目的

回答未来信息是否有用，以及低维物理摘要是否比直接输入完整未来序列更适合当前数据规模。

### 最小正式比较

1. Iterative Q，State-only；
2. Iterative Q，24/72 h summary。

### 辅助比较

3. Iterative Q，full 168 h sequence；
4. 若论文声称现实部署能力，再加入 noisy/predicted 24/72 h summary。

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

- **Table 5：Future-information ablation**
  - information used；
  - representation dimension；
  - parameter count；
  - Total cost；
  - Vent；
  - Stored；
  - Δ cost vs State-only。
- 可选 **Supplementary Figure S3**
  - 不同表示相对 State-only 的配对成本差及 95% CI。

### 注意事项

- “完整未来反而更差”不等于未来没有信息价值；
- 合理解释包括有限 roots、高维冗余、样本效率和 Q-gate 校准，但除非有直接证据，不应将其中任何一个写成已证明机制；
- 如果 24/72 h summary 使用真实未来，应称为 idealized/perfect-forecast ablation；
- 主结论可以是“结构化摘要在当前数据规模下更有效”，不能泛化成“摘要永远优于序列模型”。

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
- Failure trajectory 可从 High stress 中选择预先定义的最差配对 seed，放 Supplementary Figure S4。

---

## 支持性分析 A1. 训练与在线计算成本

### 目的

从 E1–E3 的日志中汇总不同方法在训练阶段和部署阶段的计算差异，不将 Iterative Q 的大量 counterfactual rollouts 隐藏在“roots数量”之后。本项不需要额外训练模型，也不作为独立性能实验。

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
- Event-Residual PPO 按每个高层 transition 内实际推进的 physical hours 计入，不能直接用 high-level timesteps；
- Q 的每个候选动作后续 rollout 都要计入；
- 三种学习方法均需报告是否遵守 \(B_{4800}\) 上限及实际使用比例；
- CPU-hours、GPU-hours和wall time应分开；
- 不同硬件上的 wall time不可直接比较，必须同时报告硬件；
- 即使环境交互预算匹配，也只有在展示性能随累计 simulator calls 的曲线时才能使用“sample-efficient”表述。

---

## E5. Full-horizon MILP 离线参考

### 目的

在相同三船场景上运行一个使用完整未来信息的 Full-horizon MILP，记录有限计算预算内能够达到的可行解质量和最优性界限。该实验保留，但不预设能够求到 optimal，也不作为在线控制器与 E1 六种方法直接排名。

### 设置

- 使用 perfect foresight；
- 使用与 simulator 一致的经济参数；
- 只优化船舶调度，并在模型中执行与 simulator 相同的最大可行井注入规则；
- 使用与其他方法相同的 compact trip cleanup terminal value；
- 使用与正式测试集相同的场景 seeds；若全部 seeds 的计算成本不可接受，必须在求解前锁定一个代表性 seed 子集；
- 预先固定 time limit，例如 30 min、1 h 或一个可承受的多小时预算；
- 不持续求解到 optimal；
- 对每个 seed 报告 incumbent、best bound、MIP gap、termination status 和 solve time；
- 所有 incumbent 必须经过 simulator replay，记录 replay total cost、vent、stored 和 mismatch；
- 没有经过验证的整数可行解时，不报告控制性能，只报告求解状态和 bound。

### 产出

- **Table 7 或 Supplementary Table S3：Time-limited full-horizon MILP reference**
- 是否放正文由结果决定：
  - 若多数 seeds 有有效可行解，且 bound/gap 能提供有意义参照，则在正文 Section 6.8 展示；
  - 若可行率低或 gap 很大，完整结果放 Supplementary，正文用一段话报告计算局限；
  - 即使没有有效可行解，也保留实验记录，不从实验计划中删除。

### 解释规则

- feasible incumbent 是成本最小化问题中当前已找到的可行上界，不等于最优解；
- solver best bound 是下界，只有 gap 足够小时才接近最优；
- Iterative Q 或 Rolling MILP 优于一个超时 incumbent，不代表优于理论最优；
- 若 simulator replay 与 MILP 内部 objective 不一致，以 simulator replay 结果作为执行性能；
- **禁止称为 oracle**，除非相关 seed 全部获得最优性证明。

---

## 7. 图表与数据产物总清单

### 正文建议图表

| 编号 | 内容 | 来源 |
|---|---|---|
| Figure 1 | CCS 系统拓扑和运输—terminal—注入耦合 | 物理模型 |
| Figure 2 | Iterative Action-Q 训练与在线执行流程 | 方法 |
| Table 1 | 物理、经济和 disturbance 参数 | Experimental setup |
| Table 2 | 控制器信息权限、决策频率、训练/求解设置 | Fairness protocol |
| Table 3 | 六个在线控制器主结果 | E1 |
| Figure 3 | 配对成本差和成本分解 | E1 |
| Figure 4 | Representative trajectory | E1 解释性分析 |
| Table 4 | Iteration ablation | E2 |
| Table 5 | Future-information ablation | E3 |
| Figure 5 | Low/Medium/High 综合 stress test | E4 |
| Table 6 | 训练和部署计算成本 | 支持性分析 A1 |
| Table 7 | Full-horizon MILP 结果（若结果足以进入正文） | E5 |

### Supplementary

| 编号 | 内容 |
|---|---|
| Table S1 / Figure S1 | Simulator verification |
| Figure S2 | P1–P4 的实际累计 simulator-call 曲线 |
| Figure S3 | Future representation 配对差 |
| Table S2 | Low/Medium/High 完整 robustness 数值 |
| Figure S4 | Failure trajectory |
| Table S3 | Full MILP 完整结果；Table 7 不进入正文时使用 |
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

1. **冻结物理层和经济目标**，完成 E0；
2. **冻结训练、验证和测试 seeds**；
3. **确认信息协议**：perfect forecast 还是 realistic/noisy forecast；
4. **实现并验证统一 simulator-step counter**，完成 PPO/Event-Residual PPO 目标对齐和 Rolling MILP Greedy-only warm start；
5. 使用单训练 seed 完成六种控制器的 smoke test，并确认 Rolling MILP 无 incumbent 时能明确终止和记错；
   - 实现进度：统一 runner 已完成 Fixed-Assignment、Greedy、Rolling MILP 和 Full-horizon MILP 的单 validation-seed 联调；Rolling MILP 已验证 Greedy-only warm start、无 shifted warm start、无 fallback、限时 incumbent 执行和 replay 记录。三种学习方法需在目标对齐后的 checkpoint 可用后接入同一 runner。
6. 完成 E2 和 E3 的单 seed 筛选，锁定 Iterative Q P4 配置；
7. 运行 4,800-root Iterative Q，测得并记录 \(B_{4800}\)；
8. 在 \(\le B_{4800}\) 下训练目标对齐的 Centralized PPO 和 Event-Residual PPO，并锁定 validation-best checkpoint；
9. 对三种学习方法训练至少 3 个独立训练 seeds，每个 run 均遵守相同上限；
10. 运行 E1 正式主比较；
11. 从 E1 自动选择 representative seed，生成 Figure 4；
12. 对冻结模型运行 E4 的 Low 和 High；Medium 复用 E1；
13. 使用预先锁定的时限和 seeds 运行 E5 Full-horizon MILP；
14. 汇总支持性分析 A1 的训练与在线计算成本；
15. 根据 E5 的可行率和 MIP gap 决定使用正文 Table 7 还是 Supplementary Table S3；
16. 一次性生成所有正文和 Supplementary 图表。

---

## 9. 正式实验开始前的锁定清单

- [ ] 主方法和 baseline 名称已固定；
- [x] Fixed-Assignment 与 Greedy 的行为不重复：三船正式场景的 5 个 validation seeds 中，两者在可行动状态的决策分歧率为 42.76%，Fixed-Assignment 始终保持一船一 emitter，而 Greedy 会跨 emitter 调度；
- [x] 总成本公式、经济参数和 penalty 已在机器可读协议中固定；
- [x] 共同 compact trip cleanup terminal value 已固定；
- [x] 共享的最大可行井注入函数已实现，正式协议下所有控制器的上层动作空间均已移除井注入率；
- [x] Rolling MILP 和 Full-horizon MILP 已使用相同自动注入规则，不包含额外井控制自由度；
- [ ] Forecast 来源、误差和可见范围已固定；
- [ ] 三种学习方法的 24/72 h summary 字段、horizon 和归一化已统一锁定；
- [ ] Rolling MILP 的经济目标、horizon、replan interval、time limit、Greedy-only warm start 和无 fallback 失败规则已固定；
- [ ] Full-horizon MILP time limit、测试 seed 集和 replay 验证规则已固定；
- [ ] Low/Medium/High stress 参数已在验证集检查并锁定；
- [ ] Q 门控、window 和最大 intervention 数已固定；
- [ ] Centralized PPO 已移除 stored-credit/额外塑形项，保留共同经济成本，并固定 action mask、\(\gamma=1\) 和 deterministic evaluation；
- [ ] Event-Residual PPO 已采用同一经济目标从头训练；原 tail-robust v4 仅作为可选补充结果；
- [ ] 4,800-root Iterative Q 的 \(B_{4800}\) 已由统一计数器测得，PPO 和 Event-Residual PPO 的每个训练 run 均设置相同上限；
- [x] 训练、验证、legacy-development 和正式测试 seed 范围已写入 manifest；
- [ ] 训练模型不能读取测试 seeds；
- [ ] 若 4,800 roots 曾根据当前正式 test 结果确定，则这些 seeds 已降级为 development，另建未触碰的正式 test set；
- [x] Simulator step accounting 已实现：在 `PhysicalSimulator.step()` 成功推进后统一累计 calls 与 simulated hours；深拷贝的 Q root/candidate 共享同一计数器，数据集 metadata 和 summary 均记录实际用量；
- [ ] 所有方法能输出同一套 per-seed metrics；
- [ ] 统计脚本已在 toy data 上验证；
- [ ] 轨迹案例的自动选择规则已实现；
- [ ] 所有输出目录禁止静默覆盖。
