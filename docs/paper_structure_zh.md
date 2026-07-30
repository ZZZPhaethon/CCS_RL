# 三船 Unified-Window CCS 调度论文整体架构

## 1. 论文定位

### 1.1 建议的一句话论点

> 在具有天气、capture 和注入能力扰动的三船 CCS 运输—封存系统中，本文构建一个物理约束的小时级仿真环境，并研究 Iterative Action-Q 是否能够通过迭代扩展策略访问状态、学习少量高价值干预，在较低在线决策开销下减少总成本和 CO₂ venting。

这是一条待实验验证的论点，而不是预设结论。最终摘要和标题应根据正式结果调整。

### 1.2 论文贡献边界

建议将贡献控制在三项：

1. 一个统一描述 emitter、船舶、terminal、well 和 reservoir 运行耦合的物理约束仿真与扰动协议；
2. 一个以 Greedy 为安全默认、通过 counterfactual rollouts 和 iterative state aggregation 学习稀疏修正的 Iterative Action-Q 方法；
3. 在统一三船 `unified_window_v1` 协议下，对启发式、PPO 类控制器和 Rolling MILP 的公平比较，并使用限时 Full-horizon MILP 提供条件性的离线参考，以及分析 iteration、未来信息和扰动分布变化。

不建议在同一篇论文中同时声称：

- 提出了最优 forecast encoder；
- 证明了 MILP 的理论性能上界；
- 解决了任意规模 CCS 网络；
- 证明了现实现场部署性能。

这些内容超出当前实验边界。

---

## 2. 论文整体结构

```text
Abstract
1. Introduction
2. Related Work
3. Physical CCS System and Simulation Environment
4. Iterative Action-Q
5. Experimental Design
6. Results
7. Discussion
8. Conclusion
Supplementary Information
```

各章节的任务不是简单介绍内容，而是形成以下证据链：

```text
现实运行问题
    → 为什么需要统一的物理仿真和动态控制
    → Iterative Action-Q 如何产生决策
    → 比较是否公平
    → 是否优于强基线
    → 性能来自哪里
    → 在哪些扰动下有效或失效
```

---

## Abstract

### 写作顺序

1. 船运 CCS operation 需要协调多 emitter、船舶、terminal buffer 和注入能力；
2. 天气、产量和 well availability 使固定调度或频繁优化面临困难；
3. 本文构建物理约束仿真并提出 Iterative Action-Q；
4. 给出主结果中的最强定量证据；
5. 说明 iteration 或未来摘要的关键发现；
6. 用三船和当前扰动协议限定结论边界。

### 不应出现

- 尚未得到结果时的具体提升百分比；
- “globally optimal”“oracle”；
- “适用于所有 CCS 网络”；
- 将 perfect forecast 结果描述为现实预测性能。

### 图表

摘要不放图表。

---

# 1. Introduction

## 1.1 Field-scale context

说明船运 CCS 链条的运行特点：

- 多个 emitter 持续产生 CO₂；
- 船舶容量和航行时间有限；
- terminal buffer 连接不连续运输和连续注入；
- venting 和未完成封存产生经济与环境代价。

## 1.2 Operational bottleneck

建立核心矛盾：

- 船舶到达延误会造成 emitter inventory 上升；
- 注入能力下降会造成 terminal congestion；
- 上游、运输和下游扰动通过库存相互传播；
- 控制器需要在不确定环境中协调而不是孤立优化单个环节。

## 1.3 Existing approaches and gap

按技术类别组织，而不是逐篇列论文：

1. 固定分配和 Greedy heuristics：快速、可解释，但依赖局部规则；
2. MILP/rolling optimization：可显式处理约束，但依赖预测并存在求解开销；
3. PPO 类在线学习：推理快，但在大离散动作空间中可能不稳定或偏离强启发式；
4. Offline/counterfactual action evaluation：可以利用仿真产生长期动作价值，但需要解决策略状态分布偏移。

最终收束到缺口：

> 仍缺少一种在保留强启发式安全默认的同时，利用物理仿真评估少量长期干预，并逐步覆盖自身访问状态的控制方法。

## 1.4 Present study

用一段话说明：

- 本文研究三船 fixed-network CCS operation；
- 采用统一物理仿真和 `unified_window_v1`；
- 提出 Iterative Action-Q；
- 与 Fixed-Assignment、Greedy、Maskable PPO、Event-Residual PPO 和 Rolling MILP 比较，并运行限时 Full-horizon MILP 离线参考；
- 通过 iteration、future information 和 disturbance stress tests 分析原因与边界。

### 本节图表

不建议放结果表。若期刊允许，可在 Introduction 末尾引用 **Figure 1**，但图本身放在 Section 3。

---

# 2. Related Work

## 2.1 CCS transport and storage operation optimization

介绍：

- 船运 CCS logistics；
- terminal inventory coupling；
- injection capacity constraints；
- static 与 rolling-horizon optimization。

重点指出本文研究的是固定网络的小时级 operation，而不是基础设施选址或长期船队规模规划。

## 2.2 Reinforcement learning for constrained scheduling

介绍：

- centralized RL；
- action masking；
- event-based decision making；
- residual/safe RL；
- 长时域延迟代价。

## 2.3 Offline action-value learning and iterative data aggregation

介绍：

- counterfactual action evaluation；
- distributional Q/value estimation；
- policy-induced distribution shift；
- iterative data aggregation。

## 2.4 Forecast information in operational control

简要讨论：

- 完整预测序列；
- 低维结构化摘要；
- perfect forecast 与预测误差；
- 未来信息维度与样本效率的潜在冲突。

### 写作注意

Related Work 只解释技术背景和研究缺口，不描述本文代码版本，也不在这里报告结果。

---

# 3. Physical CCS System and Simulation Environment

这一节必须在算法之前，使读者先理解控制对象、状态转移和可信性。

## 3.1 System topology and operational scope

介绍：

- 3 emitters；
- 3 vessels；
- 1 terminal；
- injection wells/reservoirs；
- 小时级决策；
- 固定网络，不优化基础设施规模。

### Figure 1：System topology and operational coupling

建议包含：

- Emitters → vessels → terminal → wells/reservoirs；
- 每个节点的库存；
- 船舶航线和容量；
- capture、loading、unloading、injection 流；
- weather、capture、maintenance 扰动入口；
- controller 读取 state 并输出 action 的接口。

图后正文解释 terminal inventory 为什么是运输和注入之间的关键耦合状态。

## 3.2 State transition and mass balance

定义：

\[
s_{t+1}=f(s_t,a_t,\xi_t),
\]

其中：

- \(s_t\)：库存、船舶模式、位置/ETA、well 状态、压力；
- \(a_t\)：上层控制器选择的船舶调度动作；
- \(\xi_t\)：外生 capture、weather 和 maintenance；
- \(f\)：满足质量守恒和容量约束的 simulator transition。

正文给出核心质量平衡方程。详细实体级更新可放 Supplementary Methods。

## 3.3 Vessel, terminal and injection constraints

介绍：

- 船舶容量；
- 满载/卸载和航行状态；
- berth 与 FIFO unload queue；
- loading/unloading rate；
- terminal buffer；
- well availability、injectivity 和 pressure constraints；
- 共享的底层井控制器：井可用时按照 terminal inventory、equipment、injectivity 和 pressure constraints 下的连续最大可行速率自动注入，维护时注入率为零；
- 井注入率不属于任何上层控制器的动作空间；
- legal-action mask。

这里应明确建模边界：当前问题不考虑时变注入电价、注入启停/爬坡，或主动降载促进压力恢复的控制收益。注入运行成本仍按 simulator 实际自动注入量计入总成本。

## 3.4 Disturbance and forecast protocol

说明 `unified_window_v1`：

- capture Gaussian noise；
- capture outage；
- high-output event；
- weather window 和 speed factor；
- well maintenance；
- initial inventory/pressure randomization；
- 168 h forecast context。

明确 forecast 是：

- perfect forecast；或
- predicted/noisy forecast。

该选择必须与实验计划一致。

## 3.5 Economic objective and evaluation boundary

定义：

\[
J =
C_{\mathrm{operating},\,0:720}
+ C_{\mathrm{vent},\,0:720}
+ C_{\mathrm{cleanup}}(s_{720})
+ C_{\mathrm{other\ locked\ penalties}},
\]

并分别解释：

- vessel fuel；
- conditioning/reconditioning；
- loading/unloading；
- vent penalty；
- common compact trip cleanup operating cost；
- 是否存在 storage shortfall penalty。

说明单位成本的分母，以及共同末端处理：720 h 后停止新增 capture、关闭 cleanup 扰动，将剩余 CO₂ 完成封存；该 cleanup cost 同时进入三种学习方法的 terminal return、Rolling/Full-horizon MILP 的 terminal objective 和所有方法的报告总成本。

## 3.6 Simulator verification

简要报告：

- 质量守恒；
- 容量和压力边界；
- 状态机；
- 扰动响应；
- MILP action replay。

### Table 1：Physical, economic and disturbance parameters

放在 Section 3.4 或 3.5 之后，包括：

- 网络容量和速率；
- 航行距离/名义时间；
- 经济参数；
- `unified_window_v1` 参数。

完整参数放 Supplementary Table。

### Supplementary Table S1 / Figure S1

放 E0 的详细验证结果。

---

# 4. Iterative Action-Q

这一节只详细介绍论文主方法。不要在这里逐个展开所有 baseline。

## 4.1 Method overview

先说明动机：

- Greedy 提供稳定可行的默认动作；
- 少量关键时刻可能存在更优长期选择；
- 单纯在 Greedy 状态上学习会遇到 policy-induced distribution shift；
- 因此使用候选动作长期 rollout 和 iterative roll-in。

### Figure 2：Iterative Action-Q pipeline

建议画成训练和部署两部分：

```text
Training:
Greedy roll-in G0
    → sample roots
    → enumerate feasible candidate actions
    → roll each action to episode end
    → long-term cost-improvement targets
    → train P1
    → P1 roll-in G1
    → ...
    → P4

Deployment:
Current state + optional future summary
    → multi-head distributional Q
    → compare candidate with FOLLOW
    → confidence/margin gate
    → override or execute Greedy
```

## 4.2 Root states and candidate actions

介绍：

- root 的定义和采样时间；
- roll-in policy；
- feasible joint actions；
- FOLLOW/default action；
- action masks；
- 每个 candidate 从同一 root 开始，避免起始状态差异。

## 4.3 Counterfactual long-horizon target

定义长期目标：

\[
y(s,a) = \alpha\left[J_{\mathrm{default}}(s)-J_a(s)\right],
\]

说明：

- \(y>0\) 表示候选动作优于默认动作；
- rollout 到 episode 结束；
- 使用 simulator future outcome 作为训练阶段 privileged supervision；
- 在线部署时不需要执行这些 rollouts。

## 4.4 Iterative state aggregation

解释 G0–G3 和 P1–P4：

- G0：Greedy roll-in；
- G1：P1 访问状态；
- G2：P2 访问状态；
- G3：P3 访问状态；
- P4：累计数据训练。

明确累计数据比例、roots 数量和训练/验证划分。详细数字可放 Implementation Details。

## 4.5 State, action and optional future representations

介绍：

- 当前系统 state；
- vessel mode/destination；
- Greedy proposal；
- episode progress；
- structured joint-action embedding；
- candidate actions 只包含船舶调度；井注入由 Section 3.3 的共享底层控制器决定；
- State-only 和 24/72 h future summary；
- full sequence 仅作为消融，不必作为主结构。

## 4.6 Distributional multi-head estimation and deployment gate

介绍：

- quantile/distributional Q；
- bootstrap heads；
- 候选相对 FOLLOW 的预测收益；
- head agreement；
- margin threshold；
- intervention windows 和最大次数；
- 无足够置信度时回退 Greedy。

## 4.7 Training and online complexity

区别：

- 训练阶段的 root/candidate rollouts；
- 网络训练；
- 在线阶段单次前向传播；
- 无在线规划求解。

### Algorithm 1

正文或 Supplementary 给出 Iterative Action-Q 训练伪代码。

### Algorithm 2

给出在线 gate 和 action execution 伪代码。

---

# 5. Experimental Design

本节说明“如何保证比较可信”，不报告结果。

## 5.1 Research questions

列出：

- RQ1：主方法是否优于在线基线？
- RQ2：iteration 是否有独立贡献？
- RQ3：未来信息及其表示如何影响结果？
- RQ4：只在 Medium stress 下训练的模型能否适应 Low 和 High 综合扰动强度？
- RQ5：限时 Full-horizon MILP 能否提供有意义的可行解或最优性界限？

训练与在线计算成本作为支持性报告，不单独设置研究问题。

## 5.2 Scenario splits and repetitions

介绍：

- 训练、验证和测试 seeds；
- 30 个 paired test seeds；
- 学习方法的训练随机种子数量；
- checkpoint selection；
- Low/Medium/High stress-test seeds 与参数；
- 所有正式配置在测试前锁定。

## 5.3 Compared online controllers and offline reference

### Fixed-Assignment Heuristic

正文约一个短段落：

- emitter 到 vessel 的固定映射；
- 满载返回 terminal；
- 固定服务对象无货时的等待/备用规则；
- 调用共享的最大可行井注入底层控制器；
- 不使用 forecast。

不要称为通用“Rule-based”，以免与动态规则混淆。

### Greedy

正文约一个短段落：

- 满载船优先返回 terminal；
- 空闲/空船根据当前 inventory 和有效 capture supply score 选择 emitter；
- 当前 emitter 可继续装载时等待；
- 井注入由共享底层控制器自动执行，不属于 Greedy 决策；
- 不使用 forecast。

### Centralized Maskable PPO

正文约一个段落：

- centralized observation；
- 只包含船舶调度的 joint MultiDiscrete action，不包含井注入率；
- legal-action masking；
- 使用与论文总成本一致的目标对齐 reward：`injection_reward=0`、`store_reward=0`，移除额外 stored-credit/塑形项，保留共同的 vent、operation 和必要约束 penalty；
- \(\gamma=1\)，避免有限 720 h 经济目标被额外时间折扣改变；
- future input；
- deterministic evaluation；
- 在 \(B_{4800}\) simulator-call 上限内从头训练，使用 validation-best checkpoint；
- 实际 simulator calls、训练 timesteps 和 seeds。

不重新推导标准 PPO；引用原算法，详细超参数放 Supplementary。

### Event-Residual PPO

正文一至两个段落，因为它不是标准 PPO：

- Greedy 产生默认动作；
- PPO 选择 FOLLOW 或 override；
- event trigger/decision interval；
- action mask；
- future summary；
- intervention windows/gate；
- residual action 只改变船舶调度，不控制井；
- 与 Centralized PPO 和 Iterative Q 对齐的经济 reward，移除原 v4 的 stored credit 和 tail-robust shaping；
- 目标对齐版本从头训练；原 tail-robust v4 只作为可选补充消融；
- curriculum、failure replay 和 checkpoint selection 的核心原则。

论文中不使用“v4”作为方法名。

### Iterative Action-Q

只引用 Section 4，不在实验设置中重复方法。

### Rolling MILP

正文约一个段落：

- 每 24 h 重规划；
- 168 h horizon；
- 使用统一 forecast；
- 目标和约束与 simulator 对齐；
- 只优化船舶调度，并在 planning transition 中复现 simulator 的最大可行井注入规则；
- 每次重规划使用从当前状态生成的完整、合法且 replay-valid Greedy 计划作为唯一 MIP start；
- 不使用 Native MPC 候选，也不使用上一次 MILP 计划平移的 warm start；
- 只执行第一个 24 h；
- 固定 time limit；
- incumbent replay；
- 无有效 incumbent 时记录 solver failure 并终止该 episode，不使用在线 fallback。

变量、约束和 solver 细节放 Supplementary Methods。

### Full-horizon MILP reference

正文约一个段落：

- 使用完整 720 h perfect foresight；
- 使用与 simulator 对齐的目标和约束；
- 与在线控制器相同，只优化船舶调度并使用确定性最大可行井注入规则；
- 预先固定 time limit 和测试 seeds；
- 记录 incumbent、best bound、MIP gap、solve time 和 termination status；
- 对整数可行 incumbent 执行 simulator replay；
- 不参与六个在线控制器的直接排名。

实验本身保留，但结果位置由求解质量决定：有意义的可行解和 bound 可进入正文；可行率低或 gap 很大时，完整表格放 Supplementary。

### 消融方法

用一段话定义：

- One-shot matched-budget 只移除 iterative roll-in；
- State-only 只移除 future summary；
- full-sequence 只更改未来表示；
- 其余设置保持一致。

## 5.4 Information and computational fairness

说明：

- 相同当前状态；
- 相同控制边界：所有方法只控制船舶调度，井由统一底层控制器自动操作；
- 相同 forecast 可访问性；
- 方法可以采用不同内部表示；
- 三种学习方法在 E1 中统一使用由同一 168 h forecast 计算的 24/72 h summary；Rolling MILP 使用同源逐小时序列；
- summary 字段、horizon 和归一化在 validation 上统一锁定，不能针对每种学习算法分别选择；
- 先测量 4,800-root Iterative Q 数据生成实际消耗的底层 1 h simulator calls，定义为 \(B_{4800}\)；
- Iterative Q P4、Centralized PPO 和 Event-Residual PPO 的每个独立训练 run 使用同一 \(B_{4800}\) 最大环境交互预算；
- Vectorized PPO 汇总所有 workers 的 hourly steps；Event-Residual PPO 汇总高层 transition 内实际推进的 physical hours；Iterative Q 汇总所有 roll-in 和候选 rollout；
- validation/test、网络重复 SGD 和纯特征计算不计入 \(B_{4800}\)，但超参数搜索与 CPU/GPU 成本另外报告；
- 最终性能比较不自动等于 sample-efficiency 比较；
- Full-horizon MILP 的 perfect foresight 和离线计算预算必须与在线控制器分开标注。

### Table 2：Controller/reference protocol and fairness

建议列：

| Method | Control scope | Current state | Forecast available | Forecast used | Decision timing | Training/simulator-call cap | Solver-failure protocol |
|---|---|---|---|---|---|---|---|

Full-horizon MILP 单独标为 `offline`, `perfect foresight`, `no online fallback`，避免读者将其与六个在线控制器混为同一运行模式。

## 5.5 Metrics and statistical analysis

主指标：

- Total cost；
- Total cost/stored t；
- Vent；
- Stored；
- Operating cost；
- Terminal cleanup operating cost；
- paired difference vs Greedy；
- decision/episode wall time。

统计：

- mean/std/median；
- paired bootstrap 95% CI；
- win/loss；
- 多训练 seed 时使用分层 bootstrap。

---

# 6. Results

Results 按“主结果 → 运行机制 → 因果消融 → 综合 stress 泛化 → 计算与优化边界”排列。

## 6.1 Simulator credibility

用一段正文概括 E0：

- 所有质量平衡和边界测试通过；
- 扰动正确作用；
- controller action replay 可执行。

详细结果指向 Supplementary Table S1 和 Figure S1。

如果物理仿真本身是主要贡献，可将这一小节扩展；否则保持简洁。

## 6.2 Main online-controller comparison

开头句式：

> To test whether Iterative Action-Q improves closed-loop CCS operation, we evaluated all online controllers on identical disturbance trajectories.

### Table 3：Main online-controller comparison

放在本小节开头或第一段之后。

列：

- Total cost；
- Δ vs Greedy，95% CI；
- Total cost/stored t；
- Operating cost；
- Terminal cleanup operating cost；
- Vent；
- Stored；
- win/loss。

### Figure 3：Paired performance and cost mechanism

- Panel a：相对 Greedy 的 paired cost difference 和 CI；
- Panel b：720 h operating cost、vent penalty 与 common terminal-cleanup operating cost 分解；完整 vessel fuel、conditioning、reconditioning、loading 和 unloading 分项放 Supplementary。

正文解释：

- 主方法是否稳定优于 Greedy；
- 成本下降来自运营成本还是 vent penalty；
- PPO 与 Rolling MILP 的相对位置；
- 哪些差异区间跨零。

不要只根据均值排序写“显著优于”。

## 6.3 Operational interpretation of the main comparison

本小节属于 E1 主比较的解释性分析，不是独立实验。它紧接主结果，而不是机械放在 Results 最后。

### Figure 4：Representative trajectory

使用中位改善 seed，展示：

- emitter inventories；
- terminal inventory；
- cumulative vent；
- disturbance intervals；
- vessel modes/destinations；
- Iterative Q intervention times。

本小节回答：

> Iterative Q 在什么状态下改变了 Greedy，并如何避免后续 vent 或 terminal congestion？

只能描述轨迹直接支持的现象，不能从单个 seed 推广普遍机制。

## 6.4 Effect of iterative state aggregation

### Table 4：Iteration ablation

比较：

- One-shot original；
- One-shot matched to \(B_{4800}\)；
- Iterative Q P1–P4，其中累计 roots 分别为 2,400、2,880、3,600 和 4,800。

正文重点：

- matched-budget One-shot 与 P4 的差异；
- iteration 是否改善访问状态覆盖；
- 性能和实际累计 simulator calls 的关系。

P1–P4 是不等长的数据聚合阶段，不应解释为 25%/50%/75%/100% checkpoint。E1 只使用最终 P4；可选 Supplementary Figure S2 以真实累计 simulator calls 为横轴进行非等距展示。PPO 类方法无需生成与 P1–P4 对齐的 checkpoint。

## 6.5 Role of future information

### Table 5：Future-information ablation

主比较：

- State-only；
- 24/72 h summary。

附录比较：

- full 168 h sequence；
- 其他 encoder 筛选。

正文回答：

- 摘要是否在配对测试中提供稳定增益；
- 完整序列是否退化；
- 结论是否只适用于当前数据规模；
- forecast 是否为 perfect。

## 6.6 Robustness to unseen disturbance severity

### Figure 5：Robustness across Low/Medium/High stress

比较：

- Low stress；
- Medium stress，即训练分布和 E1；
- High stress。

Low/Medium/High 同时调整预先锁定的 weather duration/severity、capture high-output 和 well-maintenance duration。Capture outage 和初始库存保持固定，避免“更多 outage 反而减少负荷”和初始状态混杂。

建议纵轴使用 total cost，或同时报告相对 Medium 的成本变化；Supplementary Table S2 给出绝对数值和配对区间。

正文说明：

- 随综合 stress 增强，各方法性能如何退化；
- Iterative Q 的优势是否保持；
- High stress 是否暴露方法边界。

由于多个扰动因素共同变化，不能用该实验判断天气、capture 或 well 中哪一个是唯一原因。

Failure trajectory 可放本小节末尾或 Supplementary Figure S4。

## 6.7 Supporting computational-cost report

### Table 6：Training and deployment cost

列：

- simulator calls；
- CPU/GPU-hours；
- model parameters；
- decision latency；
- episode wall time；
- Rolling MILP timeout、solver failure、Greedy warm-start acceptance 和 MIP gap。

正文应明确：

- 本表由 E1–E3 日志汇总，不需要额外训练或单独实验；
- Iterative Q 在线推理与训练数据生成的成本不同；
- Rolling MILP 使用固定时间预算；
- 三种学习方法匹配的是环境交互上限，不是 CPU/GPU 或 wall-time 预算；
- 只有展示性能随累计 simulator calls 的曲线时才声称 sample efficiency。

## 6.8 Time-limited full-horizon MILP reference

E5 实验必须运行，但不预设能够求到 optimal。

- 若多数 seeds 获得经过 replay 验证的可行解，且 bound/gap 具有解释价值，使用 **Table 7** 在正文展示；
- 若可行率低、gap 很大或仅得到较差 incumbent，完整结果使用 **Supplementary Table S3**，正文仍用一段话报告求解状态和计算局限；
- 若没有有效整数可行解，只报告 solver status、best bound 和 time limit，不报告未验证的控制性能；
- 不称为 oracle，除非相关求解获得最优性证明。

---

# 7. Discussion

## 7.1 Central finding

解释主结果对 CCS operation 的含义：

- Iterative Q 是否能稳定减少高代价 vent；
- 是否保持合理运营成本；
- 是否以稀疏干预改善强 Greedy default。

## 7.2 Why iteration matters

根据 E2 讨论：

- Greedy-only 状态覆盖；
- policy-induced states；
- matched-budget 证据；
- iteration 的收益或限制。

不要将未直接测量的 representation mechanism 写成事实。

## 7.3 Role and limits of forecast information

根据 E3 讨论：

- State-only 的能力；
- 结构化摘要；
- 完整序列退化；
- perfect forecast 与现实预测误差；
- 未来工作是否需要 noisy forecast training。

## 7.4 Optimization versus learned control

讨论：

- Rolling MILP 的预测依赖和求解开销；
- learned policy 的在线速度；
- Iterative Q 的离线 simulator cost；
- 不同方法适合的 operation 条件。

不要写成“RL全面取代MILP”。

## 7.5 Limitations

至少包括：

- 单一三船网络；
- 固定网络拓扑；
- simulator/model assumptions；
- forecast realism；
- 学习方法训练 seed 数量；
- root-generation cost；
- Full MILP 未证明最优；
- stress tests 不是所有扰动组合；
- 真实运行数据或现场验证不足。

## 7.6 Practical implication

只在结果支持时讨论：

- 稀疏干预型控制器可能适合作为强规则控制器的决策支持层；
- 在低置信度时回退 Greedy；
- forecast 不稳定时 State-only 策略可能具有价值。

---

# 8. Conclusion

按四句话组织：

1. 本文构建/采用了什么物理约束 CCS operation framework；
2. 提出了什么 Iterative Action-Q 方法；
3. 哪一项正式实验结果构成决定性证据；
4. 结论仅适用于什么网络、扰动和 forecast 条件。

不引入新的算法、结果或未来承诺。

---

# Supplementary Information

## S1. Full simulator equations and validation

- 实体级质量平衡；
- 船舶状态机；
- terminal queue；
- pressure/injectivity；
- 最大可行井注入底层控制器及跨方法一致性测试；
- E0 详细结果。

## S2. Baseline implementation details

### Fixed-Assignment

- 完整 emitter-vessel mapping；
- fallback/tie-breaking；
- pseudocode。

### Greedy

- supply score；
- 动作优先级；
- tie-breaking。

### Maskable PPO

- observation/action dimensions；
- 船舶调度动作定义，并确认不包含井控制；
- network；
- 目标对齐 reward：明确列出 vent、operation、constraint penalty，并说明 stored-credit/额外 shaping 已移除；
- learning rate、gamma、batch、timesteps；
- \(B_{4800}\) 计数和提前停止规则；
- seeds 和 checkpoint selection。

### Event-Residual PPO

- event triggers；
- residual action codec；
- residual action 不包含井控制；
- 与主经济目标对齐的 reward，以及相对原 tail-robust v4 被移除的 shaping 项；
- curriculum；
- failure replay；
- gate；
- 高层 transition 内 physical simulator hours 的预算计数；
- checkpoint selection。

### Rolling MILP

- variables；
- objective；
- constraints；
- 船舶调度变量与共享自动井注入状态转移；
- horizon/replan；
- solver/version；
- time limit；
- Greedy-only MIP start 的生成、合法性和 replay 验证；
- 不使用 Native MPC/shifted-plan warm start；
- incumbent replay 和无有效 incumbent 时的 solver-failure/episode-termination 规则。

## S3. Iterative Action-Q implementation details

- 完整 feature list；
- candidate action list；
- network/quantiles/heads；
- G0–G3 roots；
- optimization；
- early stopping；
- gate thresholds。

## S4. Complete results

- 所有 per-seed 表；
- 训练随机种子结果；
- additional encoder screening；
- stress-test 数值；
- failure trajectories；
- Full-horizon MILP 的 incumbent、bound、gap、status 和 replay 结果。

---

# 图表位置总览

| 图表 | 放置位置 | 作用 |
|---|---|---|
| Figure 1：系统拓扑 | Section 3.1 | 解释物理对象和耦合 |
| Table 1：参数 | Section 3.4–3.5 | 固定物理、经济和扰动协议 |
| Figure 2：Iterative Q pipeline | Section 4.1 | 解释训练和部署 |
| Table 2：控制器公平协议 | Section 5.4 | 说明信息和预算差异 |
| Table 3：主结果 | Section 6.2 | 支持核心性能主张 |
| Figure 3：paired difference/cost decomposition | Section 6.2 | 展示稳定性和成本来源 |
| Figure 4：代表性轨迹 | Section 6.3 | 解释 E1 主比较的运行机制 |
| Table 4：iteration ablation | Section 6.4 | 支持 iterative state aggregation |
| Table 5：future ablation | Section 6.5 | 解释未来信息作用 |
| Figure 5：Low/Medium/High robustness | Section 6.6 | 展示综合扰动强度边界 |
| Table 6：计算成本 | Section 6.7 | 支持性汇总训练—部署 trade-off |
| Table 7 或 Table S3：Full-horizon MILP | Section 6.8 或 Supplementary | 根据可行率和 gap 决定位置 |
| Supplementary S1–S4 | 对应附录 | 保证复现且不挤占正文 |

---

# 写作时必须保持一致的术语

| 工程名称 | 论文名称 |
|---|---|
| rule based 固定分配 | Fixed-Assignment Heuristic |
| greedy_shuttle_policy | Greedy |
| traditional PPO | Centralized Maskable PPO |
| event-based v4 | Event-Residual PPO |
| iterative Q | Iterative Action-Q |
| non-iterative Q | One-shot Action-Q |
| rolling milp | Rolling MILP |
| milp | Time-limited Full-horizon MILP Reference |
| 完整未来真值 | Perfect forecast |
| 24/72 h特征 | Structured future summary |

正文中避免混用“v4”“P4模型”“新摘要”等工程内部称呼。P1–P4 只在解释 iterative training stages 时使用。

---

# 论文完成前的主张—证据检查

| 论文表述 | 必须有的证据 |
|---|---|
| “Outperforms Greedy” | 配对差 95% CI 和胜负场景 |
| “Iteration improves performance” | matched-budget One-shot Q 消融 |
| “Future information helps” | State-only vs summary 的正式配对结果 |
| “Summary is better than full sequence” | 相同数据与协议下的表示比较 |
| “Robust to disturbances” | 冻结模型的未见强度 stress tests |
| “Fast online control” | 单次决策和 episode wall time |
| “Sample-efficient” | 预算匹配 learning curve；否则不能写 |
| “Near-optimal” | 经过证明的 MILP optimum 或足够紧的 bound；否则不能写 |
| “Operationally feasible” | 零硬违规和 simulator replay |
