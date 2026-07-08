# 固定船运型 CCS 网络的小时级自适应运行优化

## 1. 研究核心思想

本研究关注的不是长期 CCS 基础设施规划（例如建设多少艘船、多少终端、多少储罐或注入井），而是在**既定基础设施网络**下进行小时级运行优化（operation）。

研究目标是构建一个满足物理与运营约束的 CCS operational digital twin，并在其中训练强化学习（RL）策略，使其能够根据系统实时状态及不确定扰动，自适应地调整船舶运输与 CO₂ 注入操作，从而在满足预设 CO₂ storage-rate target 的前提下，降低运行成本、CO₂ venting、终端拥堵和注入不稳定风险。

可以概括为：

> 在固定的 ship-based CCS network 中，学习一个小时级实时调度 policy，使系统在捕集波动、船期延误、天气影响、终端故障与注入能力变化等条件下，仍能稳定、低成本地完成 CO₂ 运输与地质封存。

---

## 2. 研究边界：本研究做什么、不做什么

### 2.1 固定的基础设施

研究开始前，网络结构与主要资产已经给定，包括：

- emitter（CO₂ 捕集源）的数量、位置、捕集后储罐容量；
- CO₂ 船舶数量、容量、装卸速率与名义航行时间；
- terminal 的位置、泊位数、卸货能力和 buffer tank 容量；
- 注入井数量、最大/最小稳定注入率及井—储层连接关系；
- reservoir 的初始压力、压力上限与注入能力代理模型；
- 航线网络和基本运营成本结构。

### 2.2 研究不直接优化的内容

本研究不以以下问题为主：

- 建设多少船、储罐、terminal 或注入井；
- 网络拓扑和长期资本投资最优设计；
- 完整多年期 CCS infrastructure planning；
- 高保真多相流储层数值模拟本身。

这些内容可以作为后续扩展或情景参数，但不是当前 RL 的主决策对象。

### 2.3 本研究直接优化的 operation 决策

在每个小时，系统需要根据当前真实状态决定：

- 哪些可调度船舶应前往哪个 emitter，或是否等待；
- 何时优先收集某个接近满罐的 emitter 的 CO₂；
- terminal buffer 应如何维持安全库存；
- 哪些注入井应关闭、低速、中速或高速注入；
- 当天气、船期、terminal 或井状态发生扰动时，如何重新调整运输与注入策略。

---

## 3. 系统结构与核心耦合关系

系统结构为：

```text
Emitters → CO₂ vessels → Terminal / buffer tanks → Injection wells → Reservoirs
```

该系统的关键并非单独优化“运输”或“注入”，而是处理两者通过 terminal buffer 产生的双向耦合：

```text
船舶延误 / 捕集量上升
    → terminal 到货不足或上游储罐满溢
    → 注入降速或 emitter venting

注入能力下降 / reservoir 压力接近上限
    → terminal 库存上升
    → 船舶无法卸货、发生等待
    → emitter 无船可服务、进一步发生 venting
```

因此，terminal inventory 是运输—注入系统之间最关键的耦合状态。RL 不能仅根据某个 emitter 当前库存派船，还必须预测或学习到 terminal 库存、在途船舶 ETA、井可用性和未来扰动之间的联动关系。

---

## 4. 为什么需要 RL，而不只使用一次性 MILP

确定性 MILP 可以在已知未来捕集量、航行时间、井能力和设备状态的前提下，生成某一时间范围内的整体最优排程。它适合作为理想化、完美信息条件下的 benchmark。

但真实 operation 中会出现：

- capture rate 波动或 capture plant 临时停机；
- 天气导致航速下降和到港延误；
- terminal berth 或卸货泵暂时不可用；
- 船舶故障、排队与装卸效率下降；
- injection well maintenance；
- reservoir injectivity 下降或压力约束收紧。

一旦这些现实状态偏离初始预测，预先求解出的静态排程可能不再可行。理论上可以每小时使用 rolling-horizon MILP 重算，但随着网络规模、离散任务和不确定性增加，求解时间与建模复杂度可能成为瓶颈。

本研究中的 RL 被定位为：

> 一个在运行时根据当前状态快速生成下一步可执行操作的在线控制策略，而不是替代物理模型或替代所有优化方法。

RL 是否优于 static MILP 或 rolling-horizon MILP 不是预设结论，而是需要通过不同扰动场景下的实验证明或证伪的研究问题。

---

## 5. CCS 仿真器 / Digital Twin 的作用

仿真器是研究的核心基础设施。它不负责“替 RL 决策”，而是提供一个可重复、可随机化、满足物理约束的环境：

```text
Scenario generator
    → 生成天气、捕集波动、延误、故障、井维护等外生情景

CCS simulator
    → 根据当前状态与 action 执行一小时 operation
    → 更新库存、船位、装卸、注入、压力、成本和 venting

RL policy / MILP / heuristic
    → 根据当前状态选择 action
```

其核心状态转移为：

```text
s_(t+1) = f(s_t, a_t, ξ_t)
```

其中：

- `s_t`：当前系统状态；
- `a_t`：控制器给出的操作 action；
- `ξ_t`：外生扰动，如天气、捕集量波动、设备故障；
- `f`：满足质量守恒、容量、泊位、船舶状态和注入约束的仿真器。

仿真器应至少严格满足：

- emitter inventory 质量守恒；
- vessel onboard inventory 质量守恒；
- terminal inventory 质量守恒；
- reservoir cumulative injection 与压力/注入能力更新；
- 船舶容量、装卸速率、泊位数量与航行时间约束；
- terminal buffer 上下限与低库存 throttling；
- 井最大/最小稳定流量、维护状态与 reservoir pressure constraints。

---

## 6. 不确定情景与仿真数据生成

仿真器不是预先生成一张静态数据表，而是持续生成大量不同的运行 episode。每个 episode 例如持续 168 小时（7 天），并具有独立的外部情景。

### 6.1 每个 episode 可随机化的因素

- 初始 emitter inventory 与 terminal inventory；
- 各 emitter 的 capture-rate trajectory；
- capture plant 降负荷或临时停机；
- 天气状态及对应的航速/航行时间影响；
- terminal unload efficiency change；
- 船舶延误或临时不可用；
- well maintenance；
- injectivity decline；
- 运行成本、燃料成本或 storage target 的变化。

这些随机化因素构成 CCS 版本的 operational domain randomization。它们的作用是避免 policy 只适应一个理想、固定且无扰动的系统。

### 6.2 RL 训练数据的形式

RL 的训练经验由 policy 与 simulator 交互时产生：

```text
(state_t, action_t, reward_t, next_state_t, done_t, info_t)
```

其中 action 不由 simulator 随意生成，而是由 heuristic、MILP 或 RL policy 给出。仿真器负责执行 action 并产生其物理后果。

---

## 7. RL 问题定义

### 7.1 Observation / state

RL 在每个小时接收的 observation 应包括：

**Emitter 层面**
- 各 emitter 当前库存与 fill ratio；
- 当前 capture rate；
- 短期 capture forecast 或预测区间；
- 溢出风险。

**Vessel 层面**
- 每艘船的位置、phase、载量与 cargo ratio；
- 当前目的地、剩余航行时间、ETA；
- 装货、卸货、排队、航行或维修状态。

**Terminal 层面**
- terminal inventory 与 fill ratio；
- berth 是否占用；
- 泵效率、terminal availability；
- 距离低库存 throttle threshold 的余量。

**Injection / reservoir 层面**
- 井的可用状态、当前模式、最大可用注入率；
- injectivity factor；
- reservoir pressure、pressure headroom 与累计注入量。

**外部环境与目标层面**
- 当前天气及短期天气预测；
- 当前成本参数；
- cumulative storage rate 与目标 storage rate；
- 已发生的 venting、成本与 injection shortfall。

### 7.2 Action

第一版应采用可解释、有限的高层 action，而不是让 RL 同时控制所有精确流量和底层设备操作。

**船舶 action（离散）**

对于位于可调度状态的船舶：

```text
WAIT / GO_TO_E1 / GO_TO_E2 / GO_TO_E3 / GO_TO_TERMINAL（若适用）
```

**注入 action（离散模式）**

对于每一口井：

```text
OFF / LOW / MEDIUM / HIGH
```

仿真器将每个模式映射为目标流量，并根据终端库存、井能力、压力和 injectivity 等约束裁剪为实际可执行流量。

### 7.3 Action mask 与物理安全层

RL 不应被允许选择物理上不可能的 action。例如：

- 正在航行的船不能瞬间改去其他 emitter；
- 维护中的井不能注入；
- 空载船不能卸货；
- terminal 满罐时不能执行新的有效卸货；
- 无泊位时不能开始新的装货/卸货任务。

因此，simulator 应提供 action mask 或 safety layer，只允许 policy 在可行 action 集合中选择。RL 的职责是“在可行选择中做更好的决策”，而不是学习违反物理规则。

### 7.4 Policy 的定义

policy 是“由当前状态到当前 action 的决策规则”：

```text
action_t = πθ(state_t)
```

在本研究中，RL policy 可以理解为一个受训练的小时级 CCS operation controller：

```text
当前库存、船位、ETA、天气、井状态、压力
    → policy
    → 船舶 dispatch + 注入模式
```

它不是一次性输出未来 7 天的完整计划，而是在每个小时重新观察状态并作出当前决策。

---

## 8. 优化目标与奖励设计

研究不应仅最大化短期注入量，而应在预设 storage-rate target 下优化整体运行表现。

建议定义：

```text
storage rate = cumulative safely injected CO₂ / cumulative captured CO₂
```

并设置若干目标水平，例如 80%、90%、95%，以研究更高 storage obligation 带来的成本与运行稳定性 trade-off。

RL reward 可以由以下部分组成：

```text
reward
= - operating cost
  - venting penalty
  - storage-target shortfall penalty
  - injection instability / throttling penalty
  - reservoir pressure-risk penalty
  + permanently stored CO₂ reward
```

其中的核心原则是：

- emitter 或 terminal venting 需强惩罚；
- 未满足 storage target 需惩罚；
- terminal 低库存、频繁 throttle、井频繁启停需惩罚；
- 安全、稳定的地下封存应得到正向激励；
- 约束优先级应高于纯成本最小化。

---

## 9. 方法路线与比较框架

### 9.1 第一阶段：建立可验证的仿真环境

- 从小规模网络开始：3 emitters、2 vessels、1 terminal、3 wells、2 reservoirs；
- 使用小时级 timestep；
- 实现质量守恒、船舶状态机、泊位、库存、装卸、注入与压力代理模型；
- 建立可视化与单元测试，确保系统不会出现不合理瞬移、无源 CO₂ 或容量超限。

### 9.2 第二阶段：建立非 RL 基线

- 固定 heuristic：例如空船优先服务库存比例最高的 emitter；
- static MILP：在完美未来信息下的确定性排程 benchmark；
- rolling-horizon MILP：每小时更新状态和预测并重新求解的 adaptive optimization baseline。

### 9.3 第三阶段：训练 centralized RL

第一版优先采用一个 centralized policy，同时控制：

- fleet dispatch；
- well injection mode。

这样可以先验证 RL 是否确实能够在不确定环境下学习有效 operation policy，避免一开始将问题复杂化为大规模多智能体非平稳训练。

### 9.4 第四阶段：向 MARL 扩展

当 centralized RL 已经稳定后，可将系统扩展为：

- fleet agent：负责船舶调度；
- terminal/injection agent：负责 buffer 与井注入控制。

训练可采用 centralized training and decentralized execution（CTDE），以兼顾全局协同和实际分布式运行结构。

---

## 10. MILP 与 LLM 在项目中的角色

### 10.1 MILP

MILP 不是要被完全淘汰，而是承担三种角色：

1. **确定性最优 benchmark**：在完美信息下给出理论上高质量的整体排程；
2. **rolling optimization baseline**：衡量实时重优化的效果与计算代价；
3. **expert trajectory generator**：为 imitation learning 或 RL warm start 生成高质量 state-action demonstrations。

因此，项目并不是“RL vs MILP 谁更好”的简单比较，而是研究不同方法在不同信息、扰动和时间预算条件下的适用边界。

### 10.2 LLM / domain-knowledge model

LLM 不应直接替代 simulator 或自由生成可执行操作动作。其更合理的定位是：

- 基于领域知识提供风险提示与运行规则；
- 为 RL action 提供人类可读的解释；
- 生成或审核调度 rationale；
- 作为高层安全建议、action preference 或 constraint reminder；
- 协助将复杂状态转化为可解释的操作报告。

LLM 的主要价值是提高可解释性、领域知识引导和安全性，而不是取代 RL policy。

---

## 11. 关键研究问题

### 主问题

> 在固定船运型 CCS 网络中，能否训练一个满足物理约束的小时级 RL policy，使其在捕集波动、天气延误、terminal 故障、船舶延误和注入能力变化等不确定条件下，自适应协调船舶 dispatch 与注入 operation，并以较低成本、更少 venting 和更稳定注入实现规定的 CO₂ storage-rate target？

### 子问题

1. terminal buffer 的容量与安全库存策略如何影响运输—注入耦合及系统韧性？
2. 在不同 storage-rate target 下，成本、venting、throttling 和 pressure risk 如何变化？
3. RL 相较于 static MILP、rolling-horizon MILP 和 heuristic 的优势是否主要出现在何种扰动类型下？
4. 在相同时间决策预算下，RL policy 的实时推理速度是否足以支撑小时级 operation？
5. MILP demonstrations 与 domain-knowledge guidance 是否能提高 RL 的样本效率、可解释性与安全性？

---

## 12. 预期贡献

本研究预计形成以下贡献：

1. 一个面向 ship-based CCS operation 的小时级、物理约束 CCS digital twin；
2. 一个包含捕集、海运、terminal buffer、注入井与 reservoir proxy 的统一状态转移环境；
3. 面向不确定扰动的 adaptive RL operation policy；
4. static MILP、rolling-horizon MILP、heuristic 与 RL 的系统性比较；
5. 以 storage-rate target 为约束条件的成本—稳定性—韧性 trade-off 分析；
6. 可解释的 RL / MILP / LLM-assisted operation decision framework；
7. 对未来真实 CCUS hub（例如 Northern Lights 风格船运—终端—海上封存链条）的运行管理启示。

---

## 13. 当前最小可行版本（MVP）

```text
Network:
- 3 emitters
- 2 vessels
- 1 terminal
- 3 injection wells
- 2 reservoirs

Temporal setting:
- 1-hour decision interval
- 168-hour episode

Uncertainties:
- capture-rate fluctuation
- weather-induced travel delay
- terminal outage / unloading degradation
- well maintenance
- injectivity decline

Actions:
- vessel destination: WAIT / E1 / E2 / E3
- well mode: OFF / LOW / MEDIUM / HIGH

Evaluation:
- storage rate
- total operating cost
- venting volume
- injection shortfall
- throttling hours
- terminal congestion / berth waiting
- pressure-risk exposure
- recovery time after disruption

Baselines:
- heuristic policy
- static MILP
- rolling-horizon MILP
- centralized RL
```

---

## 14. 一句话项目定位

> 本研究提出一个物理约束、情景随机化的 CCS operational digital twin，并利用小时级 RL 在固定船运 CCS 网络中实时协调船舶运输、终端库存和井注入，以在不确定扰动下实现目标导向、低成本、低 venting、稳定且可解释的 CO₂ 地质封存 operation。

---

## 15. 相关基线文献

Shikha, S., Jilhewar, S., & Jayaram, V. (2025). *Optimization of Carbon Capture and Storage (CCS) Logistics: An Integrated MILP Approach for Transport and Storage*. SPE-228163-MS. https://doi.org/10.2118/228163-MS

该文提出了联合海运、terminal inventory 与井注入的小时级 MILP 框架，是本研究的直接确定性优化参考；本研究拟在此类耦合 operation 问题上进一步引入不确定情景、在线自适应 RL 与后续可解释性机制。
