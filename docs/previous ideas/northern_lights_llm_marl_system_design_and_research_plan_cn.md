# Northern Lights 案例驱动的 LLM 协商 + RL CCS 运营系统:研究问题、系统设计与研究路线

**日期**: 2026-06-10(v2.0,全面简化重写)
**版本记录**: v1.0 初版三层架构;v1.1 引入天然气提名机制(后证不符合行业现实);v1.2 转向合同核心(后证使研究失焦);**v2.0 定稿原则——合同只是输入,研究核心是"扰动下的多主体运营协调与 LLM 协商的可验证边际价值",机制阶梯收敛到四级,研究问题收敛到四个**。

---

## 0. 研究定位

### 0.1 一句话研究陈述

> 在一个按 Northern Lights 真实参数校准、由合同设定运营基点、信息按商业边界分散的船运 CCS 链条中:**量化扰动下跨主体协调的价值(RQ1),检验 LLM 介导的多方协商相对于行业现状机制与结构化协议的边际价值(RQ2),并用物理验证闭环保证一切协调结果可执行、可审计(RQ3)**。

### 0.2 三条设计纪律(贯穿全文)

1. **合同是输入,不是研究对象**。每个排放方一份简单合同(年度量、基准交付计划、容差、违约金),作用是给运营定基点、给结算定规则;在所有实验中固定不变,只在敏感性分析中扫描参数。
2. **每个被比较的机制都来自行业现实**:要么是现状(运营商单边调度),要么是明天就能实施的方案(结构化信息协议、LLM 协商)。不发明不存在的市场或制度。
3. **每个研究问题可证伪、有诚实基线、负结果也可发表**。

### 0.3 研究意义:人在哪里?(回应"特殊情况现实中由人协调"的根本质疑)

本研究**不主张替代人工协调**,其意义分四层:

1. **量化现状的缺口**。今天的异常处理就是人在打电话/发邮件/开周会——慢、临时、信息不全、不可审计。L1 是对这一现状的形式化;RQ1 的 G 度量"现行人工+合同流程留在桌上的钱"——这个数字对行业有用,与是否采用 AI 无关。
2. **人工协调不可扩展,而行业正在扩展**。今天 3 个客户 2–3 艘船人脑罩得住;2028 年 5+ Mt/yr,欧洲多个集群并行,远期多枢纽网络。电网与天然气都走过"调度员电话 → 协调基础设施(EMS/nomination 系统)"的路,且都是被事故逼的。机制阶梯回答的前瞻问题是:**CCS 在规模化之前应把协调基础设施建到哪一级**(L2 门户够不够?L3 有无真实增益?)。
3. **现实部署形态是人在环的决策支持**:agent 起草经物理验证的重排方案与各方损益,人批准/否决。研究评估的是该辅助路径的质量上限与价值来源;agent 的可证伪优势恰在人最弱处——同时追踪全链状态、分钟级枚举多议题打包、全程可审计。
4. **仿真做人类经验做不到的事**。NL 只运营了一个冬天,人类异常处理经验极其有限;仿真可把千百种复合扰动系统性地打在每种机制上,在事故发生前暴露失效模式——这是电力系统研究的标准价值主张(压测规程,而非替代调度员)。

---

## 1. Northern Lights 真实结构 → 仿真映射

### 1.1 项目事实(截至 2026 年中,公开资料,详细来源见 §10)

| 环节 | 真实参数 | 仿真映射 |
|---|---|---|
| 运营模式 | 全球首个第三方 CO2 运输与封存服务(Equinor/Shell/TotalEnergies 合资);2025-08 首次注入 | T&S 运营商与排放方是服务合同关系 → 信息天然按商业边界分散 |
| Phase 1 | 1.5 Mt/年,已售罄 | 基准配置 |
| Phase 2 | ≥5 Mt/年,2028 下半年投运 | 扩张配置(RQ4) |
| 排放方 | Brevik 水泥 ~0.4 Mt(挪威,近);Celsio 垃圾焚烧 ~0.35 Mt(奥斯陆,近);Yara 合成氨 ~0.8 Mt(荷兰,远);Ørsted 生物质 ~0.43 Mt(丹麦,中);Stockholm Exergi BECCS 0.9 Mt(瑞典,2028 起) | 4(+1)个异质排放方:距离、体量、工艺连续性、生物源属性各不同 |
| 船队 | 专用 LCO2 船 7,500 m³(≈8,000 t)/艘;Phase 1 运营 2–4 艘 | 船舶容量、航次、装卸速率 |
| 接收终端 | Øygarden,12 罐共约 7,500 m³,单泊位 | **终端缓冲 ≈ 一船货容** + 泊位排队 |
| 管道/封存 | 100 km 海底管道;Aurora/Johansen 组,2,600 m,Phase 1 双井 | 流量上限;压力/注入性代理模型 |

### 1.2 三个真实的研究张力

1. **终端缓冲极小(≈1 船货容)**:卸船必须与管道外输/注入近实时耦合;注入端减速会沿链条反向传播到所有排放方——跨主体协调是物理硬需求,不是软目标。
2. **排放方异质**:航程 3–8 天不等;水泥窑/垃圾焚烧不能随意降负荷;BECCS 客户的 curtail 机会成本不同——船期分配是真实的多目标问题。
3. **商业边界即信息边界**:罐存、可用率、生产计划是各方商业敏感信息;中心化全信息优化在制度上不可行——这是"为什么研究分散协调"的制度依据,也是中心化 MILP 只能当 oracle 上界的原因。

---

## 2. 系统设计

### 2.1 总体架构(四层)

```text
Layer S  场景与信息层: 扰动事件 + 文本信息流(气象叙述/检修通告),按归属分发(信息分散)
   ↓
Layer N  协调层(慢时间尺度): 周协调窗口 + 事件触发;机制可插拔(L0–L3,见 RQ2)
   ↓
Layer P  规划与控制层(快时间尺度): 排程求解器(船-航次指派)+ 各主体 RL/MPC(连续量)
   ↓
Layer V  物理验证层: 状态演化 + 动作校验;不可行 → 结构化 violation report 回流 Layer N/P
   ↓
Layer E  评估层: 系统级/协作级/主体级指标;全量日志可追溯
```

### 2.2 合同输入(半页定义完毕,所有实验固定)

每个排放方 i 一份合同,纯机器可读,作用仅两个:**定基点**(基准交付计划)与**定结算**(违约金/费用):

```yaml
contract_i:
  annual_volume:      # 年度承诺量(按 NL 客户真实量级)
  baseline_program:   # 基准交付计划: 周粒度的交付窗口序列(由配置生成器按航程和量摊开)
  tolerance: 0.05     # 交付量容差带 ±5%
  shortfall_penalty:  # 短交违约金(€/t)
  vent_liability:     # 排放方侧 vent 的费用归属
  demurrage_rate:     # 船舶滞期费率(€/h),用于成本结算
```

结算引擎按周结算违约金/滞期费,计入各主体成本。**就到此为止**——无文本条款、无 make-up 权谈判、无合同设计空间;合同参数(容差、罚金水平)只出现在 RQ1 的敏感性分析里。

### 2.3 物理层模型(Layer V,步长 2 h,参数全部可溯源)

```text
排放方 i:  q_i(t) = q_i^nom · u_i(t) · a_i(t)        u_i ∈ [u_min,1] 捕集负荷;a_i 可用率过程
           B_i(t+Δt) = B_i + q_i·Δt − L_i·Δt          0 ≤ B_i ≤ B_i^max(1–2 船货容)
           B_i 触顶 → curtail(降 u_i)→ vent(计罚)
船舶 k:    状态(航段, 货量, 阶段);τ_voyage = τ_base·(1+δ_w),δ_w 季节调制对数正态延误
           装/卸约 800 t/h,~10 h/船;Øygarden 单泊位排队
终端:      H(t+Δt) = H + U·Δt − F·Δt                 0 ≤ H ≤ ~8,000 t(≈1 船货容!)
管道:      0 ≤ F ≤ F^max;|ΔF| ≤ ΔF^max(平滑性)
储存(双时间尺度——Aurora 为 ≥100 Mt 大型咸水层,半年注入量 <1% 容量):
  慢变量:   p_res 远场压力在 episode 内近似准静态(按物质平衡缓慢漂移,
            主要作为季节/配置参数,而非逐步演化的决策状态)
  快变量:   p_bh,j = p_res + q_inj,j / II_j     井底压力随注入速率即时变化
            硬约束 p_bh ≤ p_frac − margin       ← 运营上真正绑定的约束
  运营动态: |Δq_inj,j| ≤ ramp limit(冷 CO2 井筒热应力);停注-重启惩罚
            (船运驱动的注入天然间歇,重启代价真实存在);
            II_j 随机退化事件(盐析代理)+ 井检修窗口(离散事件,
            双井下单井停 = 损失 50% 接收能力);Σq_inj = F(无井口缓冲)
全局:      质量守恒校验(<1e-6);违规 → {violation_type, magnitude, feasible_envelope,
           suggested_revision_class}
```

(储存代理模型的形态与地下储气库行业通用的"库存-可交付性曲线"一致,NETL 有 UGS→CO2 封存类比研究背书,见 §9。)

### 2.4 Agent 与信息结构

| Agent | 现实对应 | 私有观测 | 运营动作(RL/MPC) |
|---|---|---|---|
| Emitter A–D(+E) | 5 家客户 | 自身罐存 B_i、可用率、生产计划、本地文本信息 | 捕集负荷 u_i、装船速率、curtail/vent 次序 |
| T&S Operator | NL 运营商(船+终端+管道) | 船位、终端罐存 H、泊位队列、航线气象文本 | 卸船速率、外输流量;船-航次指派经排程求解器 |
| Storage Operator | 注入与储层管理 | p_res、II_j、井况 | 各井注入速率、井切换、停注 |

协调层消息协议(类型化 JSON,协议引擎校验,LLM 经结构化输出生成):

```text
STATUS   {agent, key_fields...}                    状态申报(字段由机制等级决定)
REQUEST  {agent, type: delay|swap|priority|limit, quantity, window, urgency, rationale}
PROPOSE / ACCEPT / REJECT {plan_id, ...}           计划修订提案与回应
COMMIT   {plan_id, parties, revisions[]}           落账为本周执行计划
ALERT    {issuer, risk_type, evidence, suggested_envelope}
```

每个协调窗口限 3 轮;超时回退到运营商单边裁定(系统永不停摆)。

**协调结果 → 规划层的接口**(三种,全部可被结构化协议复刻以便消融):
1. `priority_weights` — 排程求解器中各排放方的服务权重;
2. `action_envelopes` — 连续动作的建议区间(入观测+软掩码);
3. `risk_features` — 风险标志向量(入观测)。

刻意不用 reward shaping(破坏训练平稳性与可比性)。

### 2.5 决策时序

```text
T0  episode 初始化: 合同 + 基准交付计划载入(固定输入)
T1  周协调窗口(周一): 对照基准计划检视未来 1–2 周,按当前机制等级(L0–L3)
    产出计划修订 → COMMIT                         (对应行业真实的周运营会议)
T2  连续控制(每 2 h): 排程求解器执行指派;RL/MPC 出连续动作;Layer V 校验
事件触发: 船舶延误>阈值 / 井限注 / 罐存预警 → 事件协调窗口(≤2 轮)
周末结算: 违约金/滞期费/指标入账
```

Episode = 26 周(覆盖一个冬季);评估含整年长 episode。LLM 只在 T1 与事件窗口被调用(每 episode 约 30–60 次,成本可控)。

### 2.6 操作层设计:MPC 默认 + RL 消融(归因纪律的关键)

**原则**: 机制阶梯 L0–L3+ 只切换协调层;操作层在所有机制下架构、超参、训练预算严格一致(协调接口作为观测特征,L0 置零、L3 由协商填充)——否则机制差异与控制器质量混淆。

**策略清单(RL 模式下共 5 个策略)**:

| Agent | 策略 | 连续动作 | 不归 RL |
|---|---|---|---|
| Emitter ×3 | 同构共享参数+个体特征 | 捕集负荷 u_i、curtail 触发 | 装船速率(泊位/船在场决定) |
| T&S Operator | 1 | 卸船速率 U、外输流量 F | 船-航次指派(排程求解器) |
| Storage | 1 | 各井注入分配、停注/重启 | 检修窗口(协商层议题) |

**Reward**: 各 agent 用自身 P&L(结算引擎逐周产出:排放方=−vent 碳成本−减产−短交罚;运营商=服务收入−船舶成本−滞期;储存=注入收入−干预成本−裕度惩罚)。理由:与商业现实一致(目标不一致正是协调机制存在的原因);锚定合同结算,无 reward shaping 调参软肋。训练 CTDE(个体 reward 的 MAPPO/IPPO,集中 critic)。另跑共享系统 reward 对照,确认机制结论不依赖 reward 结构。

**两种评估模式(都报告)**: 冻结模式——L1 下训练一次、冻结、放到所有机制下跑(测协调信号的即插即用价值);适配模式——各机制下微调(测机制最优潜力);两者之差=协调信号的适配成本。主结果以 MPC 版为准,RL 版进 robustness 章节——核心主张不押在 MARL 训练成功上。

**LLM 与 RL 的关系(三条"不")**: 无梯度互通(不联合训练:成本/归因/稳定性);LLM 不改 reward;LLM 不出物理动作。耦合仅通过环境闭环:LLM 产出的三接口改变 RL 的信息环境与约束景观(软掩码,物理验证器有最终否决权),RL 的执行结果与 violation report 成为下一轮协商输入。松耦合是机制可插拔、归因可切干净的前提。

**动作耦合的处理(各 agent 动作互相限制对方可行域——这正是研究对象,处理分三层)**:
1. **物理层结算次序 + 投影**: 每步所有动作都是"设定值请求",物理层按固定因果序结算并裁剪到可行,例如 U_actual = min(U_request, 罐余量+本步外输, 船上货量, 泊位在位);F_actual = min(F_request, Σ井可接收);冲突不会导致死锁,而是通过缓冲与排队**自然反压传播**(井受限→F 削减→罐存升→卸船减速→船等泊→排放方 buffer 升→curtail)——这条反压链正是链条的物理真相;每次裁剪产生 clip/violation 报告。
2. **边界预测**: 各主体局部决策需要"别人会怎么做"的预测——统一用协调层 COMMIT 的计划作边界条件。机制阶梯在操作层的机械体现因此非常具体:**L0–L3+ 的差异 = 本地控制器拿到的边界预测质量**(L0 用基准计划,常错;L3 用协商后计划,更准)。协调价值由此有了清晰的传导机制,而非黑箱。
3. **MARL 非平稳性**: 个体策略互为环境的非平稳问题用 CTDE(集中 critic)+ 物理投影兜底可行性 + 协调特征降低对手不确定性来缓解;评估用冻结模式避免相互适应漂移。

**MPC 与 RL 的关系**: 同一槽位(操作层)的两种可互换实现,同输入(局部状态+协调接口+边界预测)同输出(连续设定值)。MPC = 各主体 48–72 h 滚动局部优化(仿真模型已知故模型精确,确定性等价,用 COMMIT 计划做边界),零训练、可解释、可复现;RL = 离线在场景分布上训练的反应式策略,能隐式学习对冲(如预留缓冲余量应对延误尾部风险)。**预期**: 名义/轻扰动下 MPC ≈ RL(MPC 甚至略优,因模型无误差);强随机场景(S1/S5)下 RL 可能高出数个百分点(学习型对冲 vs 点预测);**头条稳健性主张 = 机制阶梯的排序在两种操作层下不变**。RL 调 MPC 参数等混合方案超出范围。

---

## 3. 研究问题(核心,共四个)

**论证链**: 协调值多少钱(RQ1)→ LLM 协商能否赚到这笔钱、相对更便宜的机制多赚多少(RQ2)→ 如何保证协调结果物理可行、可审计(RQ3)→ 系统能否随真实扩张泛化(RQ4)。RQ1–3 = 论文 1;RQ4 = 论文 2。

---

### RQ1 — 协调差距:扰动下,跨主体协调的价值有多大、由什么决定?

**表述**: 同一目标函数(封存量 − 运营成本 − vent/违约罚)下定义:

```text
J*       全信息、完美预见、集中式滚动 MILP(oracle 上界;公式化复用 CCS 船运
         排程文献的 RTN/时间槽 MILP,不自创)
J_now    行业现状: T&S 运营商仅凭自身可见信息单边重排船期,客户被动接受
J_0      无协调: 各主体独立按基准计划机械执行
```

协调差距 **G = (J* − J_now)/J***。G 在各扰动族下多大?对终端缓冲规模、船队冗余、注入裕度、(敏感性:合同容差/罚金)的依赖如何?

**假设**: H1a 名义工况 G < 5%;H1b 复合扰动(气象群发 × 井退化)下 G > 20%,主导项是终端缓冲与船队冗余的交互;H1c 仅共享各方罐存水位即可闭合 G 的一半以上。

**J* 的计算方式与含义**: 仿真器按固定 seed 生成完整场景实现(所有扰动落定)→ 将 26 周调度写成确定性 MILP(船-航次 0/1 变量 + 连续流量/注入变量 + 全部物理约束),**把未来扰动当已知输入**求解(HiGHS;必要时 4 周滚动窗口 + 窗口内完美预见,报告 MILP gap)。J* 不是构造保证的"无损"——复合扰动下若物理容量真不够,oracle 也有损失;该残余损失把总损失干净切成"物理不可避免"与"协调可挽回"两块,后者才是机制研究的标的。J* 拥有现实中不可能的信息(完美预见+全部私有状态),**是标尺不是方案**。

**场景纪律(回应"事件是否自定义")**: 事件类型全部有据可查(北海冬季作业窗口=ERA5 波高+实船 AIS 可验证;捕集装置故障=Brevik/Celsio 公开报道;井检修=行业常规);幅度/频率从公开数据标定。方法论防线:核心结果是**比较性的**——同一条事件轨迹打在每级机制上,事件设计偏差对所有机制一视同仁,被比较的排序与差距对场景设计远比绝对数值鲁棒(TSO-DSO/电-气协调研究的标准防御);叠加扰动强度连续扫描(报告 G(强度) 曲线而非单点)、S6 held-out、全分布报告不挑 episode。

**地位**: 纯优化实验,无 LLM 无 RL,第 10 周出全部结果;既是一切后续方法的标尺,也是独立的文献空白(船运 CCS 已有集中式调度优化 [NTNU 系],但无人量化分散信息结构下的协调价值;方法论模板是电-气耦合系统的协调估值研究)。**含止损决策门**:若 G 处处 <5%,后续转"负结果 + benchmark"路线。

**指标**: 封存量缺口、vent/curtail、船等待小时、单位封存成本、G。

---

### RQ2 — LLM 协商的边际价值:四级机制阶梯,每升一级闭合多少协调差距?(方法学核心)

**表述**: 固定其余所有层,只切换 Layer N 机制:

```text
L0  无协调       各主体独立执行基准计划                      (= RQ1 的 J_0)
L1  运营商单边   T&S 凭自身信息重排,客户被动接受              (= 行业现状 J_now)
L2  结构化协议   各方按固定字段交换状态与请求(罐存、可用率、
               延迟/换班/优先级请求),运营商按规则裁定
               ——一个今天就能用运营门户实现的方案
L3  LLM 协商    各方 LLM agent 在消息协议内交换意图、提案并多轮协商,
               产出计划修订(经 Layer V 验证)
L3+ 文本增强    L3 + 非结构化文本信息流(气象叙述/检修通告,只有语言界面能消费)
```

核心读数 **R(L) = (J_L − J_L1)/(J* − J_L1)**:每级机制恢复了协调差距的百分之几。关键对比:**L3 − L2**(语义协商对结构化协议的增益)与 **L3+ − L3**(消费非结构化信息的增益)。

**假设**:
- H2a: L2 即恢复差距大半(R ≥ 60%)——若成立,结论是"信息共享是主菜,LLM 是配菜",这是对 LLM-agent 热潮的有力纠偏,可发表;
- H2b: L3 − L2 的增益集中在**需要多议题协调的复合扰动场景**(单字段请求无解、跨主体打包调整有解,如"A 让泊位窗口给 C 换下周优先权");
- H2c: L3+ − L3 的增益集中在文本预警可转化为预防动作的场景(提前 72 h 风暴叙述 → 提前装船/腾罐),且随文本噪声率上升而衰减,存在可量化的盈亏平衡噪声率。

**为什么是强问题**: 现有 LLM+能源系统工作(Grid-Agent、GridMind、RL2、CarbonDTMAS 等)都没有 L1(行业现状)与 L2(结构化协议)这两级诚实基线,因此无法回答"LLM 到底贡献了什么"。本设计把 LLM 的潜在价值锚定在两个真实机制上:多议题打包(L2 字段无法预先枚举)与文本信息消费(现实中预报/通告确实以文本到达)。**无论 H2a/H2b 哪边成立,都是清晰可发表的结论**。

**实验**: {L0…L3+} × {S1,S2,S3,S5} × 10 seeds × 2 个 LLM 规模(前沿 + 开源小模型);文本流由生成器按可控噪声率合成,另留人工撰写 held-out 集。
**指标**: 系统级(同 RQ1)+ R 恢复率 + 协作级(协商轮数、达成率、信息披露量、LLM 调用成本、文本→动作转化延迟——经反事实回滚验证可归因)。

---

### RQ3 — 物理验证闭环:violation report 驱动的再协商,能否零违规且不损吞吐?

**表述**: 对比四种安全机制(其余固定):

```text
V1  仅 reward/成本惩罚(常规做法)
V2  动作投影/shielding(裁剪到可行包络,无反馈)
V3  V2 + violation report 进入观测
V4  V3 + violation report 触发事件协调窗口(完整闭环,本架构)
```

**假设**: H3a V1 无法在训练预算内压平压力违规,V2–V4 构造性为零;H3b V4 的吞吐与扰动后恢复时间显著优于 V2/V3——机制是再协商把"局部裁剪"升级为"全链重排"(井限注时上游主动减捕,而非任由 CO2 拥堵);H3c 增益主要来自 violation report 中的 `feasible_envelope` 字段(消融即消失)。

**地位**: 把"验证"从过滤器升级为协调信号源,区别于 planner-validator 式工作(Grid-Agent);风险最低的 RQ(V4 ≥ V2 机制上近乎必然),为论文提供稳定的方法贡献。

---

### RQ4 — 扩张泛化:Phase 1 上构建的协调-控制栈能否迁移到 Phase 2?(论文 2)

**表述**: Phase 1(4 排放方、2–3 船、2 井、1.5 Mt/yr)上训练/调优;Phase 2(新增 Stockholm Exergi、5–7 船、3 井、5 Mt/yr)上零样本/少样本评估,对照从头训练与重新求解的 oracle。

**假设**: H4a 实体置换不变的观测编码使 MARL 零样本保持从头训练的 >80%;H4b LLM 协调层以类型化消息为界面、与主体数解耦,新客户"接入磨合成本"(达稳态服务水平的周数)在 L3 下显著低于 L1/L2——若成立,是语言界面价值的第二个独立证据,与 RQ2 互为犄角;H4c 容量紧张期 L3 的 curtail 分配公平性(基尼系数)不劣于按合同比例的 pro-rata 规则。

**现实对应**: 新客户分批接入、船队分批扩张正是 NL 2026–2029 的真实事件序列。

---

## 4. 实验协议

### 4.1 扰动场景族(全部有 NL 现实依据)

| 族 | 内容 |
|---|---|
| S0 | 名义运行 |
| S1 | 冬季气象群发延误(附带提前 48–96 h 文本预报,可控虚警率) |
| S2 | 排放方非计划停产 7–21 天 |
| S3 | 井注入性退化 / 单井关停(双井下损失 50% 注入能力);其中**计划性检修的窗口可协商**——把检修排进低交付周是 storage 侧最真实的协调议题 |
| S4 | 终端罐组/泊位维护 |
| S5 | 复合: S1 × S3(链条两端同时收紧) |
| S6 | held-out: 未见参数组合 + 人工文本,仅终评 |

### 4.2 方法矩阵

| 类别 | 方法 | 角色 |
|---|---|---|
| 上界 | 集中式全信息滚动 MILP(文献公式化,Pyomo+HiGHS) | J*,定义 G |
| 行业现状 | L0 无协调、L1 运营商单边 | J_0、J_now |
| 可实施协议 | L2 结构化协议 | **关键对照**:LLM 必须赢过它 |
| 本方法 | L3 / L3+ LLM 协商 + V4 闭环(操作层默认各主体 MPC,MAPPO 为消融) | — |
| 警示基线 | LLM 直接出动作(无求解器无验证) | 预期失败,论证分工 |

### 4.3 评估纪律

固定配置/预算/LLM 调用节奏与温度(T=0,结构化输出);训练评估 seed 分离,各 10 seeds,均值 ±95% CI;LLM 成本与时延作为一等公民指标;全量协商与违规日志存档支持反事实回滚;物理层单独发布校准报告(质量守恒残差、与 NL 公开数据的量级对照、参数敏感性)。

### 4.4 旗舰用例:Phase 1 冬季复合压力周(论文 1 的叙事核心)

**配置**(对应 NL Phase 1 已签约客户):3 个排放方——Brevik 水泥 0.40 Mt/yr(航程短,窑炉不可降负荷)、Celsio 垃圾焚烧 0.35 Mt/yr(航程短,buffer 小,不能停收垃圾)、Yara 合成氨 0.80 Mt/yr(航程 6–8 天,体量最大,可少量调产);2–3 艘 7,500 m³ 船;终端 ≈1 船货容,单泊位;双井注入。Ørsted 与 Stockholm Exergi 在 Phase 2 扩展配置中加入(RQ4)。

**事件脚本**(第 7 周,S5 复合扰动):
- 北海风暴关闭航行窗口 72 h,Yara 航线受影响最重;**提前 72 h 有文本气象预报**(L3+ 才能消费);
- 恰逢 2 号井计划检修(5 天,接收能力 −50%),**检修起始日有 ±3 天的可协商余地**;
- Celsio 罐存进入周期高位。

**各机制下的预期事件链**(这是论文 1 案例研究小节的底稿,也是各机制差异的具象化):

```text
L0 无协调:   船按基准计划开 → Yara 船海上滞留 3 天(滞期费)→ 同期注入减半、
            终端腾不出罐 → Brevik/Celsio 船在单泊位排队 → Celsio buffer 第 5 天
            触顶 → vent;Brevik 第 6 天 curtail 失败(窑炉刚性)→ vent
L1 运营商单边: 运营商看得见船和终端,推迟 Yara 航次、优先服务 Brevik;
            但看不见 Celsio 罐存逼近上限 → 优先级排错对象 → Celsio 仍 vent
L2 结构化协议: 各方申报罐存/可用率,storage 申报 −50% 包络 → 运营商正确排序,
            Yara 在容差内延迟 4 天;但检修窗口是预设字段之外的议题,无法联动
            → 残余损失来自"检修撞上风暴后补运高峰"
L3 LLM 协商:  多议题打包成交:storage 把检修起始日后移 3 天避开补运高峰;
            Brevik 接受 +1 天延迟换下周优先权;Celsio 获得插队卸船
            → 接近 oracle 的事件链
L3+ 文本增强: 提前 72 h 读到风暴叙述 → 风暴前抢装 + 终端预腾罐
            → 滞期费大头被预防性动作消除
```

### 4.5 预期结果模式(明确标注:这些是待检验的假设,不是承诺)

旗舰场景(S5,26 周 episode,10 seeds)上各方法的预期量级:

| 方法 | 封存量(%计划) | vent(kt/季) | 滞期(h/季) | 恢复率 R | 对应假设 |
|---|---|---|---|---|---|
| J* oracle | ~99 | ≈0 | 最低 | 1.00(定义) | — |
| L0 无协调 | ~75–85 | 高 | 高 | <0 或 ≈0 | — |
| L1 现状 | ~85–90 | 中 | 中 | 0(定义基准) | — |
| L2 结构化 | ~93–96 | 低 | 低 | **≥0.6** | H2a |
| L3 LLM | ~95–98 | 低 | 低 | 0.75–0.9,增量集中在多议题场景 | H2b |
| L3+ 文本 | ~96–98 | 低 | **显著更低** | +0.05–0.1,集中在 S1 类场景 | H2c |
| LLM 直接控制 | 不稳定 | 高方差 | 高方差 | 可能为负 | 警示基线 |

注意第一行:oracle 的"~99%"是名义-中等扰动下的预期;强复合扰动(S5)下 oracle 自身也可能降到 90% 以下——这不是问题而是信息:J* 与 100% 的差是"物理不可避免损失",J* 与各机制的差才是"协调可挽回损失",论文图中两者分开着色。

**读图方式**: 论文 1 的主图即"机制阶梯 × 场景族 → R 恢复率"的阶梯图 + 上表的分项指标。三种可能的真实结果及其论文叙事:
- 若 L3 − L2 显著 → "语义协商的价值及其来源"(主推故事);
- 若 L3 ≈ L2 → "信息共享是主菜,LLM 是配菜:对 LLM-agent 热潮的定量纠偏"(同样可发表);
- 若 L2 ≈ L1 → 连同 RQ1 的小 G 一起转"CCS 链合同化运营已足够鲁棒"的负结果+benchmark 论文(M1 决策门提前暴露此情形)。

安全机制矩阵(RQ3)的预期:V1 在训练预算内压力违规无法归零;V2–V4 违规恒为 0(构造性),V4 的吞吐与扰动恢复时间优于 V2/V3,优势随扰动强度增大。

### 4.6 数据需求清单(零保密数据依赖——这是可行性的硬保证)

| 数据 | 用途 | 来源 | 难度 |
|---|---|---|---|
| 链条结构参数(客户年量、船容 7,500 m³、终端 12 罐、单泊位、100 km 管道、双井) | 系统配置 | NL 官网 / Equinor / TotalEnergies 新闻稿(§10) | 已有 |
| 航线基准航时 | 船舶模型 | 港口坐标 + searoute 类工具计算;Brevik/鹿特丹—Øygarden 距离公开 | 容易 |
| 真实航次时间分布(可选验证) | 校准 δ_w | Northern Pioneer / Pathfinder 的公开 AIS 轨迹(MarineTraffic 等) | 容易 |
| 北海冬季气象/波高统计 | 延误分布 + 作业窗口阈值 | ERA5 再分析 / 挪威气象局公开数据;Hs 作业阈值取船级社惯例 | 容易 |
| 捕集装置可用率与最低负荷(水泥/WtE/合成氨) | 排放方模型 | IEAGHG 报告、Brevik/TCM 公开运行报告、文献 | 中等 |
| Aurora 储层量级(注入能力、深度、容量) | 储存代理标定 | 挪威近海管理局(Sodir)CO2 储存图集、EL001 许可文件、Equinor 公开材料 | 中等 |
| 注入性/可交付性曲线形态 | II(p) 函数形 | UGS 行业曲线(EIA/NETL)+ CO2 注入文献 | 容易 |
| 成本参数(船租、燃料、滞期费率、单位运储成本 ~8–10 €/t) | 成本指标 | IEAGHG/ZEP CO2 航运研究、Bjerketvedt 系论文 | 容易 |
| 合同参数(容差、罚金水平) | 输入配置 | 无公开 TSA → **设为假设并做宽幅扫描**(设计上不依赖真值) | 由设计消解 |
| 文本信息语料(预报/通告) | L3+ 与 S6 | 模板 + LLM 改写合成(格式仿 met.no 海事预报);held-out 集人工撰写 | 生成 |

关键声明:所有标定目标是**量级正确**而非精确复现(研究对象是协调机制,不是 NL 的数字孪生);凡无公开真值的参数(合同罚金、II 退化频率)一律进入敏感性扫描,使结论以"在参数区间内稳健/在何阈值翻转"的形式给出。

---

## 5. 研究路线(9–12 个月,每个里程碑以"能产出论文图表"收尾)

```text
M0 (W1–4)    物理仿真核心 + 合同输入/结算引擎 + 场景引擎 + NL 校准报告
M1 (W5–10)   J* / J_now / J_0 → RQ1 全部结果
             ◆ 决策门: G 处处<5% → 转负结果+benchmark 路线
M2 (W9–16)   L2 结构化协议引擎 + 操作层(各主体 MPC 默认;PettingZoo+MAPPO 消融)
             → L0/L1/L2 完整结果(机制阶梯下半段,LLM 尚未进场)
M3 (W13–22)  LLM 协商层(L3/L3+)+ 文本流生成器 + V1–V4 安全矩阵
             成本控制: 仅 T1/事件窗口调用、T=0、缓存、训练期协商语料重放
M4 (W21–30)  全实验矩阵 → RQ2/RQ3 全部结果 + 消融
M5 (W29–36)  论文 1 投稿 + "NL-CCS-Bench" 开源(环境/基线/协议/语料)
M6 (W33–48)  Phase 2 迁移实验 → RQ4, 论文 2
```

**技术栈**: Python+NumPy 纯函数式状态机 / PettingZoo API / Pyomo+HiGHS / 自实现 MAPPO 或 CleanRL / LLM = 前沿模型 API + 开源小模型本地对照 / Hydra + wandb + 全量 JSONL 日志。

---

## 6. 目标期刊与论文规划

**论文 1**(RQ1–3,M5): *"How much is coordination worth in ship-based CO2 transport and storage? Quantifying the marginal value of LLM-mediated negotiation against industry-standard mechanisms."*
- 首选 **Applied Energy / Advances in Applied Energy**;备选 **Energy and AI**;benchmark 部分可拆投 **NeurIPS Datasets & Benchmarks**;若 H2b+H4b 双强,可上攻 **Nature Communications**(叙事:AI agents 协调真实气候基础设施)。

**论文 2**(RQ4,M6 后): *"Scaling multi-party CCS operations: coordination policy transfer under network expansion"*,投 **Applied Energy / IJGGC**。

**写作纪律**: 贡献声明是机制量化 + 可验证架构 + 校准 benchmark,不是"apply LLM to CCS";H2a 若成立则作为主结果之一正面写出;成本-性能联合报告。

---

## 7. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| RQ1 显示 G 很小 | 中低 | M1 决策门提前暴露;终端缓冲极小+复合扰动的机制分析表明 G 大概率显著;负结果+benchmark 兜底 |
| L3 ≈ L2(LLM 无边际价值) | 中 | H2a 已被设计为可发表结论;L3+ 的文本消费给 LLM 第二条价值通道;噪声率扫描至少能刻画盈亏平衡条件 |
| LLM 成本与方差 | 中 | 仅 T1/事件窗口调用(每 episode 30–60 次)、T=0、缓存、训练期重放 |
| 审稿:仿真保真度 | 高(必然) | 参数溯源文档 + NL 公开数据校准 + 敏感性分析;声明研究对象是协调机制而非储层物理 |
| 审稿:为何不中心化 | 高(必然) | §1.2 张力 3 制度论证 + J* 全程在矩阵中作为上界 |
| 撞题(MAS-CCUS 后续、Chen & Hosseini 扩展、NTNU 组转向) | 中 | 差异化:他们无信息分散结构、无机制阶梯、无 LLM 对照;RQ1+benchmark 尽早 arXiv 占位 |

**与最近竞品的边界**(一行版): NTNU 船运 CCS 优化=集中式单决策者(其公式化被复用为我们的 oracle);MAS-CCUS=集群仿真无学习无机制对比;Chen & Hosseini=仅储存端 MARL;Grid-Agent/GridMind=单运营主体内的 LLM 工具;CCUS-Agent=静态匹配 ABM。本研究独占:**多商业主体 + 信息分散 + 行业机制阶梯 + LLM 对照 + 物理验证闭环**的组合。

---

## 8. 刻意不做的事(范围控制)

- 合同条款设计/谈判(合同是固定输入;条款价值反演留作论文 3 的可选方向);
- 市场机制(现货/拍卖/日内市场——行业不存在);
- 监管者 agent、CO2 质量规格谈判、多储存地竞争、复杂地质数值模拟;
- LLM 生成 reward(破坏可比性,且与 Text2Reward/Eureka 正面撞题)。

---

## 9. 行业依据速记(为什么这套设计贴合工业实际)

| 设计选择 | 行业依据 |
|---|---|
| 合同定基点 + 基准交付计划 | LNG 行业的 Annual Delivery Program 实践;NL 与客户的长期 T&S 服务合同 |
| 周协调窗口 + 事件触发再协调 | 航运/LNG 的周运营会议与扰动后 ad-hoc 协调(现实中以邮件/电话进行——LLM agent 化的自然对应物) |
| 运营商单边调度作为现状基线(L1) | 第三方 T&S 服务模式下运营商主导排程的行业实践 |
| 结构化协议 L2 的字段设计 | 借鉴天然气 nomination 的消息字段经验(但不引入其市场机制) |
| 储存代理模型(库存-压力-可交付性) | 地下储气库行业通用工程抽象;NETL 有 UGS→CO2 封存类比研究 |
| oracle 的 MILP 公式化 | CCS 船运排程文献(NTNU 系 RTN/时间槽 MILP)直接复用 |
| 协调价值量化范式 | 电-气耦合系统协调估值(NREL/JISEA);TSO-DSO 协调机制对比(同场景比机制) |
| 信息分散的制度正当性 | 欧盟 CO2 网络准入讨论中 negotiated TPA 路线 = 双边合同世界 |

---

## 10. 参考资料

Northern Lights 项目事实:
- [Equinor — The Northern Lights project](https://www.equinor.com/energy/northern-lights)
- [TotalEnergies — Northern Lights Phase 2 launch](https://totalenergies.com/news/press-releases/norway-totalenergies-and-partners-launch-2nd-phase-northern-lights-ccs-project)
- [Northern Lights — What we do](https://norlights.com/what-we-do/)
- [The CCUS Hub — Northern Lights/Longship](https://ccushub.ogci.com/focus_hubs/northern-lights/)
- [gCaptain — Northern Lights orders four new CO2 carriers](https://gcaptain.com/northern-lights-orders-four-new-co%E2%82%82-carriers-in-major-ccs-expansion/)
- [JPT — Phase 2 of Northern Lights gets green light](https://jpt.spe.org/phase-2-of-northern-lights-ccs-project-gets-green-light)

跨行业方法依据:
- [Rakke et al. — Rolling horizon heuristic for LNG annual delivery programs (TR-C)](https://www.sciencedirect.com/science/article/abs/pii/S0968090X10001427)
- [Bjerketvedt et al. — Optimal design and cost of ship-based CO2 transport under uncertainties (IJGGC)](https://www.sciencedirect.com/science/article/pii/S1750583620306150)
- [Bjerketvedt et al. — Deploying a shipping infrastructure for CCS from Norwegian industries (JCLP)](https://www.sciencedirect.com/science/article/abs/pii/S0959652621037641)
- [Integration of Optimization and DES in CCS Supply Logistics (I&ECR 2025)](https://pubs.acs.org/doi/10.1021/acs.iecr.5c04890)
- [NREL/JISEA — Valuing intra-day coordination of power and gas operations (Energy Policy)](https://www.sciencedirect.com/science/article/abs/pii/S0301421520302214)
- [TSO–DSO Coordination Schemes evaluation — Swedish case study (SEGAN)](https://www.sciencedirect.com/science/article/abs/pii/S2352467723001339)
- [TAP — Commercial dispatching: nomination/matching(仅借鉴消息字段)](https://www.tap-ag.com/shippers/commercial-dispatching)
- [Grid-Agent: LLM-Powered Multi-Agent System for Power Grid Control (arXiv)](https://arxiv.org/abs/2508.05702)
- [GridMind: LLMs-Powered Agents for Power System Analysis (SC'25 W)](https://dl.acm.org/doi/10.1145/3731599.3767409)
- [EIA — Basics of Underground Natural Gas Storage](https://www.eia.gov/naturalgas/storage/basics/)
- [NETL — UGS Analog Studies to Geologic Storage of CO2](https://netl.doe.gov/projects/files/UndergroundNaturalGasStorageAnalogStudiestoGeologicStorageofCO2_013019.pdf)
- [CATF — Building Future-Proof CO2 Transport Infrastructure in Europe](https://www.catf.us/resource/building-future-proof-co2-transport-infrastructure-europe/)

本仓库文献盘点:`CCUS_LLM_MultiAgent_Idea_Report.md`、`CCUS_Energy_MultiAgent_LLM_Addendum.md`(2026-06-04)。
