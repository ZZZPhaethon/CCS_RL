# `algorithms.auction_market`

排放源竞价市场(探针) / Emitter auction market (probe)

把每个 emitter 当作一个**自私竞标者**,竞争有限的空船运力:出价高者优先被服务。
这是"多方博弈 / 竞争"研究方向的第一步——一个**不学习**的贪心拍卖探针,用来回答
"按紧迫度竞价分配运力,是否比平衡规则更好、争夺中是否出现赢家/输家",从而判断这条
方向有没有肉,再决定是否投入学习出价与机制设计。

Each emitter is a self-interested bidder competing for the limited empty-vessel
dispatch slots; the highest bidder is served first. This is the first step of the
competitive multi-agent direction: a **non-learning** greedy-auction probe that
answers whether urgency-based bidding beats the balanced rule and whether
contention produces winners and losers — so you can decide if the direction is
worth learning-to-bid and mechanism design.

## 不修改原有代码 / No changes to existing code

本包只做三件事,不改动 `Simulation/` 或其他 `algorithms/` 包:

- **读取**物理环境(库存、掩码、船位、天气);
- **复用**平衡 cluster-shuttle 规则([Simulation/control/baselines.py](../../Simulation/control/baselines.py))作为基线与安全默认;
- **复用**经济账本([Simulation/economics.py](../../Simulation/economics.py))做精确的三方成本分解,复用残差 RL 的场景生成器([algorithms/residual_rl/scenario.py](../residual_rl/scenario.py))与 CVaR 评估。

## 机制 / Mechanism

出价有物理层根据,不是任意数字:

```
bid_i = carbon_price × 预计在 bid_horizon_h(默认48h)内会被迫放空的吨数
      = carbon_price × max(0, capture_tph_i × H − headroom_i)
```

与环境自身的 `CCSEnv._overflow_risk_t` 完全一致。每个决策点:

```
1. 规则先出安全默认动作(cluster-shuttle + 最高可行注入率)
2. 只挑出"规则已决定派去取货的空船"作为拍卖标的  ← 避免把正在装载的船中途改道
3. 出价最高的 emitter 优先,由最近的合法空船服务;其余船保持规则动作
4. 原物理环境执行并校验(network.step 是最终裁决)
```

只把"规则本就要派出的空船"重新指定目的地,因此拍卖与规则的差异**只在于一艘空船被派往
哪个源**——这正是排放源之间的争夺点。若所有出价为 0,则结果与规则完全相同。

## 文件 / Files

| 文件 | 作用 |
| --- | --- |
| `bidding.py` | `AuctionConfig` 与"预计放空 × 碳价"出价估值。 |
| `policies.py` | `rule_action`(基线)与 `AuctionDispatchPolicy`(竞价调度)。 |
| `payoff.py` | 精确三方成本分解(排放/船运/封存)、逐源放空成本、可选资费盈亏。 |
| `runner.py` | 复用残差 RL 场景机制构建环境并回放策略(自身不 reset,便于深拷贝配对)。 |
| `probe.py` | 命令行入口(第一步):同场景配对比较规则 vs 拍卖,打印并写出结果。 |
| `equilibrium.py` | 命令行入口(第二步):策略性竞价均衡与无序代价(Price of Anarchy)。 |
| `ceiling.py` | 命令行入口(第三步):市场 vs 中心化天花板(native MPC / rolling MILP)。 |
| `features.py` | 逐排放源局部状态特征(前瞻:天气、可达性、对手拥塞)。 |
| `bid_policy.py` | 共享、基于状态的竞价策略 `exp(w·features)`(参数共享=去中心化执行)。 |
| `train_bidding.py` | 命令行入口(第四步):用 CEM 学习共享竞价策略并对比基线与 MPC。 |
| `value_of_market.py` | 命令行入口(核心):以运营规则为基准,沿竞争/压力轴度量市场价值。 |
| `budgeted.py` | 带预算的统一价格拍卖:内生拥塞出清价 + 可耗尽预算,使价格影响分配。 |
| `budget_study.py` | 命令行入口(第五步):预算扫描与**非平凡**无序代价。 |

## 运行 / Run

```powershell
python -m algorithms.auction_market.probe `
  --scenario northern_lights_phase1_milkrun_imbalanced `
  --seeds 1 2 3 4 5 `
  --episode-hours 720 `
  --forecast-context-hours 168
```

对每个 seed 只采样一次场景、深拷贝给规则与拍卖两个控制器,并断言两者累计捕集量一致
(和项目现有 `compare_shared_*` 同样的配对方法)。可选资费盈亏:加
`--transport-tariff 5 --injection-tariff 10`。

结果默认写入 `output/auction_market_probe/<scenario>__<hours>h__seeds<a>-<b>/`:

- `comparison_raw.csv`:每 seed、每控制器的物理/经济/逐源指标;
- `comparison_summary.csv`:各控制器均值;
- `probe_metadata.json`:参数、汇总、CVaR 与(可选)资费盈亏。

## 怎么读结果 / Reading the output

- **stored / vented / total_cost**:拍卖相对规则的链条级增量。
- **CVaR vented_t**:最差 25% 场景的平均放空(项目自有的尾部风险口径)——注意均值改善
  不代表尾部改善。
- **三方成本分解**:排放方(调理+放空)、船运方(燃油+装卸)、封存方(再调理),精确、
  无虚构。
- **逐源放空**:哪个 emitter 在争夺中受益、哪个吃亏——竞争的赢家/输家。

## 第二步:均衡与无序代价 / Equilibrium & Price of Anarchy

每个 emitter 是策略性竞标者:提交前把*真实*价值乘以私有*压价系数* `shade`(1.0=如实、
<1=压价)。一价规则下赢家付自身出价(压价理性);二价规则下如实为占优策略且分配有效率。
`equilibrium.py` 用**迭代最优反应**学习一价下 `shade` 的近似纯策略纳什均衡(每个 emitter
最小化自身成本 = 放空成本 + 支付),再与如实高效基准及规则对比,报告无序代价。

Each emitter scales its true value by a private ``shade`` before bidding.
`equilibrium.py` learns an approximate first-price Nash equilibrium of the shades
by iterated best response, then reports the Price of Anarchy against the
truthful (second-price, efficient) benchmark and the rule.

```powershell
python -m algorithms.auction_market.equilibrium `
  --scenario northern_lights_phase1_milkrun_imbalanced `
  --seeds 1 2 3 `
  --episode-hours 360 `
  --shade-grid 0.25 0.5 0.75 1.0 1.5 `
  --br-rounds 3
```

**观察到的结论 / Finding.** 在该运力拍卖中,分配只取决于出价的*排序*。当各 emitter
对称地压价时,排序不变 → 物理调度不变 → 社会成本不变,因此 **Price of Anarchy ≈ 1.0**:
去中心化的自私竞价恢复了(市场内的)高效分配,压价只改变**支付**(转移),不损失效率。
市场相对平衡规则约 8–12% 的改善来自更好的分配,而非定价。

In this auction the allocation depends only on the *ranking* of bids, so
symmetric shading changes payments (transfers) but not the physical dispatch:
**Price of Anarchy ≈ 1.0**. The ~8–12% gain over the rule is a genuine
allocation gain, not a pricing artefact.

> 注意 / Caveat:这里的"高效"是*市场内*的按真实价值贪心分配,并非整回合的中心化最优
> (滚动 MILP)。市场胜过规则是实的;是否达到全局最优是另一个检验。
>
> Here "efficient" is the within-market greedy-by-true-value allocation, not the
> episode-level centralized optimum (rolling MILP).

## 第三步:对比中心化天花板 / Market vs centralized ceiling

`ceiling.py` 在**完全相同的场景**(同一环境工厂 + seed)上运行规则、市场与
[algorithms/hybrid](../hybrid) 的中心化预见控制器(native MPC,可选 rolling MILP),
比较实际的 stored / vented / cost,并报告市场捕获了"规则→天花板"改善的多少。

```powershell
python -m algorithms.auction_market.ceiling `
  --scenario northern_lights_phase1_milkrun_imbalanced `
  --seeds 1 2 3 --episode-hours 720 `
  --planning-horizon-hours 72 --replan-hours 24
```

**观察到的结论 / Finding**(`milkrun_imbalanced`, 720h, seed 1–3):

| 控制器 | stored t | vented t | total cost € | wall s |
|---|---:|---:|---:|---:|
| 规则 | 112,154 | 55,044 | 5,830,511 | 0.4 |
| 拍卖市场 | 112,562 | 46,390 | 5,129,383 | 0.4 |
| native MPC(天花板) | 118,363 | 39,783 | 4,651,488 | 14.7 |

- 规则 → 市场 → MPC 构成清晰阶梯。
- 市场相对天花板 **成本高 +10.3%**、放空多约 6,600 t,但**几乎零算力**(0.4s vs 14.7s)。
- **市场捕获了"规则→天花板"改善的约 60%**——用约 1/35 的算力拿到大部分收益。

> 天花板是当前可用的最强*中心化*控制器(native MPC),不一定是完美预见的全局最优
> (静态/滚动 MILP);故这是"相对最强中心化基线"的差距。/ The ceiling is the strongest
> available centralized controller, not necessarily the perfect-foresight optimum.

## 第四步:学习型竞价(MARL) / Learned bidding

把固定的 1-D 压价换成**共享、基于状态的竞价策略**:每个排放源(智能体)用*相同*权重
作用于其*自身*局部特征(参数共享=去中心化执行),出价乘子为 `exp(w·features)`;`w=0`
精确复现近视拍卖。权重用**交叉熵方法(CEM)**优化(无梯度、对 (参数,seed) 确定、直接
复用拍卖回放,不需 gym/PPO)。

Each emitter applies shared weights to its own local features; the multiplier
``exp(w . features)`` scales its true value and ``w = 0`` reproduces the myopic
auction. Weights are optimised by the cross-entropy method (CEM).

```powershell
python -m algorithms.auction_market.train_bidding `
  --scenario northern_lights_phase1_milkrun_imbalanced `
  --episode-hours 720 `
  --train-seeds 10 11 --val-seeds 12 --eval-seeds 1 2 3 4 5 6 `
  --with-mpc --planning-horizon-hours 72
```

**观察到的结论 / Finding**(720h;训练 seed 10–12,评估**留出** seed 1–6):

| 控制器 | stored t | vented t | social cost € |
|---|---:|---:|---:|
| 规则 | 112,070 | 55,820 | 5,893,564 |
| 近视市场 | 113,206 | 46,041 | 5,129,610 |
| **学习市场** | 114,427 | 44,237 | **4,987,031** |
| native MPC | 114,177 | 45,556 | 5,085,092 |

- **学习稳健胜过近视出价**:留出 seed 上 **成本 −2.8%、放空 −3.9%**(且在 seed 4–6 上
  独立复现 −2.4% / −3.8%),说明改善可泛化。
- 学到的权重中**只有逐排放源特征起作用**。因为分配只取决于出价的*排序*,而
  `bias`、`forecast_speed_min_72h`、`dispatchable_empty_frac` 在同一决策点对所有排放源
  取值相同,它们只贡献一个公共比例因子,**在排序中被约掉**。将这三项权重清零后,6 个
  评估 seed 的放空与成本**逐一完全相同**(已实证)。有效权重为:`capture_frac +0.81`、
  `projected_vent_frac +0.79`、`hours_to_overflow +0.64`、`fill_ratio +0.44`、
  `nearest_vessel_travel +0.28`(`max_other_fill +0.09` 近乎无效)。
  Only per-emitter features matter: allocation depends on bid *ranking*, so global
  features contribute a common factor that cancels (verified: zeroing them leaves
  every seed's result bit-identical).
- **"天花板"并不稳健**:在 6 seed 上近视市场与 native-MPC 基本持平,学习市场反而**略
  超 MPC**(约 −1.9% 成本),且算力约为 MPC 的 1/35。早前 seed 1–3 上"差 10%"只是对
  MPC 有利的小样本,不具代表性。

> MPC 是*参考*而非固定天花板:其表现随 seed 方差很大(且此处用默认 vent 权重、72h 预见)。
> 若给 MPC 更强配置(如 `vent_first_vent_eur_per_t=10000`、168h 预见)结论可能变化。
> MPC is a reference, not a fixed ceiling: it is high-variance across seeds.

## 核心:市场什么时候重要 / When the market matters (headline)

研究主线是**市场本身的价值**,基准是**运营规则(现状)**,坐标轴是**竞争结构/压力**。
核心结论:**市场的价值取决于竞争结构**——异质/不均衡/受压时市场大幅胜出,均衡/易分区
时反应式竞价反而更差。而且市场只用**局部排放源信息 + 近乎零算力**(约 0.4s/回合,无中央
规划器)。

The research focus is the *value of the market itself*, benchmarked against the
operational rule and swept along contention/stress. Headline: the market's value
is regime-dependent -- it wins large under heterogeneous / imbalanced / stressed
contention and loses where a fixed partition already suffices -- using only local
emitter information and near-zero compute.

场景扫描(近视市场,720h,seed 1–4;正 = 市场更好):

| 场景 | 放空降幅 | 成本降幅 |
|---|---:|---:|
| 3vessels(均衡,3 船 3 源) | −9.5% | −7.1% |
| milkrun(均衡,2 船 3 源) | −93.8% | −33.6% |
| **milkrun_imbalanced(不均衡)** | **+17.1%** | **+13.2%** |
| **milk_run_stress(受压)** | **+56.7%** | **+46.8%** |

压力扫描(milkrun_imbalanced;hard 0.0 / 0.5 / 1.0):放空降幅 **17.1% / 11.1% / 8.5%**。

解读 / Interpretation:
- **市场在异质/不均衡/受压(现实中最难、最需要协调)时以局部信息 + ~零算力大幅胜过固定
  规则**——这是市场重要性的核心证据。
- 在**均衡/易分区**时,规则的"承诺式固定路线"胜过反应式竞价:贪心urgency会抖动、过度
  集中,把其他源饿死(milkrun −93.8% 即此)。这刻画了**去中心化市场何时优于固定规则**,
  比"市场总是更好"更强、更诚实的结论。
- 学习型竞价(第四步)在**分布内**进一步改善不均衡场景(成本 −2.8%、放空 −3.9%)。

```powershell
python -m algorithms.auction_market.value_of_market `
  --scenario northern_lights_phase1_milkrun_imbalanced --seeds 1 2 3 4 `
  --hard-probs 0.0 0.5 1.0
```

> MPC 只是一个*参考*,不是研究主线;市场的论点是去中心化、局部信息、近零算力与在困难
> 竞争下的协调能力。/ MPC is only a reference, not the thesis.

## 第五步:让价格真正做功 / Making prices bind

在无预算拍卖中,分配只取决于出价*排序*,支付是纯转移,**钱不影响分配**(故 PoA ≡ 1,
"拍卖"实际退化为学习型优先级规则)。`budgeted.py` 引入两项耦合:

1. **统一出清价**:赢家支付最高*落选*出价 → 价格内生,运力充裕时为 0,恰在争夺稀缺
   运力时上升(拥塞定价);
2. **可耗尽预算**:`预算 = budget_factor × 碳价 × 标称捕集率 × 回合小时`;付不起出清价
   即失去名额 → **现在花钱 = 以后失去运力**。

```powershell
python -m algorithms.auction_market.budget_study `
  --scenario northern_lights_phase1_milkrun_imbalanced --seeds 1 2 3 `
  --budget-factors -1.0 1.0 0.5 0.25 0.1 0.05 --anarchy-budget-factor 0.1
```

**预算扫描**(720h, seed 1–5,承诺机制 m=10/floor=100t;规则基线 vent 58,771 t):

| budget_factor | vented t | 出清价 € | 被预算挡下次数 |
|---|---:|---:|---:|
| 无预算 / 0.5 | 48,176 | 270,555 | 0 |
| 0.25 | 49,228 | 267,114 | 2.2 |
| 0.10 | 52,383 | 275,692 | 7.6 |
| 0.05 | 54,090 | 304,704 | 7.8 |
| 0.02 | 54,959 | 292,158 | 10.0 |

**非平凡无序代价**(budget_factor = 0.05,压价网格 0.25–3.0):

| 组合 | vented t | 社会成本 € |
|---|---:|---:|
| 如实出价 | 54,090 | 5,759,817 |
| **自私纳什均衡** | **50,119** | **5,473,489** |
| 搜索到的最优组合 | 48,421 | 5,337,567 |

- **PoA = 1.026(成本)/ 1.035(放空)> 1**——自私均衡确实造成真实效率损失。
- **如实出价自身低效 = 1.079**:有预算时"说真话"反而浪费 7.9%——因为它抬高出清价、
  过快耗尽预算,导致后续紧急源被挡。**压价在此既是个体理性、也是社会改善**,这是有预算
  拍卖里一个非平凡且可发表的现象。

## 第七步:三方账本与分配公平 / Three-party ledger and distribution

`budget_study` 现在同时报告来自经济账本的**精确**三方物理成本,以及拍卖转移支付
(seed 1–5,budget_factor=0.05,如实出价):

| 主体 | 金额 € | 说明 |
|---|---:|---|
| 排放方(调理 + 放空) | 5,273,402 | 绝对主导 |
| 船运方(燃油 + 装卸) | 440,001 | |
| 封存方(再调理) | 46,414 | |
| *拍卖转移支付* | *465,173* | *转移,非物理成本* |

**逐源总负担(放空成本 + 支付)极不均衡**:

| 排放源 | 总负担 € |
|---|---:|
| celsio | 3,276,824 |
| brevik | 1,409,637 |
| yara_sluiskil | 105,894 |

- **成本几乎全部落在排放方**,船运/封存只占 ~8%——说明在此设定下,链条的经济压力来自
  碳价与放空,而非物流本身。
- **负担分布高度不均**(celsio 是 yara 的 31 倍):市场按紧迫度定价的结果是"最难服务的源
  承担最多"。这为**公平性/补偿机制**研究提供了直接抓手。

Once payments couple to allocation (uniform clearing price + depletable budgets),
selfish overbidding creates a genuine congestion externality: **PoA = 1.02 > 1**.

要点 / Takeaways:
- **市场对定价摩擦稳健**:即使预算收紧到最紧(0.05),市场放空 51,963 t 仍显著优于规则
  55,044 t——市场价值不依赖于"免费"运力。
- **预算本身是摩擦而非改进**:收紧预算使物理结果变差(46,390 → 51,963 t),因为预算耗尽
  会挡住紧急排放源。它的作用是让**价格具有真实经济含义**,从而使博弈非平凡。

> 限定:3 个 seed、粗糙的压价网格(0.25/0.5/1/2)、2 轮最优反应;PoA 幅度(2%)较小,
> 且社会最优用"搜索到的最优组合"近似。/ Caveats: 3 seeds, coarse grid, approximate optimum.

## 第六步:给市场加"承诺" / Commitment: fixing the balanced-scenario failure

**诊断**:均衡场景(milkrun)的退化**不是**"同一艘船来回改道",而是**贪心紧迫度会系统性
饿死某个源**——它从来不是当下最紧急的,于是规则原本要服务它的船被不断抢走,直到它积压
爆发。修法是给市场**承诺**:除非明显值得,否则保持规则的分区。

两个参数(均在 `policies.py` 的 `may_reassign`):

- `--commitment-margin m`:只有 `bid_winner > (1+m) × bid_rule目的地` 才允许改派;
- `--defend-floor-t X`:堵住**零出价饿死漏洞**——缓冲仍有余量时近视出价为 0,`(1+m)×0 = 0`
  意味着任何正出价都能抢走它(无论边际多大)。下限按"至少 X 吨价值处于风险中"防守规则
  目的地,代表"在源变紧急前持续服务"的**期权价值**。

**结果**(720h, seed 1–4, 放空降幅%;正 = 市场更好):

| 场景 | 纯反应式 (m=0) | **m=10, floor=100t** | m=10, floor=500t |
|---|---:|---:|---:|
| 3vessels(均衡) | −9.5 | **0.0** ✅ | 0.0 |
| milkrun(均衡) | −93.8 | **−15.3** | 0.0 ✅ |
| milkrun_imbalanced | +17.1 | **+18.0** ✅ | +3.2 |
| milk_run_stress | +56.7 | **+58.0** ✅ | +9.7 |

- **`m=10, floor=100t` 严格优于纯反应式市场**:在 3 个场景上更好(3vessels 完全修好、
  不均衡与受压场景**还略有提升**),并消除了 milkrun 84% 的损害。
- 存在清晰的**帕累托前沿**:继续加大 floor 可把 milkrun 修到 0,但会把市场推向纯规则
  (floor=2000t 时全部为 0.0,即市场完全退化为规则)。**承诺强度是可调的设计旋钮**,不是
  非黑即白。

```powershell
python -m algorithms.auction_market.value_of_market `
  --seeds 1 2 3 4 --episode-hours 720 --hard-probs 0.0 `
  --commitment-margin 10.0 --defend-floor-t 100.0
```

> 结论 / Takeaway:**纯反应式市场的短板可以用承诺机制修复,且几乎不牺牲(反而略微提升)
> 市场在困难场景的优势。** 这把"市场 vs 规则"从二选一变成了一条可调的连续谱。
> Commitment repairs the reactive market's failure mode at no cost to (indeed with a
> small gain in) its advantage where contention is real.

## 第八步:自适应承诺与"永不劣于规则"保证 / Adaptive commitment & a no-regret guarantee

目标:让市场在**任何**场景都不劣于规则,同时尽量保留困难场景的优势。试了两条路。

### 路线 A(失败):基于填充统计的自适应承诺 — `commitment.py`

用运行时信号(填充极差 + 峰值填充)自动调节 `defend_floor`:均衡时收紧、失衡时放开。
**结论:没有赢过固定承诺。**

| 场景 | 固定 floor=100 | 自适应(0.5/0.85) | 自适应(0.2/0.5) |
|---|---:|---:|---:|
| 3vessels | 0.0 | −0.2 | −0.2 |
| milkrun | **−15.3** | −20.8 | −32.3 |
| imbalanced | +18.0 | **+21.1** | +18.0 |
| milk_run_stress | **+58.0** | +9.1 | **+58.0** |

**失败原因(有价值的诊断)**:填充统计无法区分两种"高压"——`milkrun` 是**运力绝对不足**
(2 船 3 源、捕集率相同),重分配只是把短缺搬来搬去(**零和**);`stress`/`imbalanced` 是
**分配错配**,重分配是**正和**。两者填充率都高,信号看不出差别。

### 路线 B(成功):反饿死保护时域 — `--protect-horizon-h`

物理判据,局部而非全局:**若规则目的地会在 H 小时内溢出,则永不抢走它的船**——因为那只是
搬移短缺。

| 场景 | 规则 | 保护 H=72 | **保护 H=168** | 保护 H=336 |
|---|---:|---:|---:|---:|
| 3vessels | 0 | −0.2 | **0.0** ✅ | 0.0 |
| milkrun | 0 | −18.2 | **0.0** ✅ | 0.0 |
| imbalanced | 0 | +17.2 | **+9.3** | +1.3 |
| milk_run_stress | 0 | +18.6 | **+1.9** | 0.0 |

**`H=168` 达成了"永不劣于规则"**(四个场景全部 ≥ 0)。

### 代价:保证不是免费的 / The guarantee is not free

| 配置 | 净放空节省 | 说明 |
|---|---:|---|
| 固定 floor=100(容忍小幅落后) | **+38,440 t** | milkrun/3vessels 合计仅损失 1,766 t |
| 保护 H=168(永不落后) | **+6,369 t** | 全部场景 ≥ 规则 |

**坚持"永不劣于规则"要付出约 83% 的市场价值。** 这是一条清晰的**安全–性能前沿**:

> 如果场景已知或可检测,用宽松承诺拿满收益;如果必须对任何工况给出不劣于现状的保证,
> 用 `--protect-horizon-h 168`,代价是只保留约六分之一的收益。
> Insisting the market never underperforms the rule costs ~83% of its value --
> a clean safety/performance frontier rather than a free lunch.

```powershell
# 收益最大(容忍均衡场景小幅落后)
python -m algorithms.auction_market.value_of_market --commitment-margin 10 --defend-floor-t 100
# 永不劣于规则(牺牲多数收益)
python -m algorithms.auction_market.value_of_market --commitment-margin 10 --defend-floor-t 50 --protect-horizon-h 168
```

## 下一步 / Next
2. **可扩展性论证**:随排放源/船数增加(竞争更强),市场优势应扩大而算力保持平坦——
   直接支撑"市场可扩展、中央规划器不可扩展"的论点。
3. **机制与公平**:用二价/VCG 与 `payoff.py` 的三方账本,展示市场给出的价格信号与三方
   盈亏分配;引入预算/拥塞外部性研究非平凡无序代价。拍卖产出即 `DispatchGoal`,可接入
   [contracts.py](../contracts.py) 分层。
