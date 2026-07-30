# Iterative Q validation-only 配置搜索

## 最终推荐

推荐配置为 **reallocated B / P3 / margin50-cap12 / model seed 0**：

- 固定、策略无关的 G0：180 seeds × 12 roots = 2,160 nominal roots；
- 第 1 轮聚合 G1：30 seeds × 12 policy windows = 360 roots；
- 第 2 轮聚合 G2：30 seeds × 12 policy windows = 360 roots；
- 总计 2,880 nominal roots，2,879 个实际可训练 groups；
- 5-head ensemble 至少 4 heads 同意；
- Q margin = 0.50 reward units，即 €50,000；
- 12 个干预窗口，每窗口最多一次，最多 12 次干预；
- model seed 0；
- `B_selected = 7,495,692` 次底层 1 h simulator step calls。

`B_selected` 直接累加最终训练数据原始 shards 中记录的 calls：

| 数据阶段 | roll-in policy | calls |
|---|---|---:|
| G0 | Greedy；复用固定 bank | 5,639,992 |
| G1 | 候选 B 的 P1 | 963,857 |
| G2 | 候选 B 的 P2 | 891,843 |
| 合计 |  | **7,495,692** |

该预算按 paper protocol 只统计最终 checkpoint 使用的 training-data simulator calls；early-stopping validation 数据生成、controller-validation rollout、SGD 和开发搜索开销另列，不进入 `B_selected`。

## 固定 protocol

- 场景总长 888 h；
- 前 720 h 执行与计分；
- 后 168 h 仅作为只读未来信息；
- 模型只接收一个 168 h structured summary，不含 `valid_fraction`；
- 经济目标、Greedy 对照和正式测试 protocol 未改变；
- 训练 terminal target 与 validation total cost 都包含 compact terminal cleanup；
- 只使用 controller-validation seeds 8100001–8100020；
- formal test seeds 未访问；
- 未训练 PPO，未创建 git commit。

最终 checkpoint 在 20 个 controller-validation seeds 上：

| 指标 | 结果 |
|---|---:|
| 720 h episode cost | €1,655,266.95 |
| compact terminal cleanup | €222,122.56 |
| validation 总成本 | **€1,877,389.51** |
| Greedy 总成本 | €2,074,008.36 |
| 相对 Greedy | **-€196,618.85** |
| 95% bootstrap CI | [-€311,534.83, -€92,849.58] |
| 胜 / 平 / 负 | **16 / 0 / 4** |
| vent | 1,122.87 t |
| stored | 109,000.55 t |
| 平均干预次数 | 11.15 |

## 搜索方法

### 1. 固定 root-bank 代理筛选

现有 G0 是 Greedy roll-in 的策略无关数据，因此可以按 seed 取子集而不重新运行 simulator。先把所有候选的 nominal training roots 固定为 2,880，只改变轮数与每轮分配：

| 候选 | 分配 | 聚合轮数 | 代理总成本 | 胜/负 | vent | stored | 有效 bank calls |
|---|---|---:|---:|---:|---:|---:|---:|
| A（旧设计） | 2400 / 480 | 1 | €1.8880M | 17/3 | 1,371.55 | 109,027.07 | 7,598,673 |
| B | 2160 / 360 / 360 | 2 | €1.8761M | 17/3 | 1,020.54 | 110,071.21 | 7,499,694 |
| C | 1920 / 480 / 480 | 2 | €1.8967M | 14/6 | 1,240.60 | 108,007.00 | 7,549,057 |
| D | 1680 / 480 / 720 | 2 | €1.8752M | 15/5 | 1,009.76 | 109,343.31 | 7,595,764 |
| E | 1440 / 480 / 480 / 480 | 3 | €1.8447M | 16/4 | 699.69 | 109,575.15 | 7,689,323 |

这里的 B–E 只训练网络，不产生新的 simulator calls；表中的 calls 是生成所复用 bank 数据所需的实际 calls。由于 G1/G2/G3 来自旧基线策略，这一步只用于低成本排序，不能作为最终闭环结论。

### 2. 对 B 和 E 做精确闭环重生

保留代理筛选最有希望且计算结构不同的 B 与 E。P1 只依赖固定 G0，因此直接复用其精确 P1 checkpoint；之后用候选自己的 P1 生成 G1、训练 P2，再生成下一轮数据。

| 候选 | 精确总成本 | 胜/负 | vent | stored | 实际 training calls |
|---|---:|---:|---:|---:|---:|
| B | €1.8801M | 15/5 | 1,167.58 | 109,720.85 | 7,495,692 |
| E | €1.9423M | 11/9 | 1,938.95 | 107,760.55 | 7,628,972 |

E 的代理值从 €1.8447M 退化到精确闭环的 €1.9423M，说明旧 G3 bank 对四轮配置产生了明显乐观偏差。B 的代理与精确结果仅相差约 €4.0k，因此选择 B。

### 3. 只在 B 上做局部 gate 搜索

未做全网格搜索，只比较 base12、margin30/50、strict5 和 cap10：

| gate | 总成本 | 胜/负 | vent | stored |
|---|---:|---:|---:|---:|
| strict5 / margin40 / cap12 | €1.8756M | 15/5 | 1,122.81 | 110,120.03 |
| margin50 / cap12 | €1.8774M | 16/4 | 1,122.87 | 109,000.55 |
| base12 | €1.8801M | 15/5 | 1,167.58 | 109,720.85 |
| cap10 | €1.8940M | 14/6 | 1,489.17 | 107,874.99 |
| margin30 / cap12 | €1.9024M | 15/5 | 1,470.23 | 108,664.59 |

seed 0 上 strict5 与 margin50 仅差 €1.8k，配对场景差异区间跨 0。固定精确数据、完整重训 model seeds 0/1/2 后：

- margin50/cap12：平均 €1.8860M，seed 间 SD €8.8k，平均 15.67/20 胜；
- strict5/cap12：平均 €1.8868M，seed 间 SD €10.6k，平均 15.33/20 胜。

因此最终使用跨 seed 略稳的 margin50/cap12，并选择该 gate 下 validation 最好的 model seed 0。

## 相对旧推荐

旧的继承分配 P2 使用 2,400 + 480 roots、7,598,673 calls。新配置：

- training calls 减少 102,981（1.36%）；
- 三个 model seeds 的平均总成本从约 €1.8983M 降至 €1.8860M；
- 增加一次小规模策略聚合，但不增加 nominal root 总量。

旧 `selected/` 目录保留为前一轮搜索的历史产物；本轮权威产物是 `root_reallocation/selected/`，顶层 `recommendation.json` 已指向该目录。

## 文件索引

- `root_reallocation/allocation_comparison.csv`：A–E 代理与 B/E 精确闭环对比；
- `root_reallocation/gate_comparison.csv`：B 的局部 gate 搜索；
- `root_reallocation/model_seed_stability.csv`：两个最终 gate 的 seeds 0/1/2 复核；
- `root_reallocation/exact/`：逐阶段训练、逐 seed validation、calls 与 job provenance；
- `root_reallocation/proxy/`：固定 bank 代理训练和 calls 审计；
- `root_reallocation/selected/`：最终 config、checkpoint 和 validation 结果；
- `recommendation.json`：机器可读的最终推荐与 `B_selected`。
