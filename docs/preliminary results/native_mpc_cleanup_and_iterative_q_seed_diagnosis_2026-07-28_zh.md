# Terminal-cleanup 探索性对照与 Iterative Q seed 2 诊断（2026-07-28）

> 本文档仅用于内部诊断，不属于论文结果。论文正文、表格和方法列表不包含
> Native MPC。

## 1. Native MPC 加入 terminal cleanup 后的探索性对照

Native MPC 在每次重规划时，用 720 h 评估协议中的同一个 common compact
terminal cleanup 对候选末状态计价。最终总成本也统一定义为：

\[
C_{\mathrm{reported}}
= C_{\mathrm{episode},720\mathrm{h}}
+ C_{\mathrm{terminal\ cleanup}}.
\]

对照范围仅为 controller-validation seeds 8100001–8100003。Native MPC 使用
24 h replan、168 h planning horizon 和 economic objective。Greedy 与 Rolling
MPC 行沿用本轮已有探索性结果。Iterative Q 行已使用选定的
`single168 / P3 / model seed 0 / own-policy G1–G2` exact-loop checkpoint
重新评估；checkpoint SHA-256 为
`c70bc00967579594c02d233af66cf287631e6a65866a36f3209bac98678ffa75`。

| 方法 | 总成本 € | 单位成本 €/t | Vent t | Stored t |
|---|---:|---:|---:|---:|
| Greedy | 1,937,472 | 19.305 | 3,116.1 | 100,365.5 |
| Rolling MPC 30s | 1,937,472 | 19.305 | 3,116.1 | 100,365.5 |
| Rolling MPC 300s | 1,880,957 | 18.397 | 2,320.3 | 102,172.9 |
| Iterative Q（selected exact-loop） | 1,892,849 | 18.335 | 1,747.0 | 103,731.8 |
| Native MPC + cleanup（仅内部） | **1,713,108** | **16.411** | **43.3** | **104,504.7** |

Native MPC + cleanup 的平均总成本比 Greedy 低 11.58%，比 selected
exact-loop Iterative Q 低 €179,741（9.50%）。在这三个 seeds 上，它的单位成本
也低 €1.924/t，少 vent 1,703.8 t，并多存储 772.9 t。因此这次三-seed
探索性结果中 Native MPC 的四项汇总指标均优于 selected exact-loop
Iterative Q。由于样本仅有三个 validation seeds，该结果仍不能作为论文结论。

Selected exact-loop Iterative Q 的三-seed结果为 2 胜、0 平、1 负；平均
episode cost 为 €1,687,324，terminal cleanup 为 €205,525，总成本恒等式已
逐 seed 验证。本次复跑的三条记录与已归档的 20-seed exact-loop evaluation
中对应三条在所有数值字段上完全一致。此前未核实来源的
`€1,754,608 / €16.310/t / 0 vent / 107,640.9 t` 行已删除。

Native MPC 三个 seed 的逐 seed 结果、aggregate、环境检查日志和正式运行日志
保存在
`experiments_results/exploratory/native_mpc_cleanup_seeds_8100001_8100003/`。
Borg 环境检查 job 32966 和比较数组 job 32967 均以 exit code 0 完成，三个
末端候选轨迹均为 replay-exact。

## 2. 为什么 fixed-data 表中的 model seed 2 明显较差

这不是 terminal cleanup 计算错误，也没有证据表明 seed 2 的网络训练发散。
关键原因是该表属于 `fixed_data` 复训：三个 model seed 使用同一份
cleanup-aware G0–G2 数据，而不是分别用自身策略重新生成 iterative G1/G2。
因此它只能测量“同一固定数据上的初始化/SGD 敏感性”，不能完整代表
Iterative Q 的独立闭环复现性。

### 证据 1：差距主要来自 720 h 闭环 episode，而非 cleanup

| Model seed | Episode cost € | Cleanup € | 总成本 € | Vent t | Stored t |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,594,007 | 228,912 | 1,822,919 | 435.9 | 110,233.7 |
| 1 | 1,634,262 | 224,232 | 1,858,494 | 1,072.7 | 109,445.8 |
| 2 | 1,708,873 | 241,332 | 1,950,205 | 2,082.1 | 106,963.8 |

相对 seed 0，seed 2 的总成本高约 €127.3k，其中约 €114.9k 来自 episode
cost，cleanup 只贡献约 €12.4k。它还多 vent 1,646.2 t、少 stored 3,269.8 t。

### 证据 2：离线验证指标没有显示 seed 2 崩溃

三个 seed 的 validation pairwise accuracy 分别为 0.607、0.595 和 0.602；
top-1 improving fraction 分别为 0.379、0.402 和 0.392。seed 2 与另外两个
模型处于同一范围。因此，离线 composite checkpoint 指标没有捕捉到闭环中
连续干预造成的状态分布偏移。

### 证据 3：劣化集中在少数闭环场景

seed 2 相对 seed 0 成本差最大的五个 evaluation seeds 贡献了总差距的
60.3%。其中：

- seed 8100019：seed 2 比 seed 0 高 €414.2k，vent 6,157.9 t，而 seed 0
  为 0 t；
- seed 8100013：seed 2 比 seed 0 高 €342.3k，vent 4,941.1 t，而 seed 0
  为 0 t；
- seeds 8100004、8100011 和 8100009 进一步贡献了主要差距。

这说明小的 Q 排序差异经过闭环动作序列后，在少数困难场景中被放大为大量
vent，并非所有 20 个场景都一致变差。

### 证据 4：按每个 model seed 自身策略重建 G1/G2 后差距显著收窄

| Model seed | Fixed-data 总成本 € | Exact closed-loop 总成本 € |
|---:|---:|---:|
| 0 | 1,822,919 | 1,837,041 |
| 1 | 1,858,494 | 1,869,260 |
| 2 | 1,950,205 | 1,872,186 |

seed 2 在 own-policy G1/G2 下改善约 €78.0k；三 seed 的极差由约 €127.3k
降至 €35.1k。这直接支持“固定 iterative 数据造成的闭环分布失配放大了
seed 2 劣化”这一解释。剩余差距则属于真实的训练随机性和困难场景敏感性，
不应声称已完全消失。

## 建议

1. 不要把 fixed-data 表作为 model-seed 稳定性的主结果；它适合作为固定数据
   的初始化敏感性诊断。
2. 若论文需要跨 seed 结果，应采用每个 seed 独立生成 G1/G2 的 exact
   closed-loop 结果，并报告三 seed 均值、SD 和分层配对区间。
3. 对 seeds 8100019 和 8100013 做 intervention/action trace 对比，定位哪一
   个窗口的错误 override 首次导致后续 vent。
4. Native MPC 结果只保留在内部诊断材料中，不进入论文。
