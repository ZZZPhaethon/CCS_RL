# Iterative Q v2：从 P0 开始的完整递归 teacher 实验

## 结论

用户提出的 teacher 链定义是正确的：

`P0 (Greedy/FOLLOW) → P1 → P2 → P3 → P4`

此前的 P4-only selective-anchor 实验只是在已有 v1 P3 上训练 P4，不能视为一条完整的
v2 teacher 链。本实验已从 P0 开始重新生成后续各轮数据并训练全部模型，但完整递归
v2 在 20 个 controller-validation seeds 上明显退化，因此该 checkpoint 不应被选用。

本实验只访问 seeds 8100001–8100020，没有访问 formal test。

## “从头训练”的具体含义

- P0 不是神经网络 checkpoint，而是 Greedy/FOLLOW 行为策略。
- P1 随机初始化，在 G0 的每个 root 上使用 P0/FOLLOW 作为 teacher anchor。
- P2 从 P1 checkpoint 初始化，最新 G1 的 teacher 是 P1。
- P3 从 P2 checkpoint 初始化，最新 G2 的 teacher 是 P2。
- P4 从 P3 checkpoint 初始化，最新 G3 的 teacher 是 P3。
- 每轮保留累计 exact 数据：P1 用 G0；P2 用 G0+G1；P3 用 G0+G1+G2；
  P4 用 G0+G1+G2+G3。
- G1、G2、G3 均分别由新训练的 P1、P2、P3 重新 rollout，并非复用 v1 数据。

因此，“从头”不是让 P2–P4 每轮再次随机初始化，而是从 P0 开始完整重建
teacher/data 链，同时让学生继承上一轮 checkpoint。

固定配置：hard anchor coefficient 1.0、exact release margin €40,000、
anchor temperature 0.5、uniform inference margin 0.4、4/5-head gate、
12 个决策窗口、最多 12 次 override、model seed 0。

## 20-seed validation 结果

| 模型 | 平均总成本 € | 相对 Greedy | 胜/平/负 | Vent t | Stored t | 单位成本 €/t | 实际 overrides |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 2,074,008 | — | — | — | — | — | 0 |
| Recursive P1 | 1,995,865 | −3.77% | 12/0/8 | 3,332.3 | 105,967.7 | 18.894 | 3.60 |
| Recursive P2 | 1,939,933 | −6.46% | 12/1/7 | 2,585.3 | 108,216.0 | 17.986 | 3.75 |
| Recursive P3 | 2,033,211 | −1.97% | 10/0/10 | 3,635.1 | 105,785.3 | 19.297 | 3.90 |
| **Recursive P4** | **2,038,247** | **−1.72%** | **12/0/8** | **4,097.2** | **105,282.2** | **19.388** | **3.00** |
| v1 uniform P4 | 1,825,688 | −11.97% | 17/0/3 | 728.1 | 109,924.9 | 16.624 | 11.05 |
| P4-only selective-anchor | 1,790,875 | −13.65% | 18/0/2 | 330.3 | 110,919.5 | 16.157 | 8.70 |

Recursive P4 相比 P4-only selective-anchor 平均贵 €247,372，paired bootstrap
95% CI 为 [€143,836, €367,801]，仅 4/20 seeds 更好。相比 v1 P4 平均贵
€212,559，95% CI 为 [€119,077, €317,069]，同样仅 4/20 seeds 更好。

## Teacher 机制是否真正生效

| 学生 | 初始化 | teacher | protected / released roots | protected top-1 agreement |
|---|---|---|---:|---:|
| P1 | 随机初始化 | P0 Greedy/FOLLOW | 197 / 281 | 91.88% |
| P2 | P1，载入 36 个张量 | P1 | 53 / 41 | 88.68% |
| P3 | P2，载入 36 个张量 | P2 | 76 / 68 | 97.37% |
| P4 | P3，载入 36 个张量 | P3 | 119 / 121 | 95.80% |

这说明退化不是因为 teacher 没有连接、上一轮 checkpoint 没有载入，或 anchor loss
没有生效。问题在于从保守的 Greedy P0 开始施加强 anchor 后，探索行为没有充分建立。
完整递归模型每个 episode 实际只执行约 3–4 次 override；v1 P4 为 11.05 次，
P4-only selective-anchor 为 8.70 次。训练 root 上的高 teacher agreement 也不能保证
未见 validation 轨迹上的闭环稳定性。

## Seed 8100017 的闭环退化

| 阶段 | 总成本 € | 相对 Greedy € | 运行成本 € | Vent penalty € | Vent t | Stored t | overrides |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 1,923,140 | −331,419 | 1,811,939 | 111,201 | 1,390.0 | 110,823.6 | 3 |
| P2 | 1,850,063 | −404,497 | 1,799,547 | 50,516 | 631.4 | 116,904.3 | 4 |
| P3 | 2,318,867 | +64,308 | 1,731,913 | 586,954 | 7,336.9 | 105,907.1 | 4 |
| P4 | 2,857,488 | +602,929 | 1,622,860 | 1,234,628 | 15,432.9 | 98,570.7 | 5 |

P4 的运行成本相对 Greedy 仍节省约 €67,859，但 vent penalty 增加约 €670,787，
使净结果劣化 €602,929。这是闭环策略分布漂移造成的代价放大，不是 cost 汇总错误。
虽然 8100017 是最严重的 seed，但 Recursive P4 相比既有模型的 paired CI 全部位于
零以上，因此总体退化并非只由这一个 seed 决定。

## 选择决定

- `Recursive P4`：拒绝，不进入 formal test，不作为选定模型。
- `P4-only selective-anchor`：仍是当前 validation 最优 checkpoint，但应准确称为
  “P4 selective-anchor fine-tuning/ablation”，不应描述为完整递归 v2。
- 若继续改进，最直接的下一项消融是延迟启用 anchor，让前几轮先形成探索能力，
  再从较成熟的 teacher 开始保留行为；不需要修改网络架构。

## 产物与复现

- 汇总数据：`analysis_summary.json`
- 逐 seed 评估：`eval/iterative_q_v2_recursive_p1` 至
  `eval/iterative_q_v2_recursive_p4`
- 模型与训练摘要：`p1` 至 `p4`
- 数据：`g0` 至 `g3`
- 固定协议：`protocol_lock.txt`、`schedule.txt`
- 作业清单：`job_manifest.txt`
- 日志：`logs`；所有作业退出码均为 0，38 个 stderr 文件均为空。

Checkpoint SHA-256：

- P1: `2c01e318088e8680f73b8be4a3559a583423630c15b2a0ffd7d730a841e5ae2d`
- P2: `58fc18c41b3b194ba9d17886935b1cf819798be61247d50ab2bce986a180a397`
- P3: `3873e93f344029f14d9e1212aa2ea6b239c0b091230a455ce6853f77dda4cbbe`
- P4: `e77b4d7ecedf9b4158b4ecdb058ebfac281be4acd4a606596e44a080148f0d5`

SLURM jobs 33156–33171 全部成功完成。
