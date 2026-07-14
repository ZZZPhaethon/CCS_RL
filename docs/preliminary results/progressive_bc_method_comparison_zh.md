# MPC、Greedy 与递进式 BC 方法对比

## 1. 对比口径

- 所有学习策略均采用 deterministic/argmax 执行。
- 学习策略使用 5 个 model seeds（0–4），每个模型在相同的 20 个 evaluation seeds（101–120）上评估，共 100 条 720 h rollout。
- Greedy 和 Rolling MPC 使用相同的 20 个 evaluation seeds。
- “存储量”表示整个 episode 的累计 `stored_t`，不是期末未处理库存。
- 单位成本先在每个 episode 中用成本除以实际存储量，再跨 seeds 求平均。
- 总成本定义为：

\[
\text{总成本}=\text{运营成本}+80\,€/\mathrm{t}\times\text{Venting}.
\]

## 2. 完整递进结果

| 类别/步骤 | 方法 | Seed 数量 | Venting ↓ | 存储量 ↑ | 运营成本 ↓ | 运营单位成本 ↓ | 总成本 ↓ | 总单位成本 ↓ | Vent 相对 Greedy | Vent 相对 MPC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher | **Rolling MPC** | 20 eval | **514 t** | **109,583 t** | €1.597M | **€14.58/t** | **€1.638M** | **€14.96/t** | −7,500 t | — |
| 基线 | **Greedy** | 20 eval | 8,014 t | 101,341 t | **€1.481M** | €14.64/t | €2.122M | €21.12/t | — | +7,500 t |
| ① | TCN + 标准 BC | 5 model × 20 eval | 7,950 t | 100,683 t | €1.521M | €15.13/t | €2.157M | €21.69/t | −63 t | +7,437 t |
| ② | TCN + mode + 标准 BC | 5 model × 20 eval | 6,632 t | 102,566 t | €1.544M | €15.08/t | €2.075M | €20.45/t | −1,382 t | +6,118 t |
| PPO 旁支[^ppo] | TCN + mode + PPO | 5 model × 20 eval | 5,713 t | 102,937 t | €1.550M | €15.07/t | €2.007M | €19.63/t | −2,301 t | +5,199 t |
| ③ | TCN + mode + decision-only BC | 5 model × 20 eval | 6,176 t | 103,377 t | €1.552M | **€15.04/t** | €2.046M | €19.98/t | −1,838 t | +5,662 t |
| ④ | + terminal 必须卸空 | 5 model × 20 eval | 6,602 t | 103,572 t | €1.557M | €15.05/t | €2.085M | €20.30/t | −1,412 t | +6,088 t |
| ⑤ 当前最佳 | **+ current destination** | **5 model × 20 eval** | **3,726 t** | **105,118 t** | €1.585M | €15.10/t | **€1.883M** | **€18.01/t** | **−4,287 t** | +3,213 t |
| ⑥ 待实验 | 当前最佳 BC + PPO | — | 尚未训练 | — | — | — | — | — | — | — |

[^ppo]: `TCN + mode + PPO` 是旧 PPO 旁支，没有使用 decision-only loss、terminal 新规则和 current destination，不能视为当前最佳 BC 的 PPO 结果。

## 3. 逐步变化

| 新增内容 | Venting 变化 | 存储量变化 | 总成本变化 | 总单位成本变化 |
|---|---:|---:|---:|---:|
| TCN → TCN + mode | −1,319 t | +1,883 t | −€82k | −€1.24/t |
| + mode → + decision-only | −456 t | +811 t | −€29k | −€0.47/t |
| + decision-only → + terminal 必须卸空 | +425 t | +195 t | +€38k | +€0.32/t |
| + terminal 必须卸空 → + destination | **−2,875 t** | **+1,546 t** | **−€202k** | **−€2.30/t** |

terminal 必须卸空没有直接改善性能，但它修正了业务逻辑，因此不能因为 venting 略有上升而撤销。当前最大增益来自 current destination observation。

## 4. 当前最佳方法与基线

当前最佳方法为：

```text
TCN forecast
+ operation mode
+ current destination
+ decision-only BC loss
+ terminal 必须卸空
+ 100 个 MPC demonstration seeds
+ 50 BC epochs
```

它仍然是 BC-only，尚未在此 checkpoint 基础上训练 PPO。

相对 Greedy：

- venting 减少 4,287 t，改善 53.5%；
- 存储量增加约 3,777 t；
- 总成本减少约 €239k；
- 总单位成本降低 €3.11/t。

相对 Rolling MPC：

- 仍多 vent 3,213 t；
- 少存储约 4,465 t；
- 总成本高约 €245k；
- 总单位成本高 €3.04/t。

按 venting 排序：

```text
Rolling MPC：514 t
< 当前 destination BC-only：3,726 t
< 旧 mode PPO：5,713 t
< decision-only BC：6,176 t
< mode BC：6,632 t
< TCN BC：7,950 t
< Greedy：8,014 t
```
