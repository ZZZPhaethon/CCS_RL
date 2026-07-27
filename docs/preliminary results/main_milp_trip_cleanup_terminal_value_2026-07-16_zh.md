# 主 MILP 接入纯航次 cleanup 末端价值：初步验证

日期：2026-07-16

> 后续更新：本文定位的末端船舶位置计价偏差已经修复。4个窗口的内部/回放cleanup平均绝对误差由约€24,701降至€1,661，详见 `main_milp_cleanup_boundary_position_fix_2026-07-16_zh.md`。下文保留修复前结果作为诊断记录。

## 结论

纯航次 cleanup 子模型已经作为可选末端价值接入主 CPLEX MILP。实验说明它**确实能让主 MILP 考虑 168 h 规划边界后的清空责任**：4 个窗口中，回放到规划末端时的未储存 CO₂ 平均从 50,784.8 t 降到 27,725.1 t，减少 23,059.7 t（45.4%）。

但当前版本仍为实验开关，默认关闭，暂不应用于正式 720 h rolling MILP。原因不是末端价值没有生效，而是：

1. 30 s 和 120 s 下所有主问题都只返回 `Integer Feasible`，没有证明 optimal，gap 仍然很高；
2. MILP 内部末端库存与 CCSEnv 回放一致，但末端船舶位置仍有差异，造成位置相关的 cleanup 价值低估约 €6k–€36k；
3. 用回放末状态重新计算“当前窗口成本 + cleanup 成本”后，4 个窗口平均反而增加 €5.6k，仅 1/4 窗口改善。

因此，本次试验已经回答“主 MILP 会不会考虑末端状态”：**会，而且影响很强**；但还不能回答“是否稳定改善完整 rolling 经济结果”。

## 接入的目标函数

启用后，经济目标从

\[
J_{\mathrm{base}}
= C_{0:H}^{\mathrm{operating}}
+80\,M_{0:H}^{\mathrm{vent}}
\]

变为

\[
J_{\mathrm{tail}}
= C_{0:H}^{\mathrm{operating}}
+80\,M_{0:H}^{\mathrm{vent}}
+V_{\mathrm{cleanup}}(s_H).
\]

其中，`V_cleanup(s_H)` 是直接嵌入同一个主 MILP 的小型纯航次 MILP。它读取规划末端的源端库存、船上货物、终端库存和船舶位置，并在“无后续捕集、无天气扰动”的假设下协调计算：

- 清空所有剩余 CO₂ 所需的整数航次数；
- 船舶燃油成本；
- conditioning、reconditioning 成本；
- 装载和卸载期间的 hotel 成本；
- 参与清空的空船最后返回最近 emitter 的燃油成本。

它没有展开额外 144–216 个逐小时控制变量，因此 cleanup 部分仍由 MILP 联合优化，但规模增量相对有限。开关为 `terminal_cleanup_value=True`；默认值为 `False`，旧实验不受影响。

## 30 s、4 窗口成对实验

配置：seed 1；状态时刻 0、168、384、552 h；每次规划 168 h；使用同一 MPC warm start；每个求解限时 30 s；将得到的完整 168 h 动作在 CCSEnv 中回放，再对真实回放末状态独立求解纯航次 cleanup。

| 状态时刻 | 回放末端未储存 CO₂：无/有 tail (t) | 回放末状态 cleanup：无/有 tail (€) | 实际估算总成本变化 (€) | CPLEX gap：无/有 tail | 状态 |
|---:|---:|---:|---:|---:|---|
| 0 | 40,632 / 24,082 | 289,426 / 279,090 | +415 | 63.92% / 32.64% | Integer Feasible |
| 168 | 53,318 / 25,970 | 404,756 / 235,650 | +18,266 | 32.71% / 27.63% | Integer Feasible |
| 384 | 55,192 / 30,977 | 405,538 / 361,101 | -7,656 | 48.65% / 28.50% | Integer Feasible |
| 552 | 53,997 / 29,872 | 369,345 / 196,462 | +11,209 | 50.99% / 35.50% | Integer Feasible |

这里的“实际估算总成本”为：

\[
C_{0:H}^{\mathrm{replay}}+80M_{0:H}^{\mathrm{replay,vent}}
+V_{\mathrm{cleanup}}(s_H^{\mathrm{replay}}).
\]

平均结果：

- 规划末端未储存 CO₂：减少 23,059.7 t；
- 回放末状态 cleanup 价值：减少 €99,190；
- 实际估算总成本：增加 €5,558；
- 平均求解墙钟时间：33.04 s（无 tail）与 33.01 s（有 tail），限时条件下没有明显额外耗时；
- presolve 后平均增加约 71 行、586 列；
- 前 24 h vent 基本不变，执行成本平均增加约 €1,159，储存量平均增加约 801 t。

这说明 tail 价值主要改变了窗口后段的库存处理，并非简单改写第一个动作。

## 120 s 复核

对 168 h 和 552 h 两个窗口将限时提高到 120 s。两个版本仍然只返回 `Integer Feasible`：

| 状态时刻 | gap：无/有 tail | 末端未储存 CO₂：无/有 tail (t) | 实际估算总成本变化 (€) |
|---:|---:|---:|---:|
| 168 | 32.71% / 25.50% | 53,318 / 25,970 | +18,266 |
| 552 | 50.71% / 34.67% | 53,997 / 29,872 | +6,642 |

120 s 改善了部分 bound/gap，但没有实质改变 incumbent 动作和结论。因此当前问题不能只靠把 30 s 延长到 120 s 解决。

## 已定位的主要误差：末端船舶位置

启用 tail 的 4 个 30 s 窗口中，主 MILP 的末端 source/cargo/terminal 库存与 CCSEnv 完整回放在数值精度内一致；但主模型把末端位置表示为到达节点的路径弧，动作物化并回放后，有些船在第 168 h 仍处于 `sailing_to:*` 状态。

由于纯航次 cleanup 的第一段航行成本依赖船舶末端位置，这一位置差异导致：

| 状态时刻 | 主模型内 cleanup (€) | 回放末状态 cleanup (€) | 低估 (€) |
|---:|---:|---:|---:|
| 0 | 256,672 | 279,090 | 22,418 |
| 168 | 201,608 | 235,650 | 34,042 |
| 384 | 324,984 | 361,101 | 36,117 |
| 552 | 190,235 | 196,462 | 6,227 |

这不是纯航次 cleanup 本身的质量问题：独立验证中，500/500 个 cleanup 小问题均由 CPLEX 求到 Optimal，20 个协调 CCSEnv 清空回放全部 0 vent，成本 MAE 为 €1,390.72（0.552%）。当前偏差来自主 MILP 对规划边界时船舶位置的表达与动作回放不完全一致。

## 测试与产物

新增/修改：

- `src/sim/control/cplex_milp.py`：纯航次 cleanup 末端子模型和分项结果；
- `src/sim/control/rolling_milp.py`：rolling controller 可选开关和结果透传；
- `experiments/compare_terminal_cleanup_value_windows.py`：成对求解、CCSEnv 回放和回放末状态 cleanup 评价；
- `tests/test_cplex_milp.py`、`tests/test_rolling_milp.py`：目标分项及开关透传测试。

5 个相关回归测试通过。原始结果：

- `output/terminal_cleanup_value_4windows_30s_2026-07-16/`
- `output/terminal_cleanup_value_2windows_120s_2026-07-16/`

## 下一步

在运行正式 720 h rolling 对比前，应先修复或保守化“规划末端船舶位置 → cleanup 第一段航行成本”的映射，并重新跑 4–10 个窗口。验收标准应同时满足：

1. 主模型内 `V_cleanup` 与回放末状态独立 cleanup 的差异显著收敛；
2. 同一时间预算下不只降低末端库存，还能降低回放口径的完整估算总成本；
3. 然后再进行 720 h、多个 seed 的 rolling MILP 与 MPC 对比。
