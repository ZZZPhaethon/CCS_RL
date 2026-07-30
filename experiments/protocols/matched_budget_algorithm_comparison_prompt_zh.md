# 同 simulator-budget 算法比较 Prompt

把下列 `[算法名称]`、`[算法入口或实现位置]` 和 `[输出目录名]` 替换后直接使用：

```text
请对 [算法名称] 进行 validation-only 的同预算训练与比较。开始前必须完整阅读：

1. docs/paper_experiment_plan_zh.md
2. experiments/protocols/unified_window_v1_paper_protocol.json
3. experiments/protocols/unified_window_v1_seed_manifest.json

算法入口或实现位置：[算法入口或实现位置]
结果目录：experiments_results/[输出目录名]

固定实验协议：

- 使用 northern_lights_phase1_3vessels 和 unified_window_v1。
- 每个 sampled scenario 为 888 h；只允许前 720 h 执行动作和计分，后 168 h 仅为只读 forecast context。
- 学习控制器统一使用单一 168 h structured summary，不含 valid_fraction；不得改用 24/72、完整未来序列或算法专属 future features。
- 上层控制器只能决定船舶调度。井注入由共享的 continuous maximum-feasible-rate 底层控制器执行，不得给该算法额外井控制自由度。
- 使用完全相同的 action feasibility masks、物理约束、初始状态、扰动分布和经济参数。
- 训练目标必须与正式总成本对齐：gamma=1，不加入 stored-CO2 credit 或算法专属 reward shaping。训练 terminal return 和 validation 总成本都必须包含 common compact terminal cleanup。
- validation 报告总成本必须满足：
  total_cost_eur = 720h_episode_cost_eur + terminal_cleanup_operating_cost_eur。

统一训练环境交互预算：

- 每个独立 model seed 的 hard cap 为 B_selected = 9,505,319 次底层 1 h simulator step calls。
- model seeds 固定为 0、1、2，每个 model seed 分别享有相同 hard cap；不能把三个 seed 的预算合并。
- 只要实际推进底层 simulator 1 个物理小时，就计 1 call。
- vectorized/parallel workers 的 calls 全部求和。
- 实际轨迹、policy roll-in、候选动作 rollout、Greedy/counterfactual trajectory 和其他训练期 simulator rollout 全部计入。
- 已推进 simulator 的失败、截断或弃用样本也必须计入。
- 每次 simulator.step 前检查剩余预算；不足以完成算法不可拆分的成对或成组推进时提前停止，严禁超支。
- validation、formal test、网络 forward/backward、重复 SGD epoch和不推进 simulator 的 learned-model/feature computation不进入 B_selected；但应分别报告。
- 不要求强行耗尽预算；可以 early stop，但必须报告实际 calls、预算使用率和停止原因。

数据与 seed 权限：

- 训练只能使用 training-only seeds。若该算法需要新的训练 seed 流，必须先写入 manifest，并证明与 controller-validation、legacy-development 和 formal-test 范围不相交。
- checkpoint、训练轮数和算法超参数只能使用 controller-validation seeds 8100001–8100020 选择。
- 严禁在方法、checkpoint、预算和报告口径锁定前生成、读取、评估或调试未访问测试集
  `9000031–9000060`。该集合不得用于任何进一步选择。
- 不得根据 formal test 修改算法、预算、summary、checkpoint 或报告规则。

执行要求：

1. 先审计 [算法名称] 当前实现的 observation、action、reward、cleanup 和 simulator-call counter；发现不一致时，只做使其符合协议的最小修改。
2. 先用非正式测试 seed 做小预算 smoke test，验证 hard cap、cleanup 恒等式、168 h summary 和输出 schema。
3. 在可用计算资源上并行训练 model seeds 0/1/2，但保持三个 run 的数据、checkpoint 和计数器完全隔离。
4. 在每个 run 内保存预算内 validation-best checkpoint；不得用 best training seed 代替三-seed 稳定性报告。
5. 将每个最终 checkpoint 在相同的 20 个 controller-validation seeds 上，与同场景 Greedy 进行配对评估。
6. 在访问 formal test 前停止并先向我汇报结果。

每个 model seed 必须报告：

- 实际 training simulator calls、预算使用率和是否触及 hard cap；
- validation 总成本、720 h episode cost、cleanup cost和单位成本；
- 相对 Greedy 的配对成本差、95% paired-bootstrap CI、胜/平/负；
- vented CO2、stored CO2、captured CO2及末端 inventory；
- 训练稳定性、停止 epoch/step、checkpoint 选择依据；
- 数据生成 wall time、网络训练时间、CPU/GPU-hours、峰值内存；
- 在线 median/P95 决策延迟和完整 720 h episode wall time。

三 model seeds 汇总时：

- 报告 mean、between-seed SD 和 median；
- 对相对 Greedy 的差值使用训练 seed × 场景 seed 的 hierarchical bootstrap；
- 单列每个 model seed 的结果，不能只报告最好的 seed；
- 明确区分 training-budget calls、validation calls、learned-model rollouts 和开发搜索开销。

交付物至少包括：

- 配置快照和代码/环境 provenance；
- 每 seed checkpoint 与训练 summary；
- simulator-call 审计 JSON；
- 20-seed validation CSV；
- 三-seed aggregate CSV/JSON；
- 相对 Greedy 的配对统计；
- README，说明协议合规、失败任务和排除规则。

在当前算法完全锁定前不要访问修订后的正式 comparison set，不要静默覆盖已有结果，
不要提交代码。完成 validation-only 阶段后，先向我汇报 [算法名称] 的表现、计算量、
稳定性，以及它相对 Greedy 和 Iterative Action-Q 的结论。
```
