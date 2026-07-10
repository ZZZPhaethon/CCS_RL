# 基于 MPC 教师的 Forecast 编码 RL 对比实验设计

## 实验目标

在同一套受控实验协议下，对比三种 episode 长度为 720 小时的 MaskablePPO 智能体：

1. **当前状态 PPO**：接收当前运行状态和当前全域天气，但不接收未来 disturbance 序列。
2. **Flat-forecast PPO**：接收相同的当前状态，以及展平后的 `168 x 9` forecast。
3. **TCN-forecast PPO**：接收相同的当前状态和相同的 `168 x 9` forecast，并通过小型时间卷积网络将 forecast 编码为 64 维 latent。

三种智能体都使用同一批经过严格 replay 验证的 `RollingNativeMpcController` demonstrations，先进行行为克隆（BC）warm-start，再在 PPO 阶段使用相同的衰减 kickstarting 锚定。本实验只研究 forecast 信息及其编码方式的作用，不同时修改奖励函数。

## 固定实验设置

- 场景：`northern_lights_phase1_3vessels`。
- Yara buffer：使用场景文件中注册的 15,000 t，不做运行时覆盖。
- RL episode：720 个一小时时间步。
- MPC 重规划间隔：24 h。
- MPC 预测时域：168 h。
- 天气过程：每 24 h 重新采样一个所有船舶共享的全域速度系数，即 `block` 模式。
- Capture noise、capture outage、高产窗口、well maintenance、初始库存和 warm-start 设置：三种模型完全一致。
- 奖励函数：使用现有 `vent_first`，以 venting 为首要目标、overflow risk 为稠密预警、operating cost 为逐步次要目标。
- Dispatch：允许 partial-load dispatch。
- 动作掩码：三种模型使用完全相同的 MaskablePPO action masks。

第一次对比实验不加入 downstream-inventory potential shaping。若此时修改奖励，会把 forecast 架构差异与奖励差异混在一起。Episode 结束时的未储存库存作为评估指标保留；如果模型出现“不 vent 但把任务拖到 episode 结束”的行为，再把库存推进 shaping 作为独立的奖励消融实验。

## 信息一致性约束

MPC 教师和使用 forecast 的 RL 学生必须接收同一条 disturbance trajectory。应提供统一的 forecast view，让教师和学生从同一数据接口读取预测，避免教师通过环境内部状态读取隐藏未来，而学生只能接收另一套摘要。

第一阶段是 oracle-forecast 实验：未来 168 小时的值直接来自已采样 scenario，因此是精确未来值。这与当前 `RollingNativeMpcController` 的行为一致，因为它通过复制环境并沿已采样未来滚动候选策略来选动作。

后续可将统一 forecast view 替换为带误差的预测，但届时 MPC 和 RL 必须读取同一份带误差 forecast，不能让 MPC 继续读取真实未来。

### 当前状态分支

当前状态向量包含现有的归一化物理状态：

- 一周内的时间位置和系统总在途库存比例；
- 每个 emitter 的 buffer fill、当前 capture 和当前 availability；
- 每艘船的 cargo、靠泊/位置、目的地和航行进度；
- terminal fill 和泊位可用情况；
- 当前 well injection、injectivity 和 availability；
- reservoir pressure margin；
- 当前全域天气系数和当前归一化航行时间。

对于三船场景，将当前天气从未来天气摘要中分离后，预计约为 51 维。精确维度和 feature names 由环境生成并通过测试验证，不在训练代码中硬编码。

### Forecast 分支

Forecast 覆盖未来 `t+1..t+168` 小时，形状为 `[168, 9]`。当前时刻 `t` 已由当前状态分支表示，因此不在 forecast 中重复：

1. 三个归一化 emitter capture-rate 通道；
2. 三个二值 emitter-availability 通道；
3. 一个二值 well-availability 通道；
4. 一个归一化 well-injectivity 通道；
5. 一个全域 vessel-speed-factor 通道，位于通道索引 8，也就是第 9 个通道。

Emitter capture 使用对应 emitter 的最大生产率归一化。即使 outage 会令 capture 下降，仍保留独立的二值 availability 通道，以区分真正 outage 与普通低产状态。

天气是全域共享量，因此不按船舶重复复制未来 168 h 天气。各船当前到不同目的地的 travel time 继续保留在当前状态分支中。在 24 h block 天气模式下，第 9 个通道在每个物理天气 block 内保持分段常数，依次表示当前 block 的剩余部分及后续天气 blocks。

环境以 time-major 顺序输出 `[168, 9]` forecast。TCN extractor 在 batch 内将其转置为 PyTorch `Conv1d` 所需的 `[batch, 9, 168]`，底层 forecast 数据不发生变化。

所有 forecast 值必须为有限数，并使用稳定、可验证的通道顺序。Forecast metadata 需要记录通道名称、horizon、scenario 配置和归一化常数。

## Episode 尾部处理

720 h rollout 在第 719 h 仍需要完整的未来 168 h forecast，同时 SB3 的 timeout bootstrap 还需要第 720 h 的 terminal observation 包含同样完整的 forecast。因此 scenario generator 应生成 889 h 的 disturbance，但 RL 环境仍在 720 h 截断：

```text
720 h RL episode + 169 h forecast/bootstrap context = 889 h scenario trajectory
```

MPC demonstration 环境运行时拥有 889 h trajectory，但每个 episode 只收集前 720 个 state-action pairs。这样每个被收集的 MPC 决策以及 RL terminal observation 都拥有完整的 168 h lookahead。

RL 环境仍在 720 h 截断，但从相同的 889 h trajectory 中读取未来 forecast。额外的 169 h 只作为预测与 timeout-bootstrap 上下文，不计入 episode KPI，也不计入 PPO timesteps。

Timeout observation 必须是真实的第 720 h 状态：当前 emitter availability、well availability 和 injectivity 从第 720 h scenario 非破坏性读取，future forecast 则从第 721 h 开始。

这样可以避免 MPC 在 episode 后 168 h 内预测时域逐渐从 168 h 缩短到 1 h，也不需要通过重复最后一个值或补零制造虚假 forecast。

## 三种策略模型

### Variant A：当前状态 PPO

- 输入：只有当前状态分支。
- 目的：测量没有未来 disturbance 信息时的性能。
- 仍然使用 MPC demonstration 做 BC 和 kickstarting，因此该模型也可以衡量 privileged teacher 在学生看不到未来时造成的标签歧义。

### Variant B：Flat-forecast PPO

- 输入：当前状态与展平后的 1,512 个 forecast 值拼接。
- 特征提取器：普通 MLP。
- 目的：测量完整 oracle forecast 本身带来的收益，同时不加入时序归纳偏置。

### Variant C：TCN-forecast PPO

- State extractor：小型 MLP，输出 64 维特征。
- Forecast extractor：沿时间轴执行三层一维卷积，保留时间位置，然后线性投影为 64 维。
- PPO actor/critic 之前的组合特征：`64 + 64 = 128` 维。
- 不使用 global average pooling，因为它会抹去 outage 是明天发生还是在预测时域末端发生的区别。

Forecast encoder 与 Variant B 接收完全相同的原始 `[168, 9]` 张量。它压缩的是策略网络内部的有效表示，而不是减少环境提供的信息。

## Demonstration 数据流水线

MPC demonstrations 在 GPU 训练前离线生成并缓存。每条记录包含：

- 当前状态 observation；
- 完整 `[168, 9]` forecast；
- 展平后的 native action；
- 动作合法性 mask；
- 环境 seed 和当前仿真小时；
- MPC 选中的 candidate 及 replay-validation 状态；
- scenario 与 feature schema metadata。

每个 episode 都必须严格 replay 验证。存在以下任一情况时，整条有问题的 trace 不得用于训练：

- infeasible action；
- replay metric 不一致；
- feature schema 不一致；
- forecast 长度不足 168 h。

三种模型使用同一份 demonstrations。当前状态模型在输入阶段丢弃 forecast；Flat 和 TCN 模型使用 forecast。这样三种模型的教师动作和 demonstration states 完全一致。

## BC 与 Kickstarting

现有 imitation stack 只支持单个 NumPy observation array，需要扩展为支持结构化 observation。以下环节都必须支持 current-state、flat-forecast 和 TCN-forecast：

- demonstration collection；
- mini-batching；
- `evaluate_actions`；
- sample weighting；
- kickstarting callback。

继续对非 WAIT 的船舶动作使用 action-dimension-specific up-weighting。Forecast encoder 的参数必须参与 BC、PPO 和 kickstarting 更新。不增加独立 autoencoder 预训练阶段。

三种模型的 kickstarting coefficient、衰减计划、BC episodes、BC epochs、动作权重、PPO timesteps、PPO hyperparameters 和 model seeds 完全一致。

## 训练与评估协议

### Stage 1：本地和 short queue smoke test

- 一个 24 h demonstration episode，并保证每一步拥有完整 168 h forecast；
- 每种 observation variant 执行一个 BC epoch；
- 每种模型执行一个短 PPO rollout；
- 验证 CUDA device、loss 有限、action-mask 兼容、模型保存/加载和确定性 replay。

### Stage 2：Pilot 对比

- 一个训练 seed；
- 30 个 MPC demonstration episodes；
- 每种模型 100,000 PPO timesteps；
- 5 个未参与训练的 paired evaluation seeds。

### Stage 3：正式对比

- 3 个独立 model seeds；
- 所有 model seeds 使用同一份固定 demonstration cache；
- 每个模型在 10 个 held-out paired environment seeds 上评估；
- 报告逐 seed 结果、mean、standard deviation、95% confidence interval 和 paired differences。

Training seeds、demonstration seeds 和 evaluation seeds 必须互不重叠，具体列表写入 run manifest。

### Reference controllers

在同一批 held-out disturbance trajectories 上评估 idle、greedy 和 `RollingNativeMpcController`。

对每个学习模型，同时评估：

- BC-only checkpoint；
- PPO 训练后的最终 checkpoint；
- stochastic policy；
- deterministic policy。

## 评估指标

首要指标：

- 720 h 内总 vented tonnes。

次要指标：

- loss rate；
- episode 结束时的未储存库存，并分别统计 emitter、vessel 和 terminal；
- stored tonnes 和 storage rate；
- operating cost 和 actual total cost；
- cost per stored tonne；
- longest venting streak；
- berth waiting、well throttling 和 pressure-risk hours；
- MPC imitation negative log-likelihood；
- per-action accuracy、joint-action accuracy 和 non-WAIT vessel-action accuracy；
- demonstration、BC 和 PPO wall-clock time；
- policy parameter count 和 inference latency。

主要结论基于相同 evaluation seeds 上的 paired deltas，不能仅根据某个模型最好的单个 seed 宣称其更优。

## 必需消融实验

用户要求的三种模型本身就是本阶段必需的消融：

1. 无未来 forecast；
2. 相同 forecast + Flat MLP；
3. 相同 forecast + TCN-64。

本阶段不混入奖励或教师变化。

如果 TCN 胜出，后续再对 latent width 做消融。如果所有 forecast 模型都出现 episode 末端库存堆积，再单独进行 downstream-progress potential shaping 奖励实验，而不是在当前实验中临时修改奖励。

## HPC 任务组织

MPC demonstration generation 主要消耗 CPU，应与 GPU PPO training 分开：

1. environment/dependency smoke job；
2. replay-validated demonstration generation job，或按 seed 切分的 job array；
3. demonstration cache audit 和 manifest 生成；
4. 三个使用同一 cache 的 GPU pilot jobs；
5. pilot 验证通过后，再提交正式 model-seed job array；
6. paired evaluation 和汇总报告 job。

正式 GPU training 使用 Borg `root` partition、`long` QoS，每个训练任务一张 GPU，支持 checkpoint、unbuffered logs 和显式 output directory。

每个任务记录：

- git commit；
- Python/Conda environment；
- 全部配置；
- seed sets；
- observation schema hash；
- demonstration cache hash。

## 错误处理

- Forecast shape 错误、存在非有限值或通道顺序不一致时立即失败。
- Demonstration cache 的 scenario 或 normalization metadata 与训练配置不一致时拒绝加载。
- MPC replay mismatch 时失败，不允许自动回退到 greedy demonstration。
- 通过共享 run manifest 固定三种模型的 seeds 和 hyperparameters。
- 保存 demonstration shards 和 PPO checkpoints，使 SLURM timeout 后能够安全恢复。
- 不把用户已有的工作区改动混入实验提交。

## 验证与成功标准

只有满足以下全部条件，才允许进入 pilot：

1. current-state、flat-forecast 和 TCN observation 都具有经过测试的 shape 和确定性 feature order；
2. 同一个 sampled seed 给 MPC 和 RL 产生完全相同的当前状态与 forecast；
3. RL 第 719 h 仍能获得完整 168 h forecast；
4. 缓存的 MPC actions 能够精确 replay，并且满足保存的 legality masks；
5. BC 和 kickstarting 能为三种模型更新正确的 extractor 参数；
6. 三种模型都能完成短 MaskablePPO 训练、保存、重新加载和评估；
7. 三种模型之间只有 forecast exposure/encoding 不同；
8. Pilot report 包含 paired venting、inventory、storage、cost、imitation 和 runtime metrics。

本实验分别回答两个问题：

1. RL 获得未来 168 h forecast 后，控制性能是否提高？
2. 对同一份 forecast，时间编码器是否比展平后的普通 MLP 使用得更有效？
