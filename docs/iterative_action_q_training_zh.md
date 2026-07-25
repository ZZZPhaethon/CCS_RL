# 迭代式 Action-Q 训练

## 方法定位

当前算法不是 PPO，也不是单纯复制 Greedy 动作的 BC。它使用 Greedy
作为安全基线和第一批状态的来源，再反复收集模型自己会访问的新状态：

```text
Greedy 状态 G0 → 训练 P1
P1 roll-in 状态 G1 → 用 G0+G1 训练 P2
P2 roll-in 状态 G2 → 用 G0+G1+G2 训练 P3
P3 roll-in 状态 G3 → 用全部数据训练最终模型 P4
```

MPC 不参与数据生成、动作标注、训练或 checkpoint 选择，只用于最终结果对比。

## 经济目标

每个候选动作都从同一个决策状态出发，并完整模拟到 720 h 结束。标签为：

```text
target = 1e-5 ×（基准轨迹总成本 − 候选动作轨迹总成本）
总成本 = 运营成本 + Vent 罚金
```

因此 `target > 0` 表示候选动作比该状态下的基准动作更省钱。训练标签不是
下一小时 reward，也不只计算未来 168 h。

## 状态、动作与网络

- 输入：当前系统状态、船舶 mode、航行目的地、Greedy 当前建议和 episode 进度。
- 当前生产模型不读取 168 h future 数组，也没有上一动作、上一 reward、
  event 持续时间或跨 event 隐藏状态。
- 每艘船有 `WAIT / Terminal / 3个Emitter / FOLLOW` 六个局部动作。
- 三艘船形成最多 `6³ = 216` 个联合动作，物理上不合法的动作由 mask 屏蔽。
- 网络使用共享船舶 encoder、结构化动作 embedding、5 个 bootstrap heads 和
  每动作 51 个 quantiles。

模型在部署时只接受置信度足够高的 override；否则执行 `FOLLOW`。默认门控为：

- 至少 4/5 heads 同意；
- 相对 `FOLLOW` 的预测收益超过 0.40，即约 €40k；
- 每个 720 h 案例最多 8 次 override；
- 108–680 h 分为 8 个窗口，每个窗口最多一次 override。

## 固定生产配置

训练共使用 3 轮策略数据聚合、3,200 个名义训练 roots：

| 数据批次 | 生成策略 | Seeds | 每 seed roots | Roots | 占比 |
|---|---|---:|---:|---:|---:|
| G0 | Greedy | 200 | 8 | 1,600 | 50% |
| G1 | P1 | 40 | 8 | 320 | 10% |
| G2 | P2 | 60 | 8 | 480 | 15% |
| G3 | P3 | 100 | 8 | 800 | 25% |

8 个 root 位置为 episode 的 15%、25%、35%、45%、55%、65%、75% 和 85%。
策略 roll-in 会在相应窗口内保存第一个有效决策状态，因此少数案例可能因没有
有效 event 而产生少于 8 个实际 roots。

P1 从随机初始化开始，状态归一化直接由 G0 训练数据计算，不依赖任何旧模型。
P2–P4 继承上一阶段权重，并始终使用累计数据训练。每个训练阶段最多 40 epochs，
early-stop patience 为 8。

## 代码入口

本地检查配置而不提交任务：

```bash
DRY_RUN=1 bash hpc/launch_iterative_action_q.sh
```

在 Borg 提交完整训练：

```bash
bash hpc/launch_iterative_action_q.sh
```

可通过 `RUN_ROOT` 和 `CONFIG_NAME` 指定唯一输出目录和任务名前缀。启动器拒绝覆盖
已存在的运行目录，并把全部 SLURM job ID 写入 `RUN_ROOT/job_ids.txt`。

核心文件：

- `src/sim/control/iterative_action_q.py`：唯一生产网络；
- `experiments/generate_iterative_q_greedy_data.py`：G0 数据；
- `experiments/generate_iterative_q_policy_data.py`：G1–G3 数据；
- `scripts/train_iterative_action_q.py`：累计训练与 early stopping；
- `experiments/evaluate_iterative_action_q.py`：独立 seeds 上对比 Greedy；
- `hpc/launch_iterative_action_q.sh`：完整依赖链。

历史输出目录没有被清理；旧 checkpoint 可用于结果追溯，但不再是当前训练代码的依赖。
