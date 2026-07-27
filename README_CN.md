<h1 align="center">
  <img src="assets/Logo.png" alt="Logo" width="400"/>
</h1>
<h3 align="center">
面向船运 CCS 链条的物理约束仿真与学习式调度控制
</h3>
<p align="center">
  语言：<a href="README.md">English</a> | 简体中文
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="MIT"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-blue?style=flat-square" alt="Python >=3.10">
  <a href="https://drive.google.com/drive/folders/147lfZ1M1d3Am0v65fk1SX0jsXmk2lVzN"><img src="https://img.shields.io/badge/Google%20Drive-Dataset-4285F4?logo=googledrive&logoColor=white&style=flat-square" alt="Google Drive Dataset"></a>
</p>

<p align="center">
  <img src="assets/CCUSoverview.png" alt="CCUS teaser" width="95%">
</p>

CCS_RLLLM 是一个研究船运 CO₂ **运输—封存链条小时级运行调度**的代码库。它把物理约束仿真器
与一整套控制器放在同一框架下比较：启发式、Rolling MILP、多种 PPO，以及当前主方法
**Iterative Action-Q**，全部在同一份冻结的扰动与信息协议下评估。

物理链条为：

```text
Emitter -> Vessel -> Terminal -> Pipeline -> SubseaManifold -> InjectionWell -> Reservoir
```

所有控制器只决定 **船舶调度**。井注入由共享的自动控制器接管，始终取当前物理可行的最大注入率，
因此任何方法都无法通过放松物理约束获得优势。`src/sim/` 是唯一的物理事实来源：校验每个动作、
推进捕集、航行、装卸、注入与储层压力，并输出可审计的轨迹、成本账本和 KPI。

---

## 研究问题

> 在具有天气、capture 和注入能力扰动的三船 CCS 运输—封存系统中，一个以 Greedy 为安全默认、
> 通过 counterfactual rollouts 学习**少量高价值干预**的方法，能否在较低在线决策开销下降低
> 总成本和 CO₂ venting？

这是待验证的论点而非既定结论。完整论证结构见
[`docs/paper_structure_zh.md`](docs/paper_structure_zh.md)，实验设计锁定在
[`docs/paper_experiment_plan_zh.md`](docs/paper_experiment_plan_zh.md)。

## 主方法 —— Iterative Action-Q

Iterative Action-Q 既不是 PPO，也不是单纯复制 Greedy 的 BC。Greedy 提供安全默认动作和第一批状态
分布，模型随后反复收集**自己会访问到的**状态：

```text
Greedy 状态 G0        -> 训练 P1
P1 roll-in 状态 G1    -> 用 G0+G1 训练 P2
P2 roll-in 状态 G2    -> 用 G0+G1+G2 训练 P3
P3 roll-in 状态 G3    -> 用全部数据训练 P4
```

- **标签。** 每个候选动作都从同一决策状态出发，完整模拟到 720 h 结束。标签为
  `1e-5 ×（基准轨迹总成本 − 候选动作轨迹总成本）`，因此正值表示该动作在整个时域上确实更省钱 ——
  不是下一小时 reward，也不是截断的 168 h 回报。
- **动作空间。** 每艘船在 `WAIT / Terminal / 3 个 Emitter / FOLLOW` 中选择；三船最多构成
  `6³ = 216` 个联合动作，物理不合法的动作由 mask 屏蔽。
- **网络结构。** 共享船舶 encoder、结构化动作 embedding、5 个 bootstrap heads、每动作 51 个 quantiles。
- **部署门控。** 只有在一致性和收益裕度都满足时才 override Greedy（例如至少 4/5 heads 同意、
  预测收益超过约 €40k），并对 override 次数设上限、分摊到固定干预窗口；否则执行 `FOLLOW`。

方法细节与生产配置见 [`docs/iterative_action_q_training_zh.md`](docs/iterative_action_q_training_zh.md)。

## 统一比较协议 —— `unified_window_v1`

所有控制器共享完全相同的三船网络、720 h episode、1 h 物理步长、初始状态生成规则、经济参数、
动作掩码，以及相同 seed 下的扰动轨迹。协议已冻结并可机器读取：

- [`experiments/protocols/unified_window_v1_paper_protocol.json`](experiments/protocols/unified_window_v1_paper_protocol.json) —— 环境、扰动强度、井控规则、forecast 协议、成本公式、训练预算。
- [`experiments/protocols/unified_window_v1_seed_manifest.json`](experiments/protocols/unified_window_v1_seed_manifest.json) —— 训练 / 验证 / 测试 seed 划分。

| 控制器 | 论文名称 | 是否训练 | 运行时未来信息 |
|---|---|---|---|
| 固定船–emitter 分工 | Fixed-Assignment Heuristic | 否 | 不使用 |
| 动态贪心调度 | Greedy | 否 | 不使用 |
| 从零训练 PPO | Centralized Maskable PPO | 是 | 同源 24/72 h 摘要 |
| Event-based v4 架构 | Event-Residual PPO | 是 | 同源 24/72 h 摘要 |
| **当前主方法** | **Iterative Action-Q** | 是 | 同源 24/72 h 摘要 |
| 滚动优化 | Rolling MILP | 否 | 逐小时完整 168 h forecast |
| 仅作离线参考 | Full-horizon MILP（限时） | 否 | 完美信息 |

关键公平性规则：所有 forecast-capable 方法使用同一份 forecast **来源**（当前为 perfect-forecast
协议）；三种学习方法共享同一环境交互预算 `B_4800`；所有方法的报告成本统一加入 common compact trip
cleanup 末端价值；正式测试 seeds `9,000,001–9,000,030` 在方法、checkpoint、门控和报告口径全部锁定
之前不得触碰。

## 初步结果

开发 seeds 上的比较（`8,000,001–8,000,030`，30 个配对 seeds，每种学习方法**仅一个训练随机种子**）。
这些**不是**正式结果 —— 锁定的正式测试 seeds 尚未运行。完整记录见
[`docs/preliminary results/unified_window_control_comparison_2026-07-26_zh.md`](docs/preliminary%20results/unified_window_control_comparison_2026-07-26_zh.md)。

| 方法 | 总成本 (EUR) | 相对 Greedy | Vent (t) | Stored (t) | 单位成本 (EUR/t) | 胜 Greedy |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 2,059,907 | — | 7,883.1 | 100,959.4 | 21.01 | — |
| Residual PPO v4 | 1,942,032 | −117,876 | 5,263.1 | 103,421.5 | 18.87 | 13/30 |
| Iterative Q（无未来） | 1,699,864 | −360,043 | 1,704.5 | 108,989.6 | 15.68 | 23/30 |
| **Iterative Q（24/72 h 未来）** | **1,633,631** | **−426,276** | **821.1** | **109,242.1** | **14.97** | **25/30** |
| Hybrid RL（Greedy 底） | 3,134,909 | +1,075,002 | 22,990.4 | 87,728.0 | 36.05 | 2/30 |

两个 Iterative Q 变体的配对 95% bootstrap CI 均不跨 0，但 Residual PPO v4 的
CI 为 [−309,118, +47,944]，**跨 0**，因此目前不能称其稳定优于 Greedy。加入 24/72 h 未来摘要后平均
再降 66,233 EUR，CI 为 [−119,717, −15,833]。

## 安装

以下命令均在仓库根目录执行。

```powershell
uv sync
uv run python -m pip install -e .
```

包含 RL 依赖（`numpy`、`gymnasium`、`stable-baselines3`、`sb3-contrib`）：

```powershell
uv sync --extra rl
uv run python -m pip install -e ".[rl]"
```

不使用 `uv`：

```powershell
pip install -e ".[rl]"
```

其他要求：

- Python `>=3.10`（GPU 训练环境当前使用 3.12）。
- 基础物理层依赖：`searoute>=1.6`、`CoolProp>=6.6`。
- Iterative Action-Q 和 event-based 控制栈额外需要 `torch`。
- Rolling MILP / Full-horizon MILP 需要 CPLEX；CBC 在 720 h 多 seed 研究中过慢。
- Wave-height 预测训练使用独立的 conda 环境：

```powershell
conda env create -f environment-gpu.yml
conda activate ccs-rlllm-gpu
pip install -e ".[rl]"
```

## 快速开始

### 物理层演示与 dashboard

```powershell
uv run python examples\run_physical_layer_demo.py
uv run python examples\build_phase1_dashboard.py
uv run python examples\build_rule_based_dashboards.py
```

### Iterative Action-Q 流水线

四个阶段直接对应命令行入口。在集群上请改用启动器（见下一节），它会自动串起整条依赖链。

```powershell
# 1. G0 —— Greedy roots
uv run python -m experiments.generate_iterative_q_greedy_data `
  --out-path output\iq\g0_train.pt --split train `
  --seeds (1500..1699) --roots-per-seed 12

# 2. 在 G0 上从随机初始化训练 P1
uv run python scripts\train_iterative_action_q.py `
  --train-data output\iq\g0_train.pt --validation-data output\iq\g0_val.pt `
  --out-dir output\iq\p1 --observation-input v4_future_24_72

# 3. G1–G3 —— 用当前策略 roll-in，保存它自己访问到的状态
uv run python -m experiments.create_iterative_q_lock `
  --checkpoint output\iq\p1\iterative_action_q.pt --out-path output\iq\p1\lock.json `
  --protocol-id unified_window_v1 --residual-margin 0.40 --economic-margin-eur 40000
uv run python -m experiments.generate_iterative_q_policy_data `
  --lock-config output\iq\p1\lock.json --out-path output\iq\g1_train.pt `
  --split train --seeds (1500..1539)

# 4. 在累计数据上继续训练，并在未见 seeds 上对比 Greedy
uv run python scripts\train_iterative_action_q.py `
  --train-data output\iq\g0_train.pt output\iq\g1_train.pt `
  --validation-data output\iq\g0_val.pt --initial-checkpoint output\iq\p1\iterative_action_q.pt `
  --out-dir output\iq\p2 --observation-input v4_future_24_72
uv run python -m experiments.evaluate_iterative_action_q `
  --checkpoint output\iq\p4\iterative_action_q.pt --out-dir output\iq\p4\eval
```

`--observation-input` 用于切换消融分支：`state_only`、`v4_future_24_72`（E1 锁定表示）、
`forecast_168`，或各类 summary window/band 变体。`--forecast-encoder` 可在 `small_mlp`、`tcn`、
`gru` 之间切换。

### 统一控制器比较

```powershell
uv run python -m experiments.compare_unified_window_controls `
  --iterative-q-checkpoint output\iq\p4\iterative_action_q.pt `
  --v4-run-dir output\event_v4\run1 `
  --out-dir output\comparison
```

输出逐 seed CSV 和汇总 JSON，包含成本分解、vent/stored 吨数、override 次数和运行时间。

### 集群运行

`hpc/launch_iterative_action_q.sh` 把数据生成、分阶段训练和评估串成一条 SLURM 依赖链。
只检查配置而不提交任务：

```bash
DRY_RUN=1 bash hpc/launch_iterative_action_q.sh
```

可通过 `PROJECT_DIR`、`RUN_ROOT` 和 `CONFIG_NAME` 指定自己的路径、输出目录和任务名前缀。启动器
拒绝覆盖已存在的运行目录，并把全部 job ID 写入 `RUN_ROOT/job_ids.txt`。单个阶段（数据、训练、
评估、消融、环境检查）对应 `hpc/submit_*.sh`。

### 其他控制器

```powershell
# 带 BC 预热的 Centralized Maskable PPO
uv run python scripts\train_ppo_bc.py --scenario northern_lights_phase1_3vessels `
  --reward-mode economic --bc-episodes 30 --bc-epochs 20 --timesteps 150000

# Forecast encoder 对比（demos -> merge -> train -> report 子命令）
uv run python scripts\compare_forecast_encoders_rl.py --help

# Wave-height 预测模型
uv run python -m sim.scenario_generation.wave_height.prediction.train_lstm
```

## 测试

```powershell
uv run python -m unittest discover -s tests
```

未做 editable 安装时：

```powershell
$env:PYTHONPATH="$PWD\src"
python -m unittest discover -s tests
```

## 仓库结构

```text
CCS_RLLLM/
|-- data/                 # Capture-rate profile 与外部参考数据
|-- docs/                 # 论文计划、方法说明、按日期归档的初步结果
|-- examples/             # 物理层演示与 dashboard 构建脚本
|-- experiments/          # 数据生成、评估、比较、消融分析
|   `-- protocols/        # 冻结的论文协议与 seed manifest
|-- hpc/                  # SLURM 启动器与分阶段提交脚本
|-- scenarios/            # 可复现场景 JSON
|-- scripts/              # 训练入口
|-- src/sim/              # 主 Python 包
|-- tests/                # 58 个单元、结构与实验冒烟测试
|-- environment-gpu.yml   # GPU 训练环境
`-- pyproject.toml
```

### `src/sim` 结构

```text
src/sim/
|-- actions/              # ActionProposal、ActionFrame、ActionResolver
|-- control/              # 控制器，见下节
|-- entities/             # Emitter、船舶、terminal、pipeline、井、储层状态
|-- environment/          # CCSEnv、工厂函数、forecast/past 观测、Gym 适配器
|-- operations/           # 捕集、装载、卸载、运输、注入、压力限制
|-- scenario_generation/  # 扰动与 wave-height 场景生成
|-- visualization/        # Dashboard 数据与 HTML 渲染
|-- economics.py          # 成本与收益模型
|-- line_source.py        # 储层/井压力 line-source 模型
|-- metrics.py            # Rollout、KPI、评估汇总
|-- network.py            # 物理网络图与单步结算
|-- network_scenarios.py  # 从 JSON/数据构建 Northern Lights 网络
|-- routes.py             # 航线与距离计算
|-- ship_speed.py         # 海况对船速的影响
`-- simulator.py          # 顶层仿真运行器
```

### `src/sim/control` —— 控制器家族

```text
control/
|-- baselines.py                # Idle 与 greedy shuttle 策略
|-- rule_based.py               # 固定分工与规则控制器
|-- milp.py / cplex_milp.py     # 静态 MILP benchmark 与 CPLEX 后端
|-- rolling_milp.py             # 带回放校验 warm start 的滚动 MILP
|-- native_mpc.py               # 多候选原生 MPC
|-- iterative_action_q.py       # 主方法：唯一生产 Q 网络
|-- recurrent_distributional_q.py
|-- imitation.py / demonstrations.py / replay.py
`-- event_based/                # 算法层，位于物理层之外
    |-- contracts.py            # DispatchGoal 边界：高层策略 <-> 底层执行器
    |-- evaluation.py           # 用于公平比较的物理回放评估器
    |-- hybrid/                 # 规则、原生 MPC、滚动 MILP 执行器
    |-- rl/                     # 稀疏 24 h 高层 PPO
    `-- residual_rl{,_v2,_v3,_v4}/  # 残差干预 PPO，v4 即 Event-Residual PPO
```

`event_based/` 只决定“在什么目标下运行”，绝不能在其中加入容量、压力方程或流量裁剪规则；这些必须
留在 `entities/`、`operations/` 和 `network.py`，以保证所有控制器受到同一套物理约束。详见
[`src/sim/control/event_based/README.md`](src/sim/control/event_based/README.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/paper_structure_zh.md`](docs/paper_structure_zh.md) | 论文论证链条与各章节证据要求 |
| [`docs/paper_experiment_plan_zh.md`](docs/paper_experiment_plan_zh.md) | E0–E5 实验设计、公平协议、指标与统计方法 |
| [`docs/iterative_action_q_training_zh.md`](docs/iterative_action_q_training_zh.md) | 方法定义、生产配置、代码入口 |
| [`docs/preliminary results/`](docs/preliminary%20results/) | 按日期归档的结果记录：控制器比较、encoder 比较、future adapter 与可复现性消融 |
| [`docs/CCS_RL_Research_Core_Idea.md`](docs/CCS_RL_Research_Core_Idea.md) | 最初的研究构想 |
| [`docs/physical_layer_v1_cn.md`](docs/physical_layer_v1_cn.md) | 物理层模型说明 |
| [`docs/northern_lights_line_source_pressure_study.md`](docs/northern_lights_line_source_pressure_study.md) | 储层压力 line-source 研究 |
| [`docs/experiments_summary.md`](docs/experiments_summary.md) | 早期 RL/LLM 阶段的历史记录（相关脚本已移除） |
| `src/sim/scenario_generation/wave_height/prediction/README.md` | Wave-height 预测模型 |

## 数据

部分大型外部数据未完整纳入 git 跟踪，运行相关脚本前需下载并放回对应目录。

- Google Drive：<https://drive.google.com/drive/folders/147lfZ1M1d3Am0v65fk1SX0jsXmk2lVzN?usp=sharing>
- `scenarios/` —— 场景 JSON，其中 `northern_lights_phase1_3vessels.json` 是论文协议使用的网络。
- `data/capture_rates/` —— Phase 1/Phase 1+ emitter capture-rate profile 与元数据。
- `data/网络收集资料/` —— 整理后的外部参考资料，如 Climate TRACE 排放源映射。

## 二次开发

**新增控制器。** 在 `src/sim/control/` 下实现（算法层控制器放入 `event_based/`），用
`ActionProposal` / `ActionFrame` 表达动作，经 `ActionResolver` 进入 `network.step()`，在比较实验中
注册，并在 `tests/` 下补充行为测试。

**新增场景。** 在 `scenarios/` 添加 JSON，新的 capture profile 放入 `data/capture_rates/`，在
`src/sim/network_scenarios.py` 或环境工厂中添加加载入口，并用演示或比较脚本验证。

**新增扰动。** 在 `src/sim/scenario_generation/generator.py` 生成 episode 时间序列，在
`disturbance_resolver.py` 定义运行时解析规则，接入 `CCSEnv` 或对应操作模块，并补充固定 seed 测试。
修改任何 `unified_window_v1` 扰动参数都需要**建立新的协议版本**并重跑所有方法。

## Roadmap

- [x] 物理实体、操作模块、network step 与压力限制。
- [x] Action proposal/resolver 协议层。
- [x] 规则、静态 MILP、滚动 MILP 与原生 MPC 控制器。
- [x] Gymnasium/SB3 RL 环境与 PPO/BC 训练入口。
- [x] Event-based 算法层：hybrid 执行器与 residual RL v1–v4。
- [x] Iterative Action-Q 的训练、评估与门控。
- [x] 冻结的 `unified_window_v1` 协议与 seed manifest。
- [ ] 完成协议待办项：在所有控制器接口中统一自动井控规则、补齐成本/活动诊断字段、加入 1 h
      simulator step 计数器以测定 `B_4800`。
- [ ] 用目标对齐的 reward 重新训练 Centralized Maskable PPO 与 Event-Residual PPO。
- [ ] 完成 ≥3 个独立训练种子，并在锁定测试 seeds `9,000,001–9,000,030` 上报告结果。
- [ ] 将 HPC 脚本中的个人路径改为环境变量配置。
- [ ] 将大型数据与模型权重整理为可下载 release assets。

## 📝 引用

```bibtex
@software{ccs_rlllm,
  title  = {CCS_RLLLM: Physics-Constrained Simulation and Learned Dispatch Control for Ship-Based CCS},
  author = {CCS_RLLLM contributors},
  year   = {2026},
  note   = {Research code for CCS chain simulation, optimisation control, and reinforcement learning}
}
```

⭐ **如果这个项目对你有帮助，欢迎点个 star！**
