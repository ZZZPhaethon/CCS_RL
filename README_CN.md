<h1 align="center">
  <img src="assets/Logo.png" alt="Logo" width="400"/>
</h1>
<h3 align="center">
面向船运 CCS 调度的物理仿真、优化控制与强化学习平台
</h3>
<p align="center">
  语言：<a href="readme.md">English</a> | 简体中文
</p>

<p align="center">
  <img src="assets/CCUSoverview_research.png" alt="CCUS teaser" width="95%">
</p>

**CCS_RL** 是一个面向船运二氧化碳捕集、运输、接收、管输与海底注入的研究代码库。
当前以 Northern Lights 为主要参考场景，并围绕一个核心设计原则构建：把作为唯一
**物理事实来源**的物理层（`Simulation/`）与只负责决定“在什么目标下运行”的
**算法层**（`algorithms/`）彻底分开。

核心物理链条如下：

```text
Emitter -> Vessel -> Terminal -> Pipeline -> SubseaManifold -> InjectionWell -> Reservoir
```

`Simulation/` 按物理时间步（当前通常为一小时）推进网络，负责校验动作并给出实际的
封存、放空、成本、排放和压力状态。`algorithms/` 位于其上：稀疏的高层策略选择一个
`DispatchGoal`，底层执行器再把该目标转换为 `Simulation.environment.CCSEnv` 可执行的
原生动作。物理层的 `network.step()` 及其校验始终是可行性的最终判定。

## 架构

推荐的控制分层可避免 RL 绕过物理层，也避免其在每个仿真小时都学习底层动作：

```text
稀疏高层决策            （RL、启发式或 MILP/MPC 规划器）
        |  DispatchGoal：emitter<->vessel 偏好、注入率目标、重规划时域
        v
底层执行器              （规则、原生 MPC 或滚动 MILP）
        |  原生动作：{vessels: [...], wells: [...]}
        v
Simulation.environment -> simulator -> 物理约束 + 奖励
```

`algorithms/contracts.py` 用不依赖 Stable-Baselines3、CPLEX 或特定 MPC 实现的方式
定义该边界，因此可以在相同物理场景上公平比较规则、MPC、RL 和混合方法。

## 亮点

- **以物理层为唯一事实来源：** 所有提出的动作都由 `Simulation/` 执行并校验；算法层
  不重复实现物理能力、压力方程或流量裁剪规则。
- **求解器无关的目标/执行器接口：** `algorithms/contracts.py` 中的 `DispatchGoal`、
  `HighLevelPolicy` 与 `ActionExecutor` 将“追求什么目标”和“如何执行与校验”解耦。
- **混合控制器：** 目标感知规则执行器、经回放验证的原生 MPC 执行器，以及滚动 MILP
  优化基线（`algorithms/hybrid/`）。
- **多种 RL 方案：** 高层稀疏决策 PPO、事件触发残差 PPO、带规则反事实奖励与课程学习
  的掩码残差 PPO v2，以及带风险门控的 adaptive-greedy 变体（v3）。
- **公平比较框架：** `experiments/` 脚本为每个 seed 只采样一次扰动轨迹，深拷贝给每个
  控制器，并在接受结果前断言累计捕集量完全一致。
- **可复现的场景与扰动：** `data/scenarios/` 的 JSON 场景，加上 capture outage、
  maintenance、波高海况和船速影响（`Simulation/scenario_generation/`）。

## Roadmap

- [x] 物理层实体、操作模块和单步 network 结算。
- [x] Northern Lights Phase 1/Phase 2 及派生场景配置。
- [x] Action proposal/resolver 协议层。
- [x] 求解器无关的目标/执行器接口（`algorithms/contracts.py`）。
- [x] 混合规则、原生 MPC 与滚动 MILP 执行器。
- [x] 事件触发的高层稀疏决策 PPO。
- [x] 残差 PPO v1（在安全规则调度器之上的 7 动作干预）。
- [x] 掩码残差 PPO v2（动态掩码 + 持续规则反事实奖励 + 课程学习）。
- [x] 带风险门控的 adaptive-greedy 残差变体（v3）及风险门控扫描。
- [x] 严格配对的同场景比较框架。
- [ ] 统一实验配置文件和单一命令行入口。
- [ ] 将大型外部数据和训练模型权重整理为可下载 release assets。
- [ ] 补充项目打包元数据（`pyproject.toml`）和固定环境文件。
- [ ] 用 `algorithms/rl` 取代遗留的 `Simulation/training/train.py` 入口。

## 环境需求

- Python `>= 3.10`（开发环境使用 3.12）。
- 核心物理层：`numpy`、`CoolProp`、`searoute`。
- 控制 / RL：`gymnasium`、`stable-baselines3`、`sb3-contrib`、`torch`。
- 可选 MILP 基线：一个 MILP 求解器（PuLP 附带的 CBC，或用于更快滚动 MILP 研究的 CPLEX）。

仓库目前尚未提供打包元数据，因此 `Simulation/` 和 `algorithms/` 直接作为顶层包，从
仓库根目录使用。

## 安装

将依赖安装到你的环境（conda 或 venv），随后所有命令均在仓库根目录执行，以便
`Simulation` 和 `algorithms` 作为顶层包被正确解析。

```powershell
pip install numpy CoolProp searoute gymnasium stable-baselines3 sb3-contrib torch
```

如果命令找不到这些包，可将仓库根目录加入 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "$PWD"
```

## 快速开始

所有入口都是从仓库根目录运行的 Python 模块。

### 混合控制器比较（规则 vs 原生 MPC）

```powershell
python experiments\compare_hybrid_controllers.py `
  --scenario northern_lights_phase1_3vessels `
  --seeds 1 2 3 4 5 `
  --episode-hours 168 `
  --planning-horizon-hours 72
```

若有求解预算，可显式加入滚动 MILP：

```powershell
python experiments\compare_hybrid_controllers.py `
  --scenario northern_lights_phase1_3vessels `
  --seeds 1 2 3 4 5 --episode-hours 720 `
  --controllers rule native_mpc rolling_milp `
  --planning-horizon-hours 168 --milp-time-limit-seconds 30
```

结果写入 `output/hybrid_controller_comparison/`，包含原始/汇总 CSV 和元数据 JSON。
脚本默认拒绝覆盖已有结果，只有确定覆盖时才使用 `--overwrite`。

### 高层稀疏决策 PPO

```powershell
python -m algorithms.rl.train_high_level_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 --decision-interval-h 24 --event-triggered `
  --ent-coef 0.01 --timesteps 50000 --seed 0 --progress-mode lines

python -m algorithms.rl.evaluate_high_level_ppo `
  --run-dir logs\high_level_rl\YOUR_RUN_DIRECTORY `
  --seeds 1 2 3 4 5
```

训练产物（配置、实时状态、指标、检查点、最终模型）默认写入 `logs/high_level_rl/`。

### 残差 PPO v1（在安全规则调度器之上干预）

```powershell
python -m algorithms.residual_rl.train_residual_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 --forecast-context-hours 168 --decision-interval-h 24 `
  --timesteps 100000 --num-envs 4 --vec-env subproc `
  --hard-scenario-probability 0.30 --validation-every-steps 5000 `
  --seed 0 --device cpu

python -m algorithms.residual_rl.evaluate_residual_ppo `
  --run-dir logs\residual_rl\<run_name> --model best `
  --seeds 1 2 3 4 5 --hard-scenario-probability 0
```

### 掩码残差 PPO v2（规则反事实奖励 + 课程学习）

```powershell
# 静态困难场景混合
python -m algorithms.residual_rl_v2.train_masked_residual_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 --forecast-context-hours 168 --decision-interval-h 24 `
  --timesteps 20000 --num-envs 4 --vec-env subproc `
  --hard-scenario-probability 0.30 --validation-every-steps 2000 --seed 0 --device cpu

# 课程学习
python -m algorithms.residual_rl_v2.train_curriculum_masked_residual_ppo `
  --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 --forecast-context-hours 168 --decision-interval-h 24 `
  --timesteps 40000 --curriculum-stages 0.00:0.00 0.20:0.15 0.40:0.30 0.70:0.50 `
  --num-envs 4 --vec-env subproc --validation-every-steps 5000 --seed 0 --device cpu
```

训练前可沿规则轨迹枚举所有未被掩码的干预，确认存在可学习的正向动作：

```powershell
python -m algorithms.residual_rl_v2.validate_interventions `
  --seeds 1 4 --scenario northern_lights_phase1_3vessels `
  --episode-hours 720 --forecast-context-hours 168 --decision-interval-h 24 `
  --output-dir output\residual_action_validation_v2\<experiment_name>
```

### 风险门控残差 v3（adaptive-greedy 干预）

使用冻结的 v2 MaskablePPO 策略扫描 adaptive 风险门控：

```powershell
python -m algorithms.residual_rl_v3.sweep_risk_gate `
  --run-dir logs\residual_rl_v2\<run_name> --model best `
  --output-dir output\residual_action_validation_v2\<experiment_name>
```

### 严格配对的同场景比较

以下脚本为每个 seed 只采样一次 `720 h + 168 h` 扰动轨迹，深拷贝给每个控制器，
只执行前 720 h，并在报告前断言累计捕集量完全一致：

```powershell
python experiments\compare_shared_scenario_controllers.py `
  --scenario northern_lights_phase1_3vessels --seeds 1 2 3 4 5 `
  --episode-hours 720 --forecast-context-hours 168 `
  --controllers rule ppo rollout_mpc `
  --ppo-run-dir logs\high_level_rl\<run_name> `
  --output-dir output\fair_controller_comparison\<experiment_name>

python experiments\compare_shared_masked_residual_v2.py `
  --run-dir logs\residual_rl_v2\<run_name> --model best `
  --scenario northern_lights_phase1_3vessels --seeds 1 2 3 4 5 `
  --episode-hours 720 --forecast-context-hours 168 `
  --replan-hours 24 --planning-horizon-hours 168 `
  --controllers rule masked_residual_v2 rollout_mpc `
  --output-dir output\fair_controller_comparison\<experiment_name>
```

## 顶层目录

```text
CCS_RL/
|-- Simulation/     # 物理层：唯一的物理事实来源
|-- algorithms/     # 算法层：目标、执行器与 RL 策略
|-- experiments/    # 可复现、严格配对的控制器比较
|-- data/           # 场景 JSON、capture-rate profile、外部资料
|-- logs/           # 训练运行（high_level_rl、residual_rl、residual_rl_v2）
|-- output/         # 比较与验证输出
|-- assets/         # Logo 与图片
|-- README.md
`-- README_CN.md
```

## `Simulation` 包结构

```text
Simulation/
|-- entities/              # emitter、vessel、terminal、pipeline、manifold、well、storage、state
|-- actions/              # 动作协议、动作帧和实体级动作解析
|-- operations/           # 捕集、装载、运输、卸载、注入、压力限制
|-- environment/          # CCSEnv、工厂函数、forecast、Gymnasium/SB3 适配器
|-- control/              # baseline、rule-based、MILP、MPC、示范与模仿学习
|-- scenario_generation/  # 扰动生成 + 波高海况子模块
|-- training/             # 遗留 PPO 训练入口（已由 algorithms/rl 取代）
|-- visualization/        # dashboard payload 和 HTML 渲染
|-- economics.py          # 成本、碳价、放空惩罚、封存收益
|-- metrics.py            # rollout、KPI 和评估汇总
|-- network.py            # 物理网络图和单步结算
|-- network_scenarios.py  # 从 JSON/data 构造 Northern Lights 网络
|-- routes.py             # 经纬度、航线和球面距离
|-- ship_speed.py         # 海况（波高）到船速系数
`-- simulator.py          # 高层仿真执行器
```

## `algorithms` 包结构

```text
algorithms/
|-- contracts.py          # DispatchGoal、HighLevelPolicy、ActionExecutor、重规划日程
|-- evaluation.py         # 用于公平比较的物理回放评估器
|-- hybrid/               # 目标感知的规则 / 原生 MPC / 滚动 MILP 执行器
|-- rl/                   # 高层稀疏决策 PPO（Discrete(192)，事件触发）
|-- residual_rl/          # 事件触发残差 PPO（7 动作），在安全规则默认之上干预
|-- residual_rl_v2/       # 掩码残差 PPO：动态掩码、规则反事实奖励、课程学习
`-- residual_rl_v3/       # 带风险门控的 adaptive-greedy 残差变体与风险门控扫描
```

## 数据

部分外部数据体积较大，未全部纳入 git 仓库。下载后请放回对应目录再运行相关脚本。

- `data/scenarios/`：可复现场景 JSON —— `northern_lights_phase1`、
  `northern_lights_phase1_2well`、`northern_lights_phase1_3vessels`、
  `northern_lights_phase1_milkrun`、`northern_lights_phase1_milkrun_imbalanced`、
  `northern_lights_phase2`、`milk_run_stress`、`toy`。
- `data/capture_rates/`：emitter capture-rate profile 和 metadata。
- `data/Others/`：整理过的外部资料，例如三个参考排放源的 Climate TRACE source mapping
  和月度排放 profile。

## 公平比较

不同控制器族之间的 reward 数值**不可**直接横向比较：PPO 使用塑形的高层奖励，而
规则/MPC 使用仿真器奖励。公平评价应始终使用实际的 **stored_t**、**vented_t**、
运行/总成本、单位封存成本、封存/放空率、运行时间和物理违规次数，而非仅报告求解器的
计划目标值。每个比较脚本都为所有控制器固定场景、seed 和 forecast，并在接受某个 seed
前断言累计捕集量完全一致。

## 模块文档

每个包与子包都带有各自的双语 README，包含详细接口、数据流与约束：

- [`Simulation/README.md`](Simulation/README.md) 及其各子目录 README
- [`algorithms/README.md`](algorithms/README.md)
- [`algorithms/hybrid/README.md`](algorithms/hybrid/README.md)
- [`algorithms/rl/README.md`](algorithms/rl/README.md)
- [`algorithms/residual_rl/README.md`](algorithms/residual_rl/README.md)
- [`algorithms/residual_rl_v2/README.md`](algorithms/residual_rl_v2/README.md)
- [`experiments/README.md`](experiments/README.md)
- [`Simulation/scenario_generation/wave_height/README.md`](Simulation/scenario_generation/wave_height/README.md)

## Citation

如果你在论文或报告中使用本仓库，建议引用：

```bibtex
@software{ccs_rl,
  title  = {CCS_RL: Ship-Based CCS Dispatch Simulation and Reinforcement Learning},
  author = {CCS_RL contributors},
  year   = {2026},
  note   = {Research code for physical-layer CCS simulation, hybrid control, and reinforcement learning}
}
```

⭐ **如果本项目对你有帮助，欢迎 star！**
