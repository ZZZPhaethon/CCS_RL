# CCS_RLLLM
### 面向船运 CCUS 链条的物理仿真、优化控制与强化学习实验平台

语言：[English](README.md) | 简体中文

CCS_RLLLM 是一个围绕船运二氧化碳捕集、运输、接收、管输与注入流程构建的模块化研究代码库。当前代码以 Northern Lights 场景为主要原型，把物理层仿真、动作协议、规则控制器、MILP/MPC、RL 环境、扰动场景生成和可视化 dashboard 放在同一个可复现实验框架中。

核心链条如下：

```text
Emitter -> Vessel -> Terminal -> Pipeline -> SubseaManifold -> InjectionWell -> Reservoir
```

上层控制器、MILP、RL policy 或实验脚本提交动作；物理层负责校验动作、推进船舶移动、更新捕集/库存/卸载/管输/注入/压力状态，并输出可审计的轨迹和 KPI。

## 亮点

- **端到端 CCUS 物流仿真：** 覆盖 emitter、LCO2 vessel、terminal、pipeline、subsea manifold、injection well 和 reservoir。
- **动作协议与物理层解耦：** `sim.actions` 定义动作表达和解析，`sim.control` 只负责产生控制决策。
- **多类控制器对比：** 支持 idle/greedy baseline、规则控制器、静态 MILP benchmark、rolling MILP/MPC 和 RL policy。
- **可复现实验场景：** `scenarios/` 保存 Northern Lights Phase 1/Phase 2 等 JSON 场景，`data/capture_rates/` 保存 capture-rate profile。
- **扰动与天气建模：** 包含 capture outage、injectivity decline、maintenance、wave-height 场景和船速影响模块。
- **训练和评估闭环：** 提供 Gymnasium/SB3 适配器、PPO/BC 训练脚本、controller comparison 实验和 HTML dashboard 产物。

## Roadmap

- [x] 物理层实体、操作模块和 network step。
- [x] Northern Lights Phase 1/Phase 2 场景配置。
- [x] Action proposal/resolver 协议层。
- [x] Rule-based、MILP、rolling MILP/MPC controller。
- [x] Gymnasium/SB3 RL 环境与训练入口。
- [x] Controller comparison、KPI 汇总和 dashboard 生成。
- [x] Wave-height 场景、预测模型和船速扰动。
- [ ] 统一实验配置文件和命令行入口。
- [ ] 将大型外部数据和模型权重整理为可下载 release assets。
- [ ] 把 `economics.py`、`metrics.py` 进一步整理到 evaluation 包。
- [ ] 将 HPC 脚本中的个人路径改为环境变量配置。

## 环境需求

- Python `>=3.10`；GPU 训练环境当前使用 Python 3.12。
- `uv`，用于依赖解析、虚拟环境创建和运行脚本。
- 基础物理层依赖：`searoute>=1.6`、`CoolProp>=6.6`。
- RL 相关可选依赖：`numpy`、`gymnasium`、`stable-baselines3`、`sb3-contrib`。
- Wave-height GPU 训练依赖：`torch`、`torchvision`、`torchaudio`、`pandas`、`scikit-learn`、`matplotlib`、`tqdm`、`jupyterlab`。
- 运行深度学习训练时建议使用 NVIDIA GPU 和匹配的 CUDA/PyTorch 环境。

## 安装

以下命令均在仓库根目录执行。

### CPU / 基础安装

```powershell
uv sync
uv run python -m pip install -e .
```

如果需要 RL 环境：

```powershell
uv sync --extra rl
uv run python -m pip install -e ".[rl]"
```

如果不使用 `uv`，也可以直接用 pip：

```powershell
pip install -e .
pip install -e ".[rl]"
```

### GPU / Wave-Height 训练环境

Wave-height 预测模型建议使用 conda 环境：

```powershell
conda env create -f environment-gpu.yml
conda activate ccs-rlllm-gpu
pip install -e ".[rl]"
```

## 快速开始

### 物理层 Demo

```powershell
uv run python examples\run_physical_layer_demo.py
```

### Dashboard

```powershell
uv run python examples\build_phase1_dashboard.py
uv run python examples\build_rule_based_dashboards.py
```

生成的 HTML 或图片产物会写入脚本指定的 `output/` 或可视化目录。

### Controller Comparison

```powershell
uv run python experiments\compare_controllers_same_scenarios.py
```

该实验在相同 disturbance scenario 下比较不同 episode controller，并写出 CSV/report；static MILP 作为 perfect-foresight benchmark 单独报告。

### RL 训练

```powershell
uv run python -m sim.train --timesteps 200000
```

更完整的 PPO/BC 训练入口：

```powershell
uv run python scripts\train_ppo_bc.py --weather-obs --bc-episodes 30 --bc-epochs 20 --timesteps 150000
```

### Wave-Height 预测

```powershell
uv run python -m sim.scenario_generation.wave_height.prediction.train_lstm
uv run python -m sim.scenario_generation.wave_height.prediction.train_gru
```

对应说明见 `src/sim/scenario_generation/wave_height/prediction/README.md`。

## 测试

```powershell
uv run python -m unittest discover -s tests
```

如果临时不通过安装方式运行，也可以显式设置源码路径：

```powershell
$env:PYTHONPATH="$PWD\src"
python -m unittest discover -s tests
```

## 顶层目录

```text
CCS_RLLLM/
|-- data/                 # capture-rate profile、外部资料和实验数据
|-- docs/                 # 研究说明、设计文档和历史想法
|-- examples/             # 小型 demo 与 dashboard 生成脚本
|-- experiments/          # 研究实验入口，例如 controller comparison
|-- hpc/                  # 集群提交脚本与 smoke test
|-- scenarios/            # 可复现实验场景 JSON
|-- scripts/              # PPO/BC 训练和模型评估脚本
|-- src/sim/              # 主 Python 包
|-- tests/                # 单元测试、结构测试和实验 smoke test
|-- visualisation html/   # 旧可视化产物目录
|-- environment-gpu.yml   # GPU 训练环境
|-- pyproject.toml        # Python 包和依赖配置
|-- uv.lock               # uv 锁文件
`-- README.md
```

## `src/sim` 结构

```text
src/sim/
|-- actions/              # ActionProposal、ActionFrame、ActionResolver
|-- control/              # baseline、rule-based、MILP、rolling MILP、imitation
|-- entities/             # emitter、vessel、terminal、pipeline、well、state
|-- environment/          # CCSEnv、工厂函数、Gymnasium/SB3 adapter
|-- operations/           # capture、loading、unloading、transport、injection
|-- scenario_generation/  # disturbance 和 wave-height 场景生成
|-- visualization/        # dashboard payload、HTML 渲染和写出入口
|-- economics.py          # 成本和收益模型
|-- line_source.py        # reservoir/well pressure line-source 模型
|-- metrics.py            # rollout、KPI 和评估汇总
|-- network.py            # 物理网络图和单步结算
|-- network_scenarios.py  # 从 JSON/data 构造 Northern Lights 网络
|-- routes.py             # 航线和距离计算
|-- ship_speed.py         # 海况对船速的影响
|-- simulator.py          # 高层仿真执行器
`-- train.py              # RL 训练入口
```

## 数据

部分外部数据体积较大，未全部纳入 git 仓库。下载后请放回对应目录再运行相关脚本。

- Google Drive 数据目录：<https://drive.google.com/drive/folders/147lfZ1M1d3Am0v65fk1SX0jsXmk2lVzN?usp=sharing>
- `scenarios/`：可复现实验场景 JSON，例如 `northern_lights_phase1.json`、`northern_lights_phase2.json`。
- `data/capture_rates/`：Phase 1/Phase 1+ emitter capture-rate profile 和 metadata。
- `data/网络收集资料/`：网络收集并整理过的外部资料，例如 Climate TRACE source mapping 和 monthly profile。

## 主要工作流

### 新增控制器

1. 在 `src/sim/control/` 中实现控制逻辑。
2. 使用 `ActionProposal` / `ActionFrame` 表达动作。
3. 通过 `ActionResolver` 进入 `network.step()`。
4. 在 `experiments/compare_controllers_same_scenarios.py` 或新的实验脚本中注册评估。
5. 补充 `tests/` 中的行为测试或 smoke test。

### 新增场景

1. 在 `scenarios/` 中添加 JSON 配置。
2. 如果需要新的 capture profile，将数据放入 `data/capture_rates/`。
3. 在 `src/sim/network_scenarios.py` 或工厂函数中添加加载入口。
4. 用 demo、controller comparison 或 dashboard 脚本验证。

### 新增扰动

1. 在 `src/sim/scenario_generation/generator.py` 中生成 episode 时间序列。
2. 在 `disturbance_resolver.py` 中定义运行时解析规则。
3. 将扰动接入 `CCSEnv` 或物理操作模块。
4. 添加固定 seed 的单元测试，保证实验可复现。

## 相关文档

- `docs/CCS_RL_Research_Core_Idea.md`
- `docs/previous ideas/northern_lights_development_plan_cn.md`
- `docs/previous ideas/northern_lights_mechanism_ladder_L0_L3plus_cn.md`
- `src/sim/scenario_generation/wave_height/prediction/README.md`

## Citation

如果你在论文或报告中使用本仓库，建议同时注明：

```bibtex
@software{ccs_rlllm,
  title  = {CCS_RLLLM: Ship-Based CCUS Logistics Simulation and RL Playground},
  author = {CCS_RLLLM contributors},
  year   = {2026},
  note   = {Research code for physical-layer CCUS simulation, control, and reinforcement learning}
}
```
