# CCS_RLLLM

面向船运 CCS 链条的物理层仿真、控制算法、RL 环境和实验评估项目。当前代码以 Northern Lights 场景为主，核心流程是：

```text
Emitter -> Vessel -> Terminal -> Pipeline -> SubseaManifold -> InjectionWell -> Reservoir
```

上层控制器、MILP、RL policy 或实验脚本提交动作；物理层负责验证动作、推进仿真、更新库存/运输/注入/压力状态，并输出可审计的结果。

## 快速运行

PowerShell:

```powershell
$env:PYTHONPATH='E:\CCS_RLLLM\src'
python examples\run_physical_layer_demo.py
python -m unittest discover -s tests
```

生成 dashboard:

```powershell
$env:PYTHONPATH='E:\CCS_RLLLM\src'
python examples\build_physical_dashboard.py
python examples\build_phase1_dashboard.py
```

训练入口:

```powershell
$env:PYTHONPATH='E:\CCS_RLLLM\src'
python -m sim.train --timesteps 200000
```

## 顶层文件树

```text
CCS_RLLLM/
|-- data/                 # 原始数据、capture-rate 曲线、网络收集资料
|   |-- capture_rates/     # Phase 1 emitter capture-rate CSV 和元数据
|   `-- 网络收集资料/       # 从网络收集/整理的外部资料
|-- scenarios/            # 可复现实验场景 JSON
|-- docs/                 # 项目说明、研究记录、压力模型说明等文档
|-- examples/             # 小型运行示例和 dashboard 生成脚本
|-- experiments/          # 实验入口和 controller comparison 脚本
|-- hpc/                  # 集群/HPC 提交脚本和 smoke test
|-- src/sim/              # 主 Python 包
|-- tests/                # 单元测试和结构测试
|-- tmp/                  # 临时文献/PDF/截图材料；不属于核心代码
|-- visualisation html/   # 旧 HTML 产物目录；建议后续删除或归档
|-- pyproject.toml        # Python 包配置
`-- README.md
```

## `src/sim` 代码结构

```text
src/sim/
|-- actions/
|   |-- action.py
|   |-- resolver.py
|   `-- __init__.py
|-- control/
|   |-- baselines.py
|   |-- rule_based.py
|   |-- milp.py
|   |-- rolling_milp.py
|   `-- imitation.py
|-- entities/
|   |-- emitter.py
|   |-- vessel.py
|   |-- terminal.py
|   |-- pipeline.py
|   |-- manifold.py
|   |-- storage.py
|   `-- state.py
|-- environment/
|   |-- env.py
|   |-- factories.py
|   `-- gym_adapter.py
|-- operations/
|   |-- capture.py
|   |-- loading.py
|   |-- unloading.py
|   |-- transport.py
|   |-- injection.py
|   `-- snapshot.py
|-- scenario_generation/
|   |-- generator.py
|   `-- disturbance_resolver.py
|-- visualization/
|   |-- core.py
|   |-- html.py
|   `-- writers.py
|-- economics.py
|-- metrics.py
|-- line_source.py
|-- network.py
|-- network_scenarios.py
|-- routes.py
|-- simulator.py
|-- train.py
`-- __init__.py
```

## 主要目录职责

### `src/sim/actions/`

动作协议层。它不做控制决策，只规定“动作如何表达”和“动作如何进入物理层”。

- `action.py`：定义 `ActionProposal`、`ActionFrame`、`ActionDecision`、`CommittedActionFrame`。
- `resolver.py`：定义 `ActionResolver`，负责验证动作是否合法、参数是否完整、是否冲突，并把 proposal 翻译成 `network.step()` 能执行的字典。

典型导入：

```python
from sim.actions import ActionFrame, ActionProposal, ActionResolver
```

### `src/sim/control/`

控制器和算法层，负责“决定做什么”。

- `baselines.py`：简单 baseline 策略，例如 `idle_policy` 和 `greedy_shuttle_policy`。
- `rule_based.py`：规则控制器/启发式 baseline。
- `milp.py`：开环 MILP benchmark。
- `rolling_milp.py`：滚动窗口 MILP/MPC controller。
- `imitation.py`：imitation learning 相关工具。

注意：`control/` 负责产生动作，`actions/` 负责定义动作协议，二者不是同一层。

### `src/sim/entities/`

物理实体和状态定义。

- `Emitter`：排放端/捕集端。
- `Vessel`：LCO2 船。
- `Terminal`：接收终端。
- `Pipeline`：终端到海底系统的外输管线。
- `SubseaManifold`：海底分配节点。
- `InjectionWell` / `Reservoir`：注入井和储层。
- `PhysicalState` / `StepResult`：仿真状态、单步结果和违规记录。

### `src/sim/environment/`

RL 环境层。

- `env.py`：核心 `CCSEnv`，提供 `reset()` / `step()` / observation / reward / action mask。
- `factories.py`：环境工厂，例如 `build_phase1_env()`，用于快速创建 Northern Lights Phase 1 训练环境。
- `gym_adapter.py`：Gymnasium/SB3 适配器，把原生 `CCSEnv` 包装成 `gymnasium.Env`。

典型导入：

```python
from sim.environment import CCSEnv, CCSEnvConfig, build_phase1_env
from sim.environment.gym_adapter import CCSGymEnv
```

### `src/sim/operations/`

物理操作模块。它们实现具体的物理动作和约束投影。

- `capture.py`：捕集和 venting。
- `loading.py`：emitter 到 vessel 的装船。
- `unloading.py`：vessel 到 terminal 的卸船。
- `transport.py`：pipeline 和 manifold 的输送/分配。
- `injection.py`：注入井注入。
- `snapshot.py`：生成实体级观测快照。

### `src/sim/scenario_generation/`

扰动场景生成和运行时扰动解析。

- `generator.py`：生成一个 episode 的时间序列扰动，例如 capture outage、天气、well maintenance、injectivity decline。
- `disturbance_resolver.py`：运行时解析当前 step 的有效值。它不生成扰动，只负责“state 里有扰动覆盖值就用扰动值，否则回退到实体 nominal 值”。

### `src/sim/visualization/`

dashboard 和可视化生成代码。

- `core.py`：轨迹、地图 payload、仿真数据组装。
- `html.py`：HTML dashboard 渲染。
- `writers.py`：写出 Phase 1 / Phase 2 dashboard 的入口函数。

## 根目录下仍保留的核心文件

这些文件目前仍在 `src/sim/` 根目录下，因为它们是跨模块核心或尚未进一步归类：

- `network.py`：物理网络图和单步物理结算逻辑。
- `simulator.py`：仿真执行器，接收 `ActionFrame`，调用 `ActionResolver`，推进 vessel movement 和 network step。
- `routes.py`：航线和距离计算。
- `line_source.py`：储层/井底压力 line-source 模型。
- `network_scenarios.py`：从 `scenarios/` JSON 和 capture profile 数据构建 Northern Lights 物理网络。
- `economics.py`：运营成本/收益模型。
- `metrics.py`：episode rollout、KPI 和评估汇总；具体控制策略放在 `control/`。
- `train.py`：RL 训练入口。

后续如果继续整理，建议优先把 `economics.py` 和 `metrics.py` 放入 `evaluation/`，再考虑把 `network.py` / `simulator.py` / `routes.py` / `line_source.py` 拆成 `physics/`、`navigation/`、`geology/` 等包。

## 数据目录

### `scenarios/`

只放可复现实验场景 JSON：

- `northern_lights_phase1.json`
- `northern_lights_phase2.json`

### `data/capture_rates/`

Phase 1 emitter capture-rate profile 数据和元数据。

### `data/网络收集资料/`

从网络收集并整理过的外部资料，例如 Climate TRACE source mapping 和 monthly profile。它属于数据资料，不属于代码输出。

## 实验和部署目录

### `experiments/`

实验入口。这里放“如何组合已有模块跑一个研究实验”的脚本，不放底层算法实现。

当前主要脚本：

- `compare_controllers_same_scenarios.py`：在相同 disturbance scenario 下比较 episode controllers，并写出 CSV/report；static MILP 作为 perfect-foresight benchmark 单独报告。

### `hpc/`

集群运行脚本。

- `submit_env_check.sh`：检查 HPC 环境依赖和 GPU。
- `submit_rl_smoke.sh`：提交 RL smoke test。
- `submit_train_336h.sh`：提交 336h 训练任务。
- `rl_smoke.py`：最小 RL 训练/评估 smoke test。

这些脚本目前包含个人集群路径，例如 `/scratch_root/hx721/CCS_RLLLM`。如果项目要共享给其他机器使用，建议后续改成环境变量配置。

## 测试

```powershell
$env:PYTHONPATH='E:\CCS_RLLLM\src'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests
```
