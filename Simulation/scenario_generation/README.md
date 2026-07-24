# `Simulation.scenario_generation`

本目录负责为 CCS（碳捕集、运输与封存）仿真生成**外生场景**：即智能体无法
直接控制、但会影响系统可行性和经济性的初始条件与时变扰动。它不决定捕集、
运输或注入动作；这些动作应由控制器或 RL 策略选择。

典型扰动包括捕集设施停机、捕集量随机波动、天气造成的船舶减速、注入井维护
停机，以及注入能力降额。将这些不确定性明确地写入场景，有助于训练出在不同
运行条件下仍能工作的鲁棒策略。

## 目录结构

| 文件 | 主要职责 |
| --- | --- |
| `generator.py` | 定义场景配置、场景数据结构，并采样初始库存和逐时间步扰动轨迹。 |
| `disturbance_resolver.py` | 将 `PhysicalState` 中的扰动覆盖值与实体的名义参数合并，返回当前有效能力。 |
| `__init__.py` | 导出 `Scenario`、`ScenarioConfig` 和 `ScenarioGenerator`。 |

## `wave_height`：波高驱动的船运场景

`wave_height/` 是场景生成的扩展子包。它将历史海况、航段气候统计或 LSTM
波高预测转换为船舶/航段的航速系数，并复用主模块既有的 `Scenario` 扰动接口。
它不直接修改船舶实体，也不改变捕集或注入逻辑。

| 文件 | 主要职责 |
| --- | --- |
| `netcdf.py` | 读取 Classic NetCDF 波高文件，并提供变量、维度和记录访问接口。 |
| `routes.py` | 加密航线坐标、从波高网格中采样航线波高，并以平均值、最大值或分位数聚合为单条航线的时序。 |
| `scenario.py` | `WaveHeightScenarioGenerator`：从历史 NetCDF 波高场生成 `Scenario.vessel_speed_factor`。 |
| `climatology_scenario.py` | 从航段级 CSV 读取季节性平均航速系数，并生成 `Scenario.leg_speed_factor`。 |
| `forecast_scenario.py` | 从滚动 LSTM 预测 CSV 读取未来波高窗口，并生成船舶航速系数。 |
| `preprocessing.py` | 发现波高文件、构建候选航段路线，并导出航线/航段波高和航速系数 CSV 数据集。 |
| `export_leg_dataset.py` | 导出航段波高数据集的命令行入口。 |
| `visualization.py` | 绘制波高快照、航线和地点覆盖层，用于数据质量检查。 |
| `__init__.py` | 导出 NetCDF、路线处理、数据集导出、场景生成和可视化的公共接口。 |

### 波高到仿真扰动的数据流

```text
历史 NetCDF 波高场 ─┐
                     ├─► 航线/航段波高时序 ─► ship_speed.speed_factor_series()
LSTM 波高预测 CSV ──┘                                      │
                                                            ▼
                                           vessel_speed_factor 或 leg_speed_factor
                                                            │
                                                            ▼
                                         PhysicalState 的运行时扰动覆盖值
                                                            │
                                                            ▼
                                      航行时间、到达时刻、库存和泊位排队
```

波高场方案的核心转换由 `ship_speed.speed_factor_series()` 完成：它使用每艘
船舶的 `ShipSpeedParameters` 和名义航速，将显著波高转换为范围在 `[0, 1]` 的
航速系数。系数为 1 表示不减速，接近 0 表示极端海况造成的显著减速或无法航行。

### 三种波高场景生成方式

| 生成器 | 输入 | 写入的扰动 | 适用场景 |
| --- | --- | --- | --- |
| `WaveHeightScenarioGenerator` | 一个或多个历史 NetCDF 文件、每艘船舶的航线坐标 | `Scenario.vessel_speed_factor` | 历史回放、基于观测海况的训练与评估。 |
| `LegWaveClimatologyScenarioGenerator` | 航段级气候统计 CSV | `Scenario.leg_speed_factor` | 不需要逐格波高数据的长期、季节性评估。 |
| `LSTMWaveHeightScenarioGenerator` | LSTM 滚动预测 CSV、船舶航线 | `Scenario.vessel_speed_factor` | MPC、预测驱动控制或比较“预报可得/不可得”信息价值。 |

三者都会先调用基类 `ScenarioGenerator.sample()`，因此仍保留捕集扰动、注入井
维护和初始库存随机化；随后只替换或补充与海况相关的航速轨迹。

### 历史 NetCDF 波高回放

`WaveHeightScenarioGenerator` 需要以下输入：

- `nc_paths`：一个或多个 Classic NetCDF 波高文件，或自定义的 `RouteWaveReader`；
- `routes`：按 vessel ID 索引的航线映射，每条航线至少应包含 `coordinates`；
- 可选的 `ship_parameters_by_vessel`：按船舶指定的速度模型参数；
- 可选的 `RouteWaveConfig`：控制航线加密、波高变量和航线聚合方式。

生成器会在可用历史记录中随机选择连续窗口，并将该起点保存到
`last_start_record`。若波高数据的记录数少于一个回合所需步数，会引发异常。
未配置航线或坐标的船舶保持基类生成的航速轨迹。

### 航段气候统计场景

`LegWaveClimatology` 从 CSV 读取按 `leg_id` 和 `source_record` 索引的航速系数。
它将记录聚合为按小时循环的季节性序列，默认周期为 8784 小时（闰年）。
`LegWaveClimatologyScenarioGenerator` 再从周期内选择起始小时，并产生每一航段的
`leg_speed_factor`。

该生成器使用 `ScenarioConfig.leg_wave_slowdown_multiplier` 放大或减弱减速，并使用
`ScenarioConfig.leg_wave_speed_factor_floor` 设置最低航速系数。因此，总 README 中
提到的这两个字段对**基础** `ScenarioGenerator` 不生效，但在此子包的气候统计
生成器中已经生效。

### LSTM 预测场景

`LSTMWaveHeightForecastReader` 读取包含以下信息的预测 CSV：

- `vessel_id`；
- `horizon_index`；
- `global_record`；
- 默认名为 `predicted` 的波高预测列。

它将连续且从预测时域第 0 小时开始的记录组成 `ForecastWindow`。随后
`LSTMWaveHeightScenarioGenerator` 随机或固定选择一个窗口，并把预测波高转换为
每艘船的航速系数。若预测时域短于场景回合所需步数，生成器会引发异常，避免用
不完整预测隐式填充未来值。

### 数据预处理与质量检查

`preprocessing.py` 负责离线准备数据集：发现 `wam10ei_*.nc` 波高文件、建立候选
航段路线，并导出航线级或航段级 CSV。导出的数据可同时包含聚合波高和对应的
速度系数。`export_leg_dataset.py` 是相关导出的命令行入口。

`visualization.py` 可将某一记录的波高场与航线、港口/地点叠加绘制。建议在训练前
随机抽查多个时间点、不同航线和边界区域，确认经纬度方向、缺失值和航线采样点
符合预期。

### 对 RL 的影响

波高不应只被视为噪声：它通过船速影响到达时间、终端库存、泊位排队和最终注入
节奏，因而具有跨多个时间步的后果。

- 若策略能观测当前或预测未来波高/航速系数，它可以提前调整船舶指派与发船时机；
- 若只提供当前波高而不提供未来预报，策略需要在不确定性下保留库存与运力余量；
- 航段级扰动比船舶级全局扰动更精细：不同路线可同时处于不同海况；
- 训练与测试应使用不重叠的历史窗口或年份，避免天气记录泄漏导致评估过于乐观；
- 评估时应分别报告平静、中等和极端海况下的封存量、延迟、违规与成本。

## 在仿真与 RL 中的位置

```text
ScenarioConfig + random seed
              │
              ▼
ScenarioGenerator.sample(network)
              │
              ▼
Scenario（一个回合的完整扰动轨迹）
              │
       ┌──────┴──────┐
       ▼             ▼
apply_initial()  apply_to_state(time_h)
       │             │
       ▼             ▼
PhysicalState ← 扰动覆盖值（ξ_t）
       │
       ▼
disturbance_resolver → 当前有效能力 → 仿真器 / RL 环境
```

一个典型回合的执行顺序如下：

1. 创建 `ScenarioConfig`，设置回合长度和随机过程参数。
2. 调用 `ScenarioGenerator.sample(network, seed=...)` 生成一个 `Scenario`。
3. 重置环境时，调用 `scenario.apply_initial(state)` 写入初始库存。
4. 每次推进仿真前，调用 `scenario.apply_to_state(state, time_h)` 写入当前
   时间步的扰动覆盖值。
5. 仿真器通过 `disturbance_resolver` 获取有效捕集能力、注入能力、航速或泊位数。
6. 控制器或 RL 策略根据观察选择动作；场景本身不选择动作。

## 核心数据结构

### `ScenarioConfig`

`ScenarioConfig` 定义一个场景生成器的随机规则。默认值会产生温和但非平凡的
扰动，可用于一周（168 小时）的基准回合。

| 参数组 | 关键字段 | 含义 |
| --- | --- | --- |
| 时间设置 | `episode_hours`、`time_step_hours` | 回合总时长和单个仿真步长。步数按 `episode_hours / time_step_hours` 四舍五入计算，至少为 1。 |
| 捕集扰动 | `capture_noise_std` | 捕集可用率围绕 1.0 的高斯噪声标准差；负采样结果会被截断为 0。 |
| 捕集停机 | `capture_outage_rate_per_week`、`capture_outage_mean_hours` | 每周停机触发强度和平均停机时长。停机期间捕集系数为 0。 |
| 高产出窗口 | `capture_high_output_*` | 偶发高于名义产出的窗口频率、持续时长和倍率范围。高产出倍率会与噪声后的捕集系数相乘。 |
| 天气过程 | `weather_process` | 支持 `"window"` 与 `"block"` 两种生成方式。未知值会引发 `ValueError`。 |
| 天气窗口 | `weather_window_*` | `window` 模式下，随机生成持续一段时间的船舶减速窗口。 |
| 分块天气 | `weather_update_*` | `block` 模式下，每隔固定时长重新采样一个所有船舶共享的航速系数。 |
| 注入井维护 | `well_maintenance_*` | 生成注入井的可用/不可用轨迹。维护期间井无法接受注入流量。 |
| 初始库存 | `randomize_initial_inventory` 及各 `*_initial_fill_range` | 按 emitter 缓冲容量或 terminal 库容的比例随机生成初始库存。 |
| 储层热启动 | `warm_start`、`reservoir_initial_pressure_fill_range` | 可选地按压力受限容量比例预填充储层，使短回合能从中期压力状态开始。 |

参数中的 `rate_per_week` 用于计算每个时间步的事件开始概率：

```text
start_probability = clamp(rate_per_week × time_step_hours / 168, 0, 1)
end_probability   = clamp(time_step_hours / mean_hours, 0, 1)
```

这是一种简单的开始/结束随机过程。它提供可控的扰动强度，但不保证精确复现给定
的长期平均停机时间或事件数量；若需要严格的可靠性分布，应替换为校准后的过程。

### `Scenario`

`Scenario` 保存一个已经采样完成的回合。其字段不是当前时刻的单个值，而是按
实体 ID 和时间步索引保存的完整轨迹。

| 字段 | 含义 |
| --- | --- |
| `time_step_hours` | 场景的时间步长。 |
| `n_steps` | 回合中的时间步数。 |
| `initial_inventory_t` | 按实体 ID 保存的初始库存，单位为吨。 |
| `emitter_availability` | emitter 的逐时间步捕集可用率系数。 |
| `vessel_speed_factor` | vessel 的逐时间步航速系数。 |
| `leg_speed_factor` | 航段的逐时间步航速系数，键格式为 `origin_id->destination_id`。 |
| `well_available` | injection well 的逐时间步布尔可用性。 |
| `injectivity_factor` | injection well 的逐时间步注入能力系数。 |
| `seed` | 生成该场景时使用的可选随机种子。 |

`step_index(time_h)` 将仿真时间映射为场景索引，并截断到 `[0, n_steps - 1]`。
因此，若仿真时间超过场景时域，最后一个扰动值会被持续使用。

`apply_initial(state)` 只更新 `state.entity_inventory_t` 中由场景指定的实体库存。
`apply_to_state(state, time_h)` 会覆盖当前状态中的 emitter 可用率、vessel 航速、
航段航速、well 可用性及注入能力系数。

### `ScenarioGenerator`

`ScenarioGenerator` 根据 network 中已有的实体类型生成场景：

- `Emitter`：生成捕集可用率轨迹；
- `Vessel`：复制同一条全局天气航速轨迹给每艘船舶；
- `InjectionWell`：生成维护可用性轨迹，并默认生成值为 1.0 的注入能力系数；
- `Terminal`：用于生成初始库存；
- `Reservoir`：在启用 `warm_start` 时用于生成初始封存量。

当前实现依赖 network 提供：

```python
network.entities
network._entities_of_type(entity_type)
```

前者用于通过实体 ID 获取实体，后者用于按类型获取实体集合。后续实现 network
类时应保留这两个接口，或同步修改场景生成器。

## 扰动覆盖与名义参数

实体 dataclass 保存设计或名义参数；运行时扰动不会修改这些对象，而是写入
`PhysicalState`。`disturbance_resolver.py` 使用下列优先级：

```text
PhysicalState 中存在覆盖值  → 使用覆盖值
PhysicalState 中不存在覆盖值  → 使用实体的名义参数
```

| 解析函数 | 优先读取的状态字段 | 名义兜底值 | 输出 |
| --- | --- | --- | --- |
| `emitter_availability()` | `state.emitter_availability` | `Emitter.availability` | 非负捕集系数。 |
| `well_is_available()` | `state.well_available` | `InjectionWell.available` | 注入井是否可用。 |
| `well_injectivity_factor()` | `state.injectivity_factor` | `1.0` | 非负注入能力系数。 |
| `well_max_injection_tph()` | 上述两个结果 | `InjectionWell.max_injection_tph` | 有效注入上限，单位为 tph。 |
| `vessel_speed_factor()` | `state.vessel_speed_factor` | `1.0` | 非负船舶航速系数。 |
| `leg_speed_factor()` | `state.leg_speed_factor` | 调用者提供的 `fallback` | 非负航段航速系数。 |
| `terminal_berth_count()` | `state.berth_count_override` | `Terminal.berth_count` | 非负可用泊位数。 |

解析函数会将负系数截断为 0。它们不负责生成扰动，也不负责检查所有物理约束；
约束执行应由仿真器完成。

## 随机种子与可复现性

`ScenarioGenerator` 同时支持构造函数基础种子和 `sample()` 调用种子：

```python
generator = ScenarioGenerator(config, seed=42)
scenario_a = generator.sample(network)          # 使用 42
scenario_b = generator.sample(network, seed=7)  # 本回合使用 7
```

生成器从主随机流派生 capture、weather、maintenance 与 initial-inventory 的独立子流。
这样调整某一个扰动通道时，不会改变其他通道的随机序列，便于进行可复现实验和
消融对比。

## 对 RL 的影响

场景生成决定了训练分布，因此会直接影响策略的泛化能力。

### 观察（observation）

若扰动会影响未来的可行动作，策略应能看到当前扰动或其充分统计量，例如：

- 当前 emitter 可用率和预计捕集量；
- 当前 well 可用性、注入能力系数和压力裕度；
- 当前 vessel/航段航速系数、船舶位置与到达时间；
- terminal 库存、泊位可用性和排队长度；
- 回合剩余时间与近期扰动历史，尤其在存在持续停机窗口时。

若策略无法观测这些变量，环境会变成部分可观测问题；此时可能需要历史窗口、
循环网络策略或显式的信念状态估计。

### 动作可行域

扰动通过改变有效能力收缩或扩张动作可行域：

- emitter 停机会降低可捕集量；
- 注入井维护会使对应井的有效注入上限降为 0；
- 注入能力系数下降会降低允许的注入流量；
- 船速下降会延迟船运到达，进而增加库存与泊位拥塞风险；
- 泊位数量覆盖值可限制同时装卸的船舶数。

环境应将实际执行流量、未满足需求或违规情况通过 `StepResult` 和 `Violation`
反馈给奖励函数，而不应假设策略动作始终可执行。

### 奖励与评估

常见做法是奖励实际封存量，惩罚捕集后放空、运输延迟、库存溢出、约束违规和
运营成本。训练时应覆盖正常、轻度扰动和极端扰动情形；评估时应固定独立种子集，
并报告不同场景下的均值、分位数和最差表现，而不是只报告单次回合结果。

## 当前实现的范围与预留项

以下字段已经定义，但 `ScenarioGenerator.sample()` 当前尚未主动生成其非默认轨迹：

- `Scenario.leg_speed_factor`：字段会被 `apply_to_state()` 写入状态，但当前采样时
  保持为空；
- `ScenarioConfig.leg_wave_slowdown_multiplier` 与
  `ScenarioConfig.leg_wave_speed_factor_floor`：配置字段已预留，但当前生成流程未使用；
- `PhysicalState.berth_count_override`：`disturbance_resolver` 已支持解析，但当前场景
  生成器未生成泊位关闭或扩容扰动；
- `injectivity_factor`：当前所有注入井均生成常数 `1.0`，尚未根据地质或维护数据
  采样随时间衰减的注入能力。

若要启用这些功能，应在 `ScenarioConfig` 中增加对应随机或数据驱动参数，并在
`ScenarioGenerator.sample()` 中填充相应的逐时间步字典。建议同时加入固定种子
测试，以验证轨迹长度、取值范围和状态写入结果。

## 最小使用示例

```python
from Simulation.entities import PhysicalState
from Simulation.scenario_generation import ScenarioConfig, ScenarioGenerator

config = ScenarioConfig(episode_hours=168, time_step_hours=1.0)
generator = ScenarioGenerator(config, seed=42)
scenario = generator.sample(network)

state = PhysicalState()
scenario.apply_initial(state)

for step in range(scenario.n_steps):
    time_h = step * scenario.time_step_hours
    scenario.apply_to_state(state, time_h)

    # 在此读取有效能力，选择动作并推进物理仿真。
    # effective_limit = well_max_injection_tph(state, well)
    # next_state, result = simulator.step(state, action, network)
```

## 开发约定

- 新增时变扰动时，先在 `PhysicalState` 中增加覆盖字段，再在 `Scenario` 中增加
  逐时间步轨迹，并实现对应 resolver；
- 新增随机过程时，应明确其单位、时间尺度、随机种子来源和可取范围；
- 所有轨迹长度应与 `Scenario.n_steps` 一致；
- 不应通过扰动直接修改冻结的实体定义；运行时变化应写入 `PhysicalState`；
- 新功能应覆盖固定种子可复现性、边界值和状态覆盖优先级测试。
