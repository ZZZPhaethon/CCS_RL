# CCS 物理层 v1

这个版本实现一个 1 h 步长的模块化 Layer V，用于后续接 RL/MPC 控制器和 LLM 协调层。

## 范围

- 实体：排放方、船、接收终端、管道、注入井、储层。
- 实体代码按物理子系统拆分在 `src/sim/entities/`：`emitter.py`、`vessel.py`、`terminal.py`、`pipeline.py`、`storage.py`；状态与结果记录在 `state.py`。
- 兼容导入仍可用：`from sim.entities import Emitter, Vessel, PhysicalState`。
- 拓扑：通过 `PhysicalNetwork.connect()` 和 `disconnect()` 重连实体。
- 动作：使用普通字典传入，例如 `{"brevik": {"load_vessel": "northern_pioneer"}}`、`{"oygarden_terminal": {"unload_vessel": "northern_pioneer"}}`、`{"oygarden_pipeline": {"flow_tph": 200.0}}`。装船/卸船动作只指定目标船，实际速率由物理层按船舶、泊位、库存和下游能力自动取最大可行值。
- 输出：`StepResult.as_dict()` 和 `PhysicalNetwork.snapshot()` 可直接序列化为 JSON。
- 航线：`sim.routes.sea_route()` 使用 `searoute` 包生成海上航路；dashboard 生成时会强制检查 provider，避免静默退回粗糙航路。
- Phase 1 默认排放方：Norcem/Brevik 使用 `annual_target_export_tpy=400_000`、`max_production_tph=56`；Fortum Oslo Varme/Celsio 使用 `annual_target_export_tpy=350_000`、`max_production_tph=48`；仿真 `nominal_capture_tph` 按各自年目标的小时平均值写入。
- 关键物理参数：船舶标注 `volume_capacity_m3=7_500` 和 `speed_knots=14`；中间存储点为 Naturgassparken / Northern Lights Carbon Capture Plant Site，当前 `berth_count=1`；海上管道标注 `annual_capacity_tpy=5_000_000`、`length_km=100.4`，并以红色 `route_coordinates` 从 Naturgassparken 直接连到 `31/5-7 EOS` / `31/5-A-7 AH` 海底井口位置；注入井使用官方坐标：`31/5-A-7 AH` 为 `60.575913, 3.441678`，`31/5-C-1 H` 为 `60.512961, 3.468346`；Aurora 储层标注 `depth_m=2_600`。
- 路由口径：`100.4 km` 采用 Northern Lights 设施页详细管道口径；地图显示路线直接收束到 Phase 1 井口/海底分配节点坐标。

## 当前简化

- 船舶轨迹不再使用内置操作阶段表。上层调度器或多代理模型通过每步 `action_frames` 显式传入动作，例如排放端 `load_vessel`、终端 `unload_vessel`、管道 `flow_tph`，以及船舶 `sail_to`；物理层只执行并裁剪这些外部动作，不会因为仿真时间到达某个小时而自动装卸。船舶只有在 `vessel_berths[vessel_id]` 指向对应排放端或终端时才允许 loading/unloading，在航行中请求装卸会产生 `berth_required` 违规并被裁剪为 0。`load_vessel` / `unload_vessel` 不需要指定速度，默认按最大可行装卸速率执行。
- 终端允许多艘船同时停靠或等待，但所有控制器共享 `PhysicalState.terminal_unload_queues` 中的 FIFO 卸载顺序。每个终端每步最多卸载一艘队首船；部分卸载不会改变队首，半载离港会立即出队，返回后重新加入队尾。同一时步首次观测到达的船以 vessel ID 打破平局。显式请求非队首船会被物理层裁剪为 0，并记录 `fifo_unload_required`；`berth_count=0` 时暂停卸载，任何正值仍只开放一个卸载服务位。
- 储层作为 `Reservoir` 物理组件接在注入井之后，snapshot 会根据储层库存派生 `pressure_bar`、`pressure_margin_bar` 和 `fill_fraction`。
- Aurora demo 的 `Reservoir` 现在额外携带 line-source 参数。每步注入后，snapshot 会在注入井上输出 `bottomhole_pressure_bar`、`bottomhole_pressure_delta_bar` 和 `line_source_rate_tph`，并在储层上输出 `line_source_pressure_bar_by_radius_m` / `line_source_delta_bar_by_radius_m`。这些是无限径向流解析解给出的井底压力和指定半径点压力，不是封闭储层平均压力；当前按上一时间步注入速率和当前仿真时间做 constant-rate diagnostic，尚未实现变速率历史的 rate superposition。
- line-source 参数中，`well_radius_m=0.10795` 由 Concept report 的 `8 1/2'' Open Hole` 直接换算；`total_compressibility_1_pa`、`viscosity_pa_s`、`co2_density_kg_m3` 和 `skin` 仍是工程假设，已在 `line_source_parameter_status` 中标注为 `assumed`。当前压力诊断还没有反向限制注入量，后续需要最大允许井底压力、caprock fracture pressure 或 fault reactivation limit 后再接入裁剪逻辑。
- 调度器、MPC/RL、LLM 协议引擎不在本版本内。

## 快速运行

```powershell
$env:PYTHONPATH='E:\CCS_RLLLM\src'
python examples\run_physical_layer_demo.py
python -m unittest discover -s tests
```

## 动态可视化

生成可直接打开的单文件 dashboard：

```powershell
$env:PYTHONPATH='E:\CCS_RLLLM\src'
python examples\build_physical_dashboard.py
```

输出文件为 `docs/physical_layer_dashboard.html`。页面内嵌 48 h 仿真轨迹，每一帧间隔 1 h；可以拖动时间轴、播放/暂停，并点击左侧组件查看排放方、船、终端、管道、注入井和储层的库存、流入/流出、参数和违规裁剪信息。默认轨迹不自动触发装卸或航行，是无上层动作输入时的静止基线；若调用 `build_demo_trajectory(..., action_frames=...)` 或 `write_dashboard(..., action_frames=...)`，页面会展示上层多代理动作驱动后的状态。船舶卡片中的 `Cargo x / capacity t` 表示载货量随外部装卸动作变化，船舶最大容量 `capacity_t` 本身不随运行改变。

中间视图使用 Leaflet + OpenStreetMap 瓦片，支持拖动、滚轮缩放和 marker 点击。地图包含排放方、Naturgassparken 中间存储点、`31/5-7 EOS` 海底分配节点、官方井位和 Aurora 储层坐标，船运航线由 `searoute` 生成；显示层会把 searoute 海上路径补接到排放点和终端坐标，避免航线停在海上网络吸附点。return leg 默认使用同一 corridor 反向返回，标出来是为了显示空载返程仍占用船舶时间，不表示现实中返航一定不能改道。海上 CO2 管道用红色线段显示，从 Naturgassparken 起点直接连到 `31/5-7 EOS` / `31/5-A-7 AH`；分配节点到 `31/5-C-1 H` 的井间连接以紫色虚线表示。地图叠加经纬度网格、边缘标签和右下角鼠标坐标读数，便于核对图纸和井位。船位只有在上层动作传入 `{"vessel_id": {"sail_to": "destination_id"}}` 后才会按船速沿航线动态移动；当前地图包含 `northern_pioneer` 和 `northern_pathfinder` 两艘船。若完全离线，仿真数据仍可生成，但地图瓦片需要换成本地 tile 或静态底图。
