# `Simulation`

本目录实现船运 CCS（碳捕集、运输与封存）系统的物理仿真、场景扰动、控制器、
RL 环境、训练与可视化。设计上将固定物理参数、随时间变化的状态、动作解析和
策略学习分开，便于在同一网络上比较规则策略、MPC/MILP 和强化学习策略。

```text
entities + network_scenarios ─► network ─► simulator / operations
                                      ▲              │
scenario_generation ─► PhysicalState ─┘              ▼
actions / control ─────────────────────────────► StepResult + metrics
                                      ▲              │
                              environment / training ┘
                                      │
                                      ▼
                                visualization
```

## 根目录模块

| 文件 | 职责 |
| --- | --- |
| `network.py` | 将实体、连接关系和领域操作组合为 `PhysicalNetwork`。 |
| `network_scenarios.py` | 从项目数据加载固定网络场景，例如 Northern Lights Phase 1/2。 |
| `simulator.py` | 推进网络时间步、解析船舶移动，并记录动作、状态和观察。 |
| `economics.py` | 定义运行成本、碳价、放空惩罚与封存收益。 |
| `metrics.py` | 汇总回合及多回合的封存、放空、成本和服务水平指标。 |
| `routes.py` | 提供经纬度、航线和球面距离计算。 |
| `ship_speed.py` | 将海况波高映射为船舶航速系数。 |

## 子目录

| 目录 | 作用 |
| --- | --- |
| `entities/` | 静态物理实体、动态状态、违规记录及结果对象。 |
| `actions/` | 动作协议、动作帧和实体级动作解析。 |
| `operations/` | 捕集、装载、运输、卸载、注入和压力限制等物理操作。 |
| `scenario_generation/` | 初始状态和外生扰动生成；包含波高数据子模块。 |
| `environment/` | 面向 RL 的原生环境和 Gymnasium 适配器。 |
| `control/` | 规则基线、MILP、MPC、示范与模仿学习工具。 |
| `training/` | PPO 训练和与基线策略的比较入口。 |
| `visualization/` | 轨迹、地图和 HTML 仪表盘输出。 |

## 典型开发顺序

1. 在 `network_scenarios.py` 或外部配置中定义网络；
2. 用 `scenario_generation` 采样一个回合的外生条件；
3. 通过 `environment` 或 `simulator` 推进物理系统；
4. 使用 `control` 中的基线、MPC/MILP 或 `training` 中的 PPO 策略选择动作；
5. 使用 `metrics` 和 `visualization` 比较策略表现。

各子目录的详细接口、数据流和约束见其各自的 README。
