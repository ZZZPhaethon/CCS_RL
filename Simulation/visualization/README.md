# `Simulation.visualization`

本目录将网络、动作和仿真结果转换为可查看的轨迹、地图和独立 HTML 仪表盘。它仅
用于展示和诊断，不应改变物理状态或策略动作。

| 文件 | 职责 |
| --- | --- |
| `core.py` | 构建可 JSON 序列化的网络、船舶位置、流量和动作轨迹数据。 |
| `html.py` | 将数据载荷渲染为带地图和时间控制的自包含 HTML。 |
| `writers.py` | 调用核心构建器和 HTML 渲染器，将仪表盘写入目标路径。 |
| `__init__.py` | 导出常用的轨迹构建、页面渲染和写文件函数。 |

```text
network + action frames + simulation records
                    │
                    ▼
              core.py payload
                    │
                    ▼
           html.py dashboard markup
                    │
                    ▼
        writers.py → 可离线打开的 HTML 文件
```

可视化适合检查船舶路线、靠泊/卸载时序、库存积压、管道/注入流量和策略动作。若出现
异常奖励或违规，建议先生成同一回合的仪表盘，再结合 `StepResult.violations` 与
`metrics` 排查原因。
