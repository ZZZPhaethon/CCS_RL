# `Simulation.scenario_generation.wave_height`

本子包将海洋显著波高数据转化为 CCS 船运场景中的航速扰动。它与上级
`scenario_generation` 包兼容：最终结果写入 `Scenario.vessel_speed_factor` 或
`Scenario.leg_speed_factor`，再由 `PhysicalState` 和扰动解析器供仿真使用。

| 文件 | 职责 |
| --- | --- |
| `netcdf.py` | 解析 Classic NetCDF 文件中的坐标、变量和时间记录。 |
| `routes.py` | 加密经纬度航线、从网格波高场采样，并按均值/最大值/分位数聚合。 |
| `scenario.py` | 从历史 NetCDF 波高回放生成船舶级航速系数。 |
| `climatology_scenario.py` | 从航段级 CSV 气候统计生成循环的 `leg_speed_factor`。 |
| `forecast_scenario.py` | 将 LSTM 滚动预测 CSV 转换为预测驱动的船舶航速系数。 |
| `preprocessing.py` | 发现原始文件、构造候选路线，并导出航线/航段级训练数据集。 |
| `export_leg_dataset.py` | 导出 Phase 1 航段级波高数据集的命令行入口。 |
| `visualization.py` | 绘制波高快照并叠加航线、地点，验证空间数据质量。 |

```text
NetCDF 历史场 / LSTM 预测 / 气候 CSV
                  │
                  ▼
航线或航段波高时序 → ship_speed.speed_factor_series()
                  │
                  ▼
vessel_speed_factor 或 leg_speed_factor
                  │
                  ▼
船舶旅行时间、终端库存、泊位排队和 RL 回报
```

历史回放使用 `WaveHeightScenarioGenerator`；长期季节性分析使用
`LegWaveClimatologyScenarioGenerator`；MPC 或预报信息价值实验使用
`LSTMWaveHeightScenarioGenerator`。三类生成器均继承基础 `ScenarioGenerator`，
因此仍保留捕集停机、注入井维护和初始库存等非天气扰动。

波高会产生跨时间步后果。训练/评估时应明确策略能看到当前波高、气候统计还是未来
预测，并使用不重叠的历史时间窗口，避免天气数据泄漏使评估结果过于乐观。
