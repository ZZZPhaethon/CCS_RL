# `Simulation.environment`

本目录把 CCS 物理网络封装为供 RL、规则控制器和优化器使用的环境接口。它负责
回合重置、场景应用、动作掩码、观察编码、奖励计算及 Gymnasium 兼容性。

| 文件 | 职责 |
| --- | --- |
| `env.py` | 核心 `CCSEnv`、环境配置、奖励和原生动作/观察接口。 |
| `factories.py` | 从公开网络场景创建 Phase 1/2 环境的工厂函数。 |
| `gym_adapter.py` | 将原生混合动作转换为 Gymnasium `MultiDiscrete` 动作和掩码。 |
| `event_driven.py` | 仅在船舶调度事件时要求决策的稀疏决策环境。 |
| `event_gym.py` | 稀疏事件驱动环境的 Gymnasium 适配器。 |
| `forecast.py` | 生成未来预测特征和重规划周期特征。 |
| `forecast_gym.py` | 暴露当前状态与未来预测的 Gymnasium 环境。 |
| `forecast_encoder.py` | 面向 Stable-Baselines3 的结构化预测特征编码器。 |
| `vessel_mode.py` | 提取船舶 WAIT/调度相关的只读运行上下文特征。 |
| `__init__.py` | 导出公共环境、动作常量、工厂函数和适配器。 |

```text
Scenario → reset() → observation + action mask
                         │
policy ─► action ─► env.step() ─► simulator/network.step()
                         │                 │
                         └── reward, termination, info, next observation
```

## 主要接口

`CCSEnv` 是物理层上方的原生环境。`CCSEnvConfig` 控制回合长度、奖励模式、放空/
运行成本权重、封存收益、库存溢出风险和观察内容。`CCSGymEnv` 进一步适配为可由
Stable-Baselines3 等库训练的接口。

动作掩码很重要：它会随船舶位置、载货状态、终端条件和注入约束变化，避免策略
采样明显不可执行的离散动作。但掩码不能代替物理仿真；实际执行仍由网络和操作模块
决定。

## 选择环境类型

- 使用 `CCSEnv`：需要原生动作、调试物理过程或编写规则/MPC 控制器时；
- 使用 `CCSGymEnv`：使用 PPO 或其他 Gymnasium RL 算法时；
- 使用 `event_driven.py`：船舶长时间航行/等待，且希望只在调度事件发生时决策时；
- 使用 forecast 相关模块：策略或 MPC 需要看到未来天气、场景和重规划信息时。

观察中若包含未来天气或波高预测，训练、验证和测试必须采用一致的信息可得性假设，
避免把测试期的未来真实值错误地暴露给策略。
