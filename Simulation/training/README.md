# `Simulation.training`

本目录包含训练入口。当前的 `train.py` 使用 `sb3_contrib.MaskablePPO` 训练一个
集中式策略，并在相同场景种子下与闲置和贪心船运基线比较。

| 文件 | 职责 |
| --- | --- |
| `train.py` | 构建真实网络环境、训练 MaskablePPO、执行基线对比并格式化指标输出。 |

训练流程：

```text
make_native_env() → CCSGymEnv → MaskablePPO.learn()
                                      │
                                      ▼
                    compare() → metrics.evaluate() → 策略对比表
```

`make_native_env()` 集中配置回合长度、经济参数、捕集扰动、天气/波高选项和奖励。
`train_ppo()` 负责创建并训练模型。`compare()` 使用固定评估种子，对比 `idle`、
`greedy_shuttle`、随机 PPO 和确定性 PPO 的封存量、放空量及成本。

训练前应先验证物理环境和场景生成器。评估时应使用独立且固定的种子集合，避免将
训练场景表现误认为策略泛化能力。
