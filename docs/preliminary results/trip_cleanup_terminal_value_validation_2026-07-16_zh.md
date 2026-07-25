# 纯航次 cleanup 末端价值：第一阶段独立验证

日期：2026-07-16

## 验证目标

本阶段不接入168 h rolling MILP，只验证纯航次末端价值是否：

1. 对固定末状态满足质量守恒、容量和整数航次约束；
2. 使用与CCSEnv一致的经济成本分解；
3. 能正确区分几个具有明确优劣关系的状态；
4. 对应一个协调、可执行的CCSEnv cleanup，而不是复刻旧greedy的重复追船行为。

末端定义为：未来不再新增capture、无天气和设备扰动，全部剩余CO2完成储存，参与cleanup的船最终空载停靠在最近emitter。

## 500状态结构审计

对500个可达窗口末状态分别建立39变量、30约束的紧凑航次MILP，并用CPLEX求解。

| 检查项 | 结果 |
|---|---:|
| CPLEX Optimal | 500/500 |
| 最大源端质量守恒误差 | 5.0e-8 t |
| 最大航次容量违反 | 0 t |
| 最大整数性违反 | 0 |
| 最大成本分解误差 | 1.2e-10 EUR |

结构审计通过。

## 明确状态排序测试

| 测试 | 较差状态成本 | 较好状态成本 | 差值 | 结果 |
|---|---:|---:|---:|---:|
| 同为7,500 t：Yara vs Brevik | EUR 90,827.54 | EUR 83,770.16 | EUR 7,057.38 | 通过 |
| Brevik跨船容量：7,600 t vs 7,400 t | EUR 106,186.54 | EUR 82,941.05 | EUR 23,245.49 | 通过 |
| 同为7,500 t：Brevik源端 vs terminal | EUR 83,770.16 | EUR 3,075.00 | EUR 80,695.16 | 通过 |

三个测试均得到预期排序。terminal状态只需计入尚未发生的reconditioning；源端状态还需conditioning、装载、航运、卸载和最终回源。

## 20状态协调CCSEnv replay

从5个seed、5种已保存策略和4个不同末端小时中分层选取20个状态。根据纯航次MILP的吨数和航次分配，使用带库存预订的协调执行器生成逐小时native actions并在CCSEnv replay。

最初版本出现过EUR 20k--66k的额外燃油。诊断发现，未分配任务的空船在有库存emitter执行WAIT时会触发CCSEnv自动装载，从而产生非计划航次。这不是航次成本公式的误差。修复方法是：

- 将本地预订航次转移给已抵达该源端的空船；
- 暂时无任务的空船在terminal staging，不提前回到仍有库存的源端；
- 在等成本航次分配中优先使用已经位于或驶向有库存源端的船。

修复后结果：

| 检查项 | 结果 |
|---|---:|
| 完成全部储存并恢复ready | 20/20 |
| cleanup vent = 0 | 20/20 |
| 预测平均成本 | EUR 252,386.82 |
| replay平均成本 | EUR 251,992.44 |
| 平均误差（replay - 预测） | -EUR 394.38 |
| MAE | EUR 1,390.72 |
| MAE / replay平均成本 | 0.552% |
| 最大绝对误差 | EUR 4,151.40 |
| 最大单状态相对误差 | 2.066% |

conditioning、reconditioning、loading和unloading在20个状态中的最大差异均低于2.2e-10 EUR。全部剩余差异来自航行小时取整，范围为-10到+8 h；按EUR 415.14/h换算，最大为EUR 4,151.40。

## 结论

第一阶段独立验证通过。纯航次末端价值：

- 能计算完整cleanup Operating Cost；
- 在500个状态上满足质量、容量、整数性和成本一致性；
- 能识别距离、跨船容量和库存位置造成的状态优劣；
- 在20个协调可执行replay上全部清空、全部0 vent；
- 成本MAE为0.552%，剩余误差仅来自1 h离散仿真的航行取整。

该结果支持进入第二阶段：以可选配置把纯航次末端变量连接到168 h主MILP，在少量窗口上测试求解时间、状态、gap和前24 h CCSEnv replay。正式末端目标仅保留纯航次末端价值。

## 复现文件

- 航次模型：`experiments/evaluate_trip_cleanup_terminal_value.py`
- 独立验证：`experiments/validate_trip_cleanup_terminal_value.py`
- 500状态评估输入：`output/trip_cleanup_terminal_value_500_v5_2026-07-16/evaluation_states.csv`
- 500状态解：`output/trip_cleanup_terminal_value_500_v5_2026-07-16/predictions.csv`
- 20状态replay：`output/trip_cleanup_terminal_value_validation_v2_2026-07-16/coordinated_replay_20states.csv`
- 排序测试：`output/trip_cleanup_terminal_value_validation_v2_2026-07-16/ordering_tests.csv`
- 验证摘要：`output/trip_cleanup_terminal_value_validation_v2_2026-07-16/summary.json`
