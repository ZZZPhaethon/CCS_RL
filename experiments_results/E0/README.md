# E0 物理仿真层验证结果

## 结论

E0 状态：**PASS**。

- Supplementary Table S1：20/20 项通过；
- 自动化回归：179/179 项通过；
- 720 h 全系统最大质量守恒误差：
  6.158e-08 t；
- 720 h hard physical violations：0；
- 当前末状态 common compact cleanup cost：
  EUR 240,783.70；
- 总运行时间：5.53 s。

## 目录

- `supplementary_table_s1.csv/.md`：验证项目、容差、观测值和结论；
- `figures/figure_s1_mass_balance_inventory.*`：Figure S1；
- `source_data/figure_s1_timeseries.csv`：Figure S1 源数据；
- `source_data/simplified_cases.csv`：五类简化案例和正式轨迹摘要；
- `automated_tests/`：pytest stdout、JUnit XML 和测试目标；
- `summary.json`：机器可读总结果；
- `config_snapshot.json`：协议、版本和哈希；
- `terminal_cleanup_validation.json`：共同末端核算检查。

## 范围

E0 验证仿真器及共同核算边界，不比较控制器性能，也不用于选择
Iterative Q 的 future representation 或超参数。正式 test seeds 未被访问。
